"""FlexCode-style orthogonal-series conditional density estimator (non-parametric).

Expands the conditional density in an orthonormal basis (Izbicki & Lee 2017):

    p(y | x) = sum_{j=0}^{J-1} beta_j(x) * phi_j(z),     z = (y - y_min)/(y_max - y_min) in [0, 1]

where ``{phi_j}`` is the orthonormal cosine basis on ``[0, 1]``
(``phi_0 = 1``, ``phi_j(z) = sqrt(2) cos(j pi z)``).  Because
``beta_j(x) = E[phi_j(Y) | X = x]``, conditional density estimation reduces to
``J`` **regressions** of the basis-transformed response on the parents — the
key idea that lets FlexCode inherit high-dimensional regression machinery and
degrade gracefully as the number of parents grows.  Here the coefficient
functions share a small MLP trunk (torch-native, GPU, differentiable); a root
node uses the closed-form coefficients ``beta_j = mean_i phi_j(z_i)``.

The raw truncated series can dip below zero and need not integrate to one after
clipping, so (following FlexZBoost) the density is clipped at zero, optionally
sharpened by an exponent ``alpha``, and renormalised on a grid; the same grid
drives inverse-CDF sampling.

Scope: univariate child (``D_x = 1``), which covers the typical continuous BN
node.  Multivariate children raise a clear error (tensor-product bases are a
future extension).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Distribution

from nbn.learning.warm_start import check_branch, check_shapes
from nbn.learning.weighting import (
    select,
    validate_weights,
    weighted_mean,
    weighted_moments,
)
from nbn.mechanisms.base import Mechanism
from nbn.mechanisms.parametric.mdn import _build_mlp
from nbn.utils.batching import _sanitise_parents, ensure_2d, flatten_samples


class _FlexCodeDistribution(Distribution):
    has_rsample = False
    arg_constraints: dict = {}

    def __init__(self, mech: FlexCodeMechanism, parents: torch.Tensor | None, b: int) -> None:
        self._mech = mech
        self._parents = parents
        super().__init__(batch_shape=torch.Size([b]), event_shape=torch.Size([1]), validate_args=False)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._mech.log_prob(value, self._parents)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:  # type: ignore[override]
        n = 1
        for s in sample_shape:
            n *= int(s)
        out = self._mech.sample(self._parents, n=n).permute(1, 0, 2)
        return out.reshape(*sample_shape, *out.shape[1:]) if sample_shape else out[0]


class FlexCodeMechanism(Mechanism):
    """Orthogonal-series conditional density estimator (cosine basis).

    Parameters
    ----------
    n_basis:
        Number of basis terms ``J`` (truncation level; higher → lower bias,
        higher variance).
    hidden, activation:
        MLP trunk geometry for the coefficient regressors.
    epochs, lr, batch_size:
        Training schedule for the coefficient regression (ignored for root
        nodes, which use closed-form coefficients).
    grid_size:
        Resolution of the ``[0, 1]`` grid used for renormalisation / sampling.
    margin:
        Fractional padding added below ``y_min`` and above ``y_max`` so the
        observed support sits strictly inside ``[0, 1]``.
    sharpen:
        FlexZBoost sharpening exponent ``alpha`` applied to the clipped
        density before renormalisation (``1.0`` = none). Pass ``"auto"`` to
        select it by held-out negative log-likelihood at fit time.
    min_density:
        Floor for the density (numerical-stability / finite log-prob).
    """

    is_discrete: bool = False
    supports_weights: bool = True
    warm_start_is_noop: bool = False

    def __init__(
        self,
        n_basis: int = 31,
        hidden: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 512,
        grid_size: int = 256,
        margin: float = 0.05,
        sharpen: float | str = 1.0,
        min_density: float = 1e-6,
    ) -> None:
        super().__init__()
        if isinstance(sharpen, str) and sharpen != "auto":
            raise ValueError(f"sharpen must be a float or 'auto', got {sharpen!r}")
        self.n_basis = int(n_basis)
        self.hidden = tuple(hidden)
        self.activation = activation
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.grid_size = int(grid_size)
        self.margin = float(margin)
        self.sharpen = sharpen if sharpen == "auto" else float(sharpen)
        self.min_density = float(min_density)
        self.output_dim = 1
        self._d_pa = 0
        self.net: nn.Module | None = None
        self._root_coef: nn.Parameter | None = None
        self.register_buffer("_y_min", None)
        self.register_buffer("_y_max", None)
        self.register_buffer("_pa_mean", None)
        self.register_buffer("_pa_std", None)
        self.register_buffer("_grid_z", None)       # [G]
        self.register_buffer("_basis_grid", None)    # [G, J]

    # ------------------------------------------------------------------
    # Basis
    # ------------------------------------------------------------------
    def _basis(self, z: torch.Tensor) -> torch.Tensor:
        """Orthonormal cosine basis. ``z`` ``[...]`` → ``[..., J]``."""
        j = torch.arange(self.n_basis, device=z.device, dtype=z.dtype)  # [J]
        ang = math.pi * z.unsqueeze(-1) * j                              # [..., J]
        phi = math.sqrt(2.0) * torch.cos(ang)
        phi[..., 0] = 1.0
        return phi

    def _scale_to_z(self, y: torch.Tensor) -> torch.Tensor:
        return ((y - self._y_min) / (self._y_max - self._y_min)).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        weights: torch.Tensor | None = None,
        warm_start: bool = False,
        **kwargs,
    ) -> dict:
        kwargs["weights"] = weights
        if self.sharpen == "auto":
            # The search sub-fits on a split and then refits on the full data,
            # discarding whatever was there.  That is exactly the silent
            # rebuild warm_start exists to prevent, so refuse rather than do
            # it quietly.  Unreachable through the normal call sequence: the
            # first fit resolves ``sharpen`` to a float, so a fitted mechanism
            # never re-enters this branch unless the attribute was reset by
            # hand.
            if warm_start and self.is_fitted:
                raise ValueError(
                    "FlexCodeMechanism.fit_local: warm_start=True with "
                    "sharpen='auto'.  Selecting the exponent refits the "
                    "coefficients from scratch on a held-out split, which "
                    "would discard the parameters you asked to continue from. "
                    "Fit once without warm_start (that resolves sharpen to a "
                    "float), or set a fixed sharpen before warm-starting."
                )
            best = self._select_sharpen(x, parents, **kwargs)
            info = self._fit_core(x, parents, **kwargs)  # final coeffs on full data
            self.sharpen = best
            return info
        return self._fit_core(x, parents, warm_start=warm_start, **kwargs)

    def _select_sharpen(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        candidates: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5),
        val_frac: float = 0.25,
        **kwargs,
    ) -> float:
        """Pick ``sharpen`` by held-out NLL: fit coeffs on a sub-split, score the
        exponent on the held-out points (sharpen is post-hoc, so no refitting)."""
        y = ensure_2d(x).float()
        if y.shape[1] != 1:
            return 1.0  # multivariate is rejected by _fit_core anyway
        n = y.shape[0]
        gen = torch.Generator(device="cpu").manual_seed(0)
        perm = torch.randperm(n, generator=gen)
        n_val = max(8, int(round(val_frac * n)))
        if n_val >= n:
            return 1.0
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
        has_pa = parents is not None and parents.shape[-1] > 0
        pa = ensure_2d(parents).float() if has_pa else None
        # The holdout must carry its own weights on both sides: the sub-fit is
        # weighted like the real fit, and the score that picks ``sharpen`` is a
        # weighted NLL.  Selecting an exponent against an unweighted holdout
        # would tune the estimator for a data distribution the caller did not
        # ask for.
        w_all = validate_weights(
            kwargs.pop("weights", None), n, where="FlexCodeMechanism.fit_local",
        )
        sub_kwargs = dict(kwargs)
        sub_kwargs["weights"] = None if w_all is None else w_all[fit_idx]
        w_val = None if w_all is None else w_all[val_idx]
        self.sharpen = 1.0
        self._fit_core(y[fit_idx], pa[fit_idx] if has_pa else None, **sub_kwargs)
        best_s, best_nll = 1.0, float("inf")
        for s in candidates:
            self.sharpen = float(s)
            with torch.no_grad():
                nll = float(weighted_mean(
                    -self.log_prob(y[val_idx], pa[val_idx] if has_pa else None),
                    w_val,
                ))
            if nll == nll and nll < best_nll:  # nll == nll rejects NaN
                best_nll, best_s = nll, float(s)
        self.sharpen = "auto"  # restore request; caller resolves to best_s
        return best_s

    def _fit_core(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        warm_start: bool = False,
        **kwargs,
    ) -> dict:
        y = ensure_2d(x).float()
        if y.shape[1] != 1:
            raise NotImplementedError(
                "FlexCodeMechanism supports univariate children (D_x=1); "
                f"got D_x={y.shape[1]}. Use MDN/NormalizingFlow for multivariate."
            )
        n = y.shape[0]
        device = y.device
        is_root = parents is None or parents.shape[-1] == 0
        warm = bool(warm_start) and self.is_fitted
        if warm:
            where = "FlexCodeMechanism.fit_local"
            check_branch(where, was_root=self._d_pa == 0, is_root=is_root)
            if not is_root:
                check_shapes(where, {
                    "d_pa": (self._d_pa, ensure_2d(parents).shape[1]),
                })
        w_vec = validate_weights(
            kwargs.get("weights"), n, where="FlexCodeMechanism.fit_local",
        )
        if w_vec is not None:
            w_vec = w_vec.to(device)
        epochs = int(kwargs.get("epochs", self.epochs))
        lr = float(kwargs.get("lr", self.lr))
        batch_size = int(kwargs.get("batch_size", self.batch_size))

        # ``(_y_min, _y_max)`` fix the z-space the basis is evaluated in, and
        # ``net`` predicts coefficients *of that basis*.  Recomputing them
        # under a warm start would not merely rescale an input, it would
        # reinterpret every learned coefficient -- the sharper half of the
        # frozen-buffer rule.  See nbn/learning/warm_start.py.
        if not warm:
            y_min = y.min()
            y_max = y.max()
            span = (y_max - y_min).clamp_min(1e-6)
            self._y_min = y_min - self.margin * span
            self._y_max = y_max + self.margin * span
        z = self._scale_to_z(y).squeeze(-1)                  # [N]
        targets = self._basis(z)                             # [N, J]

        grid = torch.linspace(0.0, 1.0, self.grid_size, device=device)
        self._grid_z = grid
        self._basis_grid = self._basis(grid)                 # [G, J]

        if parents is None or parents.shape[-1] == 0:
            self._d_pa = 0
            self._pa_mean = torch.zeros(1, 0, device=device)
            self._pa_std = torch.ones(1, 0, device=device)
            # Closed form: beta_j = E[phi_j(Z)] -- a *weighted* expectation
            # when the caller supplies per-sample multiplicities.  This branch
            # used ``targets.mean(0)`` and dropped ``w_vec`` on the floor, so a
            # root FlexCode node silently returned the unweighted estimator
            # while the class advertised ``supports_weights = True``.  That is
            # a wrong answer rather than a missing feature: the fit converges,
            # to the wrong coefficients.  ``weighted_moments(...)[0]`` is the
            # per-column weighted mean and reduces to ``targets.mean(0)``
            # exactly when ``w_vec is None``.
            #
            # Being closed form, this branch recomputes even under
            # warm_start=True (reported as ``warm_started: False``): freezing
            # it would stop a root node responding to the E-step.  The z-space
            # it is expressed in *is* frozen above, so the recomputed
            # coefficients stay comparable with the previous ones.
            self._root_coef = nn.Parameter(weighted_moments(targets, w_vec)[0])
        else:
            pa = ensure_2d(parents).float().to(device)
            self._d_pa = pa.shape[1]
            if not warm:
                pa_m, pa_s = weighted_moments(pa, w_vec, unbiased=True)
                self._pa_mean = pa_m.unsqueeze(0)
                self._pa_std = pa_s.unsqueeze(0).clamp_min(1e-3)
            pa_std = (pa - self._pa_mean) / self._pa_std
            if not warm:
                self.net = _build_mlp(self._d_pa, self.hidden, self.n_basis, self.activation).to(device)
            # Fresh optimiser either way -- Adam's moments are not in
            # state_dict(), so persisting them would survive a caller's
            # load_state_dict revert of a rejected step.
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            self.train()
            for _ in range(epochs):
                perm = torch.randperm(n, device=device)
                for i in range(0, n, batch_size):
                    idx = perm[i:i + batch_size]
                    pred = self.net(pa_std[idx])             # [b, J]
                    # Weighted least squares on the basis coefficients: the
                    # per-row squared error is reduced by a weighted mean over
                    # the same idx the batch was drawn with.
                    loss = weighted_mean(
                        (pred - targets[idx]).pow(2).mean(dim=-1),
                        select(w_vec, idx),
                    )
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                    opt.step()
            self.eval()
        return {
            "n_train": int(n), "n_basis": self.n_basis, "d_pa": int(self._d_pa),
            "warm_started": warm and not is_root,
        }

    @property
    def is_fitted(self) -> bool:
        return self._y_min is not None

    # ------------------------------------------------------------------
    # Coefficients + density
    # ------------------------------------------------------------------
    def _coef(self, parents: torch.Tensor | None, m: int) -> torch.Tensor:
        """Return coefficients ``[m, J]``."""
        if self._d_pa == 0 or parents is None:
            return self._root_coef.unsqueeze(0).expand(m, -1)
        pa = _sanitise_parents(ensure_2d(parents).float(), mech_name="FlexCode")
        pa_std = (pa.to(self._pa_mean.device) - self._pa_mean) / self._pa_std
        assert self.net is not None
        return self.net(pa_std)

    def _density_z(self, z: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        """Unnormalised (clipped, sharpened) density at ``z`` ``[m]`` given ``coef`` ``[m, J]``."""
        phi = self._basis(z)                                 # [m, J]
        dens = (coef * phi).sum(-1).clamp_min(0.0)           # [m]
        if self.sharpen != 1.0:
            dens = dens.pow(self.sharpen)
        return dens

    def _grid_density(self, coef: torch.Tensor) -> torch.Tensor:
        """Per-query density on the grid ``[m, G]`` (clipped + sharpened)."""
        dens = (coef.unsqueeze(1) * self._basis_grid.unsqueeze(0)).sum(-1).clamp_min(0.0)  # [m, G]
        if self.sharpen != 1.0:
            dens = dens.pow(self.sharpen)
        return dens

    def _normaliser(self, coef: torch.Tensor) -> torch.Tensor:
        """Trapezoidal integral of the grid density over ``[0, 1]`` → ``[m]``."""
        gd = self._grid_density(coef)
        dz = 1.0 / (self.grid_size - 1)
        return torch.trapz(gd, dx=dz, dim=1).clamp_min(self.min_density) if hasattr(torch, "trapz") \
            else torch.trapezoid(gd, dx=dz, dim=1).clamp_min(self.min_density)

    # ------------------------------------------------------------------
    # Distribution interface
    # ------------------------------------------------------------------
    def forward(self, parents: torch.Tensor | None) -> _FlexCodeDistribution:
        b = 1 if parents is None else ensure_2d(parents).shape[0]
        return _FlexCodeDistribution(self, parents, b)

    def log_prob(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        squeeze_s = False
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_s = True
        b, s, _ = x.shape
        if parents is not None and parents.dim() == 2:
            parents = parents.unsqueeze(1).expand(-1, s, -1)
        pa_flat = None if parents is None else flatten_samples(parents)[0]
        m = b * s
        coef = self._coef(pa_flat, m)                        # [m, J]
        y = x.reshape(m, 1).to(self._y_min.device)
        z = self._scale_to_z(y).squeeze(-1)                  # [m]
        dens_z = self._density_z(z, coef)                    # [m]
        norm = self._normaliser(coef)                        # [m]
        span = (self._y_max - self._y_min)
        log_p = torch.log(dens_z.clamp_min(self.min_density)) - torch.log(norm) - torch.log(span)
        log_p = log_p.reshape(b, s)
        return log_p.squeeze(1) if squeeze_s else log_p

    def sample(self, parents: torch.Tensor | None, n: int = 1) -> torch.Tensor:
        b = 1 if parents is None else ensure_2d(parents).shape[0]
        coef = self._coef(parents, b)                        # [B, J]
        gd = self._grid_density(coef)                        # [B, G]
        cdf = torch.cumsum(gd, dim=1)
        cdf = cdf / cdf[:, -1:].clamp_min(self.min_density)  # [B, G] in [0,1]
        u = torch.rand(b, n, device=cdf.device)
        # inverse-CDF via searchsorted per row
        idx = torch.searchsorted(cdf, u)                     # [B, n]
        idx = idx.clamp(max=self.grid_size - 1)
        z = self._grid_z[idx]                                # [B, n]
        y = self._y_min + z * (self._y_max - self._y_min)
        return y.unsqueeze(-1)                               # [B, n, 1]
