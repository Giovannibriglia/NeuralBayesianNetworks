"""Structural mechanisms: ``X := f(pa, U)`` with an explicit noise law.

A :class:`StructuralMechanism` is still an NBN :class:`Mechanism` — it returns
the conditional distribution ``P(X | pa)`` from ``forward`` — so fitting,
sampling, ``do=`` queries and inference engines use it unchanged.  On top of
that it exposes the structural equation:

``location(pa)``
    The deterministic part ``g(pa)``.
``structural(pa, u)``
    ``f(pa, u)``: the value ``X`` takes for noise ``u``.
``abduct(x, pa, n_samples)``
    Noise values that explain ``x``, ``[n_samples, B, D]``: the exact residual
    (repeated) for continuous additive noise, posterior draws for discrete
    noise.  :meth:`DiscreteAdditiveNoiseMechanism.noise_posterior` and
    :meth:`AdditiveNoiseMechanism.residual` give the underlying objects.
``noise``
    The noise law (an ``nn.Module``; one object shared by tied mechanisms).

The split of ``P(X | pa)`` into ``g`` and ``U`` is a modelling convention:
data identify only the conditional distribution.  It matters for
counterfactuals and for noise tying, not for interventional queries.
"""
from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import List, Literal

import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution

from nbn.learning.weighting import validate_weights, weighted_mean, weighted_moments
from nbn.mechanisms.base import Mechanism
from nbn.scm.fitting import (
    child_law,
    fit_offset_law,
    offset_transfer,
    rows_table,
    unreachable_cells,
)
from nbn.scm.noise import GaussianMixtureNoise, GaussianNoise, OffsetNoise
from nbn.uncertainty.counts import parent_row_index, scatter_counts
from nbn.utils.batching import ensure_2d
from nbn.utils.device import accum_dtype, linalg_device, to_device_dtype

_TINY = torch.finfo(torch.float32).tiny


class StructuralMechanism(Mechanism):
    """A mechanism with an explicit structural equation and noise law."""

    noise: nn.Module

    @abstractmethod
    def location(self, parents: torch.Tensor | None) -> torch.Tensor:
        """The deterministic part ``g(pa)``."""

    @abstractmethod
    def structural(self, parents: torch.Tensor | None, noise: torch.Tensor) -> torch.Tensor:
        """``f(pa, u)``, batched over rows; ``noise`` is ``[B, D]``."""

    @abstractmethod
    def abduct(self, x: torch.Tensor, parents: torch.Tensor | None, n_samples: int = 1) -> torch.Tensor:
        """``[n_samples, B, D]`` noise values consistent with ``x`` given ``pa``."""


# ===========================================================================
# Discrete: x = clip(g(pa) + U, 0, K - 1)
# ===========================================================================


class DiscreteAdditiveNoiseMechanism(StructuralMechanism):
    """``x = clip(g(pa) + U, 0, K-1)``: a learned location table plus offset noise.

    ``g`` is one integer location per parent configuration, in
    ``-m .. K - 1 + m`` (a location past the value range saturates at the
    edge, which is a law of its own); ``U`` is a categorical law over integer
    offsets ``-m .. m`` (:class:`OffsetNoise`).  With one
    noise law for every parent configuration (and, when tied, for several
    nodes), the table has far fewer free parameters than a full CPT.

    Fitting (:func:`nbn.scm.fitting.fit_offset_law`) alternates an EM step for
    the noise law and the best location per parent configuration until the
    log-likelihood stops improving.  Rows never observed keep the middle
    location; their prediction is an extrapolation and the probability sets
    report them as ``unseen``.

    Parameters
    ----------
    max_offset:
        Largest ``|u|``.  Default ``K - 1`` (any shift is possible).
    noise:
        An existing :class:`OffsetNoise` to share (tying).  Default: a private one.
    """

    is_discrete = True
    supports_weights = True
    # The fit starts from fixed laws (multi-start), never from the previous
    # parameters, so warm_start is an accepted no-op.
    warm_start_is_noop = True

    def __init__(
        self,
        max_offset: int | None = None,
        noise: OffsetNoise | None = None,
        n_iter: int = 200,
        rtol: float = 1e-10,
    ) -> None:
        super().__init__()
        self.max_offset: int | None = max_offset
        self.noise = noise  # created at fit time when None
        self.n_iter = int(n_iter)
        self.rtol = float(rtol)
        self.output_dim = 1
        self._n_classes = 0
        self._parent_cards: List[int] = []
        self.register_buffer("_locations", None)  # [R] long
        self.register_buffer("_counts", None)     # [R, K] raw (weighted) counts

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        parent_cards: List[int] | None = None,
        n_classes: int | None = None,
        weights: torch.Tensor | None = None,
        warm_start: bool = False,
        **kwargs,
    ) -> dict:
        counts = self.prepare_counts(x, parents, parent_cards, n_classes, weights)
        (locations,), q, objective = fit_offset_law(
            [self.transfer()], [counts], n_iter=self.n_iter, rtol=self.rtol
        )
        self.locations = locations
        self.noise.set_probs(q)
        self.check_reachable(counts)
        return {"mean_log_likelihood": float(objective), "warm_started": False}

    def check_reachable(self, counts: torch.Tensor) -> None:
        """Raise if the data hold values no offset can produce from the fitted locations."""
        bad = unreachable_cells(self.transfer(), counts, self.locations)
        if bool(bad.any()):
            raise ValueError(
                f"{int(counts[bad].sum())} observations lie more than max_offset="
                f"{self.noise.max_offset} away from their fitted location; increase "
                "max_offset (or check the data)."
            )

    def prepare_counts(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        parent_cards: List[int] | None = None,
        n_classes: int | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fix the table's shape from the data and declarations; return its counts ``[R, K]``."""
        x = x.long().reshape(-1)
        n = x.shape[0]
        observed_k = int(x.max()) + 1 if n else 1
        k = observed_k if n_classes is None else max(int(n_classes), observed_k)
        pa = self._parents_2d(parents, n, x.device)
        observed_pc = [int(pa[:, d].max()) + 1 if n else 1 for d in range(pa.shape[1])]
        cards = observed_pc if parent_cards is None else [
            max(int(c), o) for c, o in zip(parent_cards, observed_pc)
        ]
        self._n_classes, self._parent_cards = k, list(cards)
        if self.noise is None:
            self.noise = OffsetNoise(self.offset_range(k)).to(x.device)

        w = validate_weights(weights, n, where=f"{type(self).__name__}.fit_local", device=x.device)
        counts = scatter_counts(parent_row_index(pa, self._parent_cards), x, self.n_rows, k, w)
        self._counts = counts
        return counts

    @property
    def noise_max_offset(self) -> int:
        """Largest ``|u|`` of the (possibly shared) noise law this mechanism uses."""
        return int(self.noise.max_offset)

    def offset_range(self, n_classes: int) -> int:
        """Largest ``|u|`` this mechanism uses for a child with ``n_classes`` values."""
        return n_classes - 1 if self.max_offset is None else int(self.max_offset)

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        n = 1
        for c in self._parent_cards:
            n *= c
        return n

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def n_locations(self) -> int:
        """Number of possible locations, ``K + 2 m``."""
        return self._n_classes + 2 * self.noise.max_offset

    @property
    def locations(self) -> torch.Tensor:
        """Location *index* of every parent configuration, ``[R]`` long (``g = index - m``)."""
        if self._locations is None:
            raise RuntimeError(f"{type(self).__name__}: call fit_local first.")
        return self._locations

    @locations.setter
    def locations(self, value: torch.Tensor) -> None:
        self._locations = value.long()

    @property
    def is_fitted(self) -> bool:
        return self._locations is not None

    def transfer(self) -> torch.Tensor:
        """``[G, K, M]`` 0/1 map from the noise law to the child law at each location."""
        return offset_transfer(self._n_classes, self.noise.offsets)

    def child_law(self) -> torch.Tensor:
        """``P(x = k | location index l)``, ``[G, K]``."""
        return child_law(self.transfer(), self.noise.probs)

    def table(self) -> torch.Tensor:
        """``P(X | pa)`` for every parent configuration, ``[R, K]``."""
        return rows_table(self.child_law(), self.locations)

    def empty_like(self, noise: OffsetNoise | None = None) -> DiscreteAdditiveNoiseMechanism:
        """An unfitted copy with the same table shape."""
        fresh = type(self)(
            max_offset=self.noise.max_offset,
            noise=OffsetNoise(self.noise.max_offset) if noise is None else noise,
            n_iter=self.n_iter, rtol=self.rtol,
        ).to(self.noise.logits.device)
        fresh._n_classes, fresh._parent_cards = self._n_classes, list(self._parent_cards)
        return fresh

    # ------------------------------------------------------------------
    # Structural interface
    # ------------------------------------------------------------------

    def location(self, parents: torch.Tensor | None) -> torch.Tensor:
        """The location value ``g(pa)`` (may lie outside ``0 .. K-1``)."""
        return self.locations[self._rows(parents)] - self.noise.max_offset

    def structural(self, parents: torch.Tensor | None, noise: torch.Tensor) -> torch.Tensor:
        loc = self.location(parents)
        return (loc + noise.long().reshape(loc.shape)).clamp(0, self._n_classes - 1)

    def noise_posterior(self, x: torch.Tensor, parents: torch.Tensor | None) -> Categorical:
        """``P(U = u | x, pa) ∝ q_u 1[clip(g(pa) + u) = x]`` over the offsets; any batch shape."""
        x_idx = self._as_index(x)
        loc = self.location(parents)
        if loc.dim() < x_idx.dim():
            loc = loc.unsqueeze(-1)
        loc = loc.expand(x_idx.shape)
        reachable = (loc.unsqueeze(-1) + self.noise.offsets).clamp(0, self._n_classes - 1)
        consistent = reachable == x_idx.unsqueeze(-1)
        if not bool(consistent.any(-1).all()):
            raise ValueError(
                f"Some x cannot be produced from g(pa) with max_offset={self.noise.max_offset}: "
                "the model gives them probability 0, so their noise has no posterior."
            )
        return Categorical(probs=consistent.to(self.noise.probs.dtype) * self.noise.probs)

    def abduct(self, x: torch.Tensor, parents: torch.Tensor | None, n_samples: int = 1) -> torch.Tensor:
        """Posterior draws of the offset, ``[n_samples, *batch, 1]`` float."""
        draws = self.noise_posterior(x, parents).sample((n_samples,))
        return self.noise.offsets[draws].to(torch.float32).unsqueeze(-1)

    # ------------------------------------------------------------------
    # Distribution interface (same shapes as CategoricalTableMechanism)
    # ------------------------------------------------------------------

    def forward(self, parents: torch.Tensor | None) -> Distribution:
        return Categorical(probs=self.table()[self._rows(parents)])

    def log_prob(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        """``log P(x | pa)`` for ``x`` of shape ``[B]``, ``[B, 1]`` or ``[B, S, 1]``."""
        x_idx = self._as_index(x)
        rows = self._rows(parents)
        if rows.dim() < x_idx.dim():
            rows = rows.unsqueeze(-1)
        log_table = torch.log(self.table().clamp_min(_TINY))
        return log_table[rows.expand_as(x_idx), x_idx]

    def sample(self, parents: torch.Tensor | None, n: int = 1) -> torch.Tensor:
        draws = Categorical(probs=self.table()[self._rows(parents)]).sample((n,))  # [n, B]
        return draws.T.unsqueeze(-1).to(torch.float32)

    def tabulate(self, parent_cards: list[int] | None = None) -> torch.Tensor:
        log_table = torch.log(self.table().clamp_min(_TINY))
        if not self._parent_cards:
            return log_table[0]
        return log_table.reshape(*self._parent_cards, self._n_classes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rows(self, parents: torch.Tensor | None) -> torch.Tensor:
        """Row index per batch element: ``[B]`` (``[B, S]`` for 3-D parents); ``[1]`` for ``None``."""
        device = self.locations.device
        if parents is None:
            return torch.zeros(1, dtype=torch.long, device=device)
        if parents.shape[-1] == 0:
            return torch.zeros(parents.shape[:-1], dtype=torch.long, device=device)
        return parent_row_index(ensure_2d(parents).to(device), self._parent_cards)

    @staticmethod
    def _as_index(x: torch.Tensor) -> torch.Tensor:
        """Child values as long indices without a trailing singleton: ``[B]`` or ``[B, S]``."""
        x_idx = x.long()
        if x_idx.dim() >= 2 and x_idx.shape[-1] == 1:
            x_idx = x_idx.squeeze(-1)
        return x_idx

    @staticmethod
    def _parents_2d(parents: torch.Tensor | None, n: int, device: torch.device) -> torch.Tensor:
        if parents is None or parents.shape[-1] == 0:
            return torch.zeros(n, 0, dtype=torch.long, device=device)
        return parents.long().reshape(n, -1)


# ===========================================================================
# Continuous: x = g(pa) + U
# ===========================================================================


class AdditiveNoiseMechanism(StructuralMechanism):
    """``x = g(pa) + U`` with ``g`` linear or an MLP and ``U`` a learned law.

    The noise is independent of the parents, so its law is one object shared
    by every parent configuration: the residuals ``x - g(pa)`` pool into a
    single fit.  The location is fitted by (weighted) least squares, which does
    not depend on the noise law, so tying several mechanisms only refits the
    shared law on their pooled residuals.

    Parameters
    ----------
    location:
        ``"linear"`` (closed-form weighted ridge regression) or ``"mlp"``
        (two hidden layers, weighted mean-squared error, Adam).
    noise:
        ``"gaussian"``, ``"mixture"``, or a noise module to share.
    epochs, lr, batch_size:
        MLP training budget (``fit(..., epochs=...)`` overrides it).
    """

    is_discrete = False
    supports_weights = True
    warm_start_is_noop = False  # the MLP location continues from its weights

    def __init__(
        self,
        location: Literal["linear", "mlp"] = "linear",
        noise: Literal["gaussian", "mixture"] | nn.Module = "gaussian",
        n_components: int = 3,
        hidden: int = 64,
        epochs: int = 300,
        lr: float = 1e-2,
        batch_size: int | None = None,
        ridge: float = 1e-6,
    ) -> None:
        super().__init__()
        if location not in ("linear", "mlp"):
            raise ValueError(f"location must be 'linear' or 'mlp', got {location!r}.")
        self.location_kind = location
        self.noise_spec = noise
        self.noise = noise if isinstance(noise, nn.Module) else None
        self.n_components = int(n_components)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = batch_size
        self.ridge = float(ridge)
        self.output_dim = 1
        self._input_dim = 0
        self.net: nn.Module | None = None  # nn.Linear or MLP, built at fit time
        self.register_buffer("_pa_mean", None)
        self.register_buffer("_pa_std", None)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        weights: torch.Tensor | None = None,
        warm_start: bool = False,
        epochs: int | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
        **kwargs,
    ) -> dict:
        x = ensure_2d(x).to(torch.float32)
        w = validate_weights(weights, x.shape[0], where=f"{type(self).__name__}.fit_local",
                             device=x.device)
        warm = self.fit_location(x, parents, w, warm_start=warm_start,
                                 epochs=epochs, lr=lr, batch_size=batch_size)
        self._ensure_noise(x.shape[1], x.device)
        metrics = self.noise.fit(self.residual(x, parents).detach(), w)
        return {**metrics, "warm_started": warm}

    def fit_location(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        weights: torch.Tensor | None,
        warm_start: bool = False,
        epochs: int | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
    ) -> bool:
        """Fit ``g`` only; ``None`` hyperparameters fall back to the constructor's.

        Returns whether parameters were carried over (warm start).
        """
        self.output_dim = x.shape[1]
        pa = self._parents_or_empty(parents, x)
        if self.location_kind == "linear" or pa.shape[1] == 0:
            self._fit_linear(x, pa, weights)
            return False
        return self._fit_mlp(
            x, pa, weights, warm_start,
            epochs=self.epochs if epochs is None else int(epochs),
            lr=self.lr if lr is None else float(lr),
            batch_size=self.batch_size if batch_size is None else batch_size,
        )

    def _fit_linear(self, x: torch.Tensor, pa: torch.Tensor, w: torch.Tensor | None) -> None:
        d_pa, d_x = pa.shape[1], x.shape[1]
        self._input_dim = d_pa
        self._pa_mean = torch.zeros(d_pa, device=x.device)
        self._pa_std = torch.ones(d_pa, device=x.device)
        if d_pa == 0:
            mean, _ = weighted_moments(x, w)
            weight, bias = torch.zeros(d_x, 0, device=x.device), mean.to(torch.float32)
        else:
            # Weighted ridge regression via the normal equations, widest dtype.
            solve_on = linalg_device(x.device)
            dtype = accum_dtype(solve_on)
            design = torch.cat([pa, torch.ones(pa.shape[0], 1, device=x.device)], 1)
            design = to_device_dtype(design, solve_on, dtype)
            target = to_device_dtype(x, solve_on, dtype)
            wd = design if w is None else design * to_device_dtype(w, solve_on, dtype).unsqueeze(-1)
            gram = design.T @ wd + self.ridge * torch.eye(d_pa + 1, device=solve_on, dtype=dtype)
            coef = torch.linalg.solve(gram, wd.T @ target).to(x.device, torch.float32)
            weight, bias = coef[:d_pa].T, coef[d_pa]
        with warnings.catch_warnings():
            # A root has no parents: nn.Linear(0, d) warns that initialising an
            # empty weight is a no-op.  The weights are overwritten just below.
            warnings.simplefilter("ignore", UserWarning)
            layer = nn.Linear(d_pa, d_x).to(x.device)
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        self.net = layer

    def _fit_mlp(
        self, x: torch.Tensor, pa: torch.Tensor, w: torch.Tensor | None,
        warm_start: bool, epochs: int, lr: float, batch_size: int | None,
    ) -> bool:
        d_pa, d_x = pa.shape[1], x.shape[1]
        warm = warm_start and isinstance(self.net, nn.Sequential) and self._input_dim == d_pa
        if warm_start and self.net is not None and not warm:
            raise ValueError(
                f"{type(self).__name__}: warm_start=True but the fitted location "
                f"does not match ({self._input_dim} parent dims fitted, {d_pa} given)."
            )
        if not warm:
            self._input_dim = d_pa
            mean, std = weighted_moments(pa, w, unbiased=True)
            self._pa_mean = mean.to(torch.float32)
            self._pa_std = std.to(torch.float32).nan_to_num(1.0).clamp_min(1e-6)
            self.net = nn.Sequential(
                nn.Linear(d_pa, self.hidden), nn.SiLU(),
                nn.Linear(self.hidden, self.hidden), nn.SiLU(),
                nn.Linear(self.hidden, d_x),
            ).to(x.device)
        z = (pa - self._pa_mean) / self._pa_std
        optimiser = torch.optim.Adam(self.net.parameters(), lr=lr)
        n = x.shape[0]
        size = n if batch_size is None else int(batch_size)
        for _ in range(epochs):
            order = torch.randperm(n, device=x.device)
            for start in range(0, n, size):
                idx = order[start:start + size]
                err = (self.net(z[idx]) - x[idx]).pow(2).sum(-1)
                loss = weighted_mean(err, None if w is None else w[idx])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        return warm

    def _ensure_noise(self, dim: int, device: torch.device) -> None:
        if self.noise is not None:
            return
        if self.noise_spec == "gaussian":
            self.noise = GaussianNoise(dim)
        elif self.noise_spec == "mixture":
            self.noise = GaussianMixtureNoise(dim, n_components=self.n_components)
        else:
            raise ValueError(f"Unknown noise spec {self.noise_spec!r}.")
        self.noise.to(device)

    # ------------------------------------------------------------------
    # Structural interface
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self.net is not None and self.noise is not None

    def location(self, parents: torch.Tensor | None) -> torch.Tensor:
        """``g(pa)``: ``[B, D_x]`` (or ``[B, S, D_x]``); ``[1, D_x]`` for ``None``."""
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__}: call fit_local first.")
        device, dtype = self._pa_mean.device, self._pa_mean.dtype  # follows .to(dtype)
        if parents is None:
            return self.net(torch.zeros(1, 0, device=device, dtype=dtype))
        if parents.shape[-1] == 0:
            return self.net(torch.zeros(*parents.shape[:-1], 0, device=device, dtype=dtype))
        z = (to_device_dtype(parents, device, dtype) - self._pa_mean) / self._pa_std
        return self.net(z)

    def structural(self, parents: torch.Tensor | None, noise: torch.Tensor) -> torch.Tensor:
        return self.location(parents) + noise

    def residual(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        """Exact noise ``u = x - g(pa)``, ``[B, D]`` (``[B, S, D]`` for 3-D ``x``)."""
        x = self._as_own(x)
        if x.dim() == 3 and parents is not None and parents.dim() == 2:
            parents = parents.unsqueeze(1).expand(-1, x.shape[1], -1)
        return x - self.location(parents)

    def abduct(self, x: torch.Tensor, parents: torch.Tensor | None, n_samples: int = 1) -> torch.Tensor:
        """The residual repeated: ``[n_samples, *batch, D]`` (the noise is identified exactly)."""
        residual = self.residual(x, parents)
        return residual.unsqueeze(0).expand(n_samples, *residual.shape)

    def forward(self, parents: torch.Tensor | None) -> Distribution:
        return self.noise.distribution(self.location(parents))

    def cdf(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        """``P(X <= x | pa) = F_U(x - g(pa))``, shape ``[B]`` (1-D children only)."""
        self._require_scalar("cdf")
        return self.noise.cdf(self.residual(x, parents)[:, 0])

    def quantile(self, p: torch.Tensor | float, parents: torch.Tensor | None) -> torch.Tensor:
        """Conditional ``p``-quantile ``g(pa) + F_U^{-1}(p)``, shape ``[B]`` (1-D children only)."""
        self._require_scalar("quantile")
        loc = self.location(parents)[:, 0]
        p_t = torch.as_tensor(p, dtype=loc.dtype, device=loc.device).reshape(-1)
        loc, p_t = torch.broadcast_tensors(loc, p_t)
        eps = torch.finfo(loc.dtype).eps  # finite quantiles at 0 and 1
        return loc + self.noise.icdf(p_t.clamp(eps, 1.0 - eps))

    def _require_scalar(self, what: str) -> None:
        if self.output_dim != 1:
            raise ValueError(
                f"{type(self).__name__}.{what} needs a 1-D child (got {self.output_dim} dims)."
            )

    def log_prob(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        """``log p(x | pa)`` for ``x`` ``[B]``, ``[B, D]`` or ``[B, S, D]``; returns ``[B]`` / ``[B, S]``."""
        x = self._as_own(x)
        if x.dim() == 3 and parents is not None and parents.dim() == 2:
            parents = parents.unsqueeze(1).expand(-1, x.shape[1], -1)
        return self.forward(parents).log_prob(x)

    def _as_own(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` as 2-D (or 3-D) on the mechanism's device and in its dtype (float32 before a fit)."""
        x = ensure_2d(x)
        if self._pa_mean is None:
            return x.to(torch.float32)
        return to_device_dtype(x, self._pa_mean.device, self._pa_mean.dtype)

    @staticmethod
    def _parents_or_empty(parents: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
        if parents is None:
            return torch.zeros(x.shape[0], 0, device=x.device)
        return to_device_dtype(ensure_2d(parents), x.device, torch.float32)
