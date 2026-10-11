from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from nbn.learning.weighting import (
    select,
    validate_weights,
    weighted_mean,
    weighted_moments,
)
from nbn.learning.warm_start import check_branch, check_shapes
from nbn.mechanisms.base import (
    Mechanism,
    _ensure_optional_buffers,
    _sync_optional_buffers,
)
from nbn.update import online_laplace
# _sanitise_parents was promoted to nbn.utils.batching (v0.14) so the flow
# mechanism can share the identical guard; re-exported via this import for the
# existing ``from nbn.mechanisms.parametric.mdn import _sanitise_parents`` callers/tests.
from nbn.utils.batching import _sanitise_parents, ensure_2d, flatten_samples  # noqa: F401

logger = logging.getLogger(__name__)


def _build_mlp(
    in_dim: int,
    hidden: Tuple[int, ...],
    out_dim: int,
    activation: str = "relu",
) -> nn.Sequential:
    act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "elu": nn.ELU}[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), act_cls()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MDNMechanism(Mechanism):
    """Mixture Density Network CPD (Bishop 1994).

    Models P(X | pa) as a Gaussian mixture whose parameters (mixing weights,
    means, scales) are predicted by an MLP from ``pa``.

    Parameters
    ----------
    num_components: int
        Number of Gaussian mixture components K.
    hidden: tuple of int
        Hidden layer widths of the MLP.
    activation: str
        Activation function: ``"relu"``, ``"tanh"``, ``"gelu"``, ``"elu"``.
    min_scale: float
        Minimum scale to prevent degenerate components.

    Shape contracts
    ---------------
    parents: ``[B, D_pa]`` or ``None``
    forward → MixtureSameFamily with batch_shape ``[B]``, event_shape ``[D_x]``
    log_prob(x, pa) → ``[B]``
    sample(pa, n) → ``[B, n, D_x]``
    """

    is_discrete: bool = False
    supports_update: bool = True
    supports_weights: bool = True
    warm_start_is_noop: bool = False

    def __init__(
        self,
        num_components: int = 5,
        hidden: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
        min_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.num_components = int(num_components)
        self.hidden = tuple(hidden)
        self.activation = activation
        self.min_scale = float(min_scale)
        # Built in fit_local once we know input/output dims:
        self.net: nn.Module | None = None
        self._d_x: int = 0
        self._d_pa: int = 0
        # Root node parameters (no parents)
        self._root_logits: nn.Parameter | None = None
        self._root_loc: nn.Parameter | None = None
        self._root_log_scale: nn.Parameter | None = None
        # Parent standardization statistics — registered as buffers so they
        # survive state_dict save/load.  Computed in fit_local from training
        # data; applied in _params_from_parents at inference time.
        # None for root nodes (no parents).
        self.register_buffer('_pa_mean', None)
        self.register_buffer('_pa_std', None)
        # Child standardisation statistics ([D_x] each), non-root only.  The
        # MLP predicts mixture locations/scales in standardised units and
        # _params_from_parents maps them back (loc = m + s*loc_z,
        # scale = s*scale_z), so log_prob carries the -log(s) Jacobian through
        # Normal's own normaliser and sample() returns raw values.  Without
        # it the MLP had to emit raw locations, and _log_scale_clamp capped
        # every component at ~1097 *raw* units: a target with a spread of a
        # few thousand was misfit (NLL 15.0 against LinearGaussian's 9.4).
        # None (identity) until fit_local runs, so a hand-built MDN
        # (nbn.bench.synthetic) keeps its raw-unit parameterisation.
        self.register_buffer('_x_mean', None)
        self.register_buffer('_x_std', None)
        # Bug C (follow-up to Bug B / PR #90): deep-chain LW sampling at
        # n≥100 can produce particles at 700σ+ after standardization.  MLP
        # log_scale outputs at that scale overflow float32 → scale=Inf →
        # sample=Inf → downstream Categorical(NaN) crash.  Clamping
        # standardized parents to ±50σ keeps MLP inputs in a regime where
        # exp(log_scale) stays finite for any learned weight < 1.76 after
        # normal training, while 50σ is already wildly OOD — predictions
        # there are noise regardless.  Tunable via _pa_clamp.
        self._pa_clamp: float = 50.0
        # Issue #95: bounds log_scale output before exp() — exp(7) ≈ 1100
        # standard deviations of the child (the clamp acts in standardised
        # units once _x_std is set), physically meaningful but prevents
        # float32 overflow on pathological input combinations that slip past
        # _pa_clamp.  Tunable via _log_scale_clamp.
        self._log_scale_clamp: float = 7.0
        # Online-EWC state for incremental update (theta* and diagonal Fisher),
        # flat concatenations over parameters().  Buffers so .to()/state_dict/
        # intervene()-deepcopy carry them; populated by online_laplace.consolidate
        # at the end of fit_local.  None until fitted.
        self.register_buffer("_ewc_mu", None)
        self.register_buffer("_ewc_fisher", None)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    # Buffers added after this mechanism's first release; ``None`` is the
    # identity map the older format used.  See nbn/mechanisms/base.py.
    _OPTIONAL_BUFFERS = ("_x_mean", "_x_std")

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        _ensure_optional_buffers(self, self._OPTIONAL_BUFFERS)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        _sync_optional_buffers(self, state_dict, prefix, self._OPTIONAL_BUFFERS)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def fit_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 512,
        weights: torch.Tensor | None = None,
        consolidate: bool = True,
        warm_start: bool = False,
        **kwargs,
    ) -> dict:
        is_root = parents is None or parents.shape[-1] == 0
        # Validate *before* mutating anything: _d_x/output_dim used to be
        # assigned above the branch, so a rejected warm start would have left
        # the mechanism describing a shape its parameters do not have.
        warm = bool(warm_start) and self.is_fitted
        # Train in the parameters' dtype (the default dtype for a fresh
        # build).  float64 data used to reach a float32 MLP and crash with
        # "mat1 and mat2 must have the same dtype".
        x = ensure_2d(x).to(self._param_dtype() if warm else torch.get_default_dtype())
        n, d_x = x.shape
        device = x.device
        w_vec = validate_weights(weights, n, where="MDNMechanism.fit_local", device=device)
        k = self.num_components

        if warm:
            where = "MDNMechanism.fit_local"
            check_branch(where, was_root=self._d_pa == 0, is_root=is_root)
            check_shapes(where, {
                "d_x": (self._d_x, d_x),
                "d_pa": (self._d_pa, 0 if is_root else ensure_2d(parents).shape[1]),
            })
        self._d_x = d_x
        self.output_dim = d_x

        if is_root:
            self._d_pa = 0
            # The root branch is parameterised in raw units (closed-form
            # moments); drop any child standardisation a previous non-root
            # fit left behind so it cannot be mistaken for this fit's.
            self._x_mean = None
            self._x_std = None
            # Learnable root parameters
            # A root MDN is initialised analytically from the data moments and
            # never enters the training loop below, so weighting has to be
            # applied to the moments themselves.
            #
            # This branch is closed form, so it recomputes even under
            # warm_start=True — there is no trajectory to continue, and
            # freezing it would stop a root node responding to the E-step
            # entirely: a silent, permanent M-step failure on exactly the
            # nodes nobody inspects.  The bitwise epochs=0 contract still
            # holds here because recomputing from unchanged data is
            # deterministic; the values only move when the weights do, which
            # is when they should.
            loc0, scale0 = weighted_moments(x, w_vec, unbiased=True)
            # One unweighted row has an undefined (NaN) unbiased std, and
            # clamp_min passes NaN through: fall back to the population value
            # (0) so the floor below applies.
            scale0 = scale0.nan_to_num(0.0)
            self._root_logits = nn.Parameter(torch.zeros(k, device=device))
            self._root_loc = nn.Parameter(loc0.unsqueeze(0).expand(k, -1).clone())
            self._root_log_scale = nn.Parameter(
                torch.log(scale0.clamp_min(1e-3)).unsqueeze(0).expand(k, -1).clone()
            )
        else:
            parents = ensure_2d(parents).to(device=device, dtype=x.dtype)
            d_pa = parents.shape[1]
            self._d_pa = d_pa
            # Child standardisation (see __init__).  Population moments: the
            # scale only fixes the units the MLP works in (the Jacobian keeps
            # log_prob exact for any choice), and the population form is
            # replication-exact and well defined for a single row or a total
            # weight <= 1.  A constant child gets s = 1.  Frozen under
            # warm_start for the same reason as _pa_mean/_pa_std below.
            if not warm:
                x_m, x_s = weighted_moments(x, w_vec)
                x_s = x_s.nan_to_num(0.0)
                self._x_mean = x_m.detach()
                self._x_std = torch.where(x_s > 0, x_s, torch.ones_like(x_s)).detach()
            # Compute parent standardization statistics from training data.
            # Extreme parent values (e.g. 3.3M in continuous_nongauss deep
            # chains) cause log_scale overflow in the MLP forward pass:
            # log_scale ≈ extreme_value * weight → exp(log_scale) = inf →
            # loss ≈ 1e14 → backward overflows to NaN → NaN params →
            # CUDA assert at LW inference (Bug B, issues #81 #24 #54).
            # Standardizing parents maps them to O(1) scale, keeping
            # log_scale finite throughout training and inference.
            # Weighted along with the loss: standardising against the
            # *unweighted* spread would silently fit a model normalised for a
            # data distribution the caller did not ask for, and would break
            # replication equivalence in a way that still converges.
            #
            # Frozen under warm_start: ``net``'s weights were trained against
            # the map these two buffers define, so recomputing them would
            # apply every learned weight to a shifted, rescaled input — the
            # warm start would be warm in name only, and worst exactly when
            # the weights moved most.  See nbn/learning/warm_start.py.
            if not warm:
                pa_m, pa_s = weighted_moments(parents, w_vec, unbiased=True)
                pa_mean = pa_m.unsqueeze(0)
                # nan_to_num: one unweighted row has a NaN unbiased std, which
                # clamp_min would pass straight into every forward call.
                pa_std = pa_s.nan_to_num(0.0).unsqueeze(0).clamp_min(1e-3)
                self._pa_mean = pa_mean.detach()
                self._pa_std = pa_std.detach()
            # Do NOT pre-standardize here: _params_from_parents applies
            # standardization at every forward call (training and inference),
            # so pre-standardizing would double-standardize during training
            # (causing NaN for constant-parent columns) and produce a model
            # trained on different inputs than it sees at inference time.
            if not warm:
                out_dim = k + k * d_x + k * d_x  # logits + locs + log_scales
                self.net = _build_mlp(d_pa, self.hidden, out_dim, self.activation).to(device)
            # A *fresh* optimiser either way, over whichever parameters now
            # exist.  Adam's moments are not in state_dict(), so persisting
            # them across calls would survive a caller's load_state_dict
            # revert and keep a rejected step's momentum alive.
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            self.train()
            for _ in range(epochs):
                perm = torch.randperm(n, device=device)
                for i in range(0, n, batch_size):
                    idx = perm[i:i + batch_size]
                    bp, bx = parents[idx], x[idx]
                    # select() with the *same* idx: the permutation is local to
                    # this loop, so the weights must be gathered here or they
                    # desynchronise silently.
                    loss = weighted_mean(
                        -self._log_prob_2d(bx, bp), select(w_vec, idx),
                    )
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                    opt.step()
            self.eval()
        # Snapshot theta* + diagonal Fisher for no-rehearsal EWC update.  Runs
        # for both root and non-root via parameters()+log_prob; parents here is
        # None (root) or the raw [N, D_pa] tensor (log_prob standardises it).
        # Opt-out (consolidate=False): the Fisher pass costs up to sample_cap
        # sequential per-sample backward passes — pure overhead for fit-only
        # workloads that never call update().
        if consolidate:
            online_laplace.consolidate(self, x, parents)
        # ``warm_started`` is True only where parameters were actually carried
        # over — never on the closed-form root branch, which recomputes.
        return {
            "d_pa": self._d_pa, "d_x": d_x, "k": k,
            "warm_started": warm and not is_root,
        }

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def update_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        *,
        forgetting: float = 1.0,
        lam: float = online_laplace.DEFAULT_LAM,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 512,
        fisher_batch_size: int = 1,
        sample_cap: int = 4096,
        weights: torch.Tensor | None = None,
        **kw,
    ) -> dict:
        """Fold new data in via online EWC (no rehearsal).

        Trains ``net`` (or the root params) on the new data under a Fisher-
        weighted quadratic anchor to the post-fit weights, then refreshes the
        running prior.  ``lam`` is the stability↔plasticity knob (see
        :data:`nbn.update.online_laplace.DEFAULT_LAM`).  ``fisher_batch_size=1``
        (default) is the exact per-sample empirical Fisher; ``>1`` is a faster,
        biased-low approximation.
        """
        if weights is not None:
            raise NotImplementedError(
                "MDNMechanism.update_local does not accept per-sample weights: "
                "weighting is supported on the fitting path (fit / fit_local) "
                "only.  A weighted fit does persist weighted sufficient "
                "statistics, so update() folds new data onto the correct "
                "prior — but the new data itself is taken at face value."
            )
        x = ensure_2d(x).to(self._param_dtype())
        if parents is not None:
            parents = ensure_2d(parents).to(device=x.device, dtype=x.dtype)
        return online_laplace.ewc_update(
            self, x, parents,
            epochs=epochs, lr=lr, batch_size=batch_size,
            lam=lam, forgetting=forgetting,
            fisher_batch_size=fisher_batch_size, sample_cap=sample_cap,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _param_dtype(self) -> torch.dtype:
        """dtype of the fitted parameters (default dtype before any fit).

        Inputs are cast to it on every entry point: the MLP's matmul does not
        type-promote, so float64 data against float32 weights crashed.
        """
        for p in self.parameters():
            return p.dtype
        return torch.get_default_dtype()

    def _params_from_parents(
        self, parents: torch.Tensor | None, shape: Tuple[int, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (logits [*shape, K], loc [*shape, K, D_x], scale [*shape, K, D_x])."""
        k = self.num_components
        d_x = self._d_x

        if self._d_pa == 0 or parents is None:
            b = shape[0] if shape else 1
            logits = self._root_logits.unsqueeze(0).expand(b, -1)
            loc = self._root_loc.unsqueeze(0).expand(b, -1, -1)
            scale = torch.exp(self._root_log_scale).clamp_min(self.min_scale)
            scale = scale.unsqueeze(0).expand(b, -1, -1)
            return logits, loc, scale

        # v0.6c-A round 2 — Finding 2 defensive guard.  Replace any
        # NaN/Inf in ``parents`` with 0 before the linear projection.
        # In PyTorch, ``0 * NaN = NaN``, so even a zero-weighted row
        # of ``self.net``'s linear (e.g. the logit slot in
        # synthetic-side MDNs whose mixture pi is parent-invariant by
        # construction) propagates an upstream NaN into ``out`` and
        # ultimately into ``Categorical(probs=softmax(logits))``,
        # tripping the Simplex constraint.  This guard is a band-aid;
        # the upstream NaN's root-cause origin in deep
        # ``continuous_nongauss`` ancestral chains is tracked in
        # v0.7 issue #24.
        parents = _sanitise_parents(
            parents.to(self._param_dtype()), mech_name="MDN._params_from_parents",
        )
        # Standardize to training distribution — prevents log_scale overflow
        # from extreme parent values (Bug B fix; see fit_local comment).
        if self._pa_mean is not None:
            parents = (parents - self._pa_mean) / self._pa_std
            parents = parents.clamp(-self._pa_clamp, self._pa_clamp)

        assert self.net is not None
        out = self.net(parents)  # [*, k + k*D_x + k*D_x]
        logits = out[..., :k]
        rest = out[..., k:].reshape(*out.shape[:-1], k, 2 * d_x)
        loc = rest[..., :d_x]
        log_scale = rest[..., d_x:]
        log_scale = log_scale.clamp(max=self._log_scale_clamp)  # issue #95
        scale = torch.exp(log_scale)
        if self._x_mean is not None:
            # Standardised -> raw units.  An affine map of a Gaussian is a
            # Gaussian, so the mixture in raw units is exact and Normal's
            # log(scale) term carries the -log(s) Jacobian.
            loc = self._x_mean + self._x_std * loc
            scale = self._x_std * scale
        # min_scale stays a floor in raw units, as documented.
        scale = scale.clamp_min(self.min_scale)
        return logits, loc, scale

    def _log_prob_2d(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        """Stable mixture log-prob for 2-D x ``[B, D_x]``."""
        logits, loc, scale = self._params_from_parents(
            parents, (x.shape[0],)
        )
        # x: [B, D_x] → [B, 1, D_x]; loc/scale: [B, K, D_x]
        x_exp = x.unsqueeze(1)
        log_comp = Independent(Normal(loc, scale), 1).log_prob(
            x_exp.expand_as(loc)
        )  # [B, K]
        log_pi = torch.log_softmax(logits, dim=-1)
        return torch.logsumexp(log_pi + log_comp, dim=-1)  # [B]

    # ------------------------------------------------------------------
    # Distribution interface
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True iff ``fit_local`` built the conditioner (or root parameters).

        Root nodes are fitted via ``_root_logits``/``_root_loc``; non-root
        nodes via the ``net`` conditioner.  Without this override the
        mechanism inherited ``Mechanism.is_fitted``'s ``False`` default and
        reported unfitted after a successful fit.
        """
        return self.net is not None or self._root_logits is not None

    def forward(self, parents: torch.Tensor | None) -> MixtureSameFamily:
        if parents is not None:
            parents = ensure_2d(parents).to(self._param_dtype())
            b = parents.shape[0]
        else:
            b = 1
        logits, loc, scale = self._params_from_parents(parents, (b,))
        mix = Categorical(logits=logits)
        comp = Independent(Normal(loc, scale), 1)
        return MixtureSameFamily(mix, comp)

    def sample(self, parents: torch.Tensor | None, n: int = 1) -> torch.Tensor:
        """Return ``[B, n, D_x]``."""
        if parents is None:
            b = 1
            parents_exp = None
        else:
            parents_2d = ensure_2d(parents)
            b = parents_2d.shape[0]
            parents_exp = parents_2d.unsqueeze(1).expand(-1, n, -1)  # [B, n, D_pa]
            flat, _, _ = flatten_samples(parents_exp)  # [B*n, D_pa]
            parents_exp = flat

        logits, loc, scale = self._params_from_parents(parents_exp, (b * n,))
        pi = torch.softmax(logits, dim=-1).clamp_min(1e-7)
        pi = pi / pi.sum(-1, keepdim=True)
        comp_idx = Categorical(probs=pi).sample()  # [B*n]
        # Gather selected component: [B*n, D_x]
        comp_idx_e = comp_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self._d_x)
        loc_sel = loc.gather(1, comp_idx_e).squeeze(1)
        scale_sel = scale.gather(1, comp_idx_e).squeeze(1)
        samp = loc_sel + scale_sel * torch.randn_like(scale_sel)
        return samp.reshape(b, n, self._d_x)

    def log_prob(
        self, x: torch.Tensor, parents: torch.Tensor | None
    ) -> torch.Tensor:
        squeeze_s = False
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_s = True
        b, s, d_x = x.shape

        if parents is None:
            pa_flat = None
        else:
            if parents.dim() == 2:
                parents = parents.unsqueeze(1).expand(-1, s, -1)
            flat, _, _ = flatten_samples(parents)
            pa_flat = flat

        x_flat = x.reshape(b * s, d_x).to(self._param_dtype())
        lp = self._log_prob_2d(x_flat, pa_flat).reshape(b, s)
        if squeeze_s:
            lp = lp.squeeze(1)
        return lp
