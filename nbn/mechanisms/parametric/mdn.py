from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from nbn.mechanisms.base import Mechanism
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
        # Bug C (follow-up to Bug B / PR #90): deep-chain LW sampling at
        # n≥100 can produce particles at 700σ+ after standardization.  MLP
        # log_scale outputs at that scale overflow float32 → scale=Inf →
        # sample=Inf → downstream Categorical(NaN) crash.  Clamping
        # standardized parents to ±50σ keeps MLP inputs in a regime where
        # exp(log_scale) stays finite for any learned weight < 1.76 after
        # normal training, while 50σ is already wildly OOD — predictions
        # there are noise regardless.  Tunable via _pa_clamp.
        self._pa_clamp: float = 50.0
        # Issue #95: bounds log_scale output before exp() — exp(7) ≈ 1100,
        # physically meaningful but prevents float32 overflow on pathological
        # input combinations that slip past _pa_clamp.  Tunable via _log_scale_clamp.
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

    def fit_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 512,
        consolidate: bool = True,
        **kwargs,
    ) -> dict:
        x = ensure_2d(x)  # [N, D_x]
        n, d_x = x.shape
        device = x.device
        self._d_x = d_x
        self.output_dim = d_x
        k = self.num_components

        if parents is None or parents.shape[-1] == 0:
            self._d_pa = 0
            # Learnable root parameters
            self._root_logits = nn.Parameter(torch.zeros(k, device=device))
            self._root_loc = nn.Parameter(x.mean(0).unsqueeze(0).expand(k, -1).clone())
            self._root_log_scale = nn.Parameter(
                torch.log(x.std(0).clamp_min(1e-3)).unsqueeze(0).expand(k, -1).clone()
            )
        else:
            parents = ensure_2d(parents).to(device=device, dtype=x.dtype)
            d_pa = parents.shape[1]
            self._d_pa = d_pa
            # Compute parent standardization statistics from training data.
            # Extreme parent values (e.g. 3.3M in continuous_nongauss deep
            # chains) cause log_scale overflow in the MLP forward pass:
            # log_scale ≈ extreme_value * weight → exp(log_scale) = inf →
            # loss ≈ 1e14 → backward overflows to NaN → NaN params →
            # CUDA assert at LW inference (Bug B, issues #81 #24 #54).
            # Standardizing parents maps them to O(1) scale, keeping
            # log_scale finite throughout training and inference.
            pa_mean = parents.mean(0, keepdim=True)
            pa_std = parents.std(0, keepdim=True).clamp_min(1e-3)
            self._pa_mean = pa_mean.detach()
            self._pa_std = pa_std.detach()
            # Do NOT pre-standardize here: _params_from_parents applies
            # standardization at every forward call (training and inference),
            # so pre-standardizing would double-standardize during training
            # (causing NaN for constant-parent columns) and produce a model
            # trained on different inputs than it sees at inference time.
            out_dim = k + k * d_x + k * d_x  # logits + locs + log_scales
            self.net = _build_mlp(d_pa, self.hidden, out_dim, self.activation).to(device)
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            self.train()
            for _ in range(epochs):
                perm = torch.randperm(n, device=device)
                for i in range(0, n, batch_size):
                    idx = perm[i:i + batch_size]
                    bp, bx = parents[idx], x[idx]
                    loss = -self._log_prob_2d(bx, bp).mean()
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
        return {"d_pa": self._d_pa, "d_x": d_x, "k": k}

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
        x = ensure_2d(x)
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
        parents = _sanitise_parents(parents, mech_name="MDN._params_from_parents")
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
        scale = torch.exp(log_scale).clamp_min(self.min_scale)
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

    def forward(self, parents: torch.Tensor | None) -> MixtureSameFamily:
        if parents is not None:
            parents = ensure_2d(parents)
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

        x_flat = x.reshape(b * s, d_x)
        lp = self._log_prob_2d(x_flat, pa_flat).reshape(b, s)
        if squeeze_s:
            lp = lp.squeeze(1)
        return lp
