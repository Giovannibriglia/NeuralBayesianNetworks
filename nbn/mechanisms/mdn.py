from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from nbn.mechanisms.base import Mechanism
from nbn.utils.batching import ensure_2d, flatten_samples

logger = logging.getLogger(__name__)


def _sanitise_parents(
    parents: torch.Tensor, *, mech_name: str = "MDN",
) -> torch.Tensor:
    """Replace NaN/Inf entries in ``parents`` with ``0``.

    Defensive guard against upstream numerical drift in deep
    ancestral sampling chains.  v0.6c-A Finding 2 surfaced one
    specific case: at ``continuous_nongauss n=5000 seed=0``, a
    single row out of 2000 in the parent input becomes ``NaN``,
    which propagates through the MDN's zero-weight logit projection
    (because PyTorch's ``0 * NaN = NaN``) and trips
    ``Categorical(probs=softmax(NaN))``'s simplex check.

    This is a band-aid.  The upstream NaN's origin — whichever
    mechanism in the chain produces it first at scale n=5000 — is
    tracked separately as a v0.7 root-cause investigation.  The
    sanitiser logs a warning the first time it triggers per
    ``(mech_name, order-of-magnitude count)`` so silent corruption
    is visible in run logs without flooding them.

    Pure no-op on already-finite input.
    """
    if torch.isfinite(parents).all():
        return parents
    invalid = ~torch.isfinite(parents)
    n_invalid = int(invalid.sum().item())
    warned = getattr(_sanitise_parents, "_warned_keys", None)
    if warned is None:
        warned = set()
        _sanitise_parents._warned_keys = warned  # type: ignore[attr-defined]
    # Deduplicate by (mech_name, decade of count) so we surface the
    # first occurrence per order of magnitude but don't spam.
    decade = 0 if n_invalid <= 0 else len(str(n_invalid)) - 1
    key = (mech_name, decade)
    if key not in warned:
        warned.add(key)
        logger.warning(
            "%s: sanitised %d non-finite parent values (NaN/Inf) at "
            "method entry.  Defensive guard for v0.6c-A Finding 2; "
            "upstream root cause tracked in a separate v0.7 issue.  "
            "(Suppressing further warnings of similar magnitude.)",
            mech_name, n_invalid,
        )
    return torch.where(invalid, torch.zeros_like(parents), parents)


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
            out_dim = k + k * d_x + k * d_x  # logits + locs + log_scales
            self.net = _build_mlp(d_pa, self.hidden, out_dim, self.activation).to(device)
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            dataset = torch.utils.data.TensorDataset(parents, x)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.train()
            for _ in range(epochs):
                for bp, bx in loader:
                    loss = -self._log_prob_2d(bx, bp).mean()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                    opt.step()
            self.eval()
        return {"d_pa": self._d_pa, "d_x": d_x, "k": k}

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
        # ``continuous_nongauss`` ancestral chains is tracked in v0.7
        # issue (see PR #23 round-2 description for the link).
        parents = _sanitise_parents(parents, mech_name="MDN._params_from_parents")

        assert self.net is not None
        out = self.net(parents)  # [*, k + k*D_x + k*D_x]
        logits = out[..., :k]
        rest = out[..., k:].reshape(*out.shape[:-1], k, 2 * d_x)
        loc = rest[..., :d_x]
        log_scale = rest[..., d_x:]
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
