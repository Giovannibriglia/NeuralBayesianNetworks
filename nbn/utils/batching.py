from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch

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
    ``Categorical(probs=softmax(NaN))``'s simplex check.  The same
    non-finite context, fed to a zuko NSF conditioner, would trip a
    device-side assert -- hence the flow mechanism uses this too.

    This is a band-aid.  The upstream NaN's origin -- whichever
    mechanism in the chain produces it first at scale n=5000 -- is
    tracked in v0.7 issue #24.  The sanitiser logs a warning the
    first time it triggers per ``(mech_name, order-of-magnitude
    count)`` so silent corruption is visible in run logs without
    flooding them.

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


def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """Return tensor with at least 2 dimensions ``[B, D]``."""
    if x.dim() == 0:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 1:
        return x.unsqueeze(-1)
    return x


def broadcast_samples(x: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Expand ``[B, D]`` → ``[B, S, D]`` by repeating along dim 1."""
    if x.dim() == 2:
        return x.unsqueeze(1).expand(-1, n_samples, -1)
    if x.dim() == 3:
        if x.shape[1] == 1:
            return x.expand(-1, n_samples, -1)
        return x
    raise ValueError(f"Expected 2-D or 3-D tensor, got shape {tuple(x.shape)}")


def flatten_samples(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Flatten ``[B, S, D]`` → ``[B*S, D]`` and return ``(flat, B, S)``."""
    if x.dim() == 2:
        return x, x.shape[0], 1
    if x.dim() == 3:
        b, s, d = x.shape
        return x.reshape(b * s, d), b, s
    raise ValueError(f"Expected 2-D or 3-D tensor, got shape {tuple(x.shape)}")


def pack_parents(
    data: Dict[str, torch.Tensor],
    parent_names: List[str],
    n_samples: int | None = None,
) -> torch.Tensor | None:
    """Concatenate parent tensors into a single ``[B, D_total]`` or ``[B, S, D_total]`` tensor.

    Parameters
    ----------
    data:
        Mapping from node name to tensor.  Expected shape: ``[B]``, ``[B, D]``,
        or ``[B, S, D]``.
    parent_names:
        Ordered list of parent node names.
    n_samples:
        If provided and data is 2-D, broadcast to ``[B, S, D]``.
    """
    if not parent_names:
        return None
    parts = []
    for name in parent_names:
        t = data[name]
        t = ensure_2d(t)
        if n_samples is not None and t.dim() == 2:
            t = broadcast_samples(t, n_samples)
        parts.append(t)
    return torch.cat(parts, dim=-1)
