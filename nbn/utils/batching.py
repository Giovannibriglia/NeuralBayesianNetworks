from __future__ import annotations

from typing import Dict, List, Tuple

import torch


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
