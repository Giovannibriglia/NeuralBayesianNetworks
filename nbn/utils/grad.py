"""Gradient housekeeping for fitted models."""
from __future__ import annotations

import torch


def release_grads(module: torch.nn.Module) -> None:
    """Set ``.grad`` to ``None`` on every parameter of ``module``.

    Mechanisms train with their own optimisers inside ``fit_local`` /
    ``update_local``, and the last step's gradients used to stay on the
    parameters after ``fit`` and ``update`` (#305).  A fitted model then
    could not tell "no gradient reached me" (``p.grad is None``), a later
    ``backward()`` through ``log_prob`` accumulated into stale values, and
    the tensors held memory.  ``None`` (not zeros) frees them, as
    ``optimizer.zero_grad(set_to_none=True)`` does.
    """
    for p in module.parameters():
        p.grad = None
