"""Single source of truth for device handling in NBN.

* ``resolve_device`` turns string specs (``'auto'``, ``'cpu'``, ``'cuda:0'``)
  and ``torch.device`` instances into a canonical ``torch.device``.
* ``to_device`` recursively moves tensors / dicts / nn.Modules to a device,
  including ``torch.distributions.Distribution`` objects.
* ``assert_on_device`` raises with a useful name if any part of an object
  is on a wrong device — used by tests to catch silent CPU fallbacks.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


def resolve_device(spec: str | torch.device | None) -> torch.device:
    """Resolve a device spec to a concrete ``torch.device``.

    Parameters
    ----------
    spec:
        ``'auto'`` → CUDA if available, else MPS if available, else CPU.
        ``None`` → ``'cpu'``.
        Otherwise: passed straight to ``torch.device``.
    """
    if spec is None:
        return torch.device("cpu")
    if isinstance(spec, torch.device):
        return spec
    spec = str(spec).strip().lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors / dicts / lists / modules to ``device``.

    Distributions are reconstructed lazily via the engine that produced them;
    here we move only the underlying parameters (e.g. ``loc``, ``scale``).
    """
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True) if obj.device != device else obj
    if isinstance(obj, nn.Module):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [to_device(v, device) for v in obj]
        return type(obj)(moved) if isinstance(obj, tuple) else moved
    return obj  # plain Python scalars etc. are device-free


def assert_on_device(obj: Any, device: torch.device, name: str = "") -> None:
    """Recursively assert every tensor/parameter in ``obj`` is on ``device``.

    Raises ``AssertionError`` with a path (``"name.foo[2]"``) on first mismatch.
    """
    target = torch.device(device)

    def _check(o, path):
        if isinstance(o, torch.Tensor):
            # Compare type and (when set) index.
            if o.device.type != target.type:
                raise AssertionError(
                    f"Device mismatch at {path or name or '<root>'}: "
                    f"expected {target}, got {o.device}"
                )
        elif isinstance(o, nn.Module):
            for n, p in o.named_parameters():
                _check(p, f"{path}.{n}")
            for n, b in o.named_buffers():
                _check(b, f"{path}.{n}")
        elif isinstance(o, Mapping):
            for k, v in o.items():
                _check(v, f"{path}[{k!r}]")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                _check(v, f"{path}[{i}]")

    _check(obj, name)
