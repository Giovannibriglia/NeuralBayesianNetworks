"""Single source of truth for device handling in NBN.

* ``resolve_device`` turns string specs (``'auto'``, ``'cpu'``, ``'cuda:0'``)
  and ``torch.device`` instances into a canonical ``torch.device``.
* ``to_device`` recursively moves tensors / dicts / nn.Modules to a device,
  including ``torch.distributions.Distribution`` objects.
* ``assert_on_device`` raises with a useful name if any part of an object
  is on a wrong device — used by tests to catch silent CPU fallbacks.
* ``supports_float64`` / ``accum_dtype`` / ``linalg_device`` encode what a
  backend can do: Apple's Metal backend (``mps``) has no float64 kernels and
  no LAPACK-backed solvers, so double-precision accumulations and small dense
  solves consult these instead of hard-coding ``torch.float64`` / the data
  device.
* ``synchronize`` / ``empty_cache`` / ``peak_memory_bytes`` /
  ``accelerator_summary`` are the backend-neutral spellings of the
  ``torch.cuda.*`` calls the benchmark suite uses.
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
        ``'auto'`` → CUDA if available, else MPS (Apple Silicon) if
        available, else CPU.  ``None`` → ``'cpu'``.  Otherwise: passed
        straight to ``torch.device``.
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


# ---------------------------------------------------------------------------
# Backend capabilities
# ---------------------------------------------------------------------------
# The library was written on CUDA, where everything torch offers on CPU is
# also available on the device.  Apple's Metal backend is not like that:
#
# * it has no float64 tensors at all (``torch.zeros(1, dtype=torch.float64,
#   device="mps")`` raises), and
# * several LAPACK-backed ``torch.linalg`` kernels (``lstsq`` among them)
#   are not implemented for it.
#
# NBN accumulates sufficient statistics in float64 on purpose (EM weights are
# long runs of repeated small values; see nbn/learning/weighting.py) and
# fits its Gaussian families by closed-form solves.  Rather than sprinkle
# ``if device.type == "mps"`` through every mechanism, those sites ask the
# helpers below.  On CPU and CUDA the answers are the historical ones, so
# results there are unchanged.


def supports_float64(device: str | torch.device) -> bool:
    """Whether ``device`` can hold float64 tensors (everything but ``mps``)."""
    return torch.device(device).type != "mps"


def accum_dtype(device: str | torch.device) -> torch.dtype:
    """Widest float dtype ``device`` supports, for precision-sensitive sums.

    ``torch.float64`` on CPU / CUDA; ``torch.float32`` on ``mps``, where
    float64 does not exist and the only alternative would be to refuse to
    run.  Call sites cast back to the data dtype afterwards, so the choice
    never leaks into a model's parameters.
    """
    return torch.float64 if supports_float64(device) else torch.float32


def linalg_device(device: str | torch.device) -> torch.device:
    """Device on which to run a small dense solve for tensors living on ``device``.

    The normal-equation / least-squares solves in the Gaussian families are
    over ``[D_pa+1, D_pa+1]`` matrices — tiny, and absent from the MPS
    backend (``torch.linalg.lstsq`` has no Metal kernel).  Run them on the CPU
    there and move the solution back; elsewhere the device itself.
    """
    d = torch.device(device)
    return torch.device("cpu") if d.type == "mps" else d


def synchronize(device: str | torch.device) -> None:
    """Block until queued work on ``device`` has finished (no-op on CPU).

    Wall-clock timings of GPU work must bracket the computation with this,
    or they measure kernel *launch* time; ``torch.cuda.synchronize`` and
    ``torch.mps.synchronize`` are the two backends' spellings.
    """
    d = torch.device(device)
    if d.type == "cuda":
        torch.cuda.synchronize(d)
    elif d.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: str | torch.device) -> None:
    """Release cached allocator blocks on ``device`` back to the driver."""
    d = torch.device(device)
    if d.type == "cuda":
        torch.cuda.empty_cache()
    elif d.type == "mps":
        torch.mps.empty_cache()


def peak_memory_bytes(device: str | torch.device) -> float | None:
    """Accelerator memory held by this process, or ``None`` when unmeasurable.

    CUDA reports the true high-water mark (``max_memory_allocated``).  The
    MPS allocator exposes no peak counter, so the *current* allocation is
    returned there — a lower bound on the peak, flagged as such wherever the
    number is recorded.  CPU devices return ``None``.
    """
    d = torch.device(device)
    if d.type == "cuda":
        return float(torch.cuda.max_memory_allocated(d))
    if d.type == "mps":
        fn = getattr(torch.mps, "current_allocated_memory", None)
        return float(fn()) if fn is not None else None
    return None


def mps_available() -> bool:
    """``torch.backends.mps.is_available()`` guarded for builds without MPS."""
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def accelerator_summary() -> str:
    """One line naming the accelerator ``resolve_device('auto')`` would pick."""
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover  (driver probe failed)
            name = "unknown"
        return f"cuda ({name}; {torch.cuda.device_count()} device(s))"
    if mps_available():
        return "mps (Apple Metal; float64 unavailable, accumulations run in float32)"
    return "none (cpu only)"
