"""Shared device-resolution helper for v0.13 adapters.

The single piece of device logic shared across adapters under design
choice β (per-adapter ownership): translate a baseline's raw ``device``
spec (``None`` / ``"auto"`` / a concrete string) into a concrete device
string.  Each adapter calls :func:`resolve_device` in its ``__init__``
and remains responsible for *how* it then pins tensors/models — pyro,
nbn, and pomegranate each have different device semantics.

Reference: investigation report 2026-06-04 (device flow fix).
"""
from __future__ import annotations


def resolve_device(raw: str | None) -> str:
    """Resolve a baseline's ``device`` spec to a concrete device string.

    Each adapter calls this in its ``__init__`` to translate the raw
    config value (``None``, ``"auto"``, ``"cpu"``, ``"cuda"``,
    ``"cuda:0"``, ``"mps"``, ...) into a concrete device string the
    adapter will pin tensors/models to.

    Resolution::

        None       -> "cuda" if CUDA available, else "mps" if available, else "cpu"
        "auto"     -> same as None
        "cpu"      -> "cpu"
        "cuda"     -> "cuda"  (caller validates availability)
        "cuda:N"   -> "cuda:N"
        "mps"      -> "mps"   (Apple Silicon; caller validates availability)
        "gpu"      -> "cuda" if CUDA available, else "mps" if available, else
                      "cuda" (so a GPU-only baseline on a CPU-only host fails
                      into an error row instead of silently running on CPU)

    ``"gpu"`` is what the shipped configs use for baselines too slow to run
    on a CPU (KDE/kNN/FlexCode LW, neural mechanisms at paper scale): it pins
    them to an accelerator without tying the config to NVIDIA hardware.

    The auto branch is :func:`nbn.utils.device.resolve_device` — the same
    preference order the library itself uses, so an adapter and the NBN
    model it wraps land on the same accelerator.

    Does NOT validate that a concrete spec actually works — that raises at
    tensor-allocation time, which is appropriate (it lets the runner record
    an error row rather than failing at import time).
    """
    if raw is None or raw == "auto":
        from nbn.utils.device import resolve_device as _resolve

        return _resolve("auto").type
    if raw == "gpu":
        import torch

        from nbn.utils.device import mps_available

        if not torch.cuda.is_available() and mps_available():
            return "mps"
        return "cuda"
    return raw
