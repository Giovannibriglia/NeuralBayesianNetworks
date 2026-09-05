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
    ``"cuda:0"``, ...) into a concrete device string the adapter will
    pin tensors/models to.

    Resolution::

        None       -> "cuda" if CUDA available else "cpu"
        "auto"     -> "cuda" if CUDA available else "cpu"
        "cpu"      -> "cpu"
        "cuda"     -> "cuda"  (caller validates availability)
        "cuda:N"   -> "cuda:N"

    Does NOT validate that ``"cuda"`` / ``"cuda:N"`` actually works —
    that raises at tensor-allocation time, which is appropriate (it lets
    the runner record an error row rather than failing at import time).
    """
    import torch

    if raw is None or raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw
