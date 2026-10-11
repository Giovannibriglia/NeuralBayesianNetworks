"""Deterministic seed derivation and RNG scoping for the benchmark (D6).

Every stochastic step of a benchmark cell -- the adapter's fit and queries,
the oracle's reference draws, the scoring-time resampling -- draws from a
seed derived from the problem seed plus a stable label (baseline name, query
index, purpose). Two rules make the derivation safe:

* **Stable across processes.** Labels are hashed with ``zlib.crc32``, never
  with Python's salted ``hash()``; cell workers are separate interpreters.
* **No collisions between neighbouring inputs.** Parts are mixed through
  ``numpy.random.SeedSequence`` (a cryptographic-quality hash of the whole
  entropy tuple), unlike linear schemes such as ``seed * 997 + i`` for which
  ``(seed=0, i=997)`` and ``(seed=1, i=0)`` collide.

Determinism scope: CPU and MPS kernels used by the benchmark are
deterministic given the seed. On CUDA, :func:`set_global_seed` seeds every
visible device (``torch.manual_seed`` covers all of them), but cuBLAS / cuDNN
reductions and ``index_add`` / ``scatter_add`` atomics may still differ in
the last bits between runs; we do not force
``torch.use_deterministic_algorithms`` because several library kernels used
by the baselines have no deterministic CUDA implementation and would raise.
"""
from __future__ import annotations

import contextlib
import random
import zlib
from typing import Iterator

import numpy as np
import torch

__all__ = ["derive_seed", "set_global_seed", "rng_scope"]


def _as_entropy(part: object) -> int:
    if isinstance(part, (bool, np.bool_)):
        return int(part)
    if isinstance(part, (int, np.integer)):
        return int(part) & 0xFFFF_FFFF_FFFF_FFFF
    return zlib.crc32(str(part).encode("utf-8"))


def derive_seed(*parts: object) -> int:
    """A 31-bit seed from ``parts`` (ints and/or strings), stable everywhere.

    The result fits every RNG the benchmark seeds (torch, numpy, ``random``).
    """
    entropy = [_as_entropy(p) for p in parts] or [0]
    state = np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)
    return int(state[0]) & 0x7FFF_FFFF


def set_global_seed(seed: int) -> None:
    """Seed Python ``random``, numpy's global RNG and torch (CPU, every CUDA
    device, MPS). Used once per cell, before the adapter is fit."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed seeds CPU, all CUDA devices (lazily, without forcing
    # CUDA initialisation) and MPS.
    torch.manual_seed(seed)


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


@contextlib.contextmanager
def rng_scope(seed: int) -> Iterator[None]:
    """Run the body under a fixed torch + numpy seed and restore the caller's
    RNG streams afterwards.

    The oracle uses this so its draws are (a) identical for every baseline of a
    cell and (b) invisible to the adapter's own RNG stream. When CUDA is
    available it is initialised on entry so every device's state can be saved
    and restored, including the case where the body is the first CUDA user.
    """
    cpu_state = torch.random.get_rng_state()
    np_state = np.random.get_state()
    cuda_states = None
    if torch.cuda.is_available():
        # Initialise CUDA BEFORE saving: if the body is the first thing to
        # touch CUDA, its lazily-created generators would otherwise keep the
        # scope's seed afterwards and leak it into the caller's (adapter's)
        # stream. In a cell worker CUDA is initialised by the fit anyway.
        torch.cuda.init()
        cuda_states = torch.cuda.get_rng_state_all()
    mps_state = None
    if _mps_available():
        try:
            mps_state = torch.mps.get_rng_state()
        except Exception:  # pragma: no cover - older torch without the API
            mps_state = None
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        np.random.set_state(np_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
