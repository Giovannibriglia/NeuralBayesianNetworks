"""The per-cell memory cap must be usable, overridable, and never silently low.

The cap is an ``RLIMIT_AS`` bound, i.e. *virtual address space*, and a torch
process reserves far more of that than it ever resides.  Measured on this
stack, importing torch under a cap of <=4 GiB fails outright, at 5 GiB
succeeds but with ``torch.cuda.is_available()`` False, and needs >=6 GiB to
import cleanly with CUDA.

The floor was 2 GiB — below the level at which a cell can start at all.  Any
time the host was busy enough for ``0.80 x available`` to approach it, every
cell died during import and the runner recorded ``status='oom'``: a spurious
OOM indistinguishable from a real one, and the mechanism behind the v0.13
runner tests failing under a parallel pytest.
"""
from __future__ import annotations

import psutil
import pytest

from nbn.bench.core.cell_runner import (
    MEMORY_LIMIT_ENV_VAR,
    _MEMORY_LIMIT_FLOOR_BYTES,
    _compute_cell_memory_limit_bytes,
)


def _pretend_available(monkeypatch, n_bytes: int) -> None:
    """Make psutil report ``n_bytes`` free, whatever the host actually has."""
    class _Mem:
        available = n_bytes

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Mem())


def test_floor_is_above_the_level_torch_can_start_at():
    """Measured: <=4 GiB cannot import torch, 5 GiB loses CUDA, >=6 GiB works."""
    assert _MEMORY_LIMIT_FLOOR_BYTES >= 6 * 1024**3


def test_explicit_override_wins(monkeypatch):
    monkeypatch.setenv(MEMORY_LIMIT_ENV_VAR, str(3 * 1024**3))
    assert _compute_cell_memory_limit_bytes() == 3 * 1024**3


def test_override_beats_the_host_view(monkeypatch):
    """A cgroup / SLURM bound must not be overwritten by psutil's host view."""
    _pretend_available(monkeypatch, 500 * 1024**3)  # a huge, wrong-for-us host
    monkeypatch.setenv(MEMORY_LIMIT_ENV_VAR, str(9 * 1024**3))
    assert _compute_cell_memory_limit_bytes() == 9 * 1024**3


@pytest.mark.parametrize("bad", ["", "garbage", "-1", "0"])
def test_unusable_override_falls_back_instead_of_crashing(monkeypatch, bad):
    monkeypatch.setenv(MEMORY_LIMIT_ENV_VAR, bad)
    assert _compute_cell_memory_limit_bytes() >= _MEMORY_LIMIT_FLOOR_BYTES


def test_cap_tracks_available_memory_when_there_is_plenty(monkeypatch):
    monkeypatch.delenv(MEMORY_LIMIT_ENV_VAR, raising=False)
    _pretend_available(monkeypatch, 100 * 1024**3)
    assert _compute_cell_memory_limit_bytes() == int(100 * 1024**3 * 0.80)


def test_cap_never_falls_below_the_floor(monkeypatch):
    """The regression: a starved host used to yield a cap no cell can run in."""
    monkeypatch.delenv(MEMORY_LIMIT_ENV_VAR, raising=False)
    _pretend_available(monkeypatch, 64 * 1024**2)  # 64 MiB
    assert _compute_cell_memory_limit_bytes() == _MEMORY_LIMIT_FLOOR_BYTES


def test_starved_host_is_reported(monkeypatch, caplog):
    """Pinned to the floor means the fraction no longer bounds anything."""
    monkeypatch.delenv(MEMORY_LIMIT_ENV_VAR, raising=False)
    _pretend_available(monkeypatch, 64 * 1024**2)
    with caplog.at_level("WARNING"):
        _compute_cell_memory_limit_bytes()
    assert "floor" in caplog.text.lower()
