"""Shared fixtures for the v0.13 runner tests.

These tests launch real subprocesses that import torch, and the per-cell
memory guard sizes its ``RLIMIT_AS`` cap from *ambient* free memory
(``0.80 x psutil.virtual_memory().available``).  Run under a parallel pytest
(``-n auto``), the workers' own memory use drives that number down, the cap
lands below the level at which torch can import, and cells die during startup
and are recorded as ``status='oom'`` — failures that look like real OOMs but
track host load, not the code under test.

Pinning the cap removes the ambient dependence.  ``_compute_cell_memory_limit_bytes``
honours this env var as an operator override precisely so a bound other than
psutil's view of the host can be supplied (cgroups, SLURM allocations, and
this).
"""
from __future__ import annotations

import pytest

from nbn.bench.core.cell_runner import MEMORY_LIMIT_ENV_VAR

#: Comfortably above the ~6 GiB at which torch imports cleanly with CUDA, and
#: still low enough to stop a genuinely runaway cell.
_TEST_CELL_MEMORY_CAP_BYTES = 12 * 1024**3


@pytest.fixture(autouse=True)
def pinned_cell_memory_cap(monkeypatch):
    """Make the per-cell memory cap deterministic for every v013 test."""
    monkeypatch.setenv(MEMORY_LIMIT_ENV_VAR, str(_TEST_CELL_MEMORY_CAP_BYTES))
