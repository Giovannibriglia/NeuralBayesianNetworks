"""v0.13 stateful baseline adapters.

Implements the v0.13 BaselineAdapter protocol from
benchmarking/core/interfaces.py. Lives alongside the legacy
benchmarking/baselines/ package during Phase 1b/1c transition.

Status (Phase 1b-i + 1b-ii): NBNAdapter, PgmpyAdapter,
PomegranateAdapter shipped. Remaining libraries (pyro, gpytorch)
pending Phase 1b-iii.

Reference: docs/v0.13-benchmark-redesign.md §4.1
"""

from benchmarking.adapters.nbn_adapter import NBNAdapter
from benchmarking.adapters.pgmpy_adapter import PgmpyAdapter
from benchmarking.adapters.pomegranate_adapter import PomegranateAdapter

__all__ = ["NBNAdapter", "PgmpyAdapter", "PomegranateAdapter"]
