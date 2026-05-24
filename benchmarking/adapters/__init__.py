"""v0.13 stateful baseline adapters.

Implements the v0.13 BaselineAdapter protocol from
benchmarking/core/interfaces.py. Lives alongside the legacy
benchmarking/baselines/ package during Phase 1b/1c transition.

Status (Phase 1b-i): NBNAdapter shipped. Other libraries
(pgmpy, pyro, gpytorch, pomegranate) pending Phase 1b-ii.

Reference: docs/v0.13-benchmark-redesign.md §4.1
"""

from benchmarking.adapters.nbn_adapter import NBNAdapter

__all__ = ["NBNAdapter"]
