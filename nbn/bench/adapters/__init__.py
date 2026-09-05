"""v0.13 stateful baseline adapters.

Implements the v0.13 BaselineAdapter protocol from
nbn/bench/core/interfaces.py. Lives alongside the legacy
nbn/bench/baselines/ package during Phase 1b/1c transition.

Status (Phase 1b-i + 1b-ii + 1b-iii): NBNAdapter, PgmpyAdapter,
PomegranateAdapter, PyroAdapter shipped. gpytorch deferred pending
issue #96 (architectural mismatch in query path). Remaining phases
(1b-iv through 1b-vi) pending.

Reference: docs/v0.13-benchmark-redesign.md §4.1
"""

from nbn.bench.adapters.nbn_adapter import NBNAdapter
from nbn.bench.adapters.pgmpy_adapter import PgmpyAdapter
from nbn.bench.adapters.pomegranate_adapter import PomegranateAdapter
from nbn.bench.adapters.pyro_adapter import PyroAdapter

__all__ = ["NBNAdapter", "PgmpyAdapter", "PomegranateAdapter", "PyroAdapter"]
