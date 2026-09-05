"""v0.13 benchmark redesign core abstractions.

This package holds the protocols, dataclasses, and orchestration layer for
the composition-based architecture.  Implementations live in
nbn/bench/problems/, nbn/bench/selectors/,
nbn/bench/measurements/, and nbn/bench/adapters/.

Phase 1c status: cutover from v0.12 complete.

Reference: docs/v0.13-benchmark-redesign.md
"""

from nbn.bench.core.applicability import (
    BASELINE_FAMILY_APPLICABILITY,
    BaselineApplicability,
    accuracy_supported,
    is_applicable,
    known_labels,
)
from nbn.bench.core.config import BaselineSpec, RunnerConfig, build_adapter
from nbn.bench.core.interfaces import (
    BaselineAdapter,
    Measurement,
    ProblemSource,
    QuerySelector,
)
from nbn.bench.core.output import JsonlWriter
from nbn.bench.core.results import CellResult
from nbn.bench.core.runner import Runner

__all__ = [
    # Protocols
    "BaselineAdapter",
    "Measurement",
    "ProblemSource",
    "QuerySelector",
    # Schema
    "CellResult",
    # Config + dispatch
    "BaselineSpec",
    "RunnerConfig",
    "build_adapter",
    # Applicability registry
    "BASELINE_FAMILY_APPLICABILITY",
    "BaselineApplicability",
    "is_applicable",
    "accuracy_supported",
    "known_labels",
    # I/O
    "JsonlWriter",
    # Orchestrator
    "Runner",
]
