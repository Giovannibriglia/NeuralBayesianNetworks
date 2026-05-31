"""v0.13 benchmark redesign core abstractions.

This package holds the protocols, dataclasses, and orchestration layer for
the composition-based architecture.  Implementations live in
benchmarking/problems/, benchmarking/selectors/,
benchmarking/measurements/, and benchmarking/adapters/.

Phase 1c status: cutover from v0.12 complete.

Reference: docs/v0.13-benchmark-redesign.md
"""

from benchmarking.core.applicability import (
    BASELINE_FAMILY_APPLICABILITY,
    BaselineApplicability,
    accuracy_supported,
    is_applicable,
    known_labels,
)
from benchmarking.core.config import BaselineSpec, RunnerConfig, build_adapter
from benchmarking.core.interfaces import (
    BaselineAdapter,
    Measurement,
    ProblemSource,
    QuerySelector,
)
from benchmarking.core.output import JsonlWriter
from benchmarking.core.results import CellResult
from benchmarking.core.runner import Runner

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
