"""v0.13 benchmark redesign core abstractions.

This package holds the protocols, dataclasses, and orchestration layer for
the composition-based architecture.  Implementations live in
benchmarking/problems/, benchmarking/selectors/,
benchmarking/measurements/, and benchmarking/adapters/.

Phase 1b-v status: runner orchestrator shipped.

Reference: docs/v0.13-benchmark-redesign.md
"""

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
    # I/O
    "JsonlWriter",
    # Orchestrator
    "Runner",
]
