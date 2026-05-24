"""v0.13 benchmark redesign core abstractions.

This package holds the protocols and dataclasses for the new
composition-based architecture. Implementations live in
benchmarking/problems/, benchmarking/selectors/,
benchmarking/measurements/, and benchmarking/adapters/.

Phase 1a status: protocols and dataclasses defined. Implementations
pending Phase 1b+.

Reference: docs/v0.13-benchmark-redesign.md
"""

from benchmarking.core.interfaces import (
    BaselineAdapter,
    Measurement,
    ProblemSource,
    QuerySelector,
)
from benchmarking.core.results import CellResult

__all__ = [
    "BaselineAdapter",
    "Measurement",
    "ProblemSource",
    "QuerySelector",
    "CellResult",
]
