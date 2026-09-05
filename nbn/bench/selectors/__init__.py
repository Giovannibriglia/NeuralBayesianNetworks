"""QuerySelector implementations for v0.13 benchmarks.

Currently shipped:
    UniformRandomSelector — selects queries by uniform random target +
        evidence assignment, returning ``list[Query]`` (one per row).

Pending (future phases):
    TopologicalAllocator (Phase 2) — hub/cut/terminal/random allocation
        with topology-aware evidence selection (#74).
    HeaviestQueryByRole (Phase 3) — 3-query scalability selector.

Reference: docs/v0.13-benchmark-redesign.md §4.1, §7
"""
from nbn.bench.selectors.uniform import UniformRandomSelector

__all__ = ["UniformRandomSelector"]
