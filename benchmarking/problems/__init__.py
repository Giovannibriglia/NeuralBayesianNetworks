"""ProblemSource implementations for v0.13 benchmarks.

Currently shipped:
    SyntheticProblemSource — wraps ``make_synthetic_bn`` from
        ``benchmarking/synthetic.py``.  Covers all four families
        (discrete, continuous_lg, continuous_nongauss, hybrid).

Pending (future phases):
    BnlearnProblemSource (Phase 4) — real-world networks from the bnlearn
        repository (#73).

Reference: docs/v0.13-benchmark-redesign.md §4.1, §5.2
"""
from benchmarking.problems.synthetic import SyntheticConfig, SyntheticProblemSource

__all__ = ["SyntheticConfig", "SyntheticProblemSource"]
