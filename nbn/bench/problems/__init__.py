"""ProblemSource implementations for v0.13 benchmarks.

Currently shipped:
    SyntheticProblemSource — wraps ``make_synthetic_bn`` from
        ``nbn/bench/synthetic.py``.  Covers all four families
        (discrete, continuous_lg, continuous_nongauss, hybrid).

    BnlearnProblemSource (Phase 4) — real-world networks from the bnlearn
        repository (#73). Stage 2 ships discrete networks; Gaussian + CLG
        land in Stage 3.

Reference: docs/v0.13-benchmark-redesign.md §4.1, §5.2
"""
from nbn.bench.problems.bnlearn import BnlearnConfig, BnlearnProblemSource
from nbn.bench.problems.synthetic import SyntheticConfig, SyntheticProblemSource

__all__ = [
    "BnlearnConfig",
    "BnlearnProblemSource",
    "SyntheticConfig",
    "SyntheticProblemSource",
]
