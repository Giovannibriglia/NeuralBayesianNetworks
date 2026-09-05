"""Plugin contract types for benchmark domains.

The v0.5 crash-test runner consumes ``nbn.bench.synthetic.SyntheticBN``
directly and does not need a domain registry.  These types are kept so
``BaselineAdapter`` subclasses (which take ``BenchmarkProblem``) and
external code that builds ``Query`` objects continue to work.
"""
from nbn.bench.domains.base import (
    BenchmarkDomain,
    BenchmarkProblem,
    GroundTruth,
    Query,
)

__all__ = [
    "BenchmarkDomain",
    "BenchmarkProblem",
    "GroundTruth",
    "Query",
]
