"""NBN benchmarking suite.

v0.5 surface (post-restructure):
    - Synthetic BN  : ``benchmarking.synthetic.make_synthetic_bn(family=..., n_nodes=...)``
    - Baselines     : ``benchmarking.baselines.get_adapter('nbn'|'pgmpy'|...)``
    - Crash test    : ``benchmarking.crash_test_runner.{run_parameter_learning, run_inference}``
    - Metrics       : ``benchmarking.metrics.{kl, js, tv, wasserstein, …}``

CLI entry: ``nbn-bench {param-learning, inference} --config <yaml>``.
"""
from benchmarking.crash_test_runner import (
    run_inference,
    run_parameter_learning,
)
from benchmarking.domains import (
    BenchmarkDomain,
    BenchmarkProblem,
    GroundTruth,
    Query,
)
from benchmarking.metrics import (
    js_divergence,
    kl_divergence,
    marginal_mae,
    tv_distance,
    wasserstein_1d,
)
from benchmarking.synthetic import (
    SyntheticBN,
    generate_synthetic_hybrid,
    make_synthetic_bn,
)

__all__ = [
    # Crash-test runners (CLI entry points)
    "run_parameter_learning",
    "run_inference",
    # Synthetic generator
    "make_synthetic_bn",
    "SyntheticBN",
    "generate_synthetic_hybrid",
    # Plugin contract types
    "BenchmarkDomain", "BenchmarkProblem", "GroundTruth", "Query",
    # Metrics
    "kl_divergence", "js_divergence", "marginal_mae",
    "tv_distance", "wasserstein_1d",
]
