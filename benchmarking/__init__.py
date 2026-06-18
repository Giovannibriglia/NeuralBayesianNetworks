"""NBN benchmarking suite — v0.13.

v0.13 surface (post-Phase 1c cutover):
    - Runner           : ``benchmarking.core.Runner``
    - RunnerConfig     : ``benchmarking.core.RunnerConfig``
    - BaselineSpec     : ``benchmarking.core.BaselineSpec``
    - build_adapter    : ``benchmarking.core.build_adapter``
    - Protocols        : ``benchmarking.core.{BaselineAdapter, Measurement, …}``
    - Applicability    : ``benchmarking.core.{is_applicable, BASELINE_FAMILY_APPLICABILITY}``
    - Synthetic        : ``benchmarking.synthetic.make_synthetic_bn``
    - Metrics          : ``benchmarking.metrics.{wasserstein_1d, …}``

CLI entry: ``nbn-bench {inference, param-learning} --config <yaml>``.
``param-learning`` scores held-out joint log-likelihood via
ParamLearningMeasurement (#109); adapters opt in with ``supports_scoring``.
"""
from benchmarking.core import (
    BASELINE_FAMILY_APPLICABILITY,
    BaselineAdapter,
    BaselineApplicability,
    BaselineSpec,
    CellResult,
    JsonlWriter,
    Measurement,
    ProblemSource,
    QuerySelector,
    Runner,
    RunnerConfig,
    accuracy_supported,
    build_adapter,
    is_applicable,
    known_labels,
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
    # v0.13 runner
    "Runner",
    "RunnerConfig",
    "BaselineSpec",
    "build_adapter",
    # v0.13 protocols
    "BaselineAdapter",
    "Measurement",
    "ProblemSource",
    "QuerySelector",
    # v0.13 schema
    "CellResult",
    "JsonlWriter",
    # Applicability registry
    "BASELINE_FAMILY_APPLICABILITY",
    "BaselineApplicability",
    "is_applicable",
    "accuracy_supported",
    "known_labels",
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
