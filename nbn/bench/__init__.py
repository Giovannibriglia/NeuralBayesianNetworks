"""NBN benchmarking suite — v0.13.

v0.13 surface (post-Phase 1c cutover):
    - Runner           : ``nbn.bench.core.Runner``
    - RunnerConfig     : ``nbn.bench.core.RunnerConfig``
    - BaselineSpec     : ``nbn.bench.core.BaselineSpec``
    - build_adapter    : ``nbn.bench.core.build_adapter``
    - Protocols        : ``nbn.bench.core.{BaselineAdapter, Measurement, …}``
    - Applicability    : ``nbn.bench.core.{is_applicable, BASELINE_FAMILY_APPLICABILITY}``
    - Synthetic        : ``nbn.bench.synthetic.make_synthetic_bn``
    - Metrics          : ``nbn.bench.metrics.{wasserstein_1d, …}``

CLI entry: ``nbn-bench {inference, param-learning} --config <yaml>``.
``param-learning`` scores held-out joint log-likelihood via
ParamLearningMeasurement (#109); adapters opt in with ``supports_scoring``.
"""
import nbn.bench._extras  # noqa: F401  -- actionable error if the bench extra is missing
from nbn.bench.core import (
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
from nbn.bench.domains import (
    BenchmarkDomain,
    BenchmarkProblem,
    GroundTruth,
    Query,
)
from nbn.bench.metrics import (
    js_divergence,
    kl_divergence,
    marginal_mae,
    tv_distance,
    wasserstein_1d,
)
from nbn.bench.synthetic import (
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
