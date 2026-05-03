"""Plugin-based benchmarking suite.

v0.2 surface:
    - Domains  : ``nbn.benchmarks.domains.get_domain('bnlearn'|'synthetic_hybrid')``
    - Baselines: ``nbn.benchmarks.baselines.get_adapter('nbn'|'pgmpy'|...)``
    - Runner   : ``nbn.benchmarks.runner.run(config)``
    - Metrics  : ``nbn.benchmarks.metrics.{kl,js,tv,wasserstein,…}``

Back-compat shims for the v0.1 API are kept below so old examples still work.
"""
from nbn.benchmarks.bnlearn_loader import BNLEARN_NETWORKS, load_bnlearn
from nbn.benchmarks.domains import (
    BenchmarkDomain,
    BenchmarkProblem,
    GroundTruth,
    Query,
    get_domain,
)
from nbn.benchmarks.metrics import (
    js_divergence,
    kl_divergence,
    marginal_mae,
    tv_distance,
    wasserstein_1d,
)
from nbn.benchmarks.synthetic import generate_synthetic_hybrid

__all__ = [
    "load_bnlearn",
    "BNLEARN_NETWORKS",
    "generate_synthetic_hybrid",
    # Plugin contract
    "BenchmarkDomain", "BenchmarkProblem", "GroundTruth", "Query",
    "get_domain",
    # Metrics
    "kl_divergence", "js_divergence", "marginal_mae",
    "tv_distance", "wasserstein_1d",
]
