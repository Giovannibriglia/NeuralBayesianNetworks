"""Plugin-based benchmarking suite.

v0.2 surface:
    - Domains  : ``benchmarking.domains.get_domain('bnlearn'|'synthetic_hybrid')``
    - Baselines: ``benchmarking.baselines.get_adapter('nbn'|'pgmpy'|...)``
    - Runner   : ``benchmarking.runner.run(config)``
    - Metrics  : ``benchmarking.metrics.{kl,js,tv,wasserstein,…}``

Back-compat shims for the v0.1 API are kept below so old examples still work.
"""
from benchmarking.bnlearn_loader import BNLEARN_NETWORKS, load_bnlearn
from benchmarking.domains import (
    BenchmarkDomain,
    BenchmarkProblem,
    GroundTruth,
    Query,
    get_domain,
)
from benchmarking.metrics import (
    js_divergence,
    kl_divergence,
    marginal_mae,
    tv_distance,
    wasserstein_1d,
)
from benchmarking.synthetic import generate_synthetic_hybrid

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
