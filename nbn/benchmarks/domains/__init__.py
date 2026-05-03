"""Domain registry — used by the runner to dispatch ``domain:`` in YAML."""
from nbn.benchmarks.domains.base import (
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
    "get_domain",
]

_DOMAIN_REGISTRY = {
    "bnlearn": ("nbn.benchmarks.domains.bnlearn", "BnlearnDomain"),
    "synthetic_hybrid": ("nbn.benchmarks.domains.synthetic_hybrid", "SyntheticHybridDomain"),
}


def get_domain(name: str, **kw) -> BenchmarkDomain:
    if name not in _DOMAIN_REGISTRY:
        raise ValueError(f"Unknown domain '{name}'. Available: {list(_DOMAIN_REGISTRY)}")
    import importlib
    mod_path, cls = _DOMAIN_REGISTRY[name]
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls)(**kw)
