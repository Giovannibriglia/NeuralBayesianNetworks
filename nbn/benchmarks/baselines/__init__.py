"""Baseline adapters. Each adapter is lazy — optional deps don't break import."""
from nbn.benchmarks.baselines.base import BaselineAdapter
from nbn.benchmarks.baselines.nbn_adapter import NBNAdapter

__all__ = ["BaselineAdapter", "NBNAdapter", "get_adapter"]

_REGISTRY = {
    "nbn": ("nbn.benchmarks.baselines.nbn_adapter", "NBNAdapter"),
    "pgmpy": ("nbn.benchmarks.baselines.pgmpy_adapter", "PgmpyAdapter"),
    "gpytorch": ("nbn.benchmarks.baselines.gpytorch_adapter", "GPyTorchAdapter"),
    "pyro": ("nbn.benchmarks.baselines.pyro_adapter", "PyroAdapter"),
}


def get_adapter(name: str, **kw):
    """Lazy-instantiate a baseline by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(_REGISTRY)}")
    import importlib
    module_path, cls = _REGISTRY[name]
    mod = importlib.import_module(module_path)
    return getattr(mod, cls)(**kw)
