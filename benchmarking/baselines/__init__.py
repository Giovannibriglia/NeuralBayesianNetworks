"""Baseline adapters. Each adapter is lazy — optional deps don't break import."""
from benchmarking.baselines.base import BaselineAdapter
from benchmarking.baselines.nbn_adapter import NBNAdapter

__all__ = ["BaselineAdapter", "NBNAdapter", "get_adapter"]

_REGISTRY = {
    # --- NBN variants — every public engine + mechanism family is exposed ---
    "nbn":                    ("benchmarking.baselines.nbn_adapter", "NBNAdapter"),
    "nbn_ve":                 ("benchmarking.baselines.nbn_adapter", "NBNTensorVEAdapter"),
    "nbn_lw":                 ("benchmarking.baselines.nbn_adapter", "NBNLikelihoodWeightingAdapter"),
    "nbn_hybrid":             ("benchmarking.baselines.nbn_adapter", "NBNHybridRouterAdapter"),
    "nbn_neural_categorical": ("benchmarking.baselines.nbn_adapter", "NBNNeuralCategoricalAdapter"),
    "nbn_linear_gaussian":    ("benchmarking.baselines.nbn_adapter", "NBNLinearGaussianAdapter"),
    # --- external baselines ---
    "pgmpy":       ("benchmarking.baselines.pgmpy_adapter",       "PgmpyAdapter"),
    "gpytorch":    ("benchmarking.baselines.gpytorch_adapter",    "GPyTorchAdapter"),
    "pyro":        ("benchmarking.baselines.pyro_adapter",        "PyroAdapter"),
    "pomegranate": ("benchmarking.baselines.pomegranate_adapter", "PomegranateAdapter"),
}


def get_adapter(name: str, **kw):
    """Lazy-instantiate a baseline by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(_REGISTRY)}")
    import importlib
    module_path, cls = _REGISTRY[name]
    mod = importlib.import_module(module_path)
    return getattr(mod, cls)(**kw)
