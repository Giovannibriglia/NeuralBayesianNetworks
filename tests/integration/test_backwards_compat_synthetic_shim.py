"""v0.3 → v0.4 back-compat: ``generate_synthetic_hybrid`` shim still works."""
from __future__ import annotations

import warnings

import torch

from nbn.core.network import NeuralBayesianNetwork


def test_v03_generate_synthetic_hybrid_still_callable() -> None:
    """Old call site (positional, v0.3 keyword names) returns valid (model, data)."""
    from benchmarking.synthetic import generate_synthetic_hybrid
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = generate_synthetic_hybrid(
            n_nodes=12, discrete_frac=0.5, edge_prob=0.20, seed=0,
        )
    # Tuple shape preserved
    assert isinstance(result, tuple) and len(result) == 2
    model, data = result
    assert isinstance(model, NeuralBayesianNetwork)
    assert isinstance(data, dict)
    assert all(isinstance(v, torch.Tensor) for v in data.values())
    # Deprecation warning was raised
    msgs = [
        str(w.message) for w in caught
        if issubclass(w.category, DeprecationWarning)
    ]
    assert any("generate_synthetic_hybrid" in m and "deprecated" in m for m in msgs), (
        f"DeprecationWarning not raised; got categories "
        f"{[w.category.__name__ for w in caught]}"
    )


def test_v03_export_path_still_works() -> None:
    """``from benchmarking import generate_synthetic_hybrid`` keeps working."""
    from benchmarking import generate_synthetic_hybrid
    assert callable(generate_synthetic_hybrid)
