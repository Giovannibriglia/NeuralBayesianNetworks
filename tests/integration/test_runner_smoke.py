"""Smoke test the runner on a tiny synthetic-hybrid problem.

Verifies the plugin pipeline end-to-end without hitting the network or
requiring optional baselines.
"""
from __future__ import annotations

import torch

from benchmarking.runner import run


def test_runner_synthetic_hybrid_smoke(tmp_path):
    out = tmp_path / "out.parquet"
    df = run({
        "domain": "synthetic_hybrid",
        "problems": ["hybrid_10"],
        "n_train": 200,
        "n_test": 50,
        "seed": 0,
        "devices": ["cpu"],
        "baselines": ["nbn"],
        "query_kinds": ["marginal"],
        "output": str(out),
    })
    assert out.exists()
    # df may be a list or a DataFrame depending on optional pandas
    if hasattr(df, "shape"):
        assert df.shape[0] > 0
    else:
        assert len(df) > 0
