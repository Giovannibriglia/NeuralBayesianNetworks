"""Smoke tests for the new benchmark metrics."""
from __future__ import annotations

import torch

from nbn.benchmarks.metrics import (
    brier_score,
    crps,
    energy_distance,
    js_divergence,
    kl_divergence,
    map_accuracy,
    mmd_rbf,
    tv_distance,
    wasserstein_1d,
)


def test_kl_self_zero():
    p = torch.tensor([0.2, 0.3, 0.5])
    assert kl_divergence(p, p).item() < 1e-6


def test_js_symmetric():
    p = torch.tensor([0.2, 0.3, 0.5])
    q = torch.tensor([0.3, 0.4, 0.3])
    assert abs(js_divergence(p, q).item() - js_divergence(q, p).item()) < 1e-6


def test_tv_bounds():
    p = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.0, 1.0])
    assert abs(tv_distance(p, q).item() - 1.0) < 1e-6


def test_wasserstein_self_zero():
    x = torch.randn(500)
    r = wasserstein_1d(x, x)
    assert r.value < 1e-3


def test_energy_distance_self_zero():
    x = torch.randn(200, 3)
    r = energy_distance(x, x)
    assert abs(r.value) < 0.5


def test_mmd_runs():
    x = torch.randn(100, 2)
    y = torch.randn(100, 2)
    r = mmd_rbf(x, y)
    # Unbiased MMD² estimator can be slightly negative; just bound it
    assert -0.5 < r.value < 1.0


def test_map_accuracy_perfect():
    pred = torch.tensor([0, 1, 2, 3])
    true = torch.tensor([0, 1, 2, 3])
    assert map_accuracy(pred, true).value == 1.0


def test_brier_score():
    probs = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    labels = torch.tensor([0, 1])
    r = brier_score(probs, labels)
    assert 0 < r.value < 1


def test_crps_runs():
    samples = torch.randn(200)
    obs = torch.tensor([0.0])
    r = crps(samples, obs)
    assert torch.is_tensor(torch.tensor(r.value))
