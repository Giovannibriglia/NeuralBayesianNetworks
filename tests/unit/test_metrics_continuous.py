"""Tests for the v0.3 continuous metrics."""
from __future__ import annotations

import math

import torch

from benchmarking.metrics import (
    js_normalized,
    kl_div_continuous_knn,
    tv_continuous,
)


def test_tv_continuous_self_zero():
    x = torch.randn(2000)
    r = tv_continuous(x, x.clone())
    assert r.value < 1e-3


def test_tv_continuous_disjoint_one():
    x = torch.randn(1000) - 10
    y = torch.randn(1000) + 10
    r = tv_continuous(x, y)
    assert r.value > 0.95   # essentially disjoint


def test_kl_knn_self_near_zero():
    torch.manual_seed(0)
    x = torch.randn(800, 1)
    r = kl_div_continuous_knn(x, x.clone(), k=5)
    assert math.isfinite(r.value)
    # Self-KL is unbiased toward zero but the k-NN estimator has variance
    assert abs(r.value) < 0.3


def test_kl_knn_shifted_gaussians_monotone():
    torch.manual_seed(0)
    x = torch.randn(800, 1)
    kl_small = kl_div_continuous_knn(x, x + 0.5, k=5).value
    kl_large = kl_div_continuous_knn(x, x + 3.0, k=5).value
    assert kl_large > kl_small


def test_js_normalized_bounds():
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.0, 1.0])
    r = js_normalized(p, q)
    assert 0.0 <= r.value <= 1.0
