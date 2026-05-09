"""Unit tests for benchmarking.metrics._compute_metrics_per_node.

Pass 9 Phase 1 introduced a parametrized helper that averages
{tv_per_node, jsd_per_node, w1_per_node} over collections of
discrete CPD pairs and continuous sample pairs, with caller
partition deciding which list each pair goes into.

This module pins the helper's behaviour on hand-computable
fixtures.  Each fixture documents its analytical derivation
alongside the expected value — a wrong derivation should trip
when the implementation produces a different result, rather
than the test silently being adjusted to match buggy output.
This matches the Phase 1 fixture-derivation discipline carry-forward.
"""
from __future__ import annotations

import math

import torch

from benchmarking.metrics import _compute_metrics_per_node, js_divergence


# ── Discrete CPD pair fixtures ────────────────────────────────────────────


def test_uniform_vs_point_mass_K2():
    """K=2 uniform vs near-point-mass.

    Analytical derivation:
      p = (0.5, 0.5), q ≈ (1, 0)
      m = (p+q)/2 = (0.75, 0.25)
      KL(p||m) = 0.5*log(0.5/0.75) + 0.5*log(0.5/0.25)
              = 0.5*(log 2 - log 3) + 0.5*log 2
              = log 2 - 0.5*log 3 ≈ 0.1438
      KL(q||m) = log(1/0.75) = log(4/3) ≈ 0.2877
      JSD = 0.5*(KL₁ + KL₂) ≈ 0.2158
      JSD/log(2) ≈ 0.3113
    TV(p,q) = 0.5*(|0.5-1| + |0.5-0|) = 0.5
    """
    true_logits = torch.log(torch.tensor([0.5, 0.5]))
    fit_logits = torch.log(torch.tensor([1.0 - 1e-9, 1e-9]))
    res = _compute_metrics_per_node(
        cpd_pairs=[(true_logits, fit_logits)],
        metrics=("tv", "jsd", "w1"),
    )
    assert abs(res["tv_per_node"] - 0.5) < 1e-3
    assert abs(res["jsd_per_node"] - 0.3113) < 1e-3
    # W1 is skipped on pure-discrete input.
    assert res["w1_per_node"] is None


def test_disjoint_point_masses_max_divergence():
    """K=2 disjoint point masses — JSD/log(2) at theoretical max=1.0.

    p ≈ (1, 0), q ≈ (0, 1).  m = (0.5, 0.5).
    KL(p||m) = log(1/0.5) = log 2.  Same for KL(q||m).
    JSD = log 2.  JSD/log(2) = 1.0 (max).
    TV = 0.5*(1 + 1) = 1.0 (max).
    """
    p = torch.tensor([1.0 - 1e-9, 1e-9])
    q = torch.tensor([1e-9, 1.0 - 1e-9])
    res = _compute_metrics_per_node(
        cpd_pairs=[(torch.log(p), torch.log(q))],
        metrics=("tv", "jsd"),
    )
    assert abs(res["tv_per_node"] - 1.0) < 1e-3
    assert res["jsd_per_node"] > 0.999


def test_symmetric_bernoulli_flip():
    """Bernoulli (0.3, 0.7) vs (0.7, 0.3).

    TV = 0.5*(|0.3-0.7| + |0.7-0.3|) = 0.5*0.8 = 0.4.
    JSD/log(2): no clean closed form on K=2 Bernoulli;
    cross-check against benchmarking.metrics.js_divergence.
    """
    p = torch.tensor([0.3, 0.7])
    q = torch.tensor([0.7, 0.3])
    res = _compute_metrics_per_node(
        cpd_pairs=[(torch.log(p), torch.log(q))],
        metrics=("tv", "jsd"),
    )
    assert abs(res["tv_per_node"] - 0.4) < 1e-3
    expected_jsd = float(js_divergence(p, q) / math.log(2))
    assert abs(res["jsd_per_node"] - expected_jsd) < 1e-6


def test_pinsker_bound_jsd_leq_tv_random():
    """Lin 1991: JSD(p,q)/log(2) ≤ TV(p,q) for any p, q on a finite alphabet.

    Sample 100 random Dirichlet pairs; assert no violations.
    """
    torch.manual_seed(0)
    n_violations = 0
    for _ in range(100):
        K = int(torch.randint(2, 6, (1,)).item())
        p = torch.distributions.Dirichlet(torch.ones(K)).sample()
        q = torch.distributions.Dirichlet(torch.ones(K)).sample()
        res = _compute_metrics_per_node(
            cpd_pairs=[(torch.log(p), torch.log(q))],
            metrics=("tv", "jsd"),
        )
        if res["jsd_per_node"] > res["tv_per_node"] + 1e-6:
            n_violations += 1
    assert n_violations == 0, (
        f"JSD/log(2) > TV in {n_violations}/100 cases — Pinsker (Lin 1991) "
        "bound violated; suggests JSD normalisation regression."
    )


# ── Continuous sample pair fixtures ───────────────────────────────────────


def test_gaussian_shifted_mean_w1_closed_form():
    """N(0, 1) vs N(1, 1): W₁ = |μ_a - μ_b| = 1.0 (closed form for
    equal-variance Gaussians).

    n=5000 samples should give W₁ within 0.05 of 1.0.
    """
    torch.manual_seed(42)
    true_s = torch.randn(5000)
    fit_s = torch.randn(5000) + 1.0
    res = _compute_metrics_per_node(
        sample_pairs=[(true_s, fit_s)],
        metrics=("tv", "jsd", "w1"),
    )
    assert abs(res["w1_per_node"] - 1.0) < 0.05
    # Binned TV/JSD: distributions overlap heavily at 1σ separation;
    # values should be moderate but non-zero.
    assert 0.2 < res["tv_per_node"] < 0.6
    assert 0.0 < res["jsd_per_node"] < 0.5


def test_identical_samples_all_metrics_zero():
    """Identical sample tensors → W₁ = TV = JSD = 0."""
    torch.manual_seed(42)
    s = torch.randn(2000)
    res = _compute_metrics_per_node(
        sample_pairs=[(s, s.clone())],
        metrics=("tv", "jsd", "w1"),
    )
    assert res["w1_per_node"] < 1e-3
    assert res["tv_per_node"] < 1e-3
    assert res["jsd_per_node"] < 1e-3


# ── Hybrid: both lists populated ──────────────────────────────────────────


def test_hybrid_both_lists_populated():
    """Hybrid case: 1 discrete pair + 1 continuous pair.

    Helper averages each metric over all contributing pairs:
    - TV averages over (discrete TV) + (binned continuous TV).
    - JSD averages over (discrete JSD) + (binned continuous JSD).
    - W1 averages over the continuous pair only (discrete-unordered skip).

    Each metric's value should be non-None and sanity-bounded.
    """
    torch.manual_seed(0)
    res = _compute_metrics_per_node(
        cpd_pairs=[(
            torch.log(torch.tensor([0.3, 0.7])),
            torch.log(torch.tensor([0.7, 0.3])),
        )],
        sample_pairs=[(torch.randn(2000), torch.randn(2000) + 1.0)],
        metrics=("tv", "jsd", "w1"),
    )
    assert res["tv_per_node"] is not None and 0 < res["tv_per_node"] < 1
    assert res["jsd_per_node"] is not None and 0 < res["jsd_per_node"] < 1
    assert res["w1_per_node"] is not None and 0.5 < res["w1_per_node"] < 1.5


# ── Contract tests ────────────────────────────────────────────────────────


def test_empty_inputs_return_none_per_metric():
    """No pairs in either list → each requested metric returns None."""
    res = _compute_metrics_per_node(metrics=("tv", "jsd", "w1"))
    assert res == {
        "tv_per_node": None,
        "jsd_per_node": None,
        "w1_per_node": None,
    }


def test_subset_of_metrics_returns_only_requested_keys():
    """metrics=('tv',) → output dict has only tv_per_node, no other keys."""
    res = _compute_metrics_per_node(
        cpd_pairs=[(
            torch.log(torch.tensor([0.5, 0.5])),
            torch.log(torch.tensor([0.6, 0.4])),
        )],
        metrics=("tv",),
    )
    assert set(res.keys()) == {"tv_per_node"}
    assert res["tv_per_node"] is not None
