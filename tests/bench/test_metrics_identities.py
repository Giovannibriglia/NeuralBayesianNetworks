"""Identity, symmetry and range checks for the benchmark metrics.

The analytic, hand-computed tests live per metric family
(test_metrics_per_node.py, test_metrics_param_recovery.py,
test_metrics_calibration.py, test_metrics_log_likelihood.py,
test_metrics_continuous_w1_ks.py); this module pins the cheap invariants every
divergence must satisfy.  Merged from test_metrics.py (v0.3 smoke tests) and
test_metrics_continuous.py.
"""
from __future__ import annotations

import math

import pytest
import torch

from nbn.bench.metrics import (
    brier_score,
    crps,
    energy_distance,
    js_divergence,
    js_normalized,
    kl_div_continuous_knn,
    kl_divergence,
    map_accuracy,
    mmd_rbf,
    tv_continuous,
    tv_distance,
    wasserstein_1d,
)


def _value(r):
    return float(r.value) if hasattr(r, "value") else float(r)


_P = torch.tensor([0.2, 0.3, 0.5])

# name -> (callable producing the divergence of a sample/distribution with
# itself, tolerance on |value|).  Estimators with sampling noise get a bound,
# not ~0: the k-NN KL is unbiased toward zero but has variance; the energy
# distance of a sample with itself is a V-statistic.
_SELF_DISTANCE = {
    "kl_divergence": (lambda: kl_divergence(_P, _P), 1e-6),
    "wasserstein_1d": (lambda: (x := torch.randn(500), wasserstein_1d(x, x))[1], 1e-3),
    "energy_distance": (lambda: (x := torch.randn(200, 3), energy_distance(x, x))[1], 0.5),
    "tv_continuous": (lambda: (x := torch.randn(2000), tv_continuous(x, x.clone()))[1], 1e-3),
    "kl_div_continuous_knn": (
        lambda: (x := torch.randn(800, 1), kl_div_continuous_knn(x, x.clone(), k=5))[1], 0.3),
}


@pytest.mark.parametrize("metric", sorted(_SELF_DISTANCE))
def test_divergence_of_a_distribution_with_itself_is_zero(metric):
    fn, tol = _SELF_DISTANCE[metric]
    v = _value(fn())
    assert math.isfinite(v) and abs(v) < tol


def test_js_divergence_is_symmetric():
    q = torch.tensor([0.3, 0.4, 0.3])
    assert abs(js_divergence(_P, q).item() - js_divergence(q, _P).item()) < 1e-6


@pytest.mark.parametrize("metric,fn,lo,hi", [
    ("tv_distance disjoint = 1",
     lambda: tv_distance(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])), 1 - 1e-6, 1 + 1e-6),
    ("tv_continuous essentially disjoint",
     lambda: tv_continuous(torch.randn(1000) - 10, torch.randn(1000) + 10), 0.95, 1.0 + 1e-9),
    ("js_normalized in [0, 1]",
     lambda: js_normalized(torch.tensor([0.5, 0.5]), torch.tensor([0.0, 1.0])), 0.0, 1.0),
    # The unbiased MMD^2 estimator can be slightly negative.
    ("mmd_rbf bounded", lambda: mmd_rbf(torch.randn(100, 2), torch.randn(100, 2)), -0.5, 1.0),
    ("brier_score in (0, 1)",
     lambda: brier_score(torch.tensor([[0.7, 0.3], [0.4, 0.6]]), torch.tensor([0, 1])),
     1e-12, 1 - 1e-12),
])
def test_metric_range(metric, fn, lo, hi):
    v = _value(fn())
    assert lo <= v <= hi, f"{metric}: {v}"


def test_kl_knn_grows_with_the_shift():
    x = torch.randn(800, 1)
    kl_small = kl_div_continuous_knn(x, x + 0.5, k=5).value
    kl_large = kl_div_continuous_knn(x, x + 3.0, k=5).value
    assert kl_large > kl_small


def test_map_accuracy_perfect_prediction_is_one():
    pred = torch.tensor([0, 1, 2, 3])
    assert map_accuracy(pred, pred.clone()).value == 1.0


def test_crps_returns_a_finite_number():
    r = crps(torch.randn(200), torch.tensor([0.0]))
    assert math.isfinite(float(r.value))
