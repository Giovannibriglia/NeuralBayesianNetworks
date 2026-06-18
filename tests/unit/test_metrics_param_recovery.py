"""Unit tests for parameter-recovery primitives (#109 PR 2).

Covers metrics.param_recovery_tv / param_recovery_kl and the
frequency_weights helper against hand-computed values:
  - TV / KL on known distributions, unweighted and frequency-weighted;
  - KL = +inf when learned puts zero mass on a true-supported class
    (NOT NaN, NOT a clamped finite number);
  - KL 0*log(0)=0 convention for true zeros (finite, no NaN);
  - KL asymmetry, direction = KL(true ‖ learned);
  - zero-weight config with an inf row contributes 0 (no 0*inf NaN);
  - node-mean reduction across nodes;
  - frequency_weights normalization.
"""
from __future__ import annotations

import math

import torch

from benchmarking.metrics import (
    frequency_weights,
    param_recovery_kl,
    param_recovery_tv,
)


# ---- TV ---------------------------------------------------------------------

def test_tv_single_config_unweighted():
    # TV([0.5,0.5], [0.25,0.75]) = 0.5*(0.25+0.25) = 0.25
    true = [torch.tensor([[0.5, 0.5]])]
    learned = [torch.tensor([[0.25, 0.75]])]
    w = [torch.tensor([1.0])]
    r = param_recovery_tv(true, learned, w)
    assert r.name == "param_recovery_tv"
    assert math.isclose(r.value, 0.25, rel_tol=1e-6, abs_tol=1e-6)


def test_tv_frequency_weighted():
    # config0: TV=0.25 (w=0.8); config1: TV([1,0],[0,1])=1.0 (w=0.2)
    # node TV = 0.8*0.25 + 0.2*1.0 = 0.4
    true = [torch.tensor([[0.5, 0.5], [1.0, 0.0]])]
    learned = [torch.tensor([[0.25, 0.75], [0.0, 1.0]])]
    w = [torch.tensor([0.8, 0.2])]
    r = param_recovery_tv(true, learned, w)
    assert math.isclose(r.value, 0.4, rel_tol=1e-6, abs_tol=1e-6)


def test_node_mean_reduction():
    # two single-config nodes: TV 0.25 and 1.0 -> mean 0.625
    true = [torch.tensor([[0.5, 0.5]]), torch.tensor([[1.0, 0.0]])]
    learned = [torch.tensor([[0.25, 0.75]]), torch.tensor([[0.0, 1.0]])]
    w = [torch.tensor([1.0]), torch.tensor([1.0])]
    r = param_recovery_tv(true, learned, w)
    assert math.isclose(r.value, 0.625, rel_tol=1e-6, abs_tol=1e-6)


def test_empty_input_returns_nan():
    assert math.isnan(param_recovery_tv([], [], []).value)
    assert math.isnan(param_recovery_kl([], [], []).value)


# ---- KL ---------------------------------------------------------------------

def test_kl_single_config_value():
    # KL([0.5,0.5] || [0.25,0.75]) = 0.5*ln2 + 0.5*ln(2/3) = 0.1438410...
    true = [torch.tensor([[0.5, 0.5]])]
    learned = [torch.tensor([[0.25, 0.75]])]
    w = [torch.tensor([1.0])]
    expected = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
    r = param_recovery_kl(true, learned, w)
    assert r.name == "param_recovery_kl"
    assert math.isclose(r.value, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_kl_diverges_to_inf_on_learned_zero():
    # learned puts 0 mass on a class the truth supports -> +inf (not NaN/finite)
    true = [torch.tensor([[0.5, 0.5]])]
    learned = [torch.tensor([[1.0, 0.0]])]
    w = [torch.tensor([1.0])]
    r = param_recovery_kl(true, learned, w)
    assert math.isinf(r.value) and r.value > 0


def test_kl_zero_times_log_zero_is_zero():
    # true has a zero entry; learned finite there -> 0*log term contributes 0,
    # KL = 1*ln(1/0.5) = ln2, finite, no NaN.
    true = [torch.tensor([[1.0, 0.0]])]
    learned = [torch.tensor([[0.5, 0.5]])]
    w = [torch.tensor([1.0])]
    r = param_recovery_kl(true, learned, w)
    assert not math.isnan(r.value)
    assert math.isclose(r.value, math.log(2.0), rel_tol=1e-6, abs_tol=1e-6)


def test_kl_both_zero_same_class_no_nan():
    # true and learned both zero on the same class -> 0*log(0)=0, no NaN.
    true = [torch.tensor([[0.5, 0.5, 0.0]])]
    learned = [torch.tensor([[0.5, 0.5, 0.0]])]
    w = [torch.tensor([1.0])]
    r = param_recovery_kl(true, learned, w)
    assert not math.isnan(r.value)
    assert math.isclose(r.value, 0.0, abs_tol=1e-6)


def test_kl_asymmetry_and_direction():
    p = torch.tensor([[0.5, 0.5]])
    q = torch.tensor([[0.25, 0.75]])
    w = [torch.tensor([1.0])]
    kl_pq = param_recovery_kl([p], [q], w).value   # KL(true=p || learned=q)
    kl_qp = param_recovery_kl([q], [p], w).value   # KL(true=q || learned=p)
    assert not math.isclose(kl_pq, kl_qp, rel_tol=1e-6)
    # Direction check: KL(true ‖ learned) = sum true * log(true/learned)
    expected_pq = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert math.isclose(kl_pq, expected_pq, rel_tol=1e-6, abs_tol=1e-6)


def test_zero_weight_inf_row_contributes_zero():
    # config0: finite KL, weight 1. config1: divergent (inf) KL but weight 0.
    # The zero-weight inf row must contribute 0 (no 0*inf NaN), so node KL is
    # config0's finite value.
    true = [torch.tensor([[0.5, 0.5], [0.5, 0.5]])]
    learned = [torch.tensor([[0.25, 0.75], [1.0, 0.0]])]
    w = [torch.tensor([1.0, 0.0])]
    r = param_recovery_kl(true, learned, w)
    expected = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
    assert not math.isnan(r.value)
    assert math.isclose(r.value, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_positive_weight_inf_row_propagates_inf():
    # Same rows but the divergent config now carries positive weight -> inf.
    true = [torch.tensor([[0.5, 0.5], [0.5, 0.5]])]
    learned = [torch.tensor([[0.25, 0.75], [1.0, 0.0]])]
    w = [torch.tensor([0.5, 0.5])]
    r = param_recovery_kl(true, learned, w)
    assert math.isinf(r.value) and r.value > 0


# ---- frequency_weights ------------------------------------------------------

def test_frequency_weights_normalizes():
    w = frequency_weights(torch.tensor([2.0, 6.0, 2.0]))
    assert torch.allclose(w, torch.tensor([0.2, 0.6, 0.2], dtype=torch.float64))
    assert math.isclose(float(w.sum()), 1.0, rel_tol=1e-12)


def test_frequency_weights_all_zero_no_div0():
    # Degenerate all-zero counts: clamp keeps it finite (all-zero weights).
    w = frequency_weights(torch.zeros(3))
    assert torch.all(w == 0)
    assert not torch.isnan(w).any()
