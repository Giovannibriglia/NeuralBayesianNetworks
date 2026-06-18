"""Unit tests for calibration primitives (#109 PR 7).

Covers metrics.calibration_pit_ks / calibration_sd_ratio and the single-node
helpers against DETERMINISTIC constructions:
  - perfectly-uniform PIT -> pit_ks == 0;
  - degenerate non-uniform PIT (all mass below y) -> pit_ks ~ 1;
  - sd_ratio is exact under scaling (pred = oracle * c -> ratio == c):
    c == 1 (right), c < 1 (over-confident), c > 1 (under-confident);
  - per-cell aggregation = mean over continuous nodes;
  - empty input -> nan (the not_applicable gate upstream).
"""
from __future__ import annotations

import math

import torch

from benchmarking.metrics import (
    _pit_ks_single,
    _sd_ratio_single,
    calibration_pit_ks,
    calibration_sd_ratio,
)


# ---- PIT-KS ----------------------------------------------------------------

def test_pit_ks_perfectly_uniform_is_zero():
    # Construct PIT_i = (i+1)/N exactly: each row's samples are [0..N-1] and
    # y_i = i, so (samples <= y_i).mean() = (i+1)/N. Sorted PIT == uniform grid.
    n = 100
    samples = torch.arange(n).float().repeat(n, 1)   # [N, N], each row 0..N-1
    y = torch.arange(n).float()                       # y_i = i
    ks = _pit_ks_single(samples, y)
    assert math.isclose(ks, 0.0, abs_tol=1e-9)


def test_pit_ks_degenerate_nonuniform_is_large():
    # All predictive mass below every y -> PIT_i = 1 for all i -> maximally
    # non-uniform -> KS ~ 1 - 1/N.
    n = 100
    samples = torch.zeros(n, 50)
    y = torch.full((n,), 5.0)
    ks = _pit_ks_single(samples, y)
    assert ks > 0.95


# ---- sd_ratio (exact under scaling) ----------------------------------------

def _oracle(n=200, s=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, s, generator=g)


def test_sd_ratio_perfect_is_one():
    o = _oracle()
    assert math.isclose(_sd_ratio_single(o.clone(), o), 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_sd_ratio_overconfident_below_one():
    o = _oracle()
    # predictive half as wide -> per-row ratio exactly 0.5 -> mean 0.5.
    assert math.isclose(_sd_ratio_single(o * 0.5, o), 0.5, rel_tol=1e-9, abs_tol=1e-9)


def test_sd_ratio_underconfident_above_one():
    o = _oracle()
    assert math.isclose(_sd_ratio_single(o * 2.0, o), 2.0, rel_tol=1e-9, abs_tol=1e-9)


def test_sd_ratio_oracle_zero_std_no_div0():
    # Degenerate oracle (point mass) -> clamp guard keeps it finite, not inf/nan.
    o = torch.zeros(10, 50)
    pred = torch.randn(10, 50)
    v = _sd_ratio_single(pred, o)
    assert math.isfinite(v)


# ---- per-cell aggregation --------------------------------------------------

def test_pit_ks_aggregates_mean_over_nodes():
    n = 100
    uniform_samples = torch.arange(n).float().repeat(n, 1)
    uniform_y = torch.arange(n).float()                 # pit_ks ~ 0
    bad_samples = torch.zeros(n, 50)
    bad_y = torch.full((n,), 5.0)                        # pit_ks ~ 1
    r = calibration_pit_ks([uniform_samples, bad_samples], [uniform_y, bad_y])
    assert r.name == "calibration_pit_ks"
    # mean of ~0 and ~0.99
    assert 0.45 < r.value < 0.55


def test_sd_ratio_aggregates_mean_over_nodes():
    o = _oracle()
    # node A perfect (1.0), node B over-confident (0.5) -> mean 0.75
    r = calibration_sd_ratio([o.clone(), o * 0.5], [o, o])
    assert r.name == "calibration_sd_ratio"
    assert math.isclose(r.value, 0.75, rel_tol=1e-6, abs_tol=1e-6)


def test_empty_input_returns_nan():
    assert math.isnan(calibration_pit_ks([], []).value)
    assert math.isnan(calibration_sd_ratio([], []).value)
