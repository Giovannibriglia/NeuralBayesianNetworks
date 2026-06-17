"""Unit tests for the held-out log-likelihood metric (issue #109 primitive)."""
from __future__ import annotations

import torch

from benchmarking.metrics import log_likelihood


def test_name_is_exactly_log_likelihood():
    """Consumer (_paper_figures) reads the literal string with no suffix."""
    r = log_likelihood(torch.zeros(4))
    assert r.name == "log_likelihood"


def test_value_matches_analytic_mean():
    """Value is the plain mean of the per-row joint log-probs (no negation)."""
    log_probs = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    # Analytic mean = (-1-2-3-4)/4 = -2.5
    assert abs(log_likelihood(log_probs).value - (-2.5)) < 1e-9


def test_positive_log_probs_give_positive_value():
    """Sign is positive mean log-prob, NOT negated (higher-is-better)."""
    log_probs = torch.tensor([0.5, 1.5])
    assert log_likelihood(log_probs).value == 1.0


def test_higher_is_better_ordering():
    """A better fit (less-negative per-row log-probs) scores strictly higher."""
    worse = torch.tensor([-3.0, -3.0, -3.0])
    better = torch.tensor([-1.0, -1.0, -1.0])
    assert log_likelihood(better).value > log_likelihood(worse).value
