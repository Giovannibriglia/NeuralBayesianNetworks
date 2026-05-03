"""Unit tests for the Dirac-tight-Gaussian intervention mechanism."""
from __future__ import annotations

import math

import pytest
import torch

from nbn import config
from nbn.mechanisms import DiracGaussianMechanism


def test_sample_concentrated_at_value():
    mech = DiracGaussianMechanism(value=2.0, sigma=1e-4)
    s = mech.sample(parents=None, n=10000)
    # All samples within ~5σ of value
    assert (s - 2.0).abs().max().item() < 5e-3


def test_log_prob_finite_at_value():
    mech = DiracGaussianMechanism(value=2.0, sigma=1e-4)
    x = torch.tensor([[2.0]])
    lp = mech.log_prob(x, parents=None)
    expected = -math.log(mech.sigma * math.sqrt(2 * math.pi))
    assert torch.isfinite(lp).all()
    assert abs(lp.item() - expected) < 1e-3


def test_value_is_learnable_parameter():
    mech = DiracGaussianMechanism(value=2.0)
    params = list(mech.parameters())
    assert len(params) == 1
    assert params[0].requires_grad


def test_autograd_through_intervention_value():
    """Loss = -log p(x|do) should give a non-None grad on the do value."""
    mech = DiracGaussianMechanism(value=2.0)
    x = torch.tensor([[2.05]])
    lp = mech.log_prob(x, parents=None)
    (-lp).sum().backward()
    assert mech._value.grad is not None
    assert torch.isfinite(mech._value.grad).all()


def test_default_sigma_uses_config():
    mech = DiracGaussianMechanism(value=0.0)
    assert mech.sigma == config.DIRAC_GAUSSIAN_SIGMA


def test_vector_value():
    val = torch.tensor([1.0, -1.0, 0.5])
    mech = DiracGaussianMechanism(value=val, output_dim=3)
    pa = torch.randn(4, 2)
    s = mech.sample(pa, n=5)
    assert s.shape == (4, 5, 3)
    assert (s.mean(dim=(0, 1)) - val).abs().max().item() < 1e-2
