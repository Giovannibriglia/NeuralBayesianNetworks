"""Regression tests for the mechanism-level review findings (fix/mechanisms).

Each section names the finding it pins (A4, A9, ...).  Every test failed on
the pre-fix code and passes after; the comments say what the failure was so a
future regression reads as the original bug, not as a mystery threshold.
"""
from __future__ import annotations

import math

import pytest
import torch

from nbn.mechanisms import LinearGaussianMechanism
from tests.conftest import available_devices


def _replicate(w: torch.Tensor) -> torch.Tensor:
    """Row indices repeating row i exactly w_i times."""
    return torch.cat([torch.full((int(k),), i) for i, k in enumerate(w)])


# ==========================================================================
# A4 -- LinearGaussian: lstsq drops the slope when a parent has a big offset
# ==========================================================================


def _offset_lg_data(n=2000, device="cpu"):
    g = torch.Generator().manual_seed(1)
    pa = 100.0 + torch.randn(n, 1, generator=g)
    x = 0.5 * pa + 100.0 + 0.1 * torch.randn(n, 1, generator=g)
    return x.to(device), pa.to(device)


@pytest.mark.parametrize("device", available_devices())
def test_a4_linear_gaussian_recovers_slope_under_large_parent_offset(device):
    # Pre-fix: the default gelsy driver treated the near-collinear
    # [parent, 1] design as rank deficient and returned w=1.50, b=0.015.
    x, pa = _offset_lg_data(device=device)
    mech = LinearGaussianMechanism()
    mech.fit_local(x, pa)
    assert abs(float(mech._weight.reshape(-1)[0]) - 0.5) < 0.02
    assert abs(float(mech._bias.reshape(-1)[0]) - 100.0) < 2.0
    assert abs(float(mech._scale().reshape(-1)[0]) - 0.1) < 0.01


def test_a4_linear_gaussian_matches_float64_ridge_normal_equations():
    # The standardised solve must keep the ridge semantics: penalty
    # ridge * ||W||^2 on the raw-unit weights, intercept unpenalised.
    g = torch.Generator().manual_seed(2)
    n = 300
    pa = torch.cat([50 + 3 * torch.randn(n, 1, generator=g),
                    -20 + 0.2 * torch.randn(n, 1, generator=g)], dim=1)
    x = pa @ torch.tensor([[1.5], [-0.7]]) + 7.0 + torch.randn(n, 1, generator=g)
    w = torch.randint(0, 4, (n,), generator=g).float()
    ridge = 5.0
    mech = LinearGaussianMechanism(ridge=ridge)
    mech.fit_local(x, pa, weights=w)

    z = torch.cat([pa, torch.ones(n, 1)], 1).double()
    pen = torch.diag(torch.tensor([ridge, ridge, 0.0], dtype=torch.float64))
    a = z.T @ (w.double()[:, None] * z) + pen
    b = z.T @ (w.double()[:, None] * x.double())
    theta = torch.linalg.solve(a, b)
    torch.testing.assert_close(mech._weight.double(), theta[:-1], atol=2e-3, rtol=1e-4)
    torch.testing.assert_close(mech._bias.double(), theta[-1], atol=5e-2, rtol=1e-4)


def test_a4_linear_gaussian_offset_fit_is_replication_and_zero_weight_exact():
    x, pa = _offset_lg_data(n=200)
    w = torch.randint(0, 4, (200,), generator=torch.Generator().manual_seed(3)).float()
    weighted = LinearGaussianMechanism()
    weighted.fit_local(x, pa, weights=w)
    replicated = LinearGaussianMechanism()
    replicated.fit_local(x[_replicate(w)], pa[_replicate(w)])
    for a, b in [(weighted._weight, replicated._weight),
                 (weighted._bias, replicated._bias),
                 (weighted._scale(), replicated._scale())]:
        torch.testing.assert_close(a, b, atol=2e-3, rtol=1e-4)
