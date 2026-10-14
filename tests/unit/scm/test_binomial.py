"""Binomial confidence intervals: formulas, nesting, coverage, devices."""
from __future__ import annotations

import math

import pytest
import torch

from nbn.uncertainty import anytime_delta, binomial_interval, kl_bernoulli, union_bound
from tests.conftest import available_devices


def _brute_force_kl_interval(k: int, n: int, delta: float) -> tuple[float, float]:
    grid = torch.linspace(0, 1, 2_000_001, dtype=torch.float64)
    p_hat = torch.tensor(k / n, dtype=torch.float64)
    inside = n * kl_bernoulli(p_hat, grid) <= math.log(2 / delta)
    return float(grid[inside].min()), float(grid[inside].max())


@pytest.mark.parametrize("k,n", [(0, 10), (3, 10), (10, 10), (17, 200)])
def test_kl_interval_matches_brute_force(k, n):
    lower, upper = binomial_interval(torch.tensor([k]), torch.tensor([n]), 0.05)
    ref_lower, ref_upper = _brute_force_kl_interval(k, n, 0.05)
    assert float(lower) == pytest.approx(ref_lower, abs=2e-6)
    assert float(upper) == pytest.approx(ref_upper, abs=2e-6)


def test_kl_interval_is_inside_hoeffding():
    """Pinsker: KL(p||q) >= 2 (p - q)^2, so the KL interval is the tighter one."""
    k = torch.arange(0, 51)
    n = torch.full_like(k, 50)
    kl_lo, kl_hi = binomial_interval(k, n, 0.05, method="kl")
    ho_lo, ho_hi = binomial_interval(k, n, 0.05, method="hoeffding")
    assert bool((kl_lo >= ho_lo - 1e-12).all()) and bool((kl_hi <= ho_hi + 1e-12).all())


def test_clopper_pearson_reference_values():
    pytest.importorskip("scipy")
    lower, upper = binomial_interval(torch.tensor([0, 5]), torch.tensor([10, 10]), 0.05,
                                     method="clopper_pearson")
    assert lower.tolist() == pytest.approx([0.0, 0.187086], abs=1e-6)
    assert upper.tolist() == pytest.approx([0.308497, 0.812914], abs=1e-6)


def test_no_trials_gives_the_whole_interval():
    lower, upper = binomial_interval(torch.tensor([0]), torch.tensor([0]), 0.05)
    assert float(lower) == 0.0 and float(upper) == 1.0


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError, match="successes"):
        binomial_interval(torch.tensor([5]), torch.tensor([3]), 0.05)
    with pytest.raises(ValueError, match="delta"):
        binomial_interval(torch.tensor([1]), torch.tensor([3]), 1.5)


@pytest.mark.parametrize("p", [0.03, 0.3, 0.5])
def test_kl_interval_covers_at_the_nominal_level(p):
    """Monte-Carlo coverage over 4000 replicates, all in one batched call."""
    replicates, n, delta = 4000, 60, 0.1
    gen = torch.Generator().manual_seed(1)
    k = torch.binomial(torch.full((replicates,), float(n)), torch.full((replicates,), p), generator=gen)
    lower, upper = binomial_interval(k, torch.full_like(k, n), delta)
    coverage = float(((lower <= p) & (p <= upper)).double().mean())
    assert coverage >= 1 - delta - 3 * math.sqrt(delta * (1 - delta) / replicates)


def test_budget_helpers():
    assert union_bound(0.05, 10) == pytest.approx(0.005)
    n = torch.arange(1, 100_001, dtype=torch.float64)
    assert float(anytime_delta(0.05, n).sum()) <= 0.05


@pytest.mark.parametrize("device", available_devices())
def test_interval_runs_on_every_device(device):
    k = torch.tensor([0.0, 3.0, 40.0], device=device)
    n = torch.tensor([10.0, 10.0, 50.0], device=device)
    lower, upper = binomial_interval(k, n, 0.05)
    ref_lower, ref_upper = binomial_interval(k.cpu(), n.cpu(), 0.05)
    assert lower.device.type == torch.device(device).type
    assert torch.allclose(lower.cpu().double(), ref_lower, atol=1e-5)
    assert torch.allclose(upper.cpu().double(), ref_upper, atol=1e-5)


def test_clopper_pearson_upper_end_not_saturated_at_tiny_levels():
    """beta.ppf(1 - d/2) rounded to 1.0 for d < 1.1e-16 (anytime levels at large n)."""
    pytest.importorskip("scipy")
    from nbn.uncertainty import binomial_interval

    k = torch.tensor([3e6], dtype=torch.float64)
    n = torch.tensor([4e6], dtype=torch.float64)
    lo, hi = binomial_interval(k, n, torch.tensor([1e-17], dtype=torch.float64), method="clopper_pearson")
    assert float(hi) < 0.76 and float(lo) > 0.74
