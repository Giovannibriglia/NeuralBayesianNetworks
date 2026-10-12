"""Neural structural mechanisms and calibrated uncertainty (flows, PIT band, conformal)."""
from __future__ import annotations

import math
import pickle

import pytest
import torch
from torch.distributions import Normal

from nbn.scm import (
    AdditiveNoiseMechanism,
    FlowNoiseMechanism,
    GaussianMixtureNoise,
    StructuralCausalModel,
    conformal_intervals,
    noise_independence_test,
    pit_band_set,
)
from tests.conftest import available_devices

pytest.importorskip("zuko", reason="flow mechanisms need the neural extra")


def _hetero(n: int, seed: int, slope: float = 1.0):
    """y = slope * x + (0.2 + 0.5 |x|) * eps: a noise scale that depends on the parent."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, generator=gen)
    y = slope * x + (0.2 + 0.5 * x.abs()) * torch.randn(n, 1, generator=gen)
    return {"x": x, "y": y}


def _homo(n: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, generator=gen)
    return {"x": x, "y": 1.5 * x + 0.4 * torch.randn(n, 1, generator=gen)}


def _scm(mech, device="cpu"):
    model = StructuralCausalModel([("x", "y")], {"x": ("continuous", 1), "y": ("continuous", 1)},
                                  device=device)
    model.set_mechanism("x", AdditiveNoiseMechanism())
    model.set_mechanism("y", mech)
    return model


@pytest.fixture(scope="module")
def fitted_flow():
    torch.manual_seed(0)
    model = _scm(FlowNoiseMechanism(num_transforms=3, hidden=(32, 32)))
    model.fit(_hetero(4000, seed=0), epochs=120)
    return model


# ---------------------------------------------------------------------------
# FlowNoiseMechanism
# ---------------------------------------------------------------------------


def test_flow_abduction_inverts_the_structural_equation(fitted_flow):
    data = _hetero(200, seed=1)
    mech = fitted_flow.mechanisms["y"]
    u = mech.abduct(data["y"], data["x"], n_samples=2)
    assert u.shape == (2, 200, 1)
    assert torch.allclose(mech.structural(data["x"], u[0]), data["y"], atol=1e-4)


def test_flow_cdf_and_quantile_are_inverse(fitted_flow):
    mech = fitted_flow.mechanisms["y"]
    pa = torch.linspace(-2, 2, 9).unsqueeze(-1)
    for p in (0.05, 0.3, 0.5, 0.9):
        q = mech.quantile(p, pa)
        assert torch.allclose(mech.cdf(q.unsqueeze(-1), pa), torch.full((9,), p), atol=1e-4)
    # monotone in x for each parent value
    x_grid = torch.linspace(-4, 4, 50)
    cdf = torch.stack([mech.cdf(torch.full((9, 1), v), pa) for v in x_grid])
    assert bool((cdf[1:] >= cdf[:-1] - 1e-6).all())


def test_flow_tracks_a_noise_scale_that_varies_with_the_parent(fitted_flow):
    mech = fitted_flow.mechanisms["y"]
    pa = torch.tensor([[0.0], [2.0]])
    width = mech.quantile(0.9, pa) - mech.quantile(0.1, pa)
    true_width = 2 * 1.2816 * (0.2 + 0.5 * pa[:, 0].abs())
    # A short raw fit is not exact (that is what calibration is for): require
    # the right order of magnitude and, above all, the right shape.
    assert bool(((width / true_width) > 0.5).all()) and bool(((width / true_width) < 2.0).all())
    assert float(width[1]) > 2.0 * float(width[0])


def test_flow_counterfactuals_run_through_the_scm(fitted_flow):
    data = _hetero(5, seed=2)
    world = fitted_flow.counterfactual_sample(data, do={"x": 0.5})
    assert world["y"].shape == (1, 5, 1)
    # Counterfactual = T(T^{-1}(y; x_obs); 0.5): the same noise pushed through the new parent.
    mech = fitted_flow.mechanisms["y"]
    u = mech.abduct(data["y"], data["x"])[0]
    expected = mech.structural(torch.full((5, 1), 0.5), u)
    assert torch.allclose(world["y"][0], expected, atol=1e-5)


def test_root_flow_and_multivariate_flow():
    torch.manual_seed(0)
    root = FlowNoiseMechanism(num_transforms=2, hidden=(16,))
    y = 2.0 + 0.5 * torch.randn(1000, 1)
    root.fit_local(y, None, epochs=60)
    assert root.quantile(0.5, None).shape == (1,)
    assert abs(float(root.quantile(0.5, None)) - 2.0) < 0.15
    two_d = FlowNoiseMechanism(d_x=2, num_transforms=2, hidden=(16,))
    pa = torch.randn(500, 1)
    x = torch.cat([pa, -pa], 1) + 0.3 * torch.randn(500, 2)
    two_d.fit_local(x, pa, epochs=20)
    u = two_d.abduct(x[:10], pa[:10])[0]
    assert torch.allclose(two_d.structural(pa[:10], u), x[:10], atol=1e-4)
    with pytest.raises(ValueError, match="1-D child"):
        two_d.cdf(x[:3], pa[:3])


def test_flows_cannot_be_tied():
    model = StructuralCausalModel([("x", "y"), ("x", "z")],
                                  dict.fromkeys("xyz", ("continuous", 1)), device="cpu")
    model.set_mechanism("y", FlowNoiseMechanism())
    model.set_mechanism("z", FlowNoiseMechanism())
    with pytest.raises(TypeError, match="fixed N"):
        model.tie_noise(["y", "z"])


def test_flow_scm_pickles(fitted_flow):
    clone = pickle.loads(pickle.dumps(fitted_flow))
    data = _hetero(50, seed=3)
    assert torch.allclose(clone.log_prob(data), fitted_flow.log_prob(data), atol=1e-5)


@pytest.mark.parametrize("device", available_devices())
def test_flow_on_every_device(device):
    torch.manual_seed(0)
    model = _scm(FlowNoiseMechanism(num_transforms=2, hidden=(16,)), device=device)
    data = _hetero(600, seed=4)
    model.fit(data, epochs=10)
    mech = model.mechanisms["y"]
    x = data["x"][:20].to(device)
    y = data["y"][:20].to(device)
    u = mech.abduct(y, x)[0]
    assert u.device.type == torch.device(device).type
    assert torch.allclose(mech.structural(x, u), y, atol=1e-3)


# ---------------------------------------------------------------------------
# Exact CDF / quantile of additive mechanisms
# ---------------------------------------------------------------------------


def test_additive_gaussian_cdf_and_quantile_match_closed_form():
    model = _scm(AdditiveNoiseMechanism())
    model.fit(_homo(4000, seed=0))
    mech = model.mechanisms["y"]
    pa = torch.tensor([[0.0], [1.0]])
    loc = mech.location(pa)[:, 0]
    scale = float(mech.noise.scale)
    expected = Normal(loc, scale).icdf(torch.tensor(0.8))
    assert torch.allclose(mech.quantile(0.8, pa), expected, atol=1e-5)
    assert torch.allclose(mech.cdf(expected.unsqueeze(-1), pa), torch.full((2,), 0.8), atol=1e-5)


def test_mixture_quantile_inverts_its_cdf():
    noise = GaussianMixtureNoise(1, n_components=2)
    u = torch.cat([torch.randn(2000) - 2.0, 0.5 * torch.randn(2000) + 1.0]).unsqueeze(-1)
    noise.fit(u)
    p = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    with torch.no_grad():
        assert torch.allclose(noise.cdf(noise.icdf(p)), p, atol=1e-5)


# ---------------------------------------------------------------------------
# Conformal intervals: finite-sample marginal coverage
# ---------------------------------------------------------------------------


def _coverage_tolerance(level: float, replicates: int) -> float:
    return 3.0 * math.sqrt(level * (1 - level) / replicates)


@pytest.mark.parametrize("method", ["cqr", "residual"])
def test_conformal_marginal_coverage_holds_for_a_misspecified_model(method):
    """A homoscedastic linear fit on heteroscedastic data is wrong, yet the
    conformal interval still covers new rows at the declared level on average."""
    alpha, replicates, hits = 0.1, 80, 0
    for r in range(replicates):
        model = _scm(AdditiveNoiseMechanism())
        model.fit(_hetero(300, seed=10_000 + r))
        cal = _hetero(200, seed=20_000 + r)
        test = _hetero(1, seed=30_000 + r)
        ci = conformal_intervals(model, cal, "y", alpha=alpha, method=method)
        lo, hi = ci.interval(model.mechanisms["y"], test["x"])
        hits += int(lo[0] <= test["y"][0, 0] <= hi[0])
    coverage = hits / replicates
    assert coverage >= (1 - alpha) - _coverage_tolerance(1 - alpha, replicates)


def test_conformal_correction_is_infinite_with_too_few_rows():
    model = _scm(AdditiveNoiseMechanism())
    model.fit(_homo(200, seed=0))
    ci = conformal_intervals(model, _homo(5, seed=1), "y", alpha=0.05)
    assert math.isinf(ci.correction)


@pytest.mark.slow
def test_cqr_with_a_flow_is_adaptive_and_covers(fitted_flow):
    cal, test = _hetero(2000, seed=40), _hetero(4000, seed=41)
    ci = conformal_intervals(fitted_flow, cal, "y", alpha=0.1, method="cqr")
    lo, hi = ci.interval(fitted_flow.mechanisms["y"], test["x"])
    y = test["y"][:, 0]
    assert float(((lo <= y) & (y <= hi)).float().mean()) >= 0.88
    narrow = (hi - lo)[test["x"][:, 0].abs() < 0.3].mean()
    wide = (hi - lo)[test["x"][:, 0].abs() > 1.5].mean()
    assert float(wide) > 2.0 * float(narrow)


# ---------------------------------------------------------------------------
# PIT band: uniform bounds on conditional probabilities
# ---------------------------------------------------------------------------


def _true_interval_probability(pa: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Exact P(a < Y <= b | x) under the _homo truth y = 1.5 x + 0.4 eps."""
    dist = Normal(1.5 * pa[:, 0], 0.4)
    return dist.cdf(torch.tensor(b)) - dist.cdf(torch.tensor(a))


def test_pit_band_covers_conditional_probabilities_at_the_declared_level():
    delta, replicates, hits = 0.1, 60, 0
    pa = torch.linspace(-2, 2, 7).unsqueeze(-1)
    a, b = torch.full((7, 1), -0.5), torch.full((7, 1), 0.8)
    truth = _true_interval_probability(pa, -0.5, 0.8)
    for r in range(replicates):
        model = _scm(AdditiveNoiseMechanism())
        model.fit(_homo(400, seed=50_000 + r))
        band = pit_band_set(model, _homo(300, seed=60_000 + r), "y", delta=delta)
        lo, hi = band.interval_probability_bounds(model.mechanisms["y"], a, b, pa)
        hits += int(bool(((lo <= truth + 1e-6) & (truth <= hi + 1e-6)).all()))
    coverage = hits / replicates
    assert coverage >= (1 - delta) - _coverage_tolerance(1 - delta, replicates)


def test_pit_band_recalibrates_a_miscalibrated_scale():
    """Fit, then inflate the noise scale 3x: raw probabilities are wrong, the band is not."""
    model = _scm(AdditiveNoiseMechanism())
    model.fit(_homo(2000, seed=0))
    mech = model.mechanisms["y"]
    with torch.no_grad():
        mech.noise.log_scale.add_(math.log(3.0))
    pa = torch.tensor([[0.0], [1.0]])
    a, b = torch.full((2, 1), -0.2), torch.full((2, 1), 0.2)
    truth = _true_interval_probability(pa + torch.tensor([[0.0], [0.0]]), -0.2, 0.2)
    raw = mech.cdf(b, pa) - mech.cdf(a, pa)
    band = pit_band_set(model, _homo(4000, seed=1), "y", delta=0.05)
    lo, hi = band.interval_probability_bounds(mech, a, b, pa)
    assert bool(((raw - truth).abs() > 0.05).any())          # the raw model is off
    assert bool(((lo <= truth) & (truth <= hi)).all())        # the band is not


# ---------------------------------------------------------------------------
# Noise independence
# ---------------------------------------------------------------------------


def test_noise_independence_detects_a_parent_dependent_noise_scale():
    homo, hetero = _scm(AdditiveNoiseMechanism()), _scm(AdditiveNoiseMechanism())
    homo.fit(_homo(1000, seed=0))
    hetero.fit(_hetero(1000, seed=0))
    gen = torch.Generator().manual_seed(0)
    assert noise_independence_test(homo, _homo(600, seed=1), "y", n_permutations=100,
                                   generator=gen).p_value > 0.01
    assert noise_independence_test(hetero, _hetero(600, seed=1), "y", n_permutations=100,
                                   generator=gen).p_value < 0.01


@pytest.mark.slow
def test_a_flow_absorbs_the_parent_dependence(fitted_flow):
    result = noise_independence_test(fitted_flow, _hetero(600, seed=5), "y", n_permutations=100)
    assert result.p_value > 0.01
