"""Neural structural mechanisms and calibrated uncertainty (flows, PIT band, conformal)."""
from __future__ import annotations

import math
import pickle

import pytest
import torch
from torch.distributions import Normal

from nbn.mechanisms.parametric.normalizing_flow import NormalizingFlowMechanism
from nbn.scm import (
    AdditiveNoiseMechanism,
    FlowNoiseMechanism,
    GaussianMixtureNoise,
    GaussianNoise,
    StructuralCausalModel,
    calibration_by_bins_test,
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


@pytest.mark.parametrize("method", ["cqr", "residual"])
def test_conformal_marginal_coverage_holds_for_a_misspecified_model(method):
    """A homoscedastic linear fit on heteroscedastic data is wrong, yet split
    conformal covers new rows at the declared level on average.

    Coverage given the fit lies in ``[1 - alpha, 1 - alpha + 1/(n + 1)]`` on
    average over calibration sets (Lei et al. 2018, Thm 2.2); we check the
    mean over replicates to within ~4 standard errors on both sides.
    """
    alpha, n_cal, n_test, replicates = 0.1, 200, 500, 150
    model = _scm(AdditiveNoiseMechanism())
    model.fit(_hetero(300, seed=0))                  # fit data independent of calibration
    mech = model.mechanisms["y"]
    coverage = []
    for r in range(replicates):
        ci = conformal_intervals(model, _hetero(n_cal, seed=10_000 + r), "y", alpha=alpha,
                                 method=method)
        test = _hetero(n_test, seed=20_000 + r)
        lo, hi = ci.interval(mech, test["x"])
        y = test["y"][:, 0]
        coverage.append(float(((lo <= y) & (y <= hi)).float().mean()))
    mean = sum(coverage) / replicates
    assert (1 - alpha) - 0.01 <= mean <= (1 - alpha) + 1 / (n_cal + 1) + 0.01


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


def _ks_distance_to(sorted_w: torch.Tensor, true_cdf) -> float:
    """sup_t |G_hat(t) - G(t)| for the empirical CDF of ``sorted_w`` (attained at the jumps)."""
    n = sorted_w.numel()
    g = true_cdf(sorted_w.double())
    i = torch.arange(1, n + 1, dtype=torch.float64)
    return float(torch.maximum(i / n - g, g - (i - 1) / n).max())


def test_pit_band_contains_the_true_calibration_map_at_the_declared_level():
    """For a linear-Gaussian fit of _homo data the PIT law is exact:
    ``G(t) = Phi((s z_t + a) / sqrt((1.5 - b)^2 + 0.16))`` (x ~ N(0, 1)).
    The DKW–Massart band must contain it with frequency >= 1 - delta."""
    delta, replicates, hits = 0.1, 300, 0
    model = _scm(AdditiveNoiseMechanism())
    model.fit(_homo(200, seed=0))
    mech = model.mechanisms["y"]
    with torch.no_grad():
        a = float(mech.location(torch.zeros(1, 1))[0, 0])
        b = float(mech.location(torch.ones(1, 1))[0, 0]) - a
        s = float(mech.noise.scale)
    std = Normal(torch.tensor(0.0, dtype=torch.float64), 1.0)
    spread = math.sqrt((1.5 - b) ** 2 + 0.16)

    def true_g(t):
        return std.cdf((s * std.icdf(t.clamp(1e-15, 1 - 1e-15)) + a) / spread)

    for r in range(replicates):
        band = pit_band_set(model, _homo(250, seed=50_000 + r), "y", delta=delta)
        hits += int(_ks_distance_to(band.sorted_pit, true_g) <= band.epsilon)
    coverage = hits / replicates
    assert coverage >= (1 - delta) - 3.0 * math.sqrt(delta * (1 - delta) / replicates)


def test_pit_band_recalibrates_a_miscalibrated_scale():
    """Fit, then inflate the noise scale 3x: raw probabilities are wrong, the band is not."""
    model = _scm(AdditiveNoiseMechanism())
    model.fit(_homo(2000, seed=0))
    mech = model.mechanisms["y"]
    with torch.no_grad():
        mech.noise.log_scale.add_(math.log(3.0))
    pa = torch.tensor([[0.0], [1.0]])
    a, b = torch.full((2, 1), -0.2), torch.full((2, 1), 0.2)
    truth = _true_interval_probability(pa, -0.2, 0.2)
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
    def gen():
        return torch.Generator().manual_seed(0)

    assert noise_independence_test(homo, _homo(600, seed=1), "y", n_permutations=100,
                                   generator=gen()).p_value > 0.01
    assert noise_independence_test(hetero, _hetero(600, seed=1), "y", n_permutations=100,
                                   generator=gen()).p_value < 0.01


@pytest.mark.slow
def test_a_flow_absorbs_the_parent_dependence(fitted_flow):
    result = noise_independence_test(fitted_flow, _hetero(600, seed=5), "y", n_permutations=100)
    assert result.p_value > 0.01


def test_noise_independence_is_reproducible_with_a_generator():
    model = _scm(AdditiveNoiseMechanism())
    model.fit(_hetero(800, seed=0))
    data = _hetero(900, seed=1)
    runs = [noise_independence_test(model, data, "y", n_permutations=50, max_rows=500,
                                    generator=torch.Generator().manual_seed(7)).p_value
            for _ in range(2)]
    assert runs[0] == runs[1]


# ---------------------------------------------------------------------------
# Calibration by bins
# ---------------------------------------------------------------------------


def test_calibration_by_bins_flags_a_region_dependent_miscalibration():
    """A homoscedastic fit is calibrated on average but not in every region of x."""
    good, bad = _scm(AdditiveNoiseMechanism()), _scm(AdditiveNoiseMechanism())
    good.fit(_homo(2000, seed=0))
    bad.fit(_hetero(2000, seed=0))
    assert calibration_by_bins_test(good, _homo(2000, seed=1), "y").p_value > 0.01
    assert calibration_by_bins_test(bad, _hetero(2000, seed=1), "y").p_value < 0.01


def test_calibration_by_bins_needs_parents():
    model = _scm(AdditiveNoiseMechanism())
    model.fit(_homo(200, seed=0))
    with pytest.raises(ValueError, match="no parents"):
        calibration_by_bins_test(model, _homo(100, seed=1), "x")


# ---------------------------------------------------------------------------
# Input validation, dtypes, devices, serialisation, binding
# ---------------------------------------------------------------------------


def test_flow_noise_dimension_follows_the_fitted_child():
    torch.manual_seed(0)
    model = StructuralCausalModel([("x", "y")], {"x": ("continuous", 1), "y": ("continuous", 2)},
                                  device="cpu")
    model.set_mechanism("x", AdditiveNoiseMechanism())
    model.set_mechanism("y", FlowNoiseMechanism(num_transforms=2, hidden=(16,)))
    x = torch.randn(400, 1)
    model.fit({"x": x, "y": torch.cat([x, -x], 1) + 0.3 * torch.randn(400, 2)}, epochs=5)
    mech = model.mechanisms["y"]
    assert mech.noise.dim == 2
    assert mech.noise.sample(3).shape == (3, 2)
    assert mech.location(x[:4]).shape == (4, 2)


def test_flow_state_dict_is_interchangeable_with_the_plain_flow():
    torch.manual_seed(0)
    pa = torch.randn(300, 1)
    x = pa + 0.3 * torch.randn(300, 1)
    plain = NormalizingFlowMechanism(num_transforms=2, hidden=(16,))
    plain.fit_local(x, pa, epochs=5)
    flow = FlowNoiseMechanism(num_transforms=2, hidden=(16,))
    flow.fit_local(x, pa, epochs=1)                 # same architecture, different weights
    flow.load_state_dict(plain.state_dict(), strict=True)
    assert torch.allclose(flow.log_prob(x[:20], pa[:20]), plain.log_prob(x[:20], pa[:20]), atol=1e-6)
    back = NormalizingFlowMechanism(num_transforms=2, hidden=(16,))
    back.fit_local(x, pa, epochs=1)
    back.load_state_dict(flow.state_dict(), strict=True)
    assert torch.allclose(back.log_prob(x[:20], pa[:20]), plain.log_prob(x[:20], pa[:20]), atol=1e-6)


def test_flow_rejects_non_finite_and_three_d_inputs(fitted_flow):
    mech = fitted_flow.mechanisms["y"]
    x, pa = torch.zeros(3, 1), torch.zeros(3, 1)
    with pytest.raises(ValueError, match="NaN or inf"):
        mech.abduct(torch.tensor([[0.0], [float("nan")], [1.0]]), pa)
    with pytest.raises(ValueError, match="NaN or inf"):
        mech.cdf(x, torch.tensor([[0.0], [float("inf")], [1.0]]))
    with pytest.raises(ValueError, match="parents"):
        mech.abduct(x, torch.zeros(2, 3, 1))
    with pytest.raises(ValueError, match="rows"):
        mech.abduct(x, torch.zeros(2, 1))


def test_calibration_rejects_non_finite_rows():
    model = _scm(AdditiveNoiseMechanism())
    model.fit(_homo(300, seed=0))
    cal = _homo(50, seed=1)
    cal["y"][3, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or inf"):
        pit_band_set(model, cal, "y")
    with pytest.raises(ValueError, match="NaN or inf"):
        conformal_intervals(model, cal, "y")


def test_multivariate_noise_has_no_scalar_cdf():
    with pytest.raises(ValueError, match="1-D noise"):
        GaussianNoise(2).cdf(torch.zeros(3, 2))
    with pytest.raises(ValueError, match="1-D noise"):
        GaussianMixtureNoise(2, n_components=2).icdf(torch.full((3,), 0.5))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extreme_quantiles_are_finite(fitted_flow, dtype):
    additive = _scm(AdditiveNoiseMechanism())
    additive.fit(_homo(300, seed=0))
    mech = additive.mechanisms["y"].to(dtype)
    pa = torch.zeros(2, 1, dtype=dtype)
    q = mech.quantile(torch.tensor([0.0, 1.0], dtype=dtype), pa)
    assert q.dtype == dtype and bool(torch.isfinite(q).all()) and float(q[0]) < float(q[1])
    root = additive.mechanisms["x"].to(dtype)          # root: a vector p broadcasts against loc
    assert root.quantile(torch.tensor([0.1, 0.5, 0.9], dtype=dtype), None).shape == (3,)
    fq = fitted_flow.mechanisms["y"].quantile(torch.tensor([0.0, 1.0]), torch.zeros(2, 1))
    assert bool(torch.isfinite(fq).all())


def test_calibration_results_refuse_a_modified_mechanism():
    model = _scm(AdditiveNoiseMechanism())
    model.fit(_homo(300, seed=0))
    band = pit_band_set(model, _homo(200, seed=1), "y")
    ci = conformal_intervals(model, _homo(200, seed=1), "y")
    pa = torch.zeros(2, 1)
    band.cdf_bounds(model.mechanisms["y"], torch.zeros(2, 1), pa)   # bound mechanism: fine
    model.fit(_homo(300, seed=2))                                    # refit changes the weights
    with pytest.raises(ValueError, match="calibrate again"):
        band.cdf_bounds(model.mechanisms["y"], torch.zeros(2, 1), pa)
    with pytest.raises(ValueError, match="calibrate again"):
        ci.interval(model.mechanisms["y"], pa)
    other = _scm(AdditiveNoiseMechanism())
    other.fit(_homo(300, seed=0))
    with pytest.raises(ValueError, match="calibrate again"):
        ci.interval(other.mechanisms["y"], pa)


def test_groups_keep_one_row_per_episode():
    model = _scm(AdditiveNoiseMechanism())
    model.fit(_homo(300, seed=0))
    cal = _homo(400, seed=1)
    episodes = torch.arange(400) // 10                       # 40 episodes of 10 steps
    band = pit_band_set(model, cal, "y", delta=0.1, groups=episodes,
                        generator=torch.Generator().manual_seed(0))
    assert band.sorted_pit.numel() == 40
    assert band.epsilon == pytest.approx(math.sqrt(math.log(2 / 0.1) / 80))
    ci = conformal_intervals(model, cal, "y", groups=episodes)
    assert ci.n_calibration == 40
    # every kept row comes from a different episode and the choice is reproducible
    again = pit_band_set(model, cal, "y", delta=0.1, groups=episodes,
                         generator=torch.Generator().manual_seed(0))
    assert torch.equal(band.sorted_pit, again.sorted_pit)


@pytest.mark.parametrize("device", available_devices())
def test_flow_location_and_quantile_on_every_device(device):
    torch.manual_seed(0)
    flow = FlowNoiseMechanism(num_transforms=2, hidden=(16,))
    pa = torch.randn(400, 1)
    flow.fit_local((pa + 0.3 * torch.randn(400, 1)).to(device), pa.to(device), epochs=5)
    q_pa = torch.linspace(-1, 1, 5, device=device).unsqueeze(-1)
    loc, med = flow.location(q_pa)[:, 0], flow.quantile(0.5, q_pa)
    assert loc.device.type == torch.device(device).type == med.device.type
    assert torch.allclose(loc, med, atol=1e-5)
    root = FlowNoiseMechanism(num_transforms=2, hidden=(16,))
    root.fit_local((1.0 + torch.randn(300, 1)).to(device), None, epochs=5)
    assert root.location(None).device.type == torch.device(device).type
    assert bool(torch.isfinite(root.quantile(torch.tensor([0.0, 0.5, 1.0]), None)).all())
