"""Discrete structural mechanisms from a categorical kernel (N2): couplings, bounds, calibration."""
from __future__ import annotations

import itertools
import math
import pickle

import pytest
import torch

from nbn.scm import (
    StructuralCategoricalTable,
    StructuralCausalModel,
    StructuralNeuralCategorical,
    calibration_by_bins_test,
    conformal_sets,
    counterfactual_bounds,
    counterfactual_bounds_from_sets,
    pit_band_set,
)
from tests.conftest import available_devices

TRUE = torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.05, 0.15, 0.8]])


def _data(n: int, seed: int, table: torch.Tensor = TRUE):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randint(0, table.shape[0], (n,), generator=gen)
    s = torch.multinomial(table[a], 1, generator=gen).reshape(-1)
    return {"a": a.float().unsqueeze(-1), "s": s.float().unsqueeze(-1)}


def _scm(mech, device="cpu"):
    model = StructuralCausalModel([("a", "s")], {"a": ("discrete", 3), "s": ("discrete", 3)},
                                  device=device)
    model.set_mechanism("a", StructuralCategoricalTable())
    model.set_mechanism("s", mech)
    return model


MECHS = [
    pytest.param(lambda c: StructuralCategoricalTable(coupling=c), id="table"),
    pytest.param(lambda c: StructuralNeuralCategorical(n_classes=3, hidden=(16,), coupling=c), id="neural"),
]
COUPLINGS = ["gumbel_max", "monotone"]


@pytest.fixture(scope="module")
def fitted():
    """One fitted SCM per (mechanism, coupling)."""
    torch.manual_seed(0)
    out = {}
    for make, name in [(p.values[0], p.id) for p in MECHS]:
        for coupling in COUPLINGS:
            model = _scm(make(coupling))
            model.fit(_data(4000, seed=0))
            out[(name, coupling)] = model
    return out


KEYS = [(m, c) for m in ("table", "neural") for c in COUPLINGS]


# ---------------------------------------------------------------------------
# N2a: structural reading of the kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", KEYS)
def test_structural_marginal_is_the_kernel(fitted, key):
    """Pushing fresh noise through f(pa, .) reproduces P(. | pa)."""
    mech = fitted[key].mechanisms["s"]
    pa = torch.tensor([[0.0], [1.0], [2.0]]).repeat_interleave(20000, 0)
    with torch.no_grad():
        x = mech.structural(pa, mech.noise.sample(pa.shape[0]))[:, 0].long()
        probs = mech.probs(torch.tensor([[0.0], [1.0], [2.0]]))
    freq = torch.stack([torch.bincount(x[i * 20000:(i + 1) * 20000], minlength=3) for i in range(3)])
    assert torch.allclose(freq.float() / 20000, probs, atol=0.015)


@pytest.mark.parametrize("key", KEYS)
def test_abduction_always_reproduces_the_observation(fitted, key):
    mech = fitted[key].mechanisms["s"]
    data = _data(500, seed=1)
    with torch.no_grad():
        u = mech.abduct(data["s"], data["a"], n_samples=8)
        for k in range(8):
            assert torch.equal(mech.structural(data["a"], u[k]), data["s"])


@pytest.mark.parametrize("mech_name", ["table", "neural"])
def test_monotone_counterfactual_is_the_comonotone_coupling(fitted, mech_name):
    """Observed x=0 under a=0, counterfactual a=1: exact answer from the two CDFs."""
    model = fitted[(mech_name, "monotone")]
    mech = model.mechanisms["s"]
    with torch.no_grad():
        p = mech.probs(torch.tensor([[0.0], [1.0]])).double()
    f0, f1 = p[0].cumsum(0), p[1].cumsum(0)
    top = f0[0]
    exact = torch.stack([(torch.minimum(f1[k], top) - torch.minimum(f1[k] - p[1, k], top)).clamp_min(0)
                         for k in range(3)]) / top
    n = 100_000
    evidence = {"a": torch.zeros(n, 1), "s": torch.zeros(n, 1)}
    torch.manual_seed(1)
    world = model.counterfactual_sample(evidence, do={"a": 1.0})["s"][0, :, 0].long()
    freq = torch.bincount(world, minlength=3).double() / n
    assert torch.allclose(freq, exact, atol=0.006)


@pytest.mark.parametrize("mech_name", ["table", "neural"])
def test_gumbel_max_is_counterfactually_stable(fitted, mech_name):
    """Oberst & Sontag (2019): if p(x|pa')/p(x|pa) >= p(k|pa')/p(k|pa) for every k,
    the counterfactual cannot move away from the observed x."""
    mech = fitted[(mech_name, "gumbel_max")].mechanisms["s"]
    with torch.no_grad():
        p = mech.probs(torch.tensor([[1.0], [2.0]]))
    ratio = p[1] / p[0]                 # class 2 has the largest ratio going a=1 -> a=2
    assert int(ratio.argmax()) == 2
    n = 20000
    u = mech.abduct(torch.full((n, 1), 2.0), torch.ones(n, 1))[0]
    with torch.no_grad():
        cf = mech.structural(torch.full((n, 1), 2.0), u)
    assert bool((cf == 2.0).all())


def test_same_parents_give_the_observation_back(fitted):
    for key in KEYS:
        mech = fitted[key].mechanisms["s"]
        data = _data(200, seed=2)
        with torch.no_grad():
            u = mech.abduct(data["s"], data["a"])[0]
            assert torch.equal(mech.structural(data["a"], u), data["s"])
            lo, hi = mech.counterfactual_bounds(data["s"], data["a"], data["a"])
        point = torch.nn.functional.one_hot(data["s"][:, 0].long(), 3).float()
        assert torch.equal(lo, point) and torch.equal(hi, point)


def test_location_is_the_mode_or_the_median(fitted):
    pa = torch.tensor([[0.0], [1.0], [2.0]])
    with torch.no_grad():
        probs = fitted[("table", "gumbel_max")].mechanisms["s"].probs(pa)
        mode = fitted[("table", "gumbel_max")].mechanisms["s"].location(pa)[:, 0]
        median = fitted[("table", "monotone")].mechanisms["s"].location(pa)[:, 0]
    assert torch.equal(mode.long(), probs.argmax(-1))
    expected_median = (probs.cumsum(-1) < 0.5).sum(-1)
    assert torch.equal(median.long(), expected_median)


def test_impossible_observations_and_bad_values_raise():
    model = _scm(StructuralCategoricalTable(coupling="gumbel_max"))
    model.fit(_data(600, seed=0))
    with pytest.raises(ValueError, match="not classes"):
        model.mechanisms["s"].abduct(torch.tensor([[1.5]]), torch.tensor([[0.0]]))
    with pytest.raises(ValueError, match="coupling"):
        StructuralCategoricalTable(coupling="rank")
    # exact zeros (tables floor at 1e-12, so this is reached through the pure function)
    with pytest.raises(ValueError, match="probability 0"):
        counterfactual_bounds(torch.tensor([[0.0, 0.5, 0.5]]), torch.full((1, 3), 1 / 3), torch.tensor([0]))


def test_zero_probability_classes_keep_their_prior_noise():
    """A class impossible under pa is not informed by the observation; under pa'
    it may be possible, and its Gumbel must then follow the prior."""
    table = torch.tensor([[0.6, 0.4, 0.0], [0.2, 0.3, 0.5], [0.3, 0.3, 0.4]])
    model = _scm(StructuralCategoricalTable(coupling="gumbel_max"))
    model.fit(_data(30000, seed=0, table=table))
    mech = model.mechanisms["s"]
    n = 40000
    u = mech.abduct(torch.zeros(n, 1), torch.zeros(n, 1))[0]
    assert bool(torch.isfinite(u).all())
    g2 = u[:, 2]  # class 2 has p = 0 under a = 0: prior standard Gumbel, mean = Euler's gamma
    assert abs(float(g2.mean()) - 0.5772) < 0.03


# ---------------------------------------------------------------------------
# N2b: bounds over all compatible SCMs
# ---------------------------------------------------------------------------


def _lp_extremes(p: torch.Tensor, q: torch.Tensor, x: int):
    """Min/max of pi(x, k) / p_x over couplings pi of p and q (brute force over a grid)."""
    k = p.numel()
    lo = torch.full((k,), math.inf, dtype=torch.float64)
    hi = torch.full((k,), -math.inf, dtype=torch.float64)
    # The row pi(x, .) ranges over {r >= 0, sum r = p_x, r <= q, q - r <= 1 - p_x}; enumerate a grid.
    steps = torch.linspace(0, 1, 81, dtype=torch.float64)
    for frac in itertools.product(steps, repeat=k - 1):
        r = torch.tensor(list(frac) + [0.0], dtype=torch.float64) * p[x]
        r[-1] = p[x] - r[:-1].sum()
        if bool((r < -1e-12).any()) or bool((r > q + 1e-12).any()) or bool(((q - r) > 1 - p[x] + 1e-12).any()):
            continue
        lo, hi = torch.minimum(lo, r / p[x]), torch.maximum(hi, r / p[x])
    return lo, hi


@pytest.mark.parametrize("seed", range(4))
def test_bounds_are_sharp_against_brute_force_couplings(seed):
    torch.manual_seed(seed)
    p = torch.distributions.Dirichlet(torch.ones(3)).sample().double()
    q = torch.distributions.Dirichlet(torch.ones(3)).sample().double()
    x = int(p.argmax())
    lo, hi = counterfactual_bounds(p.unsqueeze(0), q.unsqueeze(0), torch.tensor([x]))
    lp_lo, lp_hi = _lp_extremes(p, q, x)
    assert torch.allclose(lo[0], lp_lo, atol=0.02) and torch.allclose(hi[0], lp_hi, atol=0.02)
    # every row of the box that sums to one is attained: the vertices are feasible couplings
    assert float(lo[0].sum()) <= 1.0 + 1e-9 <= float(hi[0].sum()) + 2e-9


@pytest.mark.parametrize("key", KEYS)
def test_both_point_answers_lie_inside_the_bounds(fitted, key):
    model = fitted[key]
    mech = model.mechanisms["s"]
    n = 30000
    for a_obs, s_obs, a_cf in [(0, 0, 1), (2, 2, 0), (1, 1, 2)]:
        evidence = {"a": torch.full((n, 1), float(a_obs)), "s": torch.full((n, 1), float(s_obs))}
        world = model.counterfactual_sample(evidence, do={"a": float(a_cf)})["s"][0, :, 0].long()
        freq = torch.bincount(world, minlength=3).float() / n
        with torch.no_grad():
            lo, hi = mech.counterfactual_bounds(evidence["s"][:1], evidence["a"][:1], torch.tensor([[float(a_cf)]]))
        assert bool((freq >= lo[0] - 0.01).all()) and bool((freq <= hi[0] + 0.01).all())


def test_bounds_from_sets_contain_every_member_and_are_attained():
    gen = torch.Generator().manual_seed(0)
    p = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64)
    q = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64)
    width = 0.05
    p_lo, q_lo, q_hi = (p - width).clamp_min(0), (q - width).clamp_min(0), (q + width).clamp_max(1)
    x = torch.tensor([0])
    lo, hi = counterfactual_bounds_from_sets(p_lo, q_lo, q_hi, x)
    worst_lo, best_hi = torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    for _ in range(3000):
        pm = p + (2 * torch.rand(p.shape, generator=gen, dtype=torch.float64) - 1) * width
        qm = q + (2 * torch.rand(q.shape, generator=gen, dtype=torch.float64) - 1) * width
        pm[0, 0] = pm[0, 0].clamp(p_lo[0, 0], 1)
        l_m, h_m = counterfactual_bounds(pm, qm, x)
        assert bool((l_m >= lo - 1e-12).all()) and bool((h_m <= hi + 1e-12).all())
        worst_lo, best_hi = torch.minimum(worst_lo, l_m[0]), torch.maximum(best_hi, h_m[0])
    corner_lo, _ = counterfactual_bounds(p_lo, q_lo, x)
    _, corner_hi = counterfactual_bounds(p_lo, q_hi, x)
    assert torch.allclose(lo, corner_lo) and torch.allclose(hi, corner_hi)  # attained at the corners


def test_bounds_from_sets_are_vacuous_when_the_observation_may_be_impossible():
    lo, hi = counterfactual_bounds_from_sets(
        torch.tensor([[0.0, 0.5, 0.5]]), torch.zeros(1, 3), torch.ones(1, 3), torch.tensor([0]))
    assert torch.equal(lo, torch.zeros(1, 3)) and torch.equal(hi, torch.ones(1, 3))
    lo, hi = counterfactual_bounds_from_sets(
        torch.tensor([[0.0, 0.5, 0.5]]), torch.zeros(1, 3), torch.ones(1, 3), torch.tensor([0]),
        same_parents=True)
    assert torch.equal(lo, torch.tensor([[1.0, 0.0, 0.0]]))


# ---------------------------------------------------------------------------
# N2c: calibration for discrete children
# ---------------------------------------------------------------------------


def test_randomized_pit_is_uniform_under_the_true_kernel():
    """With the true table plugged in, W is exactly U(0, 1)."""
    model = _scm(StructuralCategoricalTable())
    model.fit(_data(3000, seed=0))
    mech = model.mechanisms["s"]
    with torch.no_grad():
        mech._logits.copy_(TRUE.log())              # plug in the true kernel
    data = _data(20000, seed=3)
    w = mech.randomized_pit(data["s"], data["a"], torch.Generator().manual_seed(0))
    sorted_w = w.sort().values.double()
    grid = torch.arange(1, w.numel() + 1, dtype=torch.float64) / w.numel()
    assert float((sorted_w - grid).abs().max()) < 0.015       # KS distance to U(0, 1)


def test_pit_band_bounds_discrete_cdfs_at_the_declared_level():
    """For a well-specified table the band contains P(X <= k | a) for every k and a."""
    delta, replicates, hits = 0.1, 60, 0
    truth = TRUE.cumsum(-1)
    pa = torch.tensor([[0.0], [1.0], [2.0]]).repeat_interleave(3, 0)
    xs = torch.tensor([[0.0], [1.0], [2.0]]).repeat(3, 1)
    for r in range(replicates):
        model = _scm(StructuralCategoricalTable())
        model.fit(_data(3000, seed=100 + r))
        band = pit_band_set(model, _data(400, seed=200 + r), "s", delta=delta,
                            generator=torch.Generator().manual_seed(r))
        lo, hi = band.cdf_bounds(model.mechanisms["s"], xs, pa)
        t = truth[pa[:, 0].long(), xs[:, 0].long()]
        hits += int(bool(((lo <= t + 1e-6) & (t <= hi + 1e-6)).all()))
    assert hits / replicates >= (1 - delta) - 3 * math.sqrt(delta * (1 - delta) / replicates)


def test_conformal_sets_have_the_declared_marginal_coverage():
    alpha, n_cal, n_test, replicates = 0.2, 300, 500, 100
    model = _scm(StructuralNeuralCategorical(n_classes=3, hidden=(16,)))
    torch.manual_seed(0)
    model.fit(_data(3000, seed=0))
    mech = model.mechanisms["s"]
    coverage = []
    for r in range(replicates):
        cs = conformal_sets(model, _data(n_cal, seed=1000 + r), "s", alpha=alpha)
        test = _data(n_test, seed=2000 + r)
        sets = cs.sets(mech, test["a"])
        coverage.append(float(sets.gather(-1, test["s"].long()).float().mean()))
    mean = sum(coverage) / replicates
    assert (1 - alpha) - 0.015 <= mean <= (1 - alpha) + 1 / (n_cal + 1) + 0.03


def test_conformal_sets_include_every_class_at_the_top_score():
    model = _scm(StructuralCategoricalTable())
    model.fit(_data(2000, seed=0))
    cs = conformal_sets(model, _data(5, seed=1), "s", alpha=0.05)   # too few rows: threshold inf
    assert math.isinf(cs.threshold)
    assert bool(cs.sets(model.mechanisms["s"], torch.tensor([[0.0], [2.0]])).all())


def test_calibration_by_bins_on_a_discrete_child():
    good, bad = _scm(StructuralCategoricalTable()), _scm(StructuralCategoricalTable())
    good.fit(_data(4000, seed=0))
    flipped = TRUE.flip(0)                     # a kernel that is wrong differently per region
    bad.fit(_data(4000, seed=0, table=flipped))
    data = _data(3000, seed=1)
    gen = torch.Generator().manual_seed(0)
    assert calibration_by_bins_test(good, data, "s", n_bins=3, generator=gen).p_value > 0.01
    assert calibration_by_bins_test(bad, data, "s", n_bins=3, generator=gen).p_value < 0.01


# ---------------------------------------------------------------------------
# Plumbing: serialisation, binding, devices
# ---------------------------------------------------------------------------


def test_state_dict_matches_the_plain_mechanisms(fitted):
    from nbn.mechanisms.parametric.neural_categorical import NeuralCategoricalMechanism

    mech = fitted[("neural", "gumbel_max")].mechanisms["s"]
    plain = NeuralCategoricalMechanism(n_classes=3, hidden=(16,))
    plain.fit_local(_data(100, seed=5)["s"][:, 0], _data(100, seed=5)["a"], epochs=1)
    plain.load_state_dict(mech.state_dict(), strict=True)
    pa = torch.tensor([[0.0], [1.0]])
    assert torch.allclose(plain.forward(pa).probs, mech.probs(pa))


def test_pickled_model_keeps_its_calibration(fitted):
    model = fitted[("table", "monotone")]
    cs = conformal_sets(model, _data(300, seed=4), "s", alpha=0.2)
    clone, cs_clone = pickle.loads(pickle.dumps((model, cs)))
    pa = torch.tensor([[0.0], [2.0]])
    assert torch.equal(cs_clone.sets(clone.mechanisms["s"], pa), cs.sets(model.mechanisms["s"], pa))


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("coupling", COUPLINGS)
def test_discrete_structural_mechanisms_on_every_device(device, coupling):
    torch.manual_seed(0)
    model = _scm(StructuralNeuralCategorical(n_classes=3, hidden=(16,), coupling=coupling), device=device)
    model.fit(_data(1500, seed=0))
    mech = model.mechanisms["s"]
    data = _data(50, seed=1)                                  # CPU inputs, model on `device`
    u = mech.abduct(data["s"], data["a"])[0]
    assert u.device.type == torch.device(device).type
    assert mech.noise.sample(2).device.type == torch.device(device).type
    with torch.no_grad():
        assert torch.equal(mech.structural(data["a"], u).cpu(), data["s"])
        lo, hi = mech.counterfactual_bounds(data["s"], data["a"], torch.ones(50, 1))
    assert bool((lo <= hi).all())
    world = model.counterfactual_sample(data, do={"a": 1.0}, n_samples=3)["s"]
    assert world.shape == (3, 50, 1)
