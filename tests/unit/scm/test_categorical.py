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


def test_same_parent_detection_handles_roots_and_dtypes():
    from nbn.scm.categorical import _same_rows

    empty = torch.zeros(2, 0)
    assert bool(_same_rows(empty, empty, 2, "cpu").all())
    assert bool(_same_rows(None, None, 2, "cpu").all())
    assert not bool(_same_rows(torch.tensor([[1]]), torch.tensor([[1.5]]), 1, "cpu").any())
    with pytest.raises(ValueError, match="both"):
        _same_rows(None, torch.ones(1, 1), 1, "cpu")


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
    it may be possible, so its Gumbel must follow the prior (mean = Euler's gamma)."""
    from nbn.scm.categorical import _gumbel_posterior

    torch.manual_seed(0)
    probs = torch.tensor([[0.6, 0.4, 0.0]]).expand(40000, -1)
    u = _gumbel_posterior(probs, torch.zeros(40000, dtype=torch.long), 1)[0]
    assert bool(torch.isfinite(u).all())
    assert abs(float(u[:, 2].mean()) - 0.5772) < 0.03
    assert bool(((torch.log(probs[:, :2]) + u[:, :2]).argmax(-1) == 0).all())


def test_monotone_abduction_is_exact_for_small_probabilities_and_refuses_unrepresentable_ones():
    torch.manual_seed(0)
    mech = StructuralNeuralCategorical(n_classes=3, hidden=(8,), coupling="monotone")
    mech.fit_local(torch.zeros(10).long(), None, epochs=1)
    with torch.no_grad():
        mech._root_logits.copy_(torch.tensor([0.5, 1e-4, 0.5]).log())      # p(1) ~ 1e-4
    n = 200_000
    u = mech.abduct(torch.ones(n, 1), None)[0]
    with torch.no_grad():
        assert torch.equal(mech.structural(None, u), torch.ones(n, 1))     # batch n vs batch 1
    with torch.no_grad():
        mech._root_logits.copy_(torch.tensor([0.5, 1e-12, 0.5]).log())     # below float32 resolution
    with pytest.raises(ValueError, match="resolution"):
        mech.abduct(torch.ones(4, 1), None)
    gumbel = mech.with_coupling("gumbel_max")                                # Gumbel-max still works
    g = gumbel.abduct(torch.ones(1000, 1), None)[0]
    with torch.no_grad():
        assert torch.equal(gumbel.structural(None, g), torch.ones(1000, 1))


# ---------------------------------------------------------------------------
# N2b: bounds over all compatible SCMs
# ---------------------------------------------------------------------------


def _lp_extremes(p: torch.Tensor, q: torch.Tensor, x: int):
    """Exact min/max of pi(x, k) / p_x over all couplings pi of p and q (linear programs)."""
    from scipy.optimize import linprog

    k = p.numel()
    a_eq, b_eq = [], []
    for i in range(k):                      # row sums = p
        row = [0.0] * (k * k)
        row[i * k:(i + 1) * k] = [1.0] * k
        a_eq.append(row)
        b_eq.append(float(p[i]))
    for j in range(k):                      # column sums = q
        col = [0.0] * (k * k)
        col[j::k] = [1.0] * k
        a_eq.append(col)
        b_eq.append(float(q[j]))
    lo, hi = torch.zeros(k, dtype=torch.float64), torch.zeros(k, dtype=torch.float64)
    for j in range(k):
        c = [0.0] * (k * k)
        c[x * k + j] = 1.0
        lo[j] = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=(0, None)).fun / float(p[x])
        hi[j] = -linprog([-v for v in c], A_eq=a_eq, b_eq=b_eq, bounds=(0, None)).fun / float(p[x])
    return lo, hi


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("k", [2, 3, 5])
def test_bounds_are_sharp_against_exact_linear_programs(seed, k):
    torch.manual_seed(seed)
    p = torch.distributions.Dirichlet(torch.ones(k)).sample().double()
    q = torch.distributions.Dirichlet(torch.ones(k)).sample().double()
    x = int(torch.randint(0, k, ()))
    lo, hi = counterfactual_bounds(p.unsqueeze(0), q.unsqueeze(0), torch.tensor([x]))
    lp_lo, lp_hi = _lp_extremes(p, q, x)
    assert torch.allclose(lo[0], lp_lo, atol=1e-7) and torch.allclose(hi[0], lp_hi, atol=1e-7)


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


def test_bounds_from_sets_use_the_simplex():
    """q in [0.3, 0.6]^3 forces q_k <= 0.4 on the simplex: upper = 0.4 / 0.5 = 0.8, not 1."""
    lo, hi = counterfactual_bounds_from_sets(
        torch.tensor([[0.5, 0.25, 0.25]]), torch.full((1, 3), 0.3), torch.full((1, 3), 0.6),
        torch.tensor([0]))
    assert torch.allclose(hi, torch.full((1, 3), 0.8))


def test_bounds_from_sets_are_vacuous_when_the_observation_may_be_impossible():
    lo, hi = counterfactual_bounds_from_sets(
        torch.tensor([[0.0, 0.5, 0.5]]), torch.zeros(1, 3), torch.ones(1, 3), torch.tensor([0]))
    assert torch.equal(lo, torch.zeros(1, 3)) and torch.equal(hi, torch.ones(1, 3))
    lo, hi = counterfactual_bounds_from_sets(          # the p -> 0 limit: impossible classes stay 0
        torch.tensor([[0.0, 0.5, 0.5]]), torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 1.0]]), torch.tensor([0]))
    assert torch.equal(hi, torch.tensor([[0.0, 1.0, 1.0]]))
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
    # Only the lower bound is guaranteed: discrete scores tie, which can over-cover.
    assert mean >= (1 - alpha) - 0.015


@pytest.mark.parametrize("seed", range(4))
def test_conformal_sets_do_not_depend_on_the_batch_size(seed):
    torch.manual_seed(seed)
    model = _scm(StructuralNeuralCategorical(n_classes=3, hidden=(16,)))
    model.fit(_data(2000, seed=seed))
    mech = model.mechanisms["s"]
    cs = conformal_sets(model, _data(400, seed=50 + seed), "s", alpha=0.2)
    test = _data(600, seed=60 + seed)
    batched = cs.sets(mech, test["a"])
    one_by_one = torch.cat([cs.sets(mech, test["a"][i:i + 1]) for i in range(0, 600, 1)])
    assert torch.equal(batched, one_by_one)


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


def test_randomized_pit_rejects_device_generators():
    accelerators = [d for d in available_devices() if d != "cpu"]
    if not accelerators:
        pytest.skip("no accelerator generator on this machine")
    model = _scm(StructuralCategoricalTable())
    model.fit(_data(500, seed=0))
    data = _data(10, seed=1)
    with pytest.raises(ValueError, match="CPU torch.Generator"):
        model.mechanisms["s"].randomized_pit(data["s"], data["a"], torch.Generator(device=accelerators[0]))
    with pytest.raises(ValueError, match=r"calibration \(generator=\)"):
        pit_band_set(model, data, "s", generator=torch.Generator(device=accelerators[0]))


def test_with_coupling_keeps_the_kernel(fitted):
    mech = fitted[("neural", "monotone")].mechanisms["s"]
    other = mech.with_coupling("gumbel_max")
    pa = torch.tensor([[0.0], [1.0], [2.0]])
    with torch.no_grad():
        assert torch.equal(other.probs(pa), mech.probs(pa))
    assert other.coupling == "gumbel_max" and mech.coupling == "monotone"
    assert other.noise.sample(4).shape == (4, 3)


def test_monotone_tables_handle_floored_cells_where_float64_exists():
    """Table lookups do not jitter, so on CPU/CUDA (float64 CDF) even a 1e-12 cell is exact."""
    table = torch.tensor([[0.6, 0.4, 0.0], [0.2, 0.3, 0.5], [0.3, 0.3, 0.4]])
    model = _scm(StructuralCategoricalTable(coupling="monotone"))
    model.fit(_data(3000, seed=0, table=table))
    mech = model.mechanisms["s"]
    x, pa = torch.full((500, 1), 2.0), torch.zeros(500, 1)              # p(2 | a=0) floored at 1e-12
    u = mech.abduct(x, pa)[0]
    with torch.no_grad():
        assert torch.equal(mech.structural(pa, u), x)


def test_small_probability_bias_is_within_the_stated_bound():
    """Exact comonotone answer vs abduction for p_x = 1e-5 (neural margin: 2 float32 eps)."""
    mech = StructuralNeuralCategorical(n_classes=2, hidden=(4,), coupling="monotone")
    mech.fit_local(torch.zeros(10).long(), None, epochs=1)
    with torch.no_grad():
        mech._root_logits.copy_(torch.tensor([1 - 1e-5, 1e-5]).log())
    torch.manual_seed(0)
    u = mech.abduct(torch.ones(400_000, 1), None)[0].double()
    lo_edge = float(mech._cdf_rows(mech.probs(None).detach())[0, 0])
    frac = float((u[:, 0] <= lo_edge + 5e-7).double().mean())          # mass of a slice of width 5e-7
    exact = 5e-7 / 1e-5
    bound = 2 * 2 * torch.finfo(torch.float32).eps / 1e-5
    assert abs(frac - exact) <= bound + 0.005


def test_bounds_from_sets_reject_an_empty_box():
    with pytest.raises(ValueError, match="simplex"):
        counterfactual_bounds_from_sets(torch.tensor([[0.5, 0.5]]), torch.tensor([[0.7, 0.7]]),
                                        torch.tensor([[0.9, 0.9]]), torch.tensor([0]))


# ---------------------------------------------------------------------------
# Conformal scores (#322)
# ---------------------------------------------------------------------------

SCORES = ["exclusive", "inclusive", "randomized"]


def _threshold_rule(probs, true, alpha, score):
    from nbn.scm.calibration import _cumulative_scores, _upper_threshold

    u = {"exclusive": 0.0, "inclusive": 1.0}[score]                     # deterministic scores only
    scores = _cumulative_scores(probs, u).gather(-1, true.unsqueeze(-1)).squeeze(-1)
    rank = math.ceil((probs.shape[0] + 1) * (1 - alpha))
    thr = float(scores.sort().values[rank - 1])
    return _cumulative_scores(probs[:1], u) <= _upper_threshold(thr, probs.dtype, probs.shape[-1])


@pytest.mark.parametrize("score, p_top", [("exclusive", 1 - 1e-7), ("inclusive", 1 - 1e-4)])
def test_issue_322_confident_rows_get_one_class(score, p_top):
    """#322: [[True, True]] with the absolute 1e-6 tolerance.  The default score fixes the exact
    reproduction (two classes two float32 units apart); the inclusive score separates classes
    farther apart than its noise margin (K units), not the two-unit case (documented)."""
    n = 1000
    probs = torch.tensor([[p_top, 1 - p_top]]).repeat(n, 1)
    true = torch.zeros(n, dtype=torch.long)
    true[:50] = 1
    assert _threshold_rule(probs, true, 0.1, score).tolist() == [[True, False]]


def _saturating_model():
    """A nearly deterministic child of a continuous parent: d = [x + V > 0], V ~ N(0, 0.2^2)."""
    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        x = 2 * torch.randn(n, generator=g)
        v = 0.2 * torch.randn(n, generator=g)
        return {"x": x.unsqueeze(-1), "d": (x + v > 0).float().unsqueeze(-1)}

    torch.manual_seed(0)
    model = StructuralCausalModel([("x", "d")], {"x": ("continuous", 1), "d": ("discrete", 2)}, device="cpu")
    model.auto_mechanisms()
    model.set_mechanism("d", StructuralNeuralCategorical(n_classes=2, hidden=(16,)))
    model.fit(data(4000, 0))
    return model, data


def test_exclusive_score_gives_informative_sets_on_a_saturated_classifier():
    model, data = _saturating_model()
    mech, test = model.mechanisms["d"], data(4000, 2)
    cs = conformal_sets(model, data(1000, 1), "d", alpha=0.1)            # default: exclusive
    assert cs.score == "exclusive" and cs.threshold == 0.0
    sets = cs.sets(mech, test["x"])
    assert bool((sets.sum(-1) == 1).all())                               # one class per row, never empty
    assert float(sets.gather(-1, test["d"].long()).float().mean()) >= 0.9


@pytest.mark.parametrize("score", SCORES)
def test_every_score_keeps_marginal_coverage(score):
    alpha, n_cal, n_test, replicates = 0.2, 300, 500, 60
    model = _scm(StructuralNeuralCategorical(n_classes=3, hidden=(16,)))
    torch.manual_seed(0)
    model.fit(_data(3000, seed=0))
    mech = model.mechanisms["s"]
    coverage = []
    for r in range(replicates):
        gen = torch.Generator().manual_seed(r)
        cs = conformal_sets(model, _data(n_cal, seed=1000 + r), "s", alpha=alpha, score=score, generator=gen)
        test = _data(n_test, seed=2000 + r)
        sets = cs.sets(mech, test["a"], generator=gen)
        coverage.append(float(sets.gather(-1, test["s"].long()).float().mean()))
    mean = sum(coverage) / replicates
    assert mean >= (1 - alpha) - 0.02
    if score == "randomized":                                            # ties broken: close to the level
        assert mean <= (1 - alpha) + 0.03


@pytest.mark.parametrize("score", ["exclusive", "inclusive"])
def test_deterministic_scores_do_not_depend_on_the_batch_size(score):
    torch.manual_seed(1)
    model = _scm(StructuralNeuralCategorical(n_classes=3, hidden=(16,)))
    model.fit(_data(2000, seed=1))
    mech = model.mechanisms["s"]
    cs = conformal_sets(model, _data(400, seed=51), "s", alpha=0.2, score=score)
    test = _data(300, seed=61)
    assert torch.equal(cs.sets(mech, test["a"]), torch.cat([cs.sets(mech, test["a"][i:i + 1]) for i in range(300)]))


def test_randomized_sets_are_reproducible_with_a_generator_and_validated():
    model = _scm(StructuralCategoricalTable())
    model.fit(_data(2000, seed=0))
    mech, pa = model.mechanisms["s"], _data(200, seed=3)["a"]
    cs = conformal_sets(model, _data(300, seed=4), "s", alpha=0.2, score="randomized",
                        generator=torch.Generator().manual_seed(0))
    a = cs.sets(mech, pa, generator=torch.Generator().manual_seed(7))
    b = cs.sets(mech, pa, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    with pytest.raises(ValueError, match="score must be one of"):
        conformal_sets(model, _data(300, seed=4), "s", score="aps")
    clone = pickle.loads(pickle.dumps(cs))
    assert clone.score == "randomized"


def test_calibrated_sets_pass_the_discrete_score_through():
    from nbn.structure import HypothesisSet

    model, data = _saturating_model()
    hs = HypothesisSet({"x": ("continuous", 1), "d": ("discrete", 2)}, {"d": [("x",)]})
    fitted = hs.fit(data(3000, 0))
    for score in SCORES:
        cal = hs.calibrated_sets(fitted, data(800, 1), level=0.9, discrete_score=score,
                                 generator=torch.Generator().manual_seed(0))
        assert cal[("d", ("x",))].score == score


def test_tied_classes_share_a_score_and_stay_together():
    """A uniform or near-tied row keeps all its tied top classes (no index or float-noise tie-break)."""
    from nbn.scm.calibration import _cumulative_scores, _upper_threshold

    probs = torch.tensor([[1 / 3, 1 / 3, 1 / 3], [0.5 + 1e-7, 0.5 - 1e-7, 0.0], [0.7, 0.2, 0.1]])
    sets = _cumulative_scores(probs, 0.0) <= _upper_threshold(0.0, probs.dtype, 3)
    assert sets.tolist() == [[True, True, True], [True, True, False], [True, False, False]]


@pytest.mark.parametrize("k", [3, 5, 10])
def test_a_class_at_the_threshold_survives_one_unit_of_noise_per_probability(k):
    """The review's check: a class whose score is the threshold must stay in the set when every
    float32 probability of its row moves by one unit (batch-size noise)."""
    from nbn.scm.calibration import _cumulative_scores, _upper_threshold

    g = torch.Generator().manual_seed(k)
    probs = torch.softmax(3 * torch.randn(4000, k, generator=g), -1).float()
    step = torch.randint(-1, 2, probs.shape, generator=g).float()
    nudged = torch.where(step == 0, probs, torch.nextafter(probs, probs + 2 * step))
    rows = torch.arange(probs.shape[0])
    cls = torch.randint(0, k, (probs.shape[0],), generator=g)
    for u in (0.0, 1.0):
        at = _cumulative_scores(probs, u)[rows, cls]                     # each row's threshold
        moved = _cumulative_scores(nudged, u)[rows, cls]
        bound = torch.tensor([_upper_threshold(float(t), torch.float32, k) for t in at],
                             dtype=torch.float64)
        assert bool((moved <= bound).all())


def test_old_pickles_keep_the_inclusive_score():
    model = _scm(StructuralCategoricalTable())
    model.fit(_data(1000, seed=0))
    cs = conformal_sets(model, _data(300, seed=4), "s", alpha=0.2, score="inclusive")
    del cs.__dict__["score"]                                           # as saved before the field existed
    assert pickle.loads(pickle.dumps(cs)).score == "inclusive"


def test_calibrated_sets_reject_an_unknown_discrete_score_up_front():
    from nbn.structure import HypothesisSet

    model, data = _saturating_model()
    hs = HypothesisSet({"x": ("continuous", 1), "y": ("continuous", 1)}, {"y": [("x",)]})
    fitted = hs.fit({"x": torch.randn(200, 1), "y": torch.randn(200, 1)})
    with pytest.raises(ValueError, match="score must be one of"):
        hs.calibrated_sets(fitted, {"x": torch.randn(100, 1), "y": torch.randn(100, 1)},
                           discrete_score="randomised")
