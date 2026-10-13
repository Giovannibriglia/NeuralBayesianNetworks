"""Continuous and mixed factors (#318): batched candidate fits and calibrated sets."""
from __future__ import annotations

import math

import pytest
import torch

from nbn.scm import StructuralCausalModel
from nbn.scm.mechanisms import AdditiveNoiseMechanism
from nbn.structure import HypothesisSet
from nbn.structure.batched import StackedMLP, train_stacked
from tests.conftest import available_devices

VARS = {"x": ("continuous", 1), "y": ("continuous", 1), "a": ("discrete", 2),
        "x'": ("continuous", 1), "d'": ("discrete", 2)}
CANDS = {"x'": [("x", "a"), ("x", "a", "y"), ("a",)], "d'": [("x",), ("y",), ("a",)]}
NOISE_SD = 0.3


def simulate(n, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g)
    y = torch.randn(n, generator=g)
    a = torch.randint(0, 2, (n,), generator=g).float()
    x1 = torch.sin(x) + 0.5 * a + NOISE_SD * torch.randn(n, generator=g)
    d1 = (x + 0.2 * torch.randn(n, generator=g) > 0).float()
    return {k: v.to(device) for k, v in {"x": x, "y": y, "a": a, "x'": x1, "d'": d1}.items()}


# ---------------------------------------------------------------------------
# The batched trainer
# ---------------------------------------------------------------------------


def _masks():
    return torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])


@pytest.mark.parametrize("loss", ["mse", "cross_entropy"])
def test_batched_training_equals_one_candidate_at_a_time(loss):
    g = torch.Generator().manual_seed(1)
    z = torch.randn(300, 3, generator=g)
    y = (z[:, :1] * 2 + z[:, 1:2]) if loss == "mse" else (z[:, 0] > 0).long()
    out_dim = 1 if loss == "mse" else 2
    masks = _masks()
    gens = lambda: [torch.Generator().manual_seed(10 + c) for c in range(3)]  # noqa: E731
    net = StackedMLP(masks, (16, 16), out_dim, activation="silu", generators=gens())
    train_stacked(net, z, y, loss, epochs=5, lr=1e-2, batch_size=64,
                  order_generator=torch.Generator().manual_seed(5))
    for c in range(3):
        alone = StackedMLP(masks[c:c + 1], (16, 16), out_dim, activation="silu",
                           generators=[gens()[c]])
        train_stacked(alone, z, y, loss, epochs=5, lr=1e-2, batch_size=64,
                      order_generator=torch.Generator().manual_seed(5))
        for w_all, w_one in zip(net.weights, alone.weights):
            torch.testing.assert_close(w_all[c], w_one[0], rtol=1e-4, atol=1e-5)


def test_unused_inputs_never_reach_a_candidate():
    g = torch.Generator().manual_seed(2)
    z = torch.randn(200, 3, generator=g)
    net = StackedMLP(_masks(), (8,), 1, activation="relu")
    train_stacked(net, z, z[:, :1], "mse", epochs=3, lr=1e-2)
    out = net(z)
    z2 = z.clone()
    z2[:, 2] = torch.randn(200, generator=g) * 100           # change the input candidates 0 and 2 ignore
    out2 = net(z2)
    torch.testing.assert_close(out[0], out2[0])
    torch.testing.assert_close(out[2], out2[2])
    assert not torch.allclose(out[1], out2[1])
    assert float(net.weights[0][0][2].detach().abs().max()) == 0.0    # masked weights stay zero


@pytest.mark.parametrize("device", available_devices())
def test_export_matches_the_stacked_forward(device):
    g = torch.Generator().manual_seed(3)
    z = torch.randn(50, 3, generator=g).to(device)
    net = StackedMLP(_masks(), (8, 8), 2, activation="silu").to(device)
    train_stacked(net, z, torch.randn(50, 2, generator=g).to(device), "mse", epochs=5, lr=1e-2)
    for c in range(3):
        active = torch.nonzero(_masks()[c]).reshape(-1).tolist()
        seq = net.export(c)
        assert next(seq.parameters()).device.type == torch.device(device).type
        torch.testing.assert_close(seq(z[:, active]), net(z)[c], rtol=1e-4, atol=1e-5)
        perm = active[::-1]                                  # inputs in another order
        torch.testing.assert_close(net.export(c, order=perm)(z[:, perm]), net(z)[c], rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match="permutation"):
        net.export(0, order=[0, 2])


@pytest.mark.parametrize("device", available_devices())
def test_exported_mlp_mechanisms_reproduce_their_training_loss(device):
    """Regression (CI, MPS): the exported location must be the trained one."""
    data = simulate(1500, device=device)
    hs = HypothesisSet(VARS, CANDS)
    fm = hs.fit(data, location="mlp", epochs=30)
    for c in CANDS["x'"]:
        mech = fm[("x'", c)]
        pa = torch.stack([data[p] for p in c], -1)
        mse = float((mech.location(pa)[:, 0] - data["x'"]).pow(2).mean())
        assert mse == pytest.approx(fm.train_loss[("x'", c)], rel=1e-3, abs=1e-5)
    mech = fm[("d'", ("x",))]
    logits_ll = float(mech.log_prob(data["d'"], data["x"].reshape(-1, 1)).mean())
    assert -logits_ll == pytest.approx(fm.train_loss[("d'", ("x",))], rel=1e-3, abs=1e-5)


# ---------------------------------------------------------------------------
# HypothesisSet.fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices())
def test_linear_subblocks_equal_separate_closed_form_fits(device):
    data = simulate(2000, device=device)
    hs = HypothesisSet(VARS, CANDS)
    fm = hs.fit(data, location="linear")
    for c in CANDS["x'"]:
        mech = fm[("x'", c)]
        ref = AdditiveNoiseMechanism(location="linear")
        pa = torch.stack([data[p] for p in c], -1)
        ref.fit_local(data["x'"].reshape(-1, 1), pa)
        torch.testing.assert_close(mech.net.weight, ref.net.weight, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(mech.net.bias, ref.net.bias, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(mech.noise.scale, ref.noise.scale, rtol=1e-4, atol=1e-5)
        assert fm.kinds[("x'", c)] == "linear"


def test_linear_fit_with_integer_weights_equals_replication():
    data = simulate(500)
    w = torch.randint(1, 4, (500,)).float()
    rep = {k: v.repeat_interleave(w.long()) for k, v in data.items()}
    hs = HypothesisSet(VARS, {"x'": [("x", "a")]})
    a, b = hs.fit(data, location="linear", weights=w), hs.fit(rep, location="linear")
    torch.testing.assert_close(a[("x'", ("x", "a"))].net.weight, b[("x'", ("x", "a"))].net.weight,
                               rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", available_devices())
def test_mlp_fit_kinds_quality_and_device(device):
    data = simulate(3000, device=device)
    hs = HypothesisSet(VARS, CANDS)
    fm = hs.fit(data, location="mlp", epochs=80)
    assert fm.kinds[("x'", ("x", "a"))] == "mlp" and fm.kinds[("d'", ("a",))] == "table"
    assert fm.kinds[("d'", ("x",))] == "categorical_mlp"
    # the true parents reach about the noise variance; the candidate without x does not
    assert fm.train_loss[("x'", ("x", "a"))] < 1.5 * NOISE_SD ** 2
    assert fm.train_loss[("x'", ("a",))] > 3 * NOISE_SD ** 2
    mech = fm[("x'", ("x", "a"))]
    assert isinstance(mech, AdditiveNoiseMechanism) and mech.location_kind == "mlp"
    assert next(mech.net.parameters()).device.type == torch.device(device).type
    assert float(mech.noise.scale.detach().reshape(-1)[0]) == pytest.approx(NOISE_SD, rel=0.25)


def test_mlp_location_matches_a_separate_additive_noise_fit_in_quality():
    data = simulate(3000)
    hs = HypothesisSet(VARS, {"x'": [("x", "a")]})
    fm = hs.fit(data, location="mlp", epochs=150)
    ref = AdditiveNoiseMechanism(location="mlp", epochs=150)
    pa = torch.stack([data["x"], data["a"]], -1)
    torch.manual_seed(0)
    ref.fit_local(data["x'"].reshape(-1, 1), pa)
    test = simulate(2000, seed=9)
    pa_t = torch.stack([test["x"], test["a"]], -1)
    mse_b = float((fm[("x'", ("x", "a"))].location(pa_t)[:, 0] - test["x'"]).pow(2).mean())
    mse_r = float((ref.location(pa_t)[:, 0] - test["x'"]).pow(2).mean())
    assert mse_b < 1.2 * mse_r + 0.01


def test_model_for_gives_a_structural_model_with_the_fitted_mechanisms():
    data = simulate(2000)
    hs = HypothesisSet(VARS, CANDS)
    fm = hs.fit(data, location="linear")
    h = {"x'": ("x", "a"), "d'": ("x",)}
    m = hs.model_for(h, fm)
    assert isinstance(m, StructuralCausalModel)
    assert set(m.variables) == {"x", "a", "x'", "d'"}
    pa = torch.stack([data["x"], data["a"]], -1)
    torch.testing.assert_close(m.mechanisms["x'"].log_prob(data["x'"], pa),
                               fm[("x'", ("x", "a"))].log_prob(data["x'"], pa))
    assert m.mechanisms["x'"] is not fm[("x'", ("x", "a"))]         # copies, not shared objects
    m.sample(5)                                                      # roots are fitted too


def test_discrete_only_counts_skip_continuous_factors():
    data = simulate(500)
    hs = HypothesisSet(VARS, CANDS)
    fc = hs.counts(data, discrete_only=True)
    assert set(fc.tables) == {("d'", ("a",))}
    with pytest.raises(ValueError, match="calibrated_sets"):
        hs.counts(data)


# ---------------------------------------------------------------------------
# Calibrated sets
# ---------------------------------------------------------------------------


def _true_interval_coverage(cs, fm, test, factor):
    res = cs[factor]
    node, parents = factor
    pa = torch.stack([test[p] for p in parents], -1)
    lo, hi = res.interval(fm[factor], pa)
    return float(((test[node] >= lo) & (test[node] <= hi)).float().mean())


def test_calibrated_sets_split_the_level_and_declare_guarantees():
    data = simulate(3000)
    cal = simulate(1500, seed=1)
    hs = HypothesisSet(VARS, CANDS)
    fm = hs.fit(data, location="linear")
    cs = hs.calibrated_sets(fm, cal, method="cqr", level=0.9)
    assert len(cs.sets) == 5 and cs.per_factor == pytest.approx(0.1 / 6)    # split over all 6 factors
    assert set(cs.guarantees.values()) == {"marginal"}
    assert ("d'", ("a",)) in cs.skipped
    pit = hs.calibrated_sets(fm, cal, method="pit", level=0.9)
    assert set(pit.guarantees.values()) == {"pac_if_pit_independent"}
    test = simulate(4000, seed=2)
    cov = _true_interval_coverage(cs, fm, test, ("x'", ("x", "a")))
    assert cov >= 1 - cs.per_factor - 0.01               # marginal coverage of one factor


def test_calibration_errors():
    data = simulate(500)
    hs = HypothesisSet(VARS, CANDS)
    fm = hs.fit(data, location="linear")
    with pytest.raises(ValueError):
        hs.calibrated_sets(fm, data, level=1.0)
    with pytest.raises(ValueError):
        hs.calibrated_sets(fm, data, method="bootstrap")
    other = HypothesisSet(VARS, CANDS)
    with pytest.raises(ValueError, match="another"):
        other.calibrated_sets(fm, data)


@pytest.mark.slow
def test_joint_coverage_across_factors():
    """All factor intervals cover a fresh row together in >= level of draws (union bound)."""
    hs = HypothesisSet(VARS, {"x'": [("x", "a"), ("x", "a", "y"), ("a",)]})
    data = simulate(3000)
    fm = hs.fit(data, location="linear")
    level, reps, hits = 0.8, 30, 0
    for r in range(reps):
        cal = simulate(400, seed=100 + r)
        cs = hs.calibrated_sets(fm, cal, method="residual", level=level)
        test = simulate(200, seed=500 + r)
        ok = torch.ones(200, dtype=torch.bool)
        for f in hs.factors:
            pa = torch.stack([test[p] for p in f[1]], -1)
            lo, hi = cs[f].interval(fm[f], pa)
            ok &= (test["x'"] >= lo) & (test["x'"] <= hi)
        hits += int(ok.float().mean() >= level - 3 * math.sqrt(level * (1 - level) / 200))
    assert hits / reps >= 0.8



# ---------------------------------------------------------------------------
# Review regressions (#323)
# ---------------------------------------------------------------------------


def test_table_factors_are_structural_so_counterfactuals_work():
    from nbn.scm.categorical import StructuralCategoricalTable

    data = simulate(2000)
    hs = HypothesisSet(VARS, CANDS)
    fm = hs.fit(data, location="linear")
    assert isinstance(fm[("d'", ("a",))], StructuralCategoricalTable)
    assert isinstance(fm.roots["a"], StructuralCategoricalTable)
    m = hs.model_for({"x'": ("x", "a"), "d'": ("a",)}, fm)
    noise = m.abduct({k: v[:5] for k, v in data.items()})
    assert set(noise) >= {"x'", "d'"}                       # every structural node is abducted


def test_mixed_factor_sets_and_calibrated_sets_share_one_budget():
    data, cal = simulate(2000), simulate(800, seed=1)
    hs = HypothesisSet(VARS, CANDS)
    fs = hs.factor_sets(data, delta=0.1)                      # count-based factors only
    assert set(fs.sets) == {("d'", ("a",))} and len(fs.skipped) == 5
    cs = hs.calibrated_sets(hs.fit(data, location="linear"), cal, level=0.9)
    assert fs.delta_per_factor == pytest.approx(cs.per_factor) == pytest.approx(0.1 / 6)
    with pytest.raises(KeyError, match="No count-based set"):
        fs.sets_for({"x'": ("x", "a"), "d'": ("a",)})


@pytest.mark.parametrize("location", ["mlp"])
def test_a_candidates_fit_does_not_depend_on_its_siblings(location):
    data = simulate(1000)
    one = HypothesisSet(VARS, {"x'": [("x", "a")]}).fit(data, location=location, epochs=20)
    many = HypothesisSet(VARS, {"x'": [("y",), ("x", "a"), ("a",)]}).fit(data, location=location, epochs=20)
    a, b = one[("x'", ("x", "a"))], many[("x'", ("x", "a"))]
    for p, q in zip(a.net.parameters(), b.net.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("location", ["linear", "mlp"])
def test_multi_dimensional_parents(location):
    g = torch.Generator().manual_seed(4)
    n = 1500
    u = torch.randn(n, 2, generator=g)
    x1 = u[:, 0] - 2 * u[:, 1] + 0.1 * torch.randn(n, generator=g)
    v = {"u": ("continuous", 2), "z": ("continuous", 1), "x'": ("continuous", 1)}
    data = {"u": u, "z": torch.randn(n, generator=g), "x'": x1}
    hs = HypothesisSet(v, {"x'": [("u",), ("u", "z")]})
    fm = hs.fit(data, location=location, epochs=60)
    mech = fm[("x'", ("u",))]
    resid = mech.residual(x1.reshape(-1, 1), u)
    assert float(resid.detach().std()) < 0.3
    if location == "linear":
        ref = AdditiveNoiseMechanism(location="linear")
        ref.fit_local(x1.reshape(-1, 1), u)
        torch.testing.assert_close(mech.net.weight, ref.net.weight, rtol=1e-4, atol=1e-5)
    w = torch.ones(n)
    hs.fit(data, location="linear", weights=w)              # weights of length N are accepted


def test_model_for_fitted_honours_device_and_rejects_alpha():
    data = simulate(500)
    hs = HypothesisSet(VARS, CANDS)
    fm = hs.fit(data, location="linear")
    m = hs.model_for({"x'": ("x", "a"), "d'": ("a",)}, fm, device="cpu")
    assert m.device.type == "cpu"
    with pytest.raises(ValueError, match="fit time"):
        hs.model_for({"x'": ("x", "a"), "d'": ("a",)}, fm, alpha=1.0)


def test_categorical_factors_are_consolidated_for_updates():
    data = simulate(1500)
    hs = HypothesisSet(VARS, {"d'": [("x",)]})
    fm = hs.fit(data, epochs=10)
    mech = fm[("d'", ("x",))]
    assert mech._ewc_mu is not None and not mech.training
    m = hs.model_for({"d'": ("x",)}, fm)
    m.update({k: v[:300] for k, v in simulate(300, seed=5).items() if k in ("x", "d'")})


def test_train_loss_is_the_full_data_loss():
    data = simulate(1000)
    hs = HypothesisSet(VARS, {"x'": [("x", "a")]})
    fm = hs.fit(data, location="mlp", epochs=20, batch_size=128)
    mech = fm[("x'", ("x", "a"))]
    pa = torch.stack([data["x"], data["a"]], -1)
    mse = float((mech.location(pa)[:, 0] - data["x'"]).pow(2).mean().detach())
    assert fm.train_loss[("x'", ("x", "a"))] == pytest.approx(mse, rel=1e-4)


# ---------------------------------------------------------------------------
# Review round 2 regressions (#323)
# ---------------------------------------------------------------------------


def test_zero_weight_mini_batches_do_not_poison_the_fit():
    data = simulate(2000)
    w = torch.zeros(2000)
    w[:40] = 1.0                                   # most mini-batches have zero total weight
    hs = HypothesisSet(VARS, {"x'": [("x", "a")], "d'": [("x",)]})
    fm = hs.fit(data, location="mlp", epochs=3, batch_size=64, weights=w)
    for f in hs.factors:
        assert all(bool(torch.isfinite(p).all()) for p in fm[f].parameters())
        assert math.isfinite(fm.train_loss[f])


def test_initial_weights_do_not_depend_on_the_union_order():
    v = {"b": ("continuous", 1), "x": ("continuous", 1), "x'": ("continuous", 1)}
    g = torch.Generator().manual_seed(0)
    data = {k: torch.randn(500, generator=g) for k in v}
    alone = HypothesisSet(v, {"x'": [("b", "x")]}).fit(data, location="mlp", epochs=0)
    with_sibling = HypothesisSet(v, {"x'": [("x",), ("b", "x")]}).fit(data, location="mlp", epochs=0)
    for p, q in zip(alone[("x'", ("b", "x"))].net.parameters(), with_sibling[("x'", ("b", "x"))].net.parameters()):
        torch.testing.assert_close(p, q)


def test_mixed_set_edge_cases():
    data = simulate(800)
    hs = HypothesisSet(VARS, CANDS)
    fc = hs.counts(data, discrete_only=True)               # continuous inputs are skipped, no crash
    assert set(fc.input_marginals) == {"a"}
    with pytest.raises(ValueError, match="fit"):
        hs.model_for({"x'": ("x", "a"), "d'": ("a",)}, fc)
    fs = hs.factor_sets(data)
    from nbn.structure import export_factored_sets
    with pytest.raises(ValueError, match="no count-based set"):
        export_factored_sets("/nonexistent/should-not-be-created", fs, hs, ["x", "y"], ["a"])
    only_continuous = HypothesisSet(VARS, {"x'": [("x",)]})
    with pytest.raises(ValueError, match="No count-based factor"):
        only_continuous.factor_sets(data)
