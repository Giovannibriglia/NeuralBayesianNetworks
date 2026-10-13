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


def test_export_matches_the_stacked_forward():
    g = torch.Generator().manual_seed(3)
    z = torch.randn(50, 3, generator=g)
    net = StackedMLP(_masks(), (8, 8), 2, activation="silu")
    for c in range(3):
        seq = net.export(c)
        active = _masks()[c].bool()
        torch.testing.assert_close(seq(z[:, active]), net(z)[c], rtol=1e-5, atol=1e-6)


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
    assert float(mech.noise.scale.reshape(-1)[0]) == pytest.approx(NOISE_SD, rel=0.25)


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
    fc = hs.counts(data, discrete_only=True, input_marginals=False)
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
    assert len(cs.sets) == 5 and cs.per_factor == pytest.approx(0.1 / 5)
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
