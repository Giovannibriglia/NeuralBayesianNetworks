"""Regression tests for the core review findings (nbn/core, learning, update,
sampling, utils).  Each test names the finding it pins (A1, A2, ... B1, C*).
"""
from __future__ import annotations

import pytest
import torch

from tests.conftest import available_devices

ACCELERATORS = [d for d in available_devices() if d != "cpu"]


# ---------------------------------------------------------------------------
# A1 — to_device must not use non_blocking for device -> CPU copies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ACCELERATORS or ["cpu"])
def test_a1_to_device_accelerator_to_cpu_reads_back_values(device):
    from nbn.utils.device import to_device

    x = torch.randn(20000, 3, device=device) + 5.0
    y = to_device(x, torch.device("cpu"))
    # Read y before anything else synchronises the device: a non_blocking
    # D2H copy is not synchronised and used to come back as zeros / garbage.
    y_snapshot = y.clone()
    torch.testing.assert_close(y_snapshot, x.cpu())
    back = to_device(y, torch.device(device))
    torch.testing.assert_close(back.cpu(), x.cpu())


# ---------------------------------------------------------------------------
# A2 — weighted_moments(unbiased=True) with fractional / normalised weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices())
def test_a2_unbiased_weighted_std_is_scale_invariant_for_fractional_weights(device):
    from nbn.learning.weighting import weighted_moments

    g = torch.Generator().manual_seed(0)
    t = (torch.randn(500, 2, generator=g) * 2.0 + 3.0).to(device)
    ref = t.std(0, unbiased=True)
    # Normalised (EM-responsibility-like) weights: sum(w) == 1, so the old
    # frequency-weight denominator sum(w) - 1 was ~0 and std exploded.
    w = torch.full((500,), 1.0 / 500, device=device)
    _, s = weighted_moments(t, w, unbiased=True)
    torch.testing.assert_close(s, ref, atol=1e-4, rtol=1e-4)
    # Any constant rescaling of fractional weights gives the same answer.
    w2 = torch.rand(500, generator=g).to(device) + 0.1
    _, s_a = weighted_moments(t, w2, unbiased=True)
    _, s_b = weighted_moments(t, w2 * 0.01, unbiased=True)
    torch.testing.assert_close(s_a, s_b, atol=1e-4, rtol=1e-4)
    assert (s_a - ref).abs().max() < 0.3


def test_a2_integer_weights_keep_replication_equivalence():
    from nbn.learning.weighting import weighted_moments

    t = torch.randn(40, 3)
    w = torch.randint(0, 4, (40,)).float()
    w[0] = 2.0
    rep = t.repeat_interleave(w.long(), dim=0)
    m, s = weighted_moments(t, w, unbiased=True)
    torch.testing.assert_close(m, rep.mean(0), atol=1e-5, rtol=0)
    torch.testing.assert_close(s, rep.std(0, unbiased=True), atol=1e-5, rtol=0)


def test_a2_effective_sample_size():
    from nbn.learning.weighting import effective_sample_size

    assert float(effective_sample_size(torch.ones(10))) == pytest.approx(10.0)
    assert float(effective_sample_size(torch.full((10,), 0.1))) == pytest.approx(10.0)
    assert float(effective_sample_size(torch.tensor([1.0, 0.0, 0.0]))) == pytest.approx(1.0)
    assert effective_sample_size(None, n=7) == 7


# ---------------------------------------------------------------------------
# A3 — recursive Gaussian update must survive a large data offset in float32
# ---------------------------------------------------------------------------


def _offset_lg_data(n=4000, offset=1000.0):
    g = torch.Generator().manual_seed(1)
    pa = torch.randn(n, 2, generator=g) + offset
    x = pa @ torch.tensor([[0.5], [-0.3]]) + offset + 0.1 * torch.randn(n, 1, generator=g)
    return pa, x


def _reference_fit(pa, x):
    """Float64 centred OLS on the pooled data: (W, b, sigma)."""
    pa, x = pa.double(), x.double()
    pc, xc = pa - pa.mean(0), x - x.mean(0)
    W = torch.linalg.solve(pc.T @ pc, pc.T @ xc)
    b = x.mean(0) - pa.mean(0) @ W
    sigma = (xc - pc @ W).pow(2).mean(0).sqrt()
    return W, b, sigma


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("forgetting", [1.0, 0.7])
def test_a3_recursive_gaussian_update_matches_pooled_fit_at_large_offset(device, forgetting):
    from nbn.update import recursive_gaussian as rg

    pa, x = _offset_lg_data()
    chunks = [(pa[i:i + 1000].to(device), x[i:i + 1000].to(device)) for i in range(0, 4000, 1000)]
    state = rg.batch_statistics(*chunks[0])
    for p_c, x_c in chunks[1:]:
        state = rg.accumulate(state, rg.batch_statistics(p_c, x_c), forgetting=forgetting)
    W, b, log_s = rg.solve(state, ridge=0.0)
    assert state.A.dtype == torch.float32
    # Pooled reference: the same forgetting expressed as row weights.
    w_rows = torch.cat([
        torch.full((1000,), forgetting ** (3 - k), dtype=torch.float64) for k in range(4)
    ])
    rw = w_rows.sqrt().reshape(-1, 1)
    p64, x64 = pa.double(), x.double()
    mp = (w_rows[:, None] * p64).sum(0) / w_rows.sum()
    mx = (w_rows[:, None] * x64).sum(0) / w_rows.sum()
    pc, xc = (p64 - mp) * rw, (x64 - mx) * rw
    W_ref = torch.linalg.solve(pc.T @ pc, pc.T @ xc)
    sig_ref = ((xc - pc @ W_ref).pow(2).sum(0) / w_rows.sum()).sqrt()
    torch.testing.assert_close(W.cpu().double(), W_ref, atol=2e-3, rtol=0)
    torch.testing.assert_close(log_s.exp().cpu().double(), sig_ref, atol=2e-3, rtol=0)
    # Predictions at the data mean agree (b alone is poorly identified).
    pred = (mp.float().to(device) @ W + b).cpu().double()
    torch.testing.assert_close(pred, mp @ W_ref + (mx - mp @ W_ref), atol=5e-3, rtol=0)


@pytest.mark.parametrize("device", available_devices())
def test_a3_linear_gaussian_update_keeps_sigma_at_large_offset(device):
    from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism

    pa, x = _offset_lg_data()
    pa, x = pa.to(device), x.to(device)
    mech = LinearGaussianMechanism()
    mech.fit_local(x[:2000], pa[:2000])
    mech.update_local(x[2000:], pa[2000:])
    _, _, sig_ref = _reference_fit(pa.cpu(), x.cpu())
    sigma = mech._log_scale.exp().detach().cpu().double()
    torch.testing.assert_close(sigma, sig_ref, atol=3e-3, rtol=0)
    W_ref = _reference_fit(pa.cpu(), x.cpu())[0]
    torch.testing.assert_close(mech._weight.detach().cpu().double(), W_ref, atol=5e-3, rtol=0)


def test_a3_legacy_raw_state_is_converted():
    """A checkpoint written with the raw normal-equation layout still updates."""
    from nbn.update import recursive_gaussian as rg

    g = torch.Generator().manual_seed(2)
    pa = torch.randn(300, 1, generator=g, dtype=torch.float64)
    x = 2.0 * pa + 1.0 + 0.5 * torch.randn(300, 1, generator=g, dtype=torch.float64)
    z = torch.cat([pa, torch.ones(300, 1, dtype=torch.float64)], 1)
    legacy = rg.NormalEquationState(
        (z.T @ z).float(), (z.T @ x).float(), (x * x).sum(0).float(),
        torch.tensor(300.0), 1,
    )
    assert not rg.is_centred(legacy)
    st = rg.accumulate(legacy, rg.batch_statistics(pa.float(), x.float()))
    pooled = rg.batch_statistics(torch.cat([pa, pa]).float(), torch.cat([x, x]).float())
    for got, want in zip(rg.solve(st), rg.solve(pooled)):
        torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# A18 — LogFactor.condition with several evidence keys
# ---------------------------------------------------------------------------


def test_a18_logfactor_condition_multiple_keys_selects_right_axes():
    from nbn.core.factor import LogFactor

    vals = torch.arange(2 * 3 * 4, dtype=torch.float).reshape(2, 3, 4)
    f = LogFactor(vals, ["A", "B", "C"], {"A": 2, "B": 3, "C": 4})
    out = f.condition({"A": 1, "C": 2})
    assert out.variables == ["B"]
    torch.testing.assert_close(out.log_values, vals[1, :, 2])
    out2 = f.condition({"C": 3, "B": 0})
    assert out2.variables == ["A"]
    torch.testing.assert_close(out2.log_values, vals[:, 0, 3])


def test_a18_logfactor_condition_rejects_out_of_range():
    from nbn.core.factor import LogFactor

    f = LogFactor(torch.zeros(2, 3), ["A", "B"], {"A": 2, "B": 3})
    with pytest.raises(ValueError, match="outside"):
        f.condition({"B": 3})
    with pytest.raises(ValueError, match="outside"):
        f.condition({"A": -1})


# ---------------------------------------------------------------------------
# A19 — log_einsum_exp with all -inf operands / underflow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices())
def test_a19_log_einsum_exp_all_neg_inf_operand(device):
    from nbn.utils.einsum import log_einsum_exp

    a = torch.full((3,), float("-inf"), device=device)
    b = torch.zeros(3, device=device)
    out = log_einsum_exp("i,i->", a, b)
    assert not torch.isnan(out).any()
    assert torch.isneginf(out).all()
    # A partly -inf operand keeps its finite part exactly.
    a2 = torch.tensor([0.0, float("-inf"), 0.0], device=device)
    out2 = log_einsum_exp("i,i->", a2, b)
    torch.testing.assert_close(out2.cpu(), torch.log(torch.tensor(2.0)))


def test_a19_log_einsum_exp_no_artificial_floor():
    from nbn.utils.einsum import log_einsum_exp

    # Product underflows after the shift: the true answer is -inf in float32,
    # never the old clamp value log(1e-40) + shift (a finite, wrong number).
    a = torch.tensor([0.0, -200.0])
    b = torch.tensor([-200.0, 0.0])
    out = log_einsum_exp("i,i->", a, b)
    assert torch.isneginf(out) or out < -150.0
    ok = log_einsum_exp("i,i->", torch.tensor([-1000.0, -1000.0]), torch.tensor([-5.0, -5.0]))
    torch.testing.assert_close(ok, torch.tensor(-1005.0 + float(torch.log(torch.tensor(2.0)))))


# ---------------------------------------------------------------------------
# B1 — node names colliding with nn.Module attributes, ints, "__DOT__"
# ---------------------------------------------------------------------------

_RESERVED_NAMES = [
    "train", "eval", "to", "cuda", "forward", "parameters", "training",
    "values", "keys", "update", "type", "float", "node__x", "a.b", "",
]


def _discrete_chain(names, device="cpu"):
    from nbn import NeuralBayesianNetwork as NBN

    edges = list(zip(names[:-1], names[1:]))
    model = NBN(edges, dict.fromkeys(names, ("discrete", 2)), device=device)
    g = torch.Generator().manual_seed(0)
    data = {n: torch.randint(0, 2, (200,), generator=g) for n in names}
    return model, data


def test_b1_reserved_node_names_fit_query_and_round_trip(tmp_path):
    from nbn import NeuralBayesianNetwork as NBN

    model, data = _discrete_chain(_RESERVED_NAMES)
    model.auto_mechanisms()
    model.fit(data)
    assert set(model.mechanisms.keys()) == set(_RESERVED_NAMES)
    # nn.Module behaviour is intact (the attributes were not shadowed).
    model.train()
    model.eval()
    p = model.query(["train"], evidence={"float": 1})
    assert p.shape == (2,)
    path = tmp_path / "m.pt"
    model.save(str(path))
    loaded = NBN.load(str(path))
    torch.testing.assert_close(loaded.query(["train"], evidence={"float": 1}), p)
    # state_dict round trip into a fresh model with the same names.
    fresh, _ = _discrete_chain(_RESERVED_NAMES)
    fresh.auto_mechanisms()
    fresh.fit(data)
    fresh.load_state_dict(model.state_dict())


def test_b1_bad_node_names_rejected_early():
    from nbn import NeuralBayesianNetwork as NBN

    with pytest.raises(TypeError, match="str"):
        NBN([(0, 1)], {0: ("discrete", 2), 1: ("discrete", 2)})
    with pytest.raises(ValueError, match="__DOT__"):
        NBN([], {"a__DOT__b": ("discrete", 2)})


# ---------------------------------------------------------------------------
# B3 / B4 — .cpu()/.cuda() refresh _device; cache version bumps
# ---------------------------------------------------------------------------


def _lg_model(device):
    from nbn import NeuralBayesianNetwork as NBN

    model = NBN([("A", "B")], {"A": ("continuous", 1), "B": ("continuous", 1)}, device=device)
    model.auto_mechanisms(default_continuous="lg")
    g = torch.Generator().manual_seed(0)
    a = torch.randn(200, 1, generator=g)
    data = {"A": a, "B": 2 * a + 0.1 * torch.randn(200, 1, generator=g)}
    return model, data


@pytest.mark.parametrize("device", ACCELERATORS or ["cpu"])
def test_b3_cpu_refreshes_device(device):
    model, data = _lg_model(device)
    model.fit(data)
    model.cpu()
    assert model.device.type == "cpu"
    assert all(p.device.type == "cpu" for p in model.parameters())
    model.fit(data)                     # data must now land on cpu, not on `device`
    assert model.mechanisms["B"]._weight.device.type == "cpu"
    s = model.sample(5)
    assert s["B"].device.type == "cpu"


@pytest.mark.parametrize("device", ACCELERATORS or ["cpu"])
def test_b3_device_refresh_for_parameterless_model(device):
    model, _ = _lg_model("cpu")          # mechanisms registered, not fitted
    if device == "cuda":
        model.cuda()
    else:
        # .to(tensor) has no explicit device spec: _apply must infer it.
        model.to(torch.empty(0, device=device))
    assert model.device.type == device
    model.cpu()
    assert model.device.type == "cpu"


def test_b4_cache_version_bumps_on_direct_writes_load_and_moves():
    from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism

    model, data = _lg_model("cpu")
    model.fit(data)
    v = model._cache_version
    model.mechanisms["B"] = LinearGaussianMechanism()
    assert model._cache_version > v
    v = model._cache_version
    del model.mechanisms["B"]
    assert model._cache_version > v
    model.set_mechanism("B", LinearGaussianMechanism())
    model.fit(data)
    sd = model.state_dict()
    v = model._cache_version
    model.load_state_dict(sd)
    assert model._cache_version > v
    v = model._cache_version
    model.cpu()
    assert model._cache_version > v
    # Setter keeps working (callers do ``_cache_version += 1``).
    v = model._cache_version
    model._cache_version += 1
    assert model._cache_version == v + 1
    import copy
    assert copy.deepcopy(model)._cache_version == model._cache_version


# ---------------------------------------------------------------------------
# B2 — float64 inputs on a device without float64 (MPS)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices())
def test_b2_float64_inputs_are_accepted_on_every_device(device):
    import numpy as np

    from nbn import NeuralBayesianNetwork as NBN

    model = NBN(
        [("D", "A"), ("A", "B")],
        {"D": ("discrete", 2), "A": ("continuous", 1), "B": ("continuous", 1)},
        device=device,
    )
    model.auto_mechanisms(default_continuous="lg")
    g = torch.Generator().manual_seed(0)
    d = torch.randint(0, 2, (300,), generator=g)
    a = torch.randn(300, 1, generator=g, dtype=torch.float64) + d.unsqueeze(1)
    data = {"D": d, "A": a, "B": 2 * a + 0.1 * torch.randn(300, 1, generator=g, dtype=torch.float64)}
    w = torch.ones(300, dtype=torch.float64)
    model.fit(data, weights=w)
    model.fit(data, weights=np.ones(300))
    model.update(data)
    lp = model.log_prob(data)
    assert torch.isfinite(lp).all()
    model.query(["D"], evidence={"A": np.array([0.5])})
    model.query_batch(
        ["D"], evidence={"A": torch.tensor([[0.5], [1.0]], dtype=torch.float64)},
    )
    m2 = model.intervene({"A": torch.tensor(1.0, dtype=torch.float64)})
    assert m2.sample(4)["A"].dtype != torch.float64 or device != "mps"
    s = model.sample(4, do={"A": torch.tensor([1.0], dtype=torch.float64)})
    torch.testing.assert_close(s["A"].cpu().double(), torch.ones(4, 1, dtype=torch.float64))


# ---------------------------------------------------------------------------
# A14 / B8 — discrete do-values: integral, in range; scalar / int do-values
# ---------------------------------------------------------------------------


def _mixed_model(device="cpu"):
    from nbn import NeuralBayesianNetwork as NBN

    model = NBN(
        [("A", "X"), ("X", "Y")],
        {"A": ("discrete", 3), "X": ("continuous", 1), "Y": ("continuous", 1)},
        device=device,
    )
    model.auto_mechanisms(default_continuous="lg")
    g = torch.Generator().manual_seed(0)
    a = torch.randint(0, 3, (400,), generator=g)
    x = a.float().unsqueeze(1) + 0.1 * torch.randn(400, 1, generator=g)
    y = 2 * x + 0.1 * torch.randn(400, 1, generator=g)
    model.fit({"A": a, "X": x, "Y": y})
    return model


@pytest.mark.parametrize("bad", [1.7, -1, 3, torch.tensor([2.5])])
def test_a14_intervene_rejects_non_integral_or_out_of_range_discrete(bad):
    model = _mixed_model()
    with pytest.raises(ValueError, match="integer|outside"):
        model.intervene({"A": bad})


@pytest.mark.parametrize("bad", [1.7, 3, torch.tensor([-1.0])])
def test_a14_sample_do_rejects_bad_discrete_values(bad):
    model = _mixed_model()
    with pytest.raises(ValueError, match="integer|outside"):
        model.sample(5, do={"A": bad})


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("val", [1, 1.0, torch.tensor(1), torch.tensor(1.0), torch.tensor([1])])
def test_b8_sample_do_accepts_scalars_and_ints(device, val):
    model = _mixed_model(device)
    s = model.sample(6, do={"A": val, "X": 2})
    assert s["A"].dtype.is_floating_point
    torch.testing.assert_close(s["A"].cpu().float(), torch.ones(6, 1))
    torch.testing.assert_close(s["X"].cpu().float(), torch.full((6, 1), 2.0))
    assert s["Y"].shape == (6, 1)


# ---------------------------------------------------------------------------
# A16 — sample(evidence=...) shapes and evidence-only log-weight
# ---------------------------------------------------------------------------


def test_a16_misaligned_batched_evidence_raises():
    model = _mixed_model()
    with pytest.raises(ValueError, match="Evidence for 'X'"):
        model.sample(10, evidence={"X": torch.zeros(3, 1)})
    with pytest.raises(ValueError, match="Evidence for 'X'"):
        model.sample(10, evidence={"X": torch.zeros(10, 2)})


def test_a16_per_row_evidence_and_log_weight():
    model = _mixed_model()
    xs = torch.linspace(-1, 1, 10).reshape(10, 1)
    s, lp, lw = model.sample(10, evidence={"X": xs}, return_log_prob=True, return_log_weight=True)
    torch.testing.assert_close(s["X"], xs)
    assert lp.shape == lw.shape == (10,)
    # The weight is exactly log p(X = x | A = a) for each sampled A.
    mech = model.mechanisms["X"]
    expected = mech.log_prob(xs, s["A"].float().reshape(10, 1)).reshape(10)
    torch.testing.assert_close(lw, expected)
    # The joint additionally carries A's and Y's terms.
    a_lp = model.mechanisms["A"].log_prob(s["A"], None).reshape(10)
    y_lp = model.mechanisms["Y"].log_prob(s["Y"], xs).reshape(10)
    torch.testing.assert_close(lp, lw + a_lp + y_lp)
    # Without evidence the weight is zero; shapes [n] / scalar evidence work.
    s2, lw2 = model.sample(4, return_log_weight=True)
    assert torch.equal(lw2, torch.zeros(4))
    s3 = model.sample(7, evidence={"X": 0.5, "A": torch.tensor([2])})
    assert s3["X"].shape == (7, 1) and s3["A"].shape == (7, 1)


# ---------------------------------------------------------------------------
# B7 — query() with documented [D] evidence on a multi-dim continuous node
# ---------------------------------------------------------------------------


def test_b7_query_vector_evidence_on_multidim_continuous_node():
    from nbn import NeuralBayesianNetwork as NBN

    model = NBN([("A", "X")], {"A": ("discrete", 2), "X": ("continuous", 3)}, device="cpu")
    model.auto_mechanisms(default_continuous="lg")
    g = torch.Generator().manual_seed(0)
    a = torch.randint(0, 2, (500,), generator=g)
    x = 3.0 * a.float().unsqueeze(1).expand(-1, 3) + 0.3 * torch.randn(500, 3, generator=g)
    model.fit({"A": a, "X": x})
    p = model.query(["A"], evidence={"X": torch.tensor([3.0, 3.0, 3.0])})
    assert p.shape == (2,)
    assert p[1] > 0.99


# ---------------------------------------------------------------------------
# B12 — update() is atomic
# ---------------------------------------------------------------------------


def _snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items() if v is not None}


def test_b12_update_failure_midway_leaves_model_untouched():
    from nbn import NeuralBayesianNetwork as NBN
    from nbn.mechanisms.non_parametric.knn_conditional import KNNConditionalMechanism

    model = NBN([("A", "B")], {"A": ("continuous", 1), "B": ("continuous", 1)}, device="cpu")
    model.set_mechanism("A", __import__("nbn").mechanisms.LinearGaussianMechanism())
    model.set_mechanism("B", KNNConditionalMechanism())
    g = torch.Generator().manual_seed(0)
    a = torch.randn(100, 1, generator=g)
    data = {"A": a, "B": a + 0.1 * torch.randn(100, 1, generator=g)}
    model.fit(data)
    before = _snapshot(model)
    mech_a = model.mechanisms["A"]
    new = {"A": data["A"] + 5.0, "B": data["B"] + 5.0}
    # A (LG) updates fine, then B (kNN) refuses forgetting < 1: must roll back A.
    with pytest.raises(NotImplementedError):
        model.update(new, forgetting=0.5)
    after = _snapshot(model)
    assert before.keys() == after.keys()
    for k in before:
        torch.testing.assert_close(after[k], before[k])
    assert model.mechanisms["A"] is mech_a


def test_b12_update_preflight_missing_column_and_ewc_state():
    from nbn import NeuralBayesianNetwork as NBN

    model, data = _lg_model("cpu")
    model.fit(data)
    before = _snapshot(model)
    with pytest.raises(ValueError, match="missing"):
        model.update({"A": data["A"]})
    for k, v in _snapshot(model).items():
        torch.testing.assert_close(v, before[k])

    m2 = NBN([("A", "B")], {"A": ("continuous", 1), "B": ("continuous", 1)}, device="cpu")
    m2.set_mechanism("A", __import__("nbn").mechanisms.LinearGaussianMechanism())
    m2.set_mechanism("B", __import__("nbn").mechanisms.MDNMechanism(num_components=2))
    m2.fit(data, consolidate=False, epochs=2)
    snap = _snapshot(m2)
    with pytest.raises(RuntimeError, match="EWC state missing.*'B'"):
        m2.update(data)
    for k, v in _snapshot(m2).items():
        torch.testing.assert_close(v, snap[k])


# ---------------------------------------------------------------------------
# A17 — method="joint" honours consolidate; C: weighted TrainHistory
# ---------------------------------------------------------------------------


def _mdn_model():
    from nbn import NeuralBayesianNetwork as NBN
    from nbn.mechanisms import LinearGaussianMechanism, MDNMechanism

    model = NBN([("A", "B")], {"A": ("continuous", 1), "B": ("continuous", 1)}, device="cpu")
    model.set_mechanism("A", LinearGaussianMechanism())
    model.set_mechanism("B", MDNMechanism(num_components=2))
    g = torch.Generator().manual_seed(0)
    a = torch.randn(300, 1, generator=g)
    data = {"A": a, "B": torch.sin(a) + 0.1 * torch.randn(300, 1, generator=g)}
    return model, data


def test_a17_joint_fit_reconsolidates_ewc_anchor():
    model, data = _mdn_model()
    model.fit(data, epochs=3)
    mech = model.mechanisms["B"]
    with pytest.warns(UserWarning, match="A \\(LinearGaussianMechanism\\)"):
        model.fit(data, method="joint", epochs=3, lr=1e-2)
    flat = torch.cat([p.detach().reshape(-1) for p in mech.parameters() if p.requires_grad])
    torch.testing.assert_close(mech._ewc_mu, flat)


def test_a17_joint_fit_without_consolidate_clears_stale_anchor():
    model, data = _mdn_model()
    model.fit(data, epochs=3)
    with pytest.warns(UserWarning):
        model.fit(data, method="joint", epochs=2, consolidate=False)
    assert model.mechanisms["B"]._ewc_mu is None
    with pytest.raises(RuntimeError, match="EWC state missing"):
        model.update(data)


def test_c_train_history_is_weighted():
    model, data = _lg_model("cpu")
    w = torch.zeros(200)
    w[:50] = 1.0
    hist = model.fit(data, weights=w)
    lp = model.mechanisms["A"].log_prob(data["A"], None).detach().reshape(-1)
    assert hist.mean_ll("A") == pytest.approx(float(lp[:50].mean()), rel=1e-5)


# ---------------------------------------------------------------------------
# C — constructor / DAG / auto_mechanisms / mixed precision / intervene /
#     save-load metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["categorical", "Discrete", "cont"])
def test_c_unknown_variable_kind_raises(kind):
    from nbn import NeuralBayesianNetwork as NBN

    with pytest.raises(ValueError, match="unknown kind"):
        NBN([], {"A": (kind, 2)})


def test_c_variables_missing_from_dag_raise():
    from nbn import NeuralBayesianNetwork as NBN
    from nbn.core.dag import DAG

    with pytest.raises(ValueError, match="not nodes of the DAG"):
        NBN(DAG([("A", "B")]), {"A": ("discrete", 2), "B": ("discrete", 2), "C": ("discrete", 2)})


def test_c_dag_does_not_mutate_callers_graph():
    import networkx as nx

    from nbn.core.dag import DAG

    g = nx.DiGraph([("A", "B")])
    DAG(g, extra_nodes=["C"])
    assert set(g.nodes) == {"A", "B"}


def test_c_auto_mechanisms_rejects_unknown_names():
    from nbn import NeuralBayesianNetwork as NBN

    model = NBN([("A", "B")], {"A": ("discrete", 2), "B": ("continuous", 1)}, device="cpu")
    with pytest.raises(ValueError, match="default_continuous"):
        model.auto_mechanisms(default_continuous="lgg")
    with pytest.raises(ValueError, match="default_discrete"):
        model.auto_mechanisms(default_discrete="table")


def test_c_set_mixed_precision_warns():
    model, _ = _lg_model("cpu")
    with pytest.warns(DeprecationWarning, match="no effect"):
        model.set_mixed_precision(True)


def test_c_chained_intervene_keeps_all_do_targets():
    model = _mixed_model()
    m = model.intervene({"A": 1}).intervene({"X": 0.5})
    assert m._do_targets == {"A", "X"}


def test_c_save_load_keeps_state_names_and_mixed_kind(tmp_path):
    from nbn import NeuralBayesianNetwork as NBN

    model = NBN([("A", "M")], {"A": ("discrete", 2), "M": ("mixed", 1)}, device="cpu")
    assert model.variables["M"].kind == "mixed"
    model.state_names = {"A": ["no", "yes"]}
    path = tmp_path / "m.pt"
    model.save(str(path))
    loaded = NBN.load(str(path))
    assert loaded.variables["M"].kind == "mixed"
    assert loaded.state_names == {"A": ["no", "yes"]}


# ---------------------------------------------------------------------------
# Round 2 — R2: weighted_quantile with fractional / normalised weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices())
def test_r2_weighted_quantile_normalised_weights_match_unweighted(device):
    from nbn.learning.weighting import weighted_quantile

    g = torch.Generator().manual_seed(0)
    t = torch.randn(101, 2, generator=g).to(device)
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    ref = torch.quantile(t.cpu(), q, dim=0)
    w = torch.full((101,), 1.0 / 101, device=device)
    torch.testing.assert_close(weighted_quantile(t, q, w).cpu(), ref, atol=1e-5, rtol=0)
    # Scale invariance for fractional weights.
    w2 = (torch.rand(101, generator=g) + 0.1).to(device)
    torch.testing.assert_close(
        weighted_quantile(t, q, w2), weighted_quantile(t, q, w2 * 1e-3), atol=1e-5, rtol=0,
    )
    iqr = weighted_quantile(t, q, w2 * 1e-3)
    assert (iqr[-1] - iqr[0] > 1.0).all()   # not collapsed onto the minimum


def test_r2_weighted_quantile_integer_weights_stay_replication_exact():
    from nbn.learning.weighting import weighted_quantile

    t = torch.randn(30, 2)
    w = torch.randint(0, 4, (30,)).float()
    w[0] = 1.0
    q = torch.tensor([0.1, 0.5, 0.9])
    rep = t.repeat_interleave(w.long(), dim=0)
    torch.testing.assert_close(weighted_quantile(t, q, w), torch.quantile(rep, q, dim=0))


# ---------------------------------------------------------------------------
# Round 2 — R4: rollback keeps parameter identities; R8: NaN rows rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("first", ["lg", "mdn"])
def test_r4_update_rollback_keeps_parameter_objects(first):
    import nbn
    from nbn import NeuralBayesianNetwork as NBN
    from nbn.mechanisms.non_parametric.knn_conditional import KNNConditionalMechanism

    model = NBN([("A", "B")], {"A": ("continuous", 1), "B": ("continuous", 1)}, device="cpu")
    mech_a = (nbn.mechanisms.LinearGaussianMechanism() if first == "lg"
              else nbn.mechanisms.MDNMechanism(num_components=2))
    model.set_mechanism("A", mech_a)
    model.set_mechanism("B", KNNConditionalMechanism())
    g = torch.Generator().manual_seed(0)
    a = torch.randn(100, 1, generator=g)
    data = {"A": a, "B": a + 0.1 * torch.randn(100, 1, generator=g)}
    model.fit(data, epochs=3)
    ewc_before = getattr(mech_a, "_ewc_mu", None)
    params_before = list(model.parameters())
    values_before = [p.detach().clone() for p in params_before]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(NotImplementedError):
        model.update({"A": a + 5.0, "B": data["B"] + 5.0}, forgetting=0.5, epochs=3)
    if ewc_before is not None:
        assert mech_a._ewc_mu is ewc_before
    params_after = list(model.parameters())
    assert len(params_after) == len(params_before)
    for p0, p1, v0 in zip(params_before, params_after, values_before):
        assert p0 is p1
        torch.testing.assert_close(p1.detach(), v0)
    assert opt.param_groups[0]["params"][0] is params_after[0]


@pytest.mark.parametrize("device", available_devices())
def test_r8_update_and_fit_reject_non_finite_rows(device):
    model, data = _lg_model(device)
    model.fit(data)
    snap = _snapshot(model)
    bad = {k: v.clone() for k, v in data.items()}
    bad["B"][3, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinite.*'B'"):
        model.update(bad)
    for k, v in _snapshot(model).items():
        torch.testing.assert_close(v, snap[k])
    bad["B"][3, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or infinite"):
        model.fit(bad)
