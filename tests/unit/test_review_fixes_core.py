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
