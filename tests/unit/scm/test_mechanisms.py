"""Structural mechanisms: recovery, NBN shape contracts, weights, devices."""
from __future__ import annotations

import copy

import pytest
import torch

from nbn import NeuralBayesianNetwork
from nbn.scm import AdditiveNoiseMechanism, DiscreteAdditiveNoiseMechanism, StructuralCausalModel
from tests.conftest import available_devices
from tests.unit.scm.simulators import TRUE_Q, rehab_iid, rehab_model, rehab_true_table


@pytest.fixture(scope="module")
def fitted_rehab():
    data = rehab_iid(20_000, generator=torch.Generator().manual_seed(0)).data
    model = rehab_model()
    model.fit(data)
    return model, data


# ---------------------------------------------------------------------------
# Discrete additive noise
# ---------------------------------------------------------------------------


def test_discrete_recovers_noise_law_and_locations(fitted_rehab):
    model, _ = fitted_rehab
    mech = model.mechanisms["h'"]
    assert torch.allclose(mech.noise.probs.detach(), TRUE_Q, atol=0.015)
    assert torch.allclose(mech.table().double(), rehab_true_table(), atol=0.015)


def test_discrete_shapes_follow_the_mechanism_contract(fitted_rehab):
    model, data = fitted_rehab
    mech = model.mechanisms["h'"]
    pa = model.parent_values(data, "h'")[:5]
    x = data["h'"][:5]
    assert mech.log_prob(x, pa).shape == (5,)
    assert mech.log_prob(x.unsqueeze(-1), pa).shape == (5,)
    pa3 = pa.unsqueeze(1).expand(-1, 7, -1)
    x3 = torch.zeros(5, 7, 1)
    assert mech.log_prob(x3, pa3).shape == (5, 7)
    assert mech.sample(pa, n=3).shape == (5, 3, 1)
    assert mech.forward(pa).batch_shape == (5,)
    tab = mech.tabulate()
    assert tab.shape == (4, 2, 2, 4)
    assert torch.allclose(tab.exp().sum(-1), torch.ones(4, 2, 2), atol=1e-5)


def test_variable_elimination_and_do_use_the_structural_table(fitted_rehab):
    model, _ = fitted_rehab
    table = model.mechanisms["h'"].table()
    row = (2 * 2 + 0) * 2 + 1  # h=2, f=0, a=1
    answer = model.query(["h'"], evidence={"h": 2, "f": 0}, do={"a": 1})
    assert torch.allclose(answer.reshape(-1).double(), table[row].double(), atol=1e-5)


def test_discrete_integer_weights_equal_replicated_rows():
    data = rehab_iid(600, generator=torch.Generator().manual_seed(3)).data
    weights = torch.randint(0, 4, (600,)).float()
    replicated = {k: v.repeat_interleave(weights.long()) for k, v in data.items()}
    weighted_model, replicated_model = rehab_model(), rehab_model()
    weighted_model.fit(data, weights=weights)
    replicated_model.fit(replicated)
    assert torch.allclose(weighted_model.mechanisms["h'"].table(),
                          replicated_model.mechanisms["h'"].table(), atol=1e-6)


def test_discrete_abduction_is_consistent_with_the_observation(fitted_rehab):
    model, data = fitted_rehab
    mech = model.mechanisms["h'"]
    pa = model.parent_values(data, "h'")[:200]
    x = data["h'"][:200]
    offsets = mech.abduct(x, pa, n_samples=3)                     # [3, B, 1]
    for draw in offsets:
        assert torch.equal(mech.structural(pa, draw).float(), x)


def test_discrete_state_dict_round_trip(fitted_rehab):
    model, _ = fitted_rehab
    mech = model.mechanisms["h'"]
    snapshot = copy.deepcopy(mech.state_dict())
    before = mech.table().clone()
    with torch.no_grad():
        mech.noise.logits.add_(torch.randn_like(mech.noise.logits))
    mech.load_state_dict(snapshot)
    assert torch.equal(mech.table(), before)


# ---------------------------------------------------------------------------
# Continuous additive noise
# ---------------------------------------------------------------------------


def _continuous_data(n=4000, seed=0, nonlinear=False, bimodal=False):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, generator=gen)
    if bimodal:
        noise = torch.where(torch.rand(n, 1, generator=gen) < 0.5, -1.0, 1.0) + 0.2 * torch.randn(n, 1, generator=gen)
    else:
        noise = 0.3 * torch.randn(n, 1, generator=gen)
    y = (torch.sin(2 * x) if nonlinear else 1.5 * x + 0.5) + noise
    return {"x": x, "y": y}


def _continuous_model(mech, device="cpu"):
    model = StructuralCausalModel([("x", "y")], {"x": ("continuous", 1), "y": ("continuous", 1)},
                                  device=device)
    model.set_mechanism("y", mech)
    model.auto_mechanisms(default_continuous="linear")
    return model


def test_linear_location_recovers_the_equation():
    data = _continuous_data()
    model = _continuous_model(AdditiveNoiseMechanism("linear"))
    model.fit(data)
    mech = model.mechanisms["y"]
    assert float(mech.net.weight) == pytest.approx(1.5, abs=0.03)
    assert float(mech.net.bias) == pytest.approx(0.5, abs=0.03)
    assert float(mech.noise.scale) == pytest.approx(0.3, abs=0.02)


def test_mlp_location_fits_a_nonlinear_mechanism():
    torch.manual_seed(0)
    data = _continuous_data(nonlinear=True)
    linear = _continuous_model(AdditiveNoiseMechanism("linear"))
    mlp = _continuous_model(AdditiveNoiseMechanism("mlp", epochs=150, batch_size=512))
    linear.fit(data)
    mlp.fit(data)
    assert float(mlp.mechanisms["y"].noise.scale) == pytest.approx(0.3, abs=0.05)
    assert float(mlp.log_prob(data).mean()) > float(linear.log_prob(data).mean()) + 0.3


def test_mixture_noise_beats_gaussian_on_bimodal_residuals():
    data = _continuous_data(bimodal=True)
    gauss = _continuous_model(AdditiveNoiseMechanism("linear", "gaussian"))
    mix = _continuous_model(AdditiveNoiseMechanism("linear", "mixture", n_components=2))
    gauss.fit(data)
    mix.fit(data)
    assert float(mix.log_prob(data).mean()) > float(gauss.log_prob(data).mean()) + 0.2


def test_continuous_abduction_inverts_the_equation():
    data = _continuous_data()
    model = _continuous_model(AdditiveNoiseMechanism("linear"))
    model.fit(data)
    mech = model.mechanisms["y"]
    u = mech.residual(data["y"], data["x"])
    assert torch.allclose(mech.structural(data["x"], u), data["y"], atol=1e-5)
    assert torch.equal(mech.abduct(data["y"], data["x"], n_samples=2)[1], u)


@pytest.mark.parametrize("noise", ["gaussian", "mixture"])
def test_continuous_batched_shapes(noise):
    data = _continuous_data()
    model = _continuous_model(AdditiveNoiseMechanism("linear", noise))
    model.fit(data)
    mech = model.mechanisms["y"]
    pa = torch.randn(5, 7, 1)
    assert mech.forward(pa).batch_shape == (5, 7)
    assert mech.log_prob(torch.randn(5, 7, 1), pa).shape == (5, 7)
    assert mech.sample(torch.randn(5, 1), n=3).shape == (5, 3, 1)


def test_linear_weights_equal_replicated_rows():
    data = _continuous_data(n=500)
    weights = torch.randint(0, 4, (500,)).float()
    replicated = {k: v.repeat_interleave(weights.long(), dim=0) for k, v in data.items()}
    a = _continuous_model(AdditiveNoiseMechanism("linear"))
    b = _continuous_model(AdditiveNoiseMechanism("linear"))
    a.fit(data, weights=weights)
    b.fit(replicated)
    ma, mb = a.mechanisms["y"], b.mechanisms["y"]
    assert torch.allclose(ma.net.weight, mb.net.weight, atol=1e-5)
    assert torch.allclose(ma.noise.scale, mb.noise.scale, atol=1e-5)


def test_mlp_warm_start_with_zero_epochs_keeps_the_parameters():
    data = _continuous_data(n=300, nonlinear=True)
    mech = AdditiveNoiseMechanism("mlp", epochs=20)
    mech.fit_local(data["y"], data["x"])
    before = copy.deepcopy(mech.net.state_dict())
    out = mech.fit_local(data["y"], data["x"], warm_start=True, epochs=0)
    assert out["warm_started"] is True
    for key, value in mech.net.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.parametrize("device", available_devices())
def test_structural_mechanisms_fit_and_query_on_every_device(device):
    data = rehab_iid(3000, generator=torch.Generator().manual_seed(0)).data
    model = rehab_model(device=device)
    model.fit(data)
    assert model.mechanisms["h'"].table().device.type == torch.device(device).type
    cont = _continuous_model(AdditiveNoiseMechanism("mlp", "mixture", epochs=5, n_components=2), device=device)
    cont.fit(_continuous_data(n=500))
    lp = cont.log_prob({k: v.to(device) for k, v in _continuous_data(n=50).items()})
    assert bool(torch.isfinite(lp).all())


def test_plain_nbn_accepts_structural_mechanisms():
    """The mechanisms are ordinary NBN mechanisms: no SCM wrapper needed."""
    data = rehab_iid(2000, generator=torch.Generator().manual_seed(0)).data
    model = NeuralBayesianNetwork(
        [("h", "h'"), ("f", "h'"), ("a", "h'")],
        {"h": ("discrete", 4), "f": ("discrete", 2), "a": ("discrete", 2), "h'": ("discrete", 4)},
        device="cpu",
    )
    model.set_mechanism("h'", DiscreteAdditiveNoiseMechanism(max_offset=1))
    model.auto_mechanisms()
    model.fit(data)
    assert bool(torch.isfinite(model.log_prob(data)).all())
