"""Verify device propagation: every mechanism, engine, and ancestral sample
runs on the model's device with no silent CPU fallback."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from nbn import NeuralBayesianNetwork
from nbn.mechanisms import (
    CategoricalTableMechanism,
    DeterministicMechanism,
    DiracGaussianMechanism,
    LinearGaussianMechanism,
    MDNMechanism,
)
from nbn.utils.device import assert_on_device, resolve_device, to_device

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def test_resolve_device_auto():
    d = resolve_device("auto")
    assert d.type in {"cpu", "cuda", "mps"}


def test_resolve_device_explicit():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device(None).type == "cpu"
    assert resolve_device(torch.device("cpu")).type == "cpu"


def test_to_device_recursive():
    d = torch.device("cpu")
    obj = {"a": torch.zeros(3), "b": [torch.ones(2), torch.zeros(1)]}
    moved = to_device(obj, d)
    assert moved["a"].device.type == "cpu"
    assert moved["b"][0].device.type == "cpu"


def test_assert_on_device_catches_mismatch():
    with pytest.raises(AssertionError, match="Device mismatch"):
        assert_on_device(torch.zeros(3, device="cpu"), torch.device("meta"), name="x")


@pytest.mark.parametrize("device", DEVICES)
def test_categorical_table_on_device(device):
    mech = CategoricalTableMechanism()
    x = torch.randint(0, 3, (50,)).to(device)
    pa = torch.randint(0, 2, (50, 1)).to(device).float()
    mech.fit_local(x, pa, parent_cards=[2])
    mech.to(device)
    lp = mech.log_prob(x[:5], pa[:5])
    assert lp.device.type == torch.device(device).type
    samp = mech.sample(pa[:3], n=4)
    assert samp.device.type == torch.device(device).type


@pytest.mark.parametrize("device", DEVICES)
def test_linear_gaussian_on_device(device):
    mech = LinearGaussianMechanism()
    x = torch.randn(80, 1).to(device)
    pa = torch.randn(80, 2).to(device)
    mech.fit_local(x, pa)
    mech.to(device)
    lp = mech.log_prob(x[:5], pa[:5])
    assert lp.device.type == torch.device(device).type


@pytest.mark.parametrize("device", DEVICES)
def test_mdn_on_device(device):
    mech = MDNMechanism(num_components=2, hidden=(8,))
    x = torch.randn(64, 1).to(device)
    pa = torch.randn(64, 1).to(device)
    mech.fit_local(x, pa, epochs=2, batch_size=32)
    mech.to(device)
    lp = mech.log_prob(x[:4], pa[:4])
    assert lp.device.type == torch.device(device).type
    s = mech.sample(pa[:4], n=3)
    assert s.device.type == torch.device(device).type


@pytest.mark.parametrize("device", DEVICES)
def test_dirac_gaussian_on_device(device):
    mech = DiracGaussianMechanism(value=2.0)
    mech.to(device)
    pa = torch.randn(5, 2).to(device)
    s = mech.sample(pa, n=4)
    assert s.device.type == torch.device(device).type
    lp = mech.log_prob(s, pa)
    assert lp.device.type == torch.device(device).type


@pytest.mark.parametrize("device", DEVICES)
def test_network_construction_on_device(device):
    model = NeuralBayesianNetwork(
        [("A", "B")],
        variables={"A": ("discrete", 2), "B": ("discrete", 3)},
        device=device,
    )
    assert model.device.type == torch.device(device).type
    mech = CategoricalTableMechanism()
    x = torch.randint(0, 2, (40,))
    mech.fit_local(x, None)
    model.set_mechanism("A", mech)
    # set_mechanism must move to device
    assert next(model.mechanisms["A"].parameters()).device.type == torch.device(device).type


@pytest.mark.parametrize("device", DEVICES)
def test_query_keyword_device_rejected(device):
    model = NeuralBayesianNetwork(
        [("A", "B")],
        variables={"A": ("discrete", 2), "B": ("discrete", 2)},
        device=device,
    )
    for n in ("A", "B"):
        m = CategoricalTableMechanism()
        x = torch.randint(0, 2, (40,))
        if n == "A":
            m.fit_local(x, None)
        else:
            m.fit_local(x, torch.randint(0, 2, (40, 1)).float(), parent_cards=[2])
        model.set_mechanism(n, m)
    with pytest.raises(TypeError, match="construction time"):
        model.query(["B"], evidence={"A": 0}, device="cpu")
    with pytest.raises(TypeError, match="construction time"):
        model.sample(n=2, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for round-trip")
def test_round_trip_cpu_cuda_preserves_results():
    """cpu → cuda round-trip must preserve query results to ≤1e-5."""
    model = NeuralBayesianNetwork(
        [("A", "B")],
        variables={"A": ("discrete", 2), "B": ("discrete", 2)},
        device="cpu",
    )
    for n in ("A", "B"):
        m = CategoricalTableMechanism()
        x = torch.randint(0, 2, (200,))
        if n == "A":
            m.fit_local(x, None)
        else:
            m.fit_local(x, torch.randint(0, 2, (200, 1)).float(), parent_cards=[2])
        model.set_mechanism(n, m)

    p_cpu = model.query(["A"]).cpu()
    model.to("cuda")
    p_cuda = model.query(["A"]).cpu()
    assert torch.allclose(p_cpu, p_cuda, atol=1e-5)
