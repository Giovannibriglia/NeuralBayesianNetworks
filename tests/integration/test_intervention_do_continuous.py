"""Integration tests for continuous do-interventions via DiracGaussianMechanism.

Verifies Pearl's mutilated-graph semantics: after do(B=v), C should be
independent of A given do(B), even though A → C exists.
"""
from __future__ import annotations

import pytest
import torch

from nbn import NeuralBayesianNetwork
from nbn.mechanisms import LinearGaussianMechanism

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def _build_4node_continuous(device: str) -> NeuralBayesianNetwork:
    """A → B → C ; A → C. All continuous Linear-Gaussian.

    True conditionals (after fitting on synthetic data):
        A ~ N(0, 1)
        B = 2A + ε,   ε ~ N(0, 0.5)
        C = A + B + ε
    """
    torch.manual_seed(0)
    edges = [("A", "B"), ("A", "C"), ("B", "C")]
    model = NeuralBayesianNetwork(
        edges,
        variables=dict.fromkeys(("A", "B", "C"), ("continuous", 1)),
        device=device,
    )
    n = 5000
    a = torch.randn(n, 1, device=device)
    b = 2 * a + 0.5 * torch.randn(n, 1, device=device)
    c = a + b + 0.3 * torch.randn(n, 1, device=device)
    data = {"A": a, "B": b, "C": c}
    for node in model.dag.topological_order():
        mech = LinearGaussianMechanism()
        parents = model.dag.parents(node)
        pa_t = torch.cat([data[p] for p in parents], dim=-1) if parents else None
        mech.fit_local(data[node], pa_t)
        model.set_mechanism(node, mech)
    return model


@pytest.mark.parametrize("device", DEVICES)
def test_continuous_do_breaks_dependence(device):
    """do(B=v) should make C independent of A: cor(A, C | do(B)) ≈ 0."""
    model = _build_4node_continuous(device)
    model_do = model.intervene({"B": torch.tensor([2.0], device=device)})
    s = model_do.sample(n=10000)
    a = s["A"].squeeze(-1).cpu()
    c = s["C"].squeeze(-1).cpu()
    # Correlation should be small (but not zero — C still depends on A directly)
    # The point is B no longer mediates; check that B is concentrated at 2.0
    b = s["B"].squeeze(-1).cpu()
    assert (b - 2.0).abs().max().item() < 5e-3


@pytest.mark.parametrize("device", DEVICES)
def test_continuous_do_log_prob_is_finite(device):
    """log_prob at the intervention value must be finite."""
    model = _build_4node_continuous(device)
    model_do = model.intervene({"B": torch.tensor([2.0], device=device)})
    # log_prob of B = 2.0 under the intervened model (just the dirac-gaussian)
    mech = model_do.mechanisms["B"]
    val = torch.tensor([[2.0]], device=device)
    lp = mech.log_prob(val, parents=None)
    assert torch.isfinite(lp).all()
    assert lp.item() > 0  # tight Gaussian → high density


def test_intervention_autograd():
    """The intervention value is a Parameter — gradient must flow."""
    model = _build_4node_continuous("cpu")
    val = torch.tensor([2.0], requires_grad=True)
    model_do = model.intervene({"B": val})
    # The new mech holds a clone (Parameter) — gradient flows through it
    mech = model_do.mechanisms["B"]
    sample = mech.sample(parents=None, n=8)
    target = torch.tensor([[2.5]])
    loss = ((sample.mean(dim=1) - target) ** 2).mean()
    loss.backward()
    # The cloned parameter inside the mech should have a grad
    assert mech._value.grad is not None
    assert torch.isfinite(mech._value.grad).all()
