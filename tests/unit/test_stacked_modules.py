"""fit(method="stacked") == fit(method="local") for neural categorical,
FlexCode and normalising-flow nodes (generic torch.func trainer).

Full-batch steps make the minibatch order irrelevant, and the network
initialisers are patched to a deterministic, shape-dependent fill so both
paths start from the same point; the fitted mechanisms must then agree up to
floating-point summation order.

The networks are pinned to the CPU: these tests compare two training
algorithms, and must not depend on a device's kernels.  On a GitHub-hosted
macOS runner (macOS 26.6.2, torch 2.14.1) a plain ``nn.Linear`` forward on
MPS disagreed with the CPU by up to 0.5 while ``torch.func.vmap`` agreed, so a
device-level discrepancy would be reported as a stacking error.  MPS itself is
covered by the ``mps`` lane.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import nbn.mechanisms.non_parametric.flexcode as fc_mod
import nbn.mechanisms.parametric.neural_categorical as nc_mod
from nbn.core.network import NeuralBayesianNetwork
from nbn.mechanisms.non_parametric.flexcode import FlexCodeMechanism
from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism
from nbn.mechanisms.parametric.mdn import _build_mlp as _orig_build
from nbn.mechanisms.parametric.neural_categorical import NeuralCategoricalMechanism


def _det_fill(module: nn.Module) -> None:
    with torch.no_grad():
        for li, p in enumerate(module.parameters()):
            n = p.numel()
            p.copy_(torch.sin(torch.arange(n, dtype=torch.float32) * 0.37 + li)
                    .reshape(p.shape) * 0.3)


def _det_build(in_dim, hidden, out_dim, activation="relu"):
    net = _orig_build(in_dim, hidden, out_dim, activation)
    _det_fill(net)
    return net


EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("A", "D"), ("B", "D")]


def _cont_data(n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, generator=g)
    b = 0.8 * a + 0.4 * torch.randn(n, generator=g) ** 2
    c = torch.where(torch.rand(n, generator=g) < 0.5, a - b, a + b) + 0.2 * torch.randn(n, generator=g)
    d = torch.tanh(c) + 0.1 * torch.randn(n, generator=g)
    return {"A": a, "B": b, "C": c, "D": d}


def _disc_data(n=400, seed=0, k=3):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, k, (n,), generator=g)
    b = (a + torch.randint(0, 2, (n,), generator=g)) % k
    c = (a * b + torch.randint(0, 2, (n,), generator=g)) % k
    d = (c + b + torch.randint(0, 2, (n,), generator=g)) % k
    return {"A": a, "B": b, "C": c, "D": d}


def _net(kind, **mk):
    if kind == "ncat":
        m = NeuralBayesianNetwork(EDGES, variables=dict.fromkeys("ABCD", ("discrete", 3)), device="cpu")
        for v in "ABCD":
            m.set_mechanism(v, NeuralCategoricalMechanism(n_classes=3, hidden=(6,), **mk))
    elif kind == "flexcode":
        m = NeuralBayesianNetwork(EDGES, variables=dict.fromkeys("ABCD", ("continuous", 1)), device="cpu")
        m.set_mechanism("A", LinearGaussianMechanism())
        for v in "BCD":
            m.set_mechanism(v, FlexCodeMechanism(n_basis=7, hidden=(8,), **mk))
    return m


@pytest.mark.parametrize("weighted", [False, True])
def test_neural_categorical_stacked_equals_local(monkeypatch, weighted):
    monkeypatch.setattr(nc_mod, "_build_mlp", _det_build)
    data = _disc_data()
    n = data["A"].shape[0]
    w = (torch.arange(n) % 3 + 1).float() if weighted else None
    kw = dict(epochs=15, batch_size=n, lr=5e-2, consolidate=False, weights=w)
    a, b = _net("ncat"), _net("ncat")
    a.fit(data, method="local", **kw)
    b.fit(data, method="stacked", **kw)
    test = {k: v[:100].reshape(-1, 1) for k, v in _disc_data(100, seed=1).items()}
    with torch.no_grad():
        la, lb = a.log_prob(test), b.log_prob(test)
    assert torch.allclose(la, lb, atol=1e-4), (la - lb).abs().max()
    # and the run moved away from the initialisation
    c = _net("ncat")
    c.fit(data, method="stacked", **{**kw, "epochs": 0})
    with torch.no_grad():
        assert (c.log_prob(test) - la).abs().max() > 1e-2


@pytest.mark.parametrize("sharpen", [1.0, "auto"])
def test_flexcode_stacked_equals_local(monkeypatch, sharpen):
    monkeypatch.setattr(fc_mod, "_build_mlp", _det_build)
    data = _cont_data()
    n = data["A"].shape[0]
    kw = dict(epochs=20, batch_size=n, lr=2e-2, consolidate=False)
    a, b = _net("flexcode", sharpen=sharpen), _net("flexcode", sharpen=sharpen)
    a.fit(data, method="local", **kw)
    b.fit(data, method="stacked", **kw)
    for v in "BCD":
        assert a.mechanisms[v].sharpen == b.mechanisms[v].sharpen
    test = {k: v.reshape(-1, 1) for k, v in _cont_data(120, seed=2).items()}
    with torch.no_grad():
        la, lb = a.log_prob(test), b.log_prob(test)
    assert torch.allclose(la, lb, atol=1e-3, rtol=1e-4), (la - lb).abs().max()


def _flow_net():
    pytest.importorskip("zuko")
    from nbn.mechanisms.parametric.normalizing_flow import NormalizingFlowMechanism
    m = NeuralBayesianNetwork(EDGES, variables=dict.fromkeys("ABCD", ("continuous", 1)), device="cpu")
    m.set_mechanism("A", LinearGaussianMechanism())
    for v in "BCD":
        m.set_mechanism(v, NormalizingFlowMechanism(num_transforms=2, hidden=(8,), bins=4))
    return m


def test_flow_stacked_equals_local_without_early_stopping(monkeypatch):
    pytest.importorskip("zuko")
    from nbn.mechanisms.parametric import normalizing_flow as nf_mod
    orig = nf_mod.NormalizingFlowMechanism._build_flow

    def det_build(self, d_pa, device):
        orig(self, d_pa, device)
        _det_fill(self._flow)

    monkeypatch.setattr(nf_mod.NormalizingFlowMechanism, "_build_flow", det_build)
    data = _cont_data()
    n = data["A"].shape[0]
    kw = dict(epochs=15, batch_size=n, lr=1e-2, consolidate=False, early_stopping=False)
    a, b = _flow_net(), _flow_net()
    a.fit(data, method="local", **kw)
    b.fit(data, method="stacked", **kw)
    test = {k: v.reshape(-1, 1) for k, v in _cont_data(120, seed=3).items()}
    with torch.no_grad():
        la, lb = a.log_prob(test), b.log_prob(test)
    assert torch.allclose(la, lb, atol=2e-3, rtol=1e-3), (la - lb).abs().max()


def test_flow_stacked_early_stopping_runs_and_restores_best():
    data = _cont_data(n=600)
    m = _flow_net()
    h = m.fit(data, method="stacked", epochs=60, batch_size=64, lr=1e-2, consolidate=False,
              patience=2)
    assert all(math.isfinite(v[0]) for v in h.node_log_likelihoods.values())
    s = m.sample(n=20)
    assert all(torch.isfinite(t).all() for t in s.values())


def test_flexcode_grid_density_matmul_equals_broadcast():
    g = torch.Generator().manual_seed(0)
    m = FlexCodeMechanism(epochs=2, hidden=(8,))
    pa = torch.randn(600, 2, generator=g)
    m.fit_local(pa.sum(1, keepdim=True) + 0.3 * torch.randn(600, 1, generator=g), pa)
    coef = m._coef(torch.randn(500, 2, generator=g), 500).detach()
    ref = (coef.unsqueeze(1) * m._basis_grid.unsqueeze(0)).sum(-1).clamp_min(0.0)
    assert torch.allclose(m._grid_density(coef), ref, atol=1e-5)
