"""fit(method="stacked") trains the same per-node estimator as method="local".

With full-batch steps the minibatch order is irrelevant, so given identical
initial weights the two paths must give the same fitted mechanisms up to
floating-point summation order.  The initialiser is patched to a
deterministic, shape-dependent fill so both paths start from the same point.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import nbn.learning.stacked as stacked_mod
import nbn.mechanisms.parametric.mdn as mdn_mod
from nbn.core.network import NeuralBayesianNetwork
from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism
from nbn.mechanisms.parametric.mdn import MDNMechanism

_orig_build = mdn_mod._build_mlp


def _det_build(in_dim, hidden, out_dim, activation="relu"):
    net = _orig_build(in_dim, hidden, out_dim, activation)
    with torch.no_grad():
        for li, m in enumerate(x for x in net if isinstance(x, nn.Linear)):
            n = m.weight.numel()
            m.weight.copy_(torch.sin(torch.arange(n, dtype=torch.float32) * 0.7 + li)
                           .reshape(m.weight.shape) / math.sqrt(m.in_features))
            m.bias.copy_(torch.cos(torch.arange(m.bias.numel(), dtype=torch.float32) + li) * 0.1)
    return net


def _data(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, generator=g)
    b = 0.8 * a + 0.3 * torch.randn(n, generator=g) ** 2
    c = torch.where(torch.rand(n, generator=g) < 0.5, a - b, a + 2 * b) \
        + 0.2 * torch.randn(n, generator=g)
    d = torch.tanh(c) + 0.1 * torch.randn(n, generator=g)
    e = torch.randn(n, generator=g)
    return {"A": a, "B": b, "C": c, "D": d, "E": e}


EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("A", "D"), ("B", "D")]


def _model(hidden=(8,), K=3, extra_lg=False):
    m = NeuralBayesianNetwork(EDGES + ([("E", "D")] if extra_lg else []),
                              variables=dict.fromkeys("ABCDE", ("continuous", 1)),
                              device="cpu")
    for v in "ABCDE":
        if extra_lg and v == "E":
            m.set_mechanism(v, LinearGaussianMechanism())
        else:
            m.set_mechanism(v, MDNMechanism(num_components=K, hidden=hidden))
    return m


@pytest.mark.parametrize("hidden", [(), (8,), (6, 5)])
@pytest.mark.parametrize("weighted", [False, True])
def test_stacked_equals_local_full_batch(monkeypatch, hidden, weighted):
    monkeypatch.setattr(mdn_mod, "_build_mlp", _det_build)
    monkeypatch.setattr(stacked_mod, "_build_mlp", _det_build)
    data = _data()
    n = data["A"].shape[0]
    w = (torch.arange(n) % 3 + 1).float() if weighted else None
    kw = dict(epochs=25, batch_size=n, lr=1e-2, consolidate=False, weights=w)
    m_loc, m_stk = _model(hidden), _model(hidden)
    m_loc.fit(data, method="local", **kw)
    m_stk.fit(data, method="stacked", **kw)
    test = {k: v.reshape(-1, 1) for k, v in _data(200, seed=1).items()}
    for v in "ABCDE":
        pa = list(m_loc.dag.parents(v))
        p = torch.stack([test[x][:, 0] for x in pa], -1) if pa else None
        with torch.no_grad():
            lp_l = m_loc.mechanisms[v].log_prob(test[v], p)
            lp_s = m_stk.mechanisms[v].log_prob(test[v], p)
        assert torch.allclose(lp_l, lp_s, atol=2e-4, rtol=1e-4), (v, (lp_l - lp_s).abs().max())
    with torch.no_grad():
        assert torch.allclose(m_loc.log_prob(test), m_stk.log_prob(test), atol=1e-3)


def test_stacked_falls_back_for_other_mechanisms_and_trains_minibatch():
    data = _data()
    m = _model((8,), extra_lg=True)
    hist = m.fit(data, method="stacked", epochs=3, batch_size=64, consolidate=False)
    assert set(hist.node_log_likelihoods) == set("ABCDE")
    assert all(math.isfinite(h[0]) for h in hist.node_log_likelihoods.values())
    # fitted mechanisms are ordinary MDNs: sampling and queries work
    s = m.sample(n=10)
    assert all(torch.isfinite(t).all() for t in s.values())


def test_stacked_node_chunking_is_equivalent(monkeypatch):
    monkeypatch.setattr(mdn_mod, "_build_mlp", _det_build)
    monkeypatch.setattr(stacked_mod, "_build_mlp", _det_build)
    data = _data()
    n = data["A"].shape[0]
    kw = dict(epochs=10, batch_size=n, lr=1e-2, consolidate=False)
    a, b = _model(), _model()
    a.fit(data, method="stacked", **kw)
    b.fit(data, method="stacked", node_chunk=1, **kw)
    test = {k: v.reshape(-1, 1) for k, v in _data(100, seed=2).items()}
    with torch.no_grad():
        assert torch.allclose(a.log_prob(test), b.log_prob(test), atol=1e-4)


def test_mixture_nll_closed_form_backward_matches_autograd():
    from nbn.learning.stacked import _MixtureNLL

    torch.manual_seed(0)
    B, J, K, D = 6, 4, 3, 2
    lg = torch.randn(B, J, K, dtype=torch.float64, requires_grad=True)
    lr = torch.randn(B, J, K, D, dtype=torch.float64, requires_grad=True)
    ls = (torch.randn(B, J, K, D, dtype=torch.float64) * 3).requires_grad_()
    x = torch.randn(B, J, D, dtype=torch.float64)
    xm = torch.randn(J, D, dtype=torch.float64)
    xs = torch.rand(J, D, dtype=torch.float64) + 0.5

    def ref():
        sc = (xs[:, None, :] * torch.exp(ls.clamp(max=1.0))).clamp_min(0.3)
        z = (x.unsqueeze(-2) - (xm[:, None, :] + xs[:, None, :] * lr)) / sc
        lc = (-0.5 * z.pow(2) - torch.log(sc) - 0.5 * math.log(2 * math.pi)).sum(-1)
        return -torch.logsumexp(torch.log_softmax(lg, -1) + lc, -1)

    w = torch.randn(B, J, dtype=torch.float64)
    a = ref()
    ga = torch.autograd.grad((a * w).sum(), [lg, lr, ls])
    b = _MixtureNLL.apply(lg, lr, ls, x, xm, xs, 1.0, 0.3)
    gb = torch.autograd.grad((b * w).sum(), [lg, lr, ls])
    assert torch.allclose(a, b, atol=1e-12)
    for p, q in zip(ga, gb):
        assert torch.allclose(p, q, atol=1e-12)
