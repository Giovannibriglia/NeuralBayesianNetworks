"""Amortized VI: the vectorised all-table ELBO equals the per-node loop."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_certified_and_lw_pruning import _random_bn  # noqa: E402

from nbn.inference.amortized_vi import AmortizedVIEngine  # noqa: E402
from nbn.inference.recognition_net import RecognitionNetwork  # noqa: E402
from nbn.sampling.ancestral import ancestral_sample  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vectorised_elbo_equals_loop(seed):
    m, names = _random_bn(seed, n=9)
    eng = AmortizedVIEngine()
    net = RecognitionNetwork(m)
    eng._parent_names = {v: list(m.dag.parents(v)) for v in net.node_order}
    torch.manual_seed(seed)
    samples = ancestral_sample(m, n=256, device="cpu")
    xb = eng._stack_values(samples, net.node_order, torch.device("cpu")).detach()
    mask = (torch.rand(xb.shape, generator=torch.Generator().manual_seed(seed)) < 0.4).float()
    t = eng._table_layout(m, net)
    assert t is not None
    # the loop's draws (one Categorical per node, in node order) ...
    torch.manual_seed(100 + seed)
    with torch.no_grad():
        params = net(xb * mask, mask)
        draws = torch.stack([
            torch.distributions.Categorical(
                logits=torch.log_softmax(net.node_param_slice(params, v), -1)).sample()
            for v in net.node_order], dim=1)
    out = {}
    net.zero_grad()
    eng.vectorize_tables = False
    torch.manual_seed(100 + seed)
    loss = eng._elbo_loss(m, net, xb, mask)
    loss.backward()
    out["loop"] = (float(loss), [p.grad.clone() for p in net.parameters()])
    # ... fed to the batched loss
    net.zero_grad()
    loss = eng._elbo_loss_tables(net, xb, mask, t, draws=draws)
    loss.backward()
    out["vec"] = (float(loss), [p.grad.clone() for p in net.parameters()])
    assert abs(out["vec"][0] - out["loop"][0]) < 1e-4 * max(1.0, abs(out["loop"][0]))
    for a, b in zip(out["vec"][1], out["loop"][1]):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_batched_draw_has_the_q_law():
    m, names = _random_bn(4, n=6)
    eng = AmortizedVIEngine()
    net = RecognitionNetwork(m)
    eng._parent_names = {v: list(m.dag.parents(v)) for v in net.node_order}
    t = eng._table_layout(m, net)
    xb = torch.zeros(20000, len(names))
    mask = torch.zeros_like(xb)
    with torch.no_grad():
        params = net(xb * mask, mask)
        p = params[:, t["pidx"]].masked_fill(~t["kvalid"], float("-inf"))
        q = torch.softmax(p, -1)[0]
        draws = torch.distributions.Categorical(logits=torch.log_softmax(p, -1)).sample()
    freq = torch.stack([torch.bincount(draws[:, j], minlength=3).float() / 20000
                        for j in range(len(names))])
    assert (freq - q).abs().max() < 0.02


def test_mixed_network_keeps_loop():
    from nbn.core.network import NeuralBayesianNetwork
    from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism
    m = NeuralBayesianNetwork([("A", "B")], variables={"A": ("continuous", 1),
                                                        "B": ("continuous", 1)})
    m.set_mechanism("A", LinearGaussianMechanism())
    m.set_mechanism("B", LinearGaussianMechanism())
    m.fit({"A": torch.randn(50), "B": torch.randn(50)})
    eng = AmortizedVIEngine()
    net = RecognitionNetwork(m)
    eng._parent_names = {v: list(m.dag.parents(v)) for v in net.node_order}
    assert eng._table_layout(m, net) is None
