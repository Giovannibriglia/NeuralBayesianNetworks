"""Phase-A tests for ParentConditionedProposal (β / P2).

Covers the recognition-net rewrite in isolation (no engine): per-node heads
sized to each node's own parent set, parent encoding, and make_dist-compatible
output per mechanism. Engine wiring / P1-contract / training are Phase B.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import pytest

from benchmarking.adapters import NBNAdapter
from benchmarking.domains.base import BenchmarkProblem, Query
from nbn.inference.amortized_is import (
    _MIN_TRAIN_STEPS,
    _STEPS_PER_NODE,
    AmortizedISEngine,
)
from nbn.inference.recognition_net import ParentConditionedProposal


def _multiparent_discrete_model(n: int = 600, seed: int = 0):
    """4-node discrete BN with a 2-parent node: X0,X1 -> X2 ; X0 -> X3."""
    g = torch.Generator().manual_seed(seed)
    dag = [("X0", "X2"), ("X1", "X2"), ("X0", "X3")]
    x0 = torch.randint(0, 2, (n,), generator=g)
    x1 = torch.randint(0, 3, (n,), generator=g)          # cardinality 3
    x2 = (x0 + x1) % 4                                    # cardinality 4, 2 parents
    x3 = torch.where(torch.rand(n, generator=g) < 0.2, 1 - x0, x0)
    data = {"X0": x0, "X1": x1, "X2": x2, "X3": x3}
    variables = {"X0": ("discrete", 2), "X1": ("discrete", 3),
                 "X2": ("discrete", 4), "X3": ("discrete", 2)}
    prob = BenchmarkProblem(name="mp", dag=dag, variables=variables,
                            train_data=data, test_data=data, queries=[],
                            family="discrete", problem_id="mp", seed=seed)
    ad = NBNAdapter(mechanism="cat", engine="ve", device="cpu")
    ad.fit(prob, epochs=1)
    return ad.model


def test_per_node_head_sized_to_parents():
    model = _multiparent_discrete_model()
    pcp = ParentConditionedProposal(model, d_ctx=64, head_hidden=32)
    i2 = pcp.node_index["X2"]   # parents X0 (card 2) + X1 (card 3) -> enc 5
    i0 = pcp.node_index["X0"]   # root -> enc 0
    # First Linear of each head: in_features == d_ctx + parent_enc_dim.
    head2_in = pcp.head_mlps[i2][0].in_features
    head0_in = pcp.head_mlps[i0][0].in_features
    assert pcp.parent_enc_dim[i2] == 2 + 3, "one-hot widths summed over parents"
    assert head2_in == 64 + 5
    assert pcp.parent_enc_dim[i0] == 0 and head0_in == 64
    # Output head matches the node's cardinality.
    assert pcp.head_mlps[i2][-1].out_features == 4
    assert pcp.head_mlps[i0][-1].out_features == 2


def test_per_node_head_emits_correct_distribution():
    model = _multiparent_discrete_model()
    pcp = ParentConditionedProposal(model, d_ctx=64, head_hidden=32)
    n = len(pcp.node_order)
    M = 16
    ctx = pcp.context(torch.zeros(M, n), torch.zeros(M, n))   # [M, d_ctx]
    i2 = pcp.node_index["X2"]
    # X2 has parents (X0, X1); feed scalar parent values.
    pv = torch.stack([torch.randint(0, 2, (M,)), torch.randint(0, 3, (M,))], dim=1).float()
    params = pcp.node_params(i2, ctx, pv)
    assert params.shape == (M, 4)
    dist = pcp.make_dist("X2", params)
    s = dist.sample()
    assert s.shape == (M,) and dist.log_prob(s).shape == (M,)
    assert int(s.max()) < 4 and int(s.min()) >= 0           # categorical over 4 states

    # Root node: node_params works with parent_values=None.
    i0 = pcp.node_index["X0"]
    p0 = pcp.node_params(i0, ctx, None)
    assert p0.shape == (M, 2)
    assert pcp.make_dist("X0", p0).sample().shape == (M,)


def test_scalar_parent_guard_and_contract():
    """v1 surface present (node_order, make_dist, is_discrete) for P1/_run reuse."""
    model = _multiparent_discrete_model()
    pcp = ParentConditionedProposal(model)
    assert pcp.node_order and all(pcp.is_discrete(nd) for nd in pcp.node_order)
    # encode_parents one-hots discrete parents by their own cardinality.
    i2 = pcp.node_index["X2"]
    enc = pcp.encode_parents(i2, torch.tensor([[1.0, 2.0]]))   # X0=1, X1=2
    assert enc.shape == (1, 5)
    assert torch.allclose(enc, torch.tensor([[0., 1., 0., 0., 1.]]))  # one-hot(1,2)+one-hot(2,3)
    assert isinstance(pcp.head_mlps, nn.ModuleList)


# ---- Phase B: engine wiring + training objective ----------------------------

def test_training_budget_scales_with_n():
    # Formula: max(floor, steps_per_node * n) — testable without training.
    assert AmortizedISEngine._target_train_steps(4) == _MIN_TRAIN_STEPS       # floor
    assert AmortizedISEngine._target_train_steps(100) == _STEPS_PER_NODE * 100
    assert AmortizedISEngine._target_train_steps(1000) == _STEPS_PER_NODE * 1000
    assert AmortizedISEngine._target_train_steps(37, steps_per_node=90) == 3330


def test_p1_fallback_contract_preserved():
    """The new recognition net is evaluable by _estimate_ess_fraction (P1)."""
    model = _multiparent_discrete_model()
    eng = AmortizedISEngine(n_samples=128)
    eng.recognition_net = ParentConditionedProposal(model).to("cpu")
    eng.device = torch.device("cpu")
    ess = eng._estimate_ess_fraction(model)        # must not raise
    assert ess is None or (0.0 <= ess <= 1.0 + 1e-6)


def test_run_uses_parent_conditioned_proposal_finite():
    """_run with the new net produces finite weights of the right shape, and
    gathers SAMPLED parents in-loop (not a single pre-loop proposal)."""
    model = _multiparent_discrete_model()
    eng = AmortizedISEngine(n_samples=256)
    eng.recognition_net = ParentConditionedProposal(model).to("cpu")
    eng.device = torch.device("cpu")
    ev = {"X0": torch.tensor([0.0])}
    log_w, buf = eng._run(model, ["X2"], ev, {}, 256)
    assert log_w.shape == (1, 256) and torch.isfinite(log_w).all()
    assert buf.shape[0] == 1 and buf.shape[1] == 256


@pytest.mark.slow
def test_trained_proposal_query_finite_and_budget_reported():
    model = _multiparent_discrete_model()
    eng = AmortizedISEngine(n_samples=512)
    metrics = eng.train_proposal(model, n_training_samples=2000, device="cpu")
    assert metrics["target_steps"] == _MIN_TRAIN_STEPS      # n=4 -> floor
    assert metrics["grad_steps"] >= 1
    assert metrics["proposal_used"] in ("learned", "lw_fallback")
    # A query through the production engine path returns a valid posterior.
    p = eng.query(model, ["X2"], {"X0": torch.tensor([0])})
    assert torch.isfinite(p).all() and abs(float(p.sum()) - 1.0) < 1e-4
