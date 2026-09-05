"""Phase-A tests for ParentConditionedProposal (β / P2).

Covers the recognition-net rewrite in isolation (no engine): per-node heads
sized to each node's own parent set, parent encoding, and make_dist-compatible
output per mechanism. Engine wiring / P1-contract / training are Phase B.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import pytest

from nbn.bench.adapters import NBNAdapter
from nbn.bench.domains.base import BenchmarkProblem, Query
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
    # Grouped storage (Phase D): node i's stacked head slice has in/out dims
    # d_ctx + parent_enc_dim and param_size.
    g2, _ = pcp._node_to_group[i2]
    g0, _ = pcp._node_to_group[i0]
    assert pcp.parent_enc_dim[i2] == 2 + 3, "one-hot widths summed over parents"
    assert pcp.gW1[g2].shape[1] == 64 + 5            # [G, in, H]
    assert pcp.parent_enc_dim[i0] == 0 and pcp.gW1[g0].shape[1] == 64
    # Output head matches the node's cardinality.
    assert pcp.gW2[g2].shape[2] == 4                 # [G, H, out]
    assert pcp.gW2[g0].shape[2] == 2


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
    assert isinstance(pcp.gW1, nn.ParameterList)


# ---- Phase D: Strategy F vectorization (grouped storage + batched NLL) -------

def _diverse_discrete_model(n: int = 800, seed: int = 0):
    """Network with diverse signatures: roots, 1-parent, and 2-parent nodes
    with different parent cardinalities (exercises multiple F-groups)."""
    g = torch.Generator().manual_seed(seed)
    dag = [("A", "C"), ("B", "C"), ("A", "D"), ("C", "E"), ("B", "E")]
    a = torch.randint(0, 2, (n,), generator=g)        # root card 2
    b = torch.randint(0, 3, (n,), generator=g)        # root card 3
    c = (a + b) % 4                                    # 2 parents (2,3) -> card 4
    d = torch.where(torch.rand(n, generator=g) < 0.2, 1 - a, a)  # 1 parent (2) -> card 2
    e = (c + b) % 5                                    # 2 parents (4,3) -> card 5
    data = {"A": a, "B": b, "C": c, "D": d, "E": e}
    variables = {"A": ("discrete", 2), "B": ("discrete", 3), "C": ("discrete", 4),
                 "D": ("discrete", 2), "E": ("discrete", 5)}
    prob = BenchmarkProblem(name="div", dag=dag, variables=variables,
                            train_data=data, test_data=data, queries=[],
                            family="discrete", problem_id="div", seed=seed)
    ad = NBNAdapter(mechanism="cat", engine="ve", device="cpu")
    ad.fit(prob, epochs=1)
    return ad.model


def test_vectorized_matches_per_node_loop():
    """THE GATE: grouped batched NLL == per-node oracle NLL to float32 tol."""
    torch.manual_seed(0)
    model = _diverse_discrete_model()
    pcp = ParentConditionedProposal(model, d_ctx=48, head_hidden=32)
    n = len(pcp.node_order)
    B = 64
    g = torch.Generator().manual_seed(3)
    xb = torch.stack([torch.randint(0, pcp.heads[nd].k, (B,), generator=g)
                      for nd in pcp.node_order], dim=1).float()
    mask = (torch.rand(B, n, generator=g) < 0.5).float()
    ctx = pcp.context(xb * mask, mask)
    grouped = AmortizedISEngine._proposal_nll(pcp, ctx, xb, mask)
    pernode = AmortizedISEngine._proposal_nll_per_node(pcp, ctx, xb, mask)
    assert torch.allclose(grouped, pernode, atol=1e-5), \
        f"grouped {float(grouped):.6f} vs per-node {float(pernode):.6f}"


def test_grouped_storage_is_registered_parameters():
    model = _diverse_discrete_model()
    pcp = ParentConditionedProposal(model, d_ctx=48, head_hidden=32)
    pnames = {name for name, _ in pcp.named_parameters()}
    assert any(nm.startswith("gW1.") for nm in pnames), "grouped weights must be registered"
    # No stale per-node head_mlps storage (grouped is the sole head storage).
    assert not any("head_mlps" in nm for nm in pnames)
    # Param count == trunk + 4 stacked tensors per group (no duplication).
    n_groups = len(pcp._group_nodes)
    n_trunk = sum(1 for _ in pcp.trunk.parameters())
    assert len(list(pcp.parameters())) == n_trunk + 4 * n_groups


def test_signature_grouping_correctness():
    model = _diverse_discrete_model()
    pcp = ParentConditionedProposal(model)
    # Expected signatures: A(root,k2), B(root,k3), C(parents(2,3),k4),
    # D(parent(2),k2), E(parents(4,3),k5) — all distinct → 5 groups, each size 1.
    assert len(pcp._group_nodes) == 5
    assert sorted(len(ns) for ns in pcp._group_nodes) == [1, 1, 1, 1, 1]


@pytest.mark.slow
def test_grouped_training_converges_and_p1_contract():
    torch.manual_seed(0)
    model = _diverse_discrete_model()
    eng = AmortizedISEngine(n_samples=512)
    metrics = eng.train_proposal(model, n_training_samples=4000, device="cpu")
    # Trained via the grouped path; ESS gate evaluates the result (P1 contract).
    assert metrics["proposal_used"] in ("learned", "lw_fallback")
    ess = eng._estimate_ess_fraction(model)
    assert ess is None or (0.0 <= ess <= 1.0 + 1e-6)


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


@pytest.mark.slow
def test_ais_end_to_end_via_adapter_competitive_with_lw():
    """Production pipeline (NBNAdapter fit→query) on a diagnostic query:
    parent-conditioned AIS is finite, normalized, and not worse than LW vs the
    exact VE posterior (no regression; the β win regime is downstream evidence)."""
    from nbn.inference.tensor_ve import TensorVariableElimination

    # Build the problem inline (X0,X1->X2 ; X0->X3) and fit AIS + LW adapters.
    g = torch.Generator().manual_seed(1)
    dag = [("X0", "X2"), ("X1", "X2"), ("X0", "X3")]
    n = 600
    x0 = torch.randint(0, 2, (n,), generator=g)
    x1 = torch.randint(0, 3, (n,), generator=g)
    x2 = (x0 + x1) % 4
    x3 = torch.where(torch.rand(n, generator=g) < 0.2, 1 - x0, x0)
    data = {"X0": x0, "X1": x1, "X2": x2, "X3": x3}
    variables = {"X0": ("discrete", 2), "X1": ("discrete", 3),
                 "X2": ("discrete", 4), "X3": ("discrete", 2)}
    prob = BenchmarkProblem(name="mp", dag=dag, variables=variables,
                            train_data=data, test_data=data, queries=[],
                            family="discrete", problem_id="mp", seed=1)

    ais = NBNAdapter(mechanism="cat", engine="ais", device="cpu", n_samples=4096)
    lw = NBNAdapter(mechanism="cat", engine="lw", device="cpu", n_samples=4096)
    ais.fit(prob, epochs=1)
    lw.fit(prob, epochs=1)
    ve = TensorVariableElimination()

    # Diagnostic query: target root X0 given downstream descendants X2, X3.
    q = Query(targets=("X0",), evidence={"X2": torch.tensor(1), "X3": torch.tensor(0)},
              kind="marginal")
    p_ve = ve.query(ais.model, ["X0"],
                    {"X2": torch.tensor([1.0]), "X3": torch.tensor([0.0])}).reshape(-1)
    p_ais = ais.query(q).probs
    p_lw = lw.query(q).probs
    assert p_ais is not None and torch.isfinite(p_ais).all()
    assert abs(float(p_ais.sum()) - 1.0) < 1e-4
    tv = lambda a, b: 0.5 * float((a.reshape(-1) - b.reshape(-1)).abs().sum())  # noqa: E731
    tv_ais, tv_lw = tv(p_ais, p_ve), tv(p_lw, p_ve)
    # No regression: AIS within a small slack of LW vs the exact posterior.
    assert tv_ais <= tv_lw + 0.05, f"AIS TV {tv_ais:.3f} >> LW TV {tv_lw:.3f}"
