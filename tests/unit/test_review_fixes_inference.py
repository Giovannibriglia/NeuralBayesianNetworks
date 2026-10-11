"""Regression tests for the inference-side review findings (A5-A8, B4-B6, C, #279).

Ground truth is brute-force enumeration of the joint of small random discrete
networks: every configuration is scored with ``model.log_prob(per_node=True)``,
the do-nodes' own CPT terms are dropped (graph surgery) and the configurations
inconsistent with the evidence / do-values are masked out.  Nothing here
trusts one engine to check another.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from nbn import NeuralBayesianNetwork as NBN
from nbn.bench.synthetic import make_synthetic_bn
from nbn.inference.amortized_is import AmortizedISEngine
from nbn.inference.recognition_net import ParentConditionedProposal
from nbn.mechanisms.parametric.neural_categorical import NeuralCategoricalMechanism

# --------------------------------------------------------------------------- #
# Brute-force oracle
# --------------------------------------------------------------------------- #


def _random_discrete(seed: int = 1, n_nodes: int = 6, card: int = 3):
    bn = make_synthetic_bn(
        family="discrete", n_nodes=n_nodes, edge_density=0.5, max_in_degree=3,
        cardinality=card, n_train=50, n_test=10, n_reference=10, seed=seed,
        device="cpu",
    )
    return bn.true_model


def _joint_table(model):
    """``(configs [C, n] long, per_node {node: [C]} log-terms, nodes)``."""
    nodes = list(model.dag.topological_order())
    cards = [model.mechanisms[n].n_classes for n in nodes]
    configs = torch.tensor(list(itertools.product(*[range(c) for c in cards])))
    data = {n: configs[:, i].float().to(model.device) for i, n in enumerate(nodes)}
    with torch.no_grad():
        per_node = model.log_prob(data, per_node=True)
    return configs, {k: v.cpu() for k, v in per_node.items()}, nodes


def brute_posterior(model, target, evidence=None, do=None) -> torch.Tensor:
    """Exact ``P(target | evidence, do(...))`` by joint enumeration ``[K]``."""
    evidence, do = dict(evidence or {}), dict(do or {})
    configs, per_node, nodes = _joint_table(model)
    idx = {n: i for i, n in enumerate(nodes)}
    # Graph surgery: the intervened nodes' CPTs leave the product.
    log_joint = sum(v for n, v in per_node.items() if n not in do)
    keep = torch.ones(configs.shape[0], dtype=torch.bool)
    for n, v in {**evidence, **do}.items():
        keep &= configs[:, idx[n]] == int(v)
    k = model.mechanisms[target].n_classes
    out = torch.zeros(k)
    for c in range(k):
        m = keep & (configs[:, idx[target]] == c)
        out[c] = torch.logsumexp(log_joint[m], 0) if m.any() else -float("inf")
    return torch.softmax(out, 0)


# --------------------------------------------------------------------------- #
# #279 — AIS proposal training on hybrid networks
# --------------------------------------------------------------------------- #


def _hybrid_mdn_parent_model():
    """Continuous MDN root ``C`` and discrete ``A`` both parents of discrete ``D``."""
    torch.manual_seed(0)
    m = NBN(
        [("C", "D"), ("A", "D")],
        variables={"C": ("continuous", 1), "A": ("discrete", 4), "D": ("discrete", 4)},
        device="cpu",
    )
    m.set_mechanism("D", NeuralCategoricalMechanism(n_classes=4))
    m.auto_mechanisms(default_continuous="mdn")
    n = 1000
    c = torch.randn(n)
    a = torch.randint(0, 4, (n,)).float()
    d = ((c > 0).long() + a.long()).clamp(max=3).float()
    m.fit({"C": c, "A": a, "D": d}, epochs=2)
    return m


def test_279_grouped_nll_encodes_continuous_parents_raw():
    """The batched training NLL must match the per-node (inference) encoding.

    It used to one-hot an MDN parent by its component count: a width
    mismatch in ``bmm`` (the "[1, 142] vs [1, 138]" of #279) or a crash on
    negative parent values.
    """
    model = _hybrid_mdn_parent_model()
    pcp = ParentConditionedProposal(model, d_ctx=16, head_hidden=8)
    B = 32
    g = torch.Generator().manual_seed(0)
    cols = []
    for nd in pcp.node_order:
        if pcp.is_discrete(nd):
            cols.append(torch.randint(0, pcp.heads[nd].k, (B,), generator=g).float())
        else:
            cols.append(torch.randn(B, generator=g) * 3)  # negatives included
    xb = torch.stack(cols, dim=1)
    mask = (torch.rand(B, len(pcp.node_order), generator=g) < 0.5).float()
    ctx = pcp.context(xb * mask, mask)
    grouped = AmortizedISEngine._proposal_nll(pcp, ctx, xb, mask)
    pernode = AmortizedISEngine._proposal_nll_per_node(pcp, ctx, xb, mask)
    torch.testing.assert_close(grouped, pernode, atol=1e-5, rtol=1e-5)


def test_279_ais_trains_and_queries_on_hybrid():
    model = _hybrid_mdn_parent_model()
    eng = AmortizedISEngine(n_samples=64)
    eng.train_proposal(model, n_training_samples=256, steps_per_node=1,
                       time_budget_s=10)
    out = eng.query(model, ["D"], {"C": torch.tensor([0.5])})
    assert out.shape == (4,) and torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# A5 — target that is also evidence
# --------------------------------------------------------------------------- #


def test_oracle_agrees_with_ve_on_ordinary_queries():
    """Sanity: the enumeration oracle and VE agree where VE was never broken."""
    from nbn import TensorVariableElimination

    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    eng = TensorVariableElimination()
    for t, ev in [(nodes[0], {nodes[-1]: 2}), (nodes[3], {nodes[1]: 0, nodes[5]: 1})]:
        out = eng.query(model, [t], {k: torch.tensor(v) for k, v in ev.items()})
        torch.testing.assert_close(out, brute_posterior(model, t, ev), atol=1e-5, rtol=1e-5)


def test_a5_ve_query_target_in_evidence_is_delta():
    from nbn import TensorVariableElimination

    model = _random_discrete()
    t = list(model.dag.topological_order())[-1]
    out = TensorVariableElimination().query(model, [t], {t: torch.tensor(1)})
    torch.testing.assert_close(out, brute_posterior(model, t, {t: 1}))
    torch.testing.assert_close(out, torch.tensor([0.0, 1.0, 0.0]))


def test_a5_ve_query_batch_target_in_evidence_is_delta_per_row():
    from nbn import TensorVariableElimination

    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    t, other = nodes[-1], nodes[0]
    ev = {t: torch.tensor([0, 2, 1]), other: torch.tensor([1, 1, 0])}
    out = TensorVariableElimination().query_batch(model, [t], ev)
    expected = torch.eye(3)[[0, 2, 1]]
    torch.testing.assert_close(out, expected)


def test_a5_map_query_returns_observed_state():
    model = _random_discrete()
    model._engine = None
    model._engine_spec = "tensor_ve"
    t = list(model.dag.topological_order())[-1]
    assert int(model.map_query([t], {t: torch.tensor(2)})[t]) == 2


# --------------------------------------------------------------------------- #
# C — out-of-range discrete evidence / do values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [-1, 3])
@pytest.mark.parametrize("kind", ["evidence", "do"])
def test_c_ve_rejects_out_of_range_states(bad, kind):
    from nbn import TensorVariableElimination

    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    eng = TensorVariableElimination()
    spec = {nodes[2]: torch.tensor(bad)}
    kw = {"do": spec} if kind == "do" else {}
    ev = {} if kind == "do" else spec
    with pytest.raises(ValueError, match="out of range"):
        eng.query(model, [nodes[0]], ev, **kw)
    spec_b = {nodes[2]: torch.tensor([0, bad])}
    kw = {"do": spec_b} if kind == "do" else {}
    ev = {} if kind == "do" else spec_b
    with pytest.raises(ValueError, match="out of range"):
        eng.query_batch(model, [nodes[0]], ev, **kw)


# --------------------------------------------------------------------------- #
# A8 — chunked query_batch must keep do= as an intervention
# --------------------------------------------------------------------------- #


def test_a8_chunked_query_batch_keeps_do_semantics():
    from nbn import TensorVariableElimination

    model = _random_discrete(seed=3)
    nodes = list(model.dag.topological_order())
    # Target = a parent of the do-node: intervening on X leaves its parent's
    # distribution alone while observing X does not, so do != conditioning.
    x = next(n for n in nodes if model.dag.parents(n))
    y = model.dag.parents(x)[0]
    e = next(n for n in reversed(nodes) if n not in (x, y))
    B = 7
    g = torch.Generator().manual_seed(0)
    do_v = torch.randint(0, 3, (B,), generator=g)
    ev_v = torch.randint(0, 3, (B,), generator=g)
    eng = TensorVariableElimination()
    ev_norm = {e: ev_v, x: do_v}  # what query_batch hands the chunker
    out = eng._query_batch_chunked(
        model, [y], ev_norm, B=B, chunk=3, order="min_fill", do_keys=(x,),
    )
    ref = torch.stack([
        brute_posterior(model, y, {e: int(ev_v[i])}, {x: int(do_v[i])})
        for i in range(B)
    ])
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    # Non-vacuous: the observational answer (the old bug) differs here.
    obs = torch.stack([
        brute_posterior(model, y, {e: int(ev_v[i]), x: int(do_v[i])})
        for i in range(B)
    ])
    assert (obs - ref).abs().max() > 1e-3


# --------------------------------------------------------------------------- #
# C — LW/AIS output contract: [1, K] batches, ordinary (non-inference) tensors
# --------------------------------------------------------------------------- #


def test_c_lw_query_batch_b1_keeps_batch_axis():
    from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine

    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    lw = LikelihoodWeightingEngine(n_samples=256)
    out = lw.query_batch(model, [nodes[-1]], {nodes[0]: torch.tensor([1])})
    assert out.shape == (1, 3)
    out_do = lw.query_batch(model, [nodes[-1]], {}, do={nodes[0]: torch.tensor([1])})
    assert out_do.shape == (1, 3)
    # Single-query contract unchanged.
    assert lw.query(model, [nodes[-1]], {nodes[0]: torch.tensor([1])}).shape == (3,)


@pytest.mark.parametrize("engine_cls", ["lw", "ais"])
def test_c_lw_outputs_usable_in_autograd_and_in_place(engine_cls):
    from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine

    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    eng = (LikelihoodWeightingEngine(256) if engine_cls == "lw"
           else AmortizedISEngine(n_samples=256))
    out = eng.query(model, [nodes[-1]], {nodes[0]: torch.tensor(1)})
    assert not out.is_inference() and not out.requires_grad
    w = torch.ones(3, requires_grad=True)
    (out * w).sum().backward()          # posterior as a constant in a loss
    assert w.grad is not None
    out.mul_(2.0)                       # in-place post-processing

