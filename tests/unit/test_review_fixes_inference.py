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
