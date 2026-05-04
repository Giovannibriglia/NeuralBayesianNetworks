"""Correctness gate for the v0.3.1 vectorised ``query_batch``.

For every (network, B) pair we compare the batched output against the
reference single-row ``query()`` looped over the batch dim. Results must
match to ``allclose(atol=1e-6, rtol=1e-6)`` — the batched path is just
the same algebra on a leading B axis.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from nbn import NeuralBayesianNetwork, TensorVariableElimination
from nbn.mechanisms.categorical_table import CategoricalTableMechanism

pgmpy = pytest.importorskip("pgmpy")


def _build_nbn_from_pgmpy(bn) -> NeuralBayesianNetwork:
    """Construct an NBN with CategoricalTable mechanisms exactly mirroring pgmpy CPDs."""
    import networkx as nx

    edges = list(bn.edges())
    nodes = list(bn.nodes())
    cards = {n: int(bn.get_cpds(n).cardinality[0]) for n in nodes}
    variables = {n: ("discrete", cards[n]) for n in nodes}
    model = NeuralBayesianNetwork(edges, variables=variables, device="cpu")

    for node in nx.topological_sort(bn):
        cpd = bn.get_cpds(node)
        # cpd.variables = [target, *evidence_in_pgmpy_order]
        # cpd.values shape = [k_target, *parent_cards] (pgmpy 1.x convention)
        target_card = int(cpd.cardinality[0])
        pgmpy_parents = list(cpd.variables[1:])
        # Our DAG.parents() may iterate in a different order — align both.
        nbn_parents = list(model.dag.parents(node))
        # cpd.values has axes (target, *pgmpy_parents). We want
        # (*nbn_parents, target). Move target to end then permute parents
        # to nbn order.
        vals = np.asarray(cpd.values)
        # Move target axis (axis 0) to last
        if vals.ndim > 1:
            vals = np.moveaxis(vals, 0, -1)  # shape: (*pgmpy_parents, k)
            # Now permute pgmpy_parents axes -> nbn_parents order
            perm = [pgmpy_parents.index(p) for p in nbn_parents]
            vals = np.transpose(vals, axes=(*perm, vals.ndim - 1))
            vals_t = torch.tensor(vals.copy(), dtype=torch.float)
            parent_cards = [cards[p] for p in nbn_parents]
            log_cpt = torch.log(vals_t.clamp_min(1e-12)).reshape(-1, target_card)
        else:
            # Root node
            log_cpt = torch.log(torch.tensor(vals, dtype=torch.float).clamp_min(1e-12)).reshape(1, target_card)
            parent_cards = []

        mech = CategoricalTableMechanism()
        mech._logits = nn.Parameter(log_cpt)
        mech._n_classes = target_card
        mech._parent_cards = parent_cards
        strides: list[int] = []
        stride = 1
        for c in reversed(parent_cards):
            strides.append(stride)
            stride *= c
        mech._parent_strides = list(reversed(strides))
        mech._class_values = torch.arange(target_card, dtype=torch.float)
        mech.output_dim = 1
        model.set_mechanism(node, mech)

    return model


def _load_bnlearn(name: str) -> NeuralBayesianNetwork:
    from benchmarking.domains.bnlearn import _load_pgmpy_model
    bn = _load_pgmpy_model(name)
    return _build_nbn_from_pgmpy(bn)


def _sample_targets_and_evidence(
    model: NeuralBayesianNetwork,
    *,
    n_targets: int = 1,
    n_evidence: int = 3,
    B: int,
    seed: int,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Pick a random target + ``n_evidence`` distinct evidence variables.

    For every evidence node, draw ``[B]`` integer samples uniformly from the
    valid index range ``[0, cardinality)``.
    """
    g = torch.Generator().manual_seed(seed)
    nodes = list(model.dag.topological_order())
    perm = torch.randperm(len(nodes), generator=g).tolist()
    chosen = [nodes[i] for i in perm[: n_targets + n_evidence]]
    targets = chosen[:n_targets]
    evidence_nodes = chosen[n_targets:]
    evidence: dict[str, torch.Tensor] = {}
    for v in evidence_nodes:
        k = model.mechanisms[v].n_classes
        evidence[v] = torch.randint(0, k, (B,), generator=g, dtype=torch.long)
    return targets, evidence


@pytest.mark.parametrize("net", ["asia", "cancer", "child", "alarm"])
@pytest.mark.parametrize("B", [1, 16, 256, 1024])
def test_vectorized_matches_loop(net: str, B: int) -> None:
    model = _load_bnlearn(net)
    engine = TensorVariableElimination()

    # Use same RNG seed per net so the (target, evidence_keys) tuple is identical
    # across the B sweep — keeps the comparison apples-to-apples.
    targets, evidence = _sample_targets_and_evidence(
        model, n_targets=1, n_evidence=3, B=B, seed=hash(net) & 0xFFFF,
    )

    # Loop reference
    loop_rows = []
    for b in range(B):
        ev_b = {k: evidence[k][b].reshape(1) for k in evidence}
        loop_rows.append(engine.query(model, targets, ev_b))
    loop_result = torch.stack(loop_rows, dim=0)  # [B, K]

    # Vectorised
    batched_result = engine.query_batch(model, targets, evidence)

    assert batched_result.shape == loop_result.shape, (
        f"shape mismatch: batched {batched_result.shape} vs loop {loop_result.shape}"
    )
    assert torch.allclose(batched_result, loop_result, atol=1e-6, rtol=1e-6), (
        f"max |Δ| = {(batched_result - loop_result).abs().max().item():.3e}"
    )


def test_vectorized_no_evidence_broadcasts() -> None:
    """When evidence is empty, query_batch should return [1, K] = the prior."""
    model = _load_bnlearn("asia")
    engine = TensorVariableElimination()
    target = "asia"
    prior = engine.query(model, [target])
    batched = engine.query_batch(model, [target], {})
    assert batched.shape == (1, prior.shape[-1])
    assert torch.allclose(batched[0], prior, atol=1e-6, rtol=1e-6)
