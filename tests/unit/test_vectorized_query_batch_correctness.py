"""Correctness gate for the v0.3.1 vectorised ``query_batch``.

For every (network, B) pair we compare the batched output against the
reference single-row ``query()`` looped over the batch dim.  Results
must match to ``allclose(atol=1e-6, rtol=1e-6)`` — the batched path is
just the same algebra on a leading B axis.

Test fixtures are synthetic discrete BNs of varying size produced by
:func:`benchmarking.synthetic.make_synthetic_bn`.
"""
from __future__ import annotations

import pytest
import torch

from benchmarking.synthetic import make_synthetic_bn
from nbn import TensorVariableElimination


def _sample_targets_and_evidence(
    model,
    *,
    n_targets: int = 1,
    n_evidence: int = 3,
    B: int,
    seed: int,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Pick a random target + ``n_evidence`` distinct evidence variables.

    For every evidence node, draw ``[B]`` integer samples uniformly from
    the valid index range ``[0, cardinality)``.
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


@pytest.mark.parametrize("n_nodes", [8, 12, 20])
@pytest.mark.parametrize("B", [1, 16, 256, 1024])
def test_vectorized_matches_loop(n_nodes: int, B: int) -> None:
    bn = make_synthetic_bn(
        family="discrete", n_nodes=n_nodes, edge_density=0.20,
        max_in_degree=3, cardinality=3,
        n_train=200, n_test=50, n_reference=200, seed=n_nodes, device="cpu",
    )
    model = bn.true_model
    engine = TensorVariableElimination()

    targets, evidence = _sample_targets_and_evidence(
        model, n_targets=1, n_evidence=3, B=B, seed=n_nodes * 11,
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
    """When evidence is empty, ``query_batch`` returns ``[1, K]`` = the prior."""
    bn = make_synthetic_bn(
        family="discrete", n_nodes=8, edge_density=0.20,
        max_in_degree=3, cardinality=3,
        n_train=200, n_test=50, n_reference=200, seed=0, device="cpu",
    )
    model = bn.true_model
    engine = TensorVariableElimination()
    target = list(model.dag.topological_order())[0]
    prior = engine.query(model, [target])
    batched = engine.query_batch(model, [target], {})
    assert batched.shape == (1, prior.shape[-1])
    assert torch.allclose(batched[0], prior, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------- #
# v0.6b round-2: plan-independence under min-fill vs topological orders
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("order", ["topological", "min_fill"])
def test_query_batch_correctness_at_n20_under_both_orders(order: str) -> None:
    """``query_batch``'s output must be numerically identical (within
    float tolerance) regardless of elimination order.  Tests the v0.6b
    min-fill ordering at ``n=20`` against the legacy topological
    baseline; the result of both must match the looped single-row
    reference.

    ``B=4`` keeps the topological run feasible on cpu memory (the
    naive plan's peak is ~1 GiB at ``B=16`` on this DAG; ``B=4``
    drops it to 256 MiB, comfortably absorbed by the sandbox)."""
    bn = make_synthetic_bn(
        family="discrete", n_nodes=20, edge_density=0.20,
        max_in_degree=4, cardinality=4,
        n_train=200, n_test=50, n_reference=500, seed=0, device="cpu",
    )
    model = bn.true_model
    engine = TensorVariableElimination()
    target = "X2"
    B = 4
    evidence = {
        "X0":  torch.zeros(B, dtype=torch.long),
        "X10": torch.zeros(B, dtype=torch.long),
    }

    # Reference: loop the single-row query (no order kwarg passed —
    # uses the default min_fill, which is plan-independent for output).
    loop_rows = [
        engine.query(model, [target], {k: v[b].reshape(1) for k, v in evidence.items()})
        for b in range(B)
    ]
    loop_result = torch.stack(loop_rows, dim=0)

    batched = engine.query_batch(model, [target], evidence, order=order)
    assert batched.shape == loop_result.shape, (
        f"shape mismatch: batched {batched.shape} vs loop {loop_result.shape}"
    )
    assert torch.allclose(batched, loop_result, atol=1e-6, rtol=1e-5), (
        f"order={order!r}: max |Δ| = "
        f"{(batched - loop_result).abs().max().item():.3e}"
    )
