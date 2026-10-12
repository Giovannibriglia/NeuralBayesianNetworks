"""Tests for the v0.6b round-2 pre-allocation memory-budget guard in
``TensorVariableElimination.query_batch``.

The estimator is exercised end-to-end on the n=20 case (where naive
ordering predicts a multi-GiB peak and min-fill predicts ~256 MiB);
the runtime guard is exercised by mocking ``torch.cuda.mem_get_info``
to simulate a constrained device — the actual cuda code path is
device-only and is verified by Giovanni's local re-run, not in this
sandbox.
"""
from __future__ import annotations

from unittest.mock import PropertyMock, patch

import pytest
import torch

from nbn.bench.synthetic import make_synthetic_bn
import nbn.inference.tensor_ve as ve_mod
from nbn.inference.tensor_ve import (
    TensorVariableElimination,
    _estimate_peak_bytes,
    _memory_budget_message,
)


def _all_nodes(graph, target, evidence=None):
    """Disable Bug 2 subnetwork pruning by returning the full node set.

    Bug 2 / PR #131 (Stage 2) wired ``relevant_subnetwork`` into VE, which
    shrinks the elimination scope to the variables m-connected to the
    target.  The two tests below were calibrated to the *full* 20-node
    elimination and measure pre-pruning behaviour (the min-fill-vs-
    topological estimator and the OOM guard) that pruning by design now
    bypasses; patching the relevant set back to all nodes restores the
    elimination scope they were written to exercise.
    """
    return set(graph.nodes())


def test_estimator_predicts_smaller_peak_for_min_fill_than_topological() -> None:
    """On the v0.5b §A.5 case (n=20, K=4, max_in_degree=4, B=16),
    min-fill must predict a peak strictly smaller than topological —
    by the 64× factor measured in PR #22 round-1's follow-up, give or
    take whatever the algebraic walk produces under each ordering."""
    bn = make_synthetic_bn(
        family="discrete", n_nodes=20,
        cardinality=4, max_in_degree=4, edge_density=0.20,
        n_train=200, n_test=50, n_reference=500,
        seed=0, device="cpu",
    )
    eng = TensorVariableElimination()
    target = "X2"
    ev_keys = ("X0", "X10")
    factors = eng._extract_factors(bn.true_model)
    # Bug 2 / PR #131: pruning would reduce this query to a 1-variable
    # elimination (where min-fill and topological coincide).  Disable it so
    # the estimator is exercised on the full 20-node scope this test targets.
    with patch.object(ve_mod, "relevant_subnetwork", _all_nodes):
        plan_topo = eng._plan(bn.true_model, target, ev_keys, order="topological")
        plan_mf = eng._plan(bn.true_model, target, ev_keys, order="min_fill")

    peak_topo = _estimate_peak_bytes(
        plan_topo, factors, evidence_keys=ev_keys, B=16,
    )
    peak_mf = _estimate_peak_bytes(
        plan_mf, factors, evidence_keys=ev_keys, B=16,
    )

    assert peak_mf < peak_topo, (
        f"min-fill peak {peak_mf / 1024 ** 2:.1f} MiB should be smaller "
        f"than topological peak {peak_topo / 1024 ** 2:.1f} MiB"
    )
    # The 64× round-1 prediction; allow generous slack since the exact
    # ratio depends on which DAG nodes happen to be target/evidence.
    ratio = peak_topo / peak_mf
    assert ratio >= 8, (
        f"expected min-fill to reduce peak by ≥8× on this case; got {ratio:.1f}×"
    )


def _n20_plan():
    bn = make_synthetic_bn(
        family="discrete", n_nodes=20,
        cardinality=4, max_in_degree=4, edge_density=0.20,
        n_train=200, n_test=50, n_reference=500,
        seed=0, device="cpu",
    )
    eng = TensorVariableElimination()
    ev_keys = ("X0", "X10")
    factors = eng._extract_factors(bn.true_model)
    with patch.object(ve_mod, "relevant_subnetwork", _all_nodes):
        plan = eng._plan(bn.true_model, "X2", ev_keys, order="min_fill")
    return plan, factors, ev_keys


def test_estimate_prices_the_live_set_not_one_factor() -> None:
    """#178 (replacing #177's ``_LIVESET_MULTIPLIER`` test): the estimate is
    the live set at the costliest allocation, not one union-scope factor.

    The largest elimination step holds its product *and* ``logsumexp``'s
    full-size temporary, so the estimate is at least twice the largest single
    tensor; the step's other live tensors are reduced or smaller, so it stays
    below #177's three copies of it (the 3.0 multiplier over-charged a peak
    made of one product).
    """
    plan, factors, ev_keys = _n20_plan()
    B = 16
    cost = ve_mod._plan_cost(plan, factors, evidence_keys=ev_keys)
    largest = max(cost.tensor[0], B * cost.tensor[1]) * 4
    est = _estimate_peak_bytes(plan, factors, evidence_keys=ev_keys, B=B)
    assert largest > 0
    assert est >= 2 * largest
    assert est <= 3 * largest


def test_guard_rejects_query_whose_live_set_exceeds_headroom() -> None:
    """#177 safety property, now from the live-set model (#178): a query whose
    largest single factor fits the headroom but whose live set does not must
    be rejected — the pre-#177 guard compared the single factor and let it
    through, then it OOMed mid-elimination.

    Exercised through ``_memory_budget_message`` — the guard-decision helper
    — so it runs on CPU in CI without a real cuda device (the end-to-end
    cuda path is covered by
    ``test_query_batch_oom_guard_raises_on_constrained_cuda_device``).
    """
    plan, factors, ev_keys = _n20_plan()
    B = 16
    cost = ve_mod._plan_cost(plan, factors, evidence_keys=ev_keys)
    bare = max(cost.tensor[0], B * cost.tensor[1]) * 4
    est = _estimate_peak_bytes(plan, factors, evidence_keys=ev_keys, B=B)

    # Mock free memory into the band: the single factor fits 0.9 x free, the
    # live set does not.
    safety = 0.9
    free_bytes = int(bare / safety * 1.5)  # -> 0.9·free = 1.5·bare
    assert bare <= safety * free_bytes, "design check: one factor fits"
    assert est > safety * free_bytes, "design check: the live set does not"

    # #264: the guard also counts PyTorch's reusable cached blocks
    # (reserved - allocated) as headroom; pin them to 0 so CUDA work done
    # earlier in this worker cannot lift the budget above the mock (#298).
    with patch("torch.cuda.mem_get_info",
               return_value=(free_bytes, free_bytes)), \
         patch("torch.cuda.memory_reserved", return_value=0), \
         patch("torch.cuda.memory_allocated", return_value=0):
        msg = _memory_budget_message(
            plan, factors, evidence_keys=ev_keys, B=B,
            device=torch.device("cuda:0"), order="min_fill", safety=safety,
        )
    assert msg is not None and "pre-allocation guard" in msg


def test_estimator_prices_the_final_product_of_an_empty_plan() -> None:
    """Nothing to eliminate still multiplies the remaining factors into one
    table (#178): with three unconditioned factors that table is their full
    joint, so the peak is at least its size.  (Before #178 this test pinned
    an estimate of 0, which the guard then waved through.)"""
    bn = make_synthetic_bn(
        family="discrete", n_nodes=3, cardinality=4, max_in_degree=2,
        edge_density=0.50,
        n_train=100, n_test=20, n_reference=100,
        seed=0, device="cpu",
    )
    eng = TensorVariableElimination()
    factors = eng._extract_factors(bn.true_model)
    peak = _estimate_peak_bytes(
        plan=[], factors=factors, evidence_keys=(), B=1,
    )
    assert peak >= 4 * 4 ** 3  # the joint over three card-4 variables


# Needs a real CUDA device: the guard's ``device.type == 'cuda'`` branch sits
# behind the evidence-to-device move, which initialises the driver, so a
# fake-cuda mock fails before the guard runs.  The estimator's algebra is
# covered on CPU above; this runs in the CUDA CI lane (``-m cuda``).
@pytest.mark.cuda
def test_query_batch_oom_guard_raises_on_constrained_cuda_device() -> None:
    """When even a SINGLE evidence row's estimated peak exceeds 90% of free
    cuda memory, ``query_batch`` must raise ``torch.cuda.OutOfMemoryError``
    cleanly *before* the elimination loop starts — protecting the cuda
    allocator from a partial-OOM that would fragment the heap.

    PR C (batch chunking): the guard's contract changed — a batch whose
    full-B peak exceeds the budget but whose per-row peak fits is now
    CHUNKED instead of rejected (``_max_chunk_rows``), so this test mocks
    free memory low enough that B=1 does not fit either, which is the only
    regime where the guard still raises."""
    bn = make_synthetic_bn(
        family="discrete", n_nodes=20,
        cardinality=4, max_in_degree=4, edge_density=0.20,
        n_train=200, n_test=50, n_reference=500,
        seed=0, device="cpu",
    )
    eng = TensorVariableElimination()
    target = "X2"
    B = 16
    evidence = {
        "X0":  torch.zeros(B, dtype=torch.long),
        "X10": torch.zeros(B, dtype=torch.long),
    }

    # ``model.device`` is a property — patch it at the class level via
    # ``PropertyMock`` so the guard's ``device.type == 'cuda'`` branch
    # is exercised.  We don't actually move tensors to cuda; the guard
    # raises before any tensor op that would touch the device.
    fake_device = torch.device("cuda:0")
    model_cls = type(bn.true_model)
    # Bug 2 / PR #131: disable pruning so the guard sees the full 20-node
    # elimination peak this test calibrates against; with pruning the relevant
    # scope shrinks and the guard correctly would not fire (pruning solves the
    # OOM rather than guarding it).
    #
    # PR 11 (#226): query_batch defaults to order="min_fill"; the realised
    # min-fill peak for this scale is tens of MiB (#178's live-set estimate,
    # after #131 pruning shrank the old ~256 MiB topological-shaped one).
    #
    # PR C (batch chunking): 32 MiB free (the previous mock) now triggers
    # CHUNKING, not rejection — the ~3 MiB per-row peak fits the 28.8 MiB
    # threshold, so the batch is split rather than refused.  Mock 1 MiB free
    # (0.9 MiB threshold < per-row peak ≥ 48/16 = 3 MiB) so even B=1 exceeds
    # the budget and the guard raises the pre-existing error.
    with patch.object(ve_mod, "relevant_subnetwork", _all_nodes), \
         patch.object(model_cls, "device",
                      new_callable=PropertyMock,
                      return_value=fake_device), \
         patch("torch.cuda.mem_get_info",
               return_value=(1 * 1024 ** 2, 8 * 1024 ** 3)), \
         patch("torch.cuda.memory_reserved", return_value=0), \
         patch("torch.cuda.memory_allocated", return_value=0):
        # #264: the guard also counts PyTorch's reusable cached blocks
        # (reserved - allocated) as headroom; pin them to 0 so earlier CUDA
        # work in this worker cannot lift the budget above the mock.
        # 1 MiB free; even one row's peak exceeds the 0.9 MiB budget →
        # guard rejects the query pre-allocation (chunking cannot help).
        with pytest.raises(
            (torch.cuda.OutOfMemoryError, RuntimeError),
            match="pre-allocation guard",
        ):
            eng.query_batch(bn.true_model, [target], evidence)
