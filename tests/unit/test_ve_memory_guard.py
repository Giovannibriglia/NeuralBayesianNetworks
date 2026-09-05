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


def test_guard_includes_broadcast_liveset_multiplier() -> None:
    """Issue #177: ``_estimate_peak_bytes`` scales the bare per-step
    max-factor walk by ``_LIVESET_MULTIPLIER`` (3.0) to model the
    broadcast live-set — ``_log_factor_product_batched`` holds
    ``a_aligned`` + ``b_aligned`` + their sum simultaneously, plus
    conditioned factors that persist across steps.

    Without this multiplier the guard under-estimates actual peak by
    ~2.85× (measured 2026-06-08), letting queries that will OOM in
    execution pass the pre-allocation check.
    """
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
    with patch.object(ve_mod, "relevant_subnetwork", _all_nodes):
        plan = eng._plan(bn.true_model, target, ev_keys, order="min_fill")

    scaled = _estimate_peak_bytes(plan, factors, evidence_keys=ev_keys, B=16)
    # The multiplier is read as a module global at call time, so patching
    # it to 1.0 recovers the bare per-step max-factor estimate.
    with patch.object(ve_mod, "_LIVESET_MULTIPLIER", 1.0):
        bare = _estimate_peak_bytes(plan, factors, evidence_keys=ev_keys, B=16)

    assert bare > 0, "non-trivial plan should predict a positive peak"
    assert ve_mod._LIVESET_MULTIPLIER >= 2.85, (
        "multiplier must cover the measured 2.85× under-estimate"
    )
    assert scaled == pytest.approx(ve_mod._LIVESET_MULTIPLIER * bare, rel=1e-6), (
        f"scaled peak {scaled} should be {ve_mod._LIVESET_MULTIPLIER}× the "
        f"bare per-step max {bare}"
    )


def test_guard_rejects_query_that_would_oom_without_multiplier() -> None:
    """Issue #177 safety property: a query whose *actual* peak would
    exceed free GPU memory must be rejected by the pre-allocation guard.

    The 2026-06-08 investigation measured a borderline case (B=256): the
    bare per-step estimate sat *below* the 90% threshold (so the pre-#177
    guard would have let it through), while the realised peak was ~3×
    larger and would crash mid-execution.  The 3.0× liveset multiplier
    lifts the estimate above the threshold so the query is rejected.

    This test reconstructs that band: free memory is mocked so that the
    *bare* estimate passes the guard but the *scaled* estimate (×3.0)
    exceeds it.  With the multiplier (current code) the guard rejects;
    with the multiplier patched to 1.0 (pre-#177 behaviour) it does not —
    proving the test exercises the multiplier's safety contribution, not
    just its arithmetic.

    Exercised through ``_memory_budget_message`` — the guard-decision
    helper ``query_batch`` delegates to — so it runs on CPU in CI without
    a real cuda device (the end-to-end cuda path is covered by
    ``test_query_batch_oom_guard_raises_on_constrained_cuda_device``).
    """
    bn = make_synthetic_bn(
        family="discrete", n_nodes=20,
        cardinality=4, max_in_degree=4, edge_density=0.20,
        n_train=200, n_test=50, n_reference=500,
        seed=0, device="cpu",
    )
    eng = TensorVariableElimination()
    target = "X2"
    ev_keys = ("X0", "X10")
    B = 16
    factors = eng._extract_factors(bn.true_model)
    with patch.object(ve_mod, "relevant_subnetwork", _all_nodes):
        plan = eng._plan(bn.true_model, target, ev_keys, order="min_fill")

    # Bare estimate = what the guard would have compared against pre-#177.
    with patch.object(ve_mod, "_LIVESET_MULTIPLIER", 1.0):
        bare = _estimate_peak_bytes(plan, factors, evidence_keys=ev_keys, B=B)
    assert bare > 0, "design check: non-trivial plan must predict a positive peak"

    # Mock free memory into the band: bare passes (bare <= 0.9·free) but
    # the 3× scaled estimate exceeds the budget (3·bare > 0.9·free).
    safety = 0.9
    free_bytes = int(bare / safety * 1.5)  # -> 0.9·free = 1.5·bare
    assert bare <= safety * free_bytes, "design check: bare must pass the guard"
    assert 3 * bare > safety * free_bytes, "design check: scaled must fail the guard"

    fake_device = torch.device("cuda:0")

    def _call() -> str | None:
        return _memory_budget_message(
            plan, factors, evidence_keys=ev_keys, B=B,
            device=fake_device, order="min_fill", safety=safety,
        )

    with patch("torch.cuda.mem_get_info",
               return_value=(free_bytes, free_bytes)):
        # Current code (multiplier=3.0): the query is rejected.
        msg = _call()
        assert msg is not None and "pre-allocation guard" in msg, (
            "with the liveset multiplier, a query whose actual peak exceeds "
            "free memory must be rejected"
        )

        # Pre-#177 behaviour (multiplier=1.0): the same query passes —
        # this is precisely the crash the multiplier prevents.
        with patch.object(ve_mod, "_LIVESET_MULTIPLIER", 1.0):
            assert _call() is None, (
                "without the multiplier the guard would let this query "
                "through, then it would OOM in execution — the regression "
                "this fix guards against"
            )


def test_estimator_returns_zero_when_plan_is_empty() -> None:
    """Trivial query (no variables to eliminate): peak is 0."""
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
    assert peak == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "guard's `device.type == 'cuda'` branch requires a real cuda "
        "device — the upstream evidence-to-device move tries to "
        "initialise the cuda driver, which fails before the guard runs "
        "on a fake-cuda mock.  Cuda end-to-end verification is deferred "
        "to Giovanni's laptop per the v0.6b round-2 brief; the "
        "estimator's algebra is exercised by the two cpu tests above."
    ),
)
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
    # min-fill peak for this scale is ~48 MiB (= ~16 MiB bare × 3.0
    # _LIVESET_MULTIPLIER from #179, after #131 pruning shrank the old ~256 MiB
    # topological-shaped estimate).
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
               return_value=(1 * 1024 ** 2, 8 * 1024 ** 3)):
        # 1 MiB free; even one row's peak exceeds the 0.9 MiB budget →
        # guard rejects the query pre-allocation (chunking cannot help).
        with pytest.raises(
            (torch.cuda.OutOfMemoryError, RuntimeError),
            match="pre-allocation guard",
        ):
            eng.query_batch(bn.true_model, [target], evidence)
