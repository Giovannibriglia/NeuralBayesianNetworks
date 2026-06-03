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

from benchmarking.synthetic import make_synthetic_bn
import nbn.inference.tensor_ve as ve_mod
from nbn.inference.tensor_ve import (
    TensorVariableElimination,
    _estimate_peak_bytes,
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
    """When the estimator predicts a peak above 90% of free cuda memory,
    ``query_batch`` must raise ``torch.cuda.OutOfMemoryError`` cleanly
    *before* the elimination loop starts — protecting the cuda
    allocator from a partial-OOM that would fragment the heap."""
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
    # elimination peak (~256 MiB) this test calibrates against; with pruning
    # the relevant scope shrinks below the 64 MiB threshold and the guard
    # correctly would not fire (pruning solves the OOM rather than guarding it).
    with patch.object(ve_mod, "relevant_subnetwork", _all_nodes), \
         patch.object(model_cls, "device",
                      new_callable=PropertyMock,
                      return_value=fake_device), \
         patch("torch.cuda.mem_get_info",
               return_value=(64 * 1024 ** 2, 8 * 1024 ** 3)):
        # 64 MiB free; min-fill peak at this scale is ~256 MiB → guard
        # should reject the query pre-allocation.
        with pytest.raises(
            (torch.cuda.OutOfMemoryError, RuntimeError),
            match="pre-allocation guard",
        ):
            eng.query_batch(bn.true_model, [target], evidence)
