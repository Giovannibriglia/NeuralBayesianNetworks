"""Tests for the PR-C batch-chunking path in ``TensorVariableElimination``.

Paper-scale evidence (batch_speed): nbn-cat-ve and nbn-neuralcat-ve each
lost 10 cells to the pre-allocation guard's OOM rejection at large batch
sizes while pomegranate handled B=1024.  Since ``_estimate_peak_bytes`` is
linear in ``B`` in the regime where the guard fires (the peak elimination
step carries the batch axis), the guard now splits the batch into the
largest row-chunks that fit the existing 0.9 × free budget instead of
rejecting it, and raises the pre-existing ``OutOfMemoryError`` only when
even a single row does not fit.

Two layers:
- ``_max_chunk_rows`` — the pure chunk-size decision (no cuda needed);
- ``_query_batch_chunked`` — the chunked execution path, exercised
  directly on CPU and pinned row-identical to the single-pass result.
"""
from __future__ import annotations

import torch

from nbn.bench.synthetic import make_synthetic_bn
from nbn.inference.tensor_ve import (
    TensorVariableElimination,
    _max_chunk_rows,
)


# ---- _max_chunk_rows: pure chunk-size decision -------------------------------

def test_max_chunk_rows_full_batch_fits() -> None:
    """Estimate within budget → the whole batch runs in a single pass."""
    assert _max_chunk_rows(peak_at_B=100, peak_at_1=10, B=10, budget=200) == 10
    # Exactly on the budget still fits (guard uses a strict > comparison).
    assert _max_chunk_rows(peak_at_B=100, peak_at_1=10, B=10, budget=100) == 10


def test_max_chunk_rows_tight_budget_yields_smaller_chunk() -> None:
    """Batch-dominated peak over budget → largest chunk with per-row cost
    ``peak_at_B / B`` that fits; the chunked peak must itself fit."""
    # per_row = 10; budget 45 → floor(45/10) = 4 rows per chunk.
    chunk = _max_chunk_rows(peak_at_B=100, peak_at_1=10, B=10, budget=45)
    assert chunk == 4
    assert chunk * (100 / 10) <= 45          # the chosen chunk fits the budget
    # A budget that barely holds one row degrades gracefully to chunk=1.
    assert _max_chunk_rows(peak_at_B=100, peak_at_1=10, B=10, budget=10) == 1


def test_max_chunk_rows_impossible_budget_signals_zero() -> None:
    """Even B=1 exceeds the budget → 0, i.e. the caller raises the
    pre-existing guard error (chunking cannot help)."""
    assert _max_chunk_rows(peak_at_B=100, peak_at_1=15, B=10, budget=9) == 0
    # Degenerate single-row batch over budget: nothing to split.
    assert _max_chunk_rows(peak_at_B=7, peak_at_1=7, B=1, budget=5) == 0


def test_max_chunk_rows_batch_free_peak_cannot_be_chunked() -> None:
    """When the peak step carries no batch axis (peak_at_1 == peak_at_B >
    budget), splitting rows does not shrink the peak → 0."""
    assert _max_chunk_rows(peak_at_B=100, peak_at_1=100, B=16, budget=50) == 0


def test_max_chunk_rows_never_returns_full_batch_when_over_budget() -> None:
    """Over-budget input must always yield a chunk strictly below B (or 0),
    so the caller's chunk-vs-single-pass branch cannot loop."""
    for budget in (10, 25, 50, 99):
        chunk = _max_chunk_rows(peak_at_B=100, peak_at_1=10, B=10, budget=budget)
        assert 0 <= chunk < 10


# ---- _query_batch_chunked: chunked execution == single pass ------------------

def _small_discrete_bn():
    return make_synthetic_bn(
        family="discrete", n_nodes=8, cardinality=3, max_in_degree=3,
        edge_density=0.40,
        n_train=100, n_test=20, n_reference=100,
        seed=0, device="cpu",
    )


def test_chunked_query_batch_matches_single_pass() -> None:
    """The chunking code path (row-slice → query_batch per chunk → concat)
    must be row-identical to the unchunked single pass, including an uneven
    final chunk (B=7, chunk=3 → 3+3+1)."""
    bn = _small_discrete_bn()
    model = bn.true_model
    eng = TensorVariableElimination()
    topo = model.dag.topological_order()
    target, ev_a, ev_b = topo[-1], topo[0], topo[2]

    B = 7
    g = torch.Generator().manual_seed(0)
    evidence = {
        ev_a: torch.randint(0, 3, (B,), generator=g),
        ev_b: torch.randint(0, 3, (B,), generator=g),
    }

    full = eng.query_batch(model, [target], evidence)
    for chunk in (1, 3, 7):
        chunked = eng._query_batch_chunked(
            model, [target], evidence, B=B, chunk=chunk, order="min_fill",
        )
        assert chunked.shape == full.shape == (B, 3)
        assert torch.allclose(chunked, full, atol=1e-6), (
            f"chunk={chunk}: chunked query_batch diverged from single pass "
            f"(max abs diff {(chunked - full).abs().max().item():.3e})"
        )


def test_chunked_query_batch_handles_broadcast_evidence() -> None:
    """Evidence given as a broadcast [1] row alongside [B] rows must chunk
    identically to the single pass (normalisation expands it to [B] before
    slicing)."""
    bn = _small_discrete_bn()
    model = bn.true_model
    eng = TensorVariableElimination()
    topo = model.dag.topological_order()
    target, ev_a, ev_b = topo[-1], topo[0], topo[2]

    B = 5
    g = torch.Generator().manual_seed(1)
    raw = {
        ev_a: torch.randint(0, 3, (B,), generator=g),
        ev_b: torch.zeros(1, dtype=torch.long),  # broadcast row
    }
    full = eng.query_batch(model, [target], raw)

    # Mirror query_batch's normalisation: broadcast [1] evidence to [B]
    # (the chunk path always receives the already-expanded ev_norm).
    ev_norm = {ev_a: raw[ev_a], ev_b: raw[ev_b].expand(B)}
    chunked = eng._query_batch_chunked(
        model, [target], ev_norm, B=B, chunk=2, order="min_fill",
    )
    assert torch.allclose(chunked, full, atol=1e-6)
