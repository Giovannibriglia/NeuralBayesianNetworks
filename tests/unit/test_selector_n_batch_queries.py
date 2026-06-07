"""Tests for selector n_batch_queries + select_groups (PR 4 stage a, #148).

Covers all three config-selectable selectors (UniformRandomSelector,
TopologicalAllocator, HeaviestQueryByRole):

  1. Identity at defaults — n_batch_queries=1, batch_size=1 emits the
     exact Query objects select() returns, each in a length-1 inner list
  2. Chunking — N=10240, B=512 → 20 inner lists of length 512 per position
  3. Last-chunk handling — N=100, B=64 → lengths [64, 36] per position
  4. Sampling with replacement when n_train < N
  5. Determinism — same seed → identical query sequence

Reference: docs/v0.14-batched-queries-design.md §1.3, §2.
"""
from __future__ import annotations

import pytest
import torch

from benchmarking.domains.base import BenchmarkProblem
from benchmarking.selectors import UniformRandomSelector
from benchmarking.selectors.heaviest import HeaviestQueryByRole
from benchmarking.selectors.topological import TopologicalAllocator


def _make_problem(n_train: int = 200, n_test: int = 50, seed: int = 0) -> BenchmarkProblem:
    """6-node binary chain/fan BN with train + test data."""
    g = torch.Generator().manual_seed(seed)
    nodes = [f"X{i}" for i in range(6)]
    dag = [("X0", "X1"), ("X1", "X2"), ("X2", "X3"), ("X1", "X4"), ("X4", "X5")]

    def _data(n: int) -> dict[str, torch.Tensor]:
        return {v: torch.randint(0, 2, (n,), generator=g) for v in nodes}

    variables = dict.fromkeys(nodes, ("discrete", 2))
    return BenchmarkProblem(
        name="batch_sel_test",
        dag=dag,
        variables=variables,
        train_data=_data(n_train),
        test_data=_data(n_test),
        queries=[],
        family="discrete",
        problem_id="batch_sel_test",
        seed=seed,
    )


def _selectors(n_batch_queries: int):
    """One instance of each selector type with the given multiplicity."""
    return [
        UniformRandomSelector(n_batch_queries=n_batch_queries),
        TopologicalAllocator(n_batch_queries=n_batch_queries),
        HeaviestQueryByRole(n_batch_queries=n_batch_queries),
    ]


_IDS = ["uniform", "topological", "heaviest"]


# ---- 1. Identity at defaults ---------------------------------------------------

@pytest.mark.parametrize("make_sel", range(3), ids=_IDS)
def test_identity_at_defaults(make_sel):
    sel = _selectors(n_batch_queries=1)[make_sel]
    problem = _make_problem()

    flat = sel.select(problem, 8, seed=3)
    groups = sel.select_groups(problem, 8, seed=3, batch_size=1)

    assert len(groups) == len(flat)
    for g, q in zip(groups, flat):
        assert len(g) == 1
        ref = g[0]
        # Same content: identity behavior at defaults.
        assert ref.targets == q.targets
        assert ref.kind == q.kind
        assert set(ref.evidence.keys()) == set(q.evidence.keys())
        for k, v in q.evidence.items():
            if v is None:
                assert ref.evidence[k] is None
            else:
                assert torch.equal(
                    torch.as_tensor(ref.evidence[k]), torch.as_tensor(v)
                )


# ---- 2. Chunking -----------------------------------------------------------------

@pytest.mark.parametrize("make_sel", range(3), ids=_IDS)
def test_chunking_n10240_b512(make_sel):
    sel = _selectors(n_batch_queries=10240)[make_sel]
    problem = _make_problem(n_train=200)  # < N → replacement kicks in too

    k = len(sel.select(problem, 4, seed=1))  # query positions
    groups = sel.select_groups(problem, 4, seed=1, batch_size=512)

    assert k > 0
    assert len(groups) == k * 20  # ceil(10240/512) = 20 per position
    assert all(len(g) == 512 for g in groups)
    # All queries in any group share (targets, evidence_keys) — §1.3.
    for g in groups:
        assert len({q.targets for q in g}) == 1
        assert len({frozenset(q.evidence.keys()) for q in g}) == 1


@pytest.mark.parametrize("make_sel", range(3), ids=_IDS)
def test_last_chunk_handling_n100_b64(make_sel):
    sel = _selectors(n_batch_queries=100)[make_sel]
    problem = _make_problem(n_train=200)

    k = len(sel.select(problem, 4, seed=1))
    groups = sel.select_groups(problem, 4, seed=1, batch_size=64)

    assert len(groups) == k * 2
    # Per position: [64, 36], consecutively emitted.
    for pos in range(k):
        assert len(groups[2 * pos]) == 64
        assert len(groups[2 * pos + 1]) == 36


# ---- 4. Sampling with replacement ------------------------------------------------

@pytest.mark.parametrize("make_sel", range(3), ids=_IDS)
def test_replacement_when_train_smaller_than_n(make_sel):
    sel = _selectors(n_batch_queries=50)[make_sel]
    problem = _make_problem(n_train=10)

    k = len(sel.select(problem, 2, seed=5))
    groups = sel.select_groups(problem, 2, seed=5, batch_size=50)

    assert len(groups) == k  # one chunk per position (B == N)
    assert all(len(g) == 50 for g in groups)  # no exception, 50 variants


# ---- 5. Determinism ---------------------------------------------------------------

def _evidence_fingerprint(groups):
    out = []
    for g in groups:
        for q in g:
            out.append((
                q.targets,
                tuple(sorted(
                    (k, None if v is None else float(v))
                    for k, v in q.evidence.items()
                )),
            ))
    return out


@pytest.mark.parametrize("make_sel", range(3), ids=_IDS)
def test_determinism_same_seed(make_sel):
    problem = _make_problem()
    a = _selectors(n_batch_queries=16)[make_sel]
    b = _selectors(n_batch_queries=16)[make_sel]

    ga = a.select_groups(problem, 4, seed=11, batch_size=4)
    gb = b.select_groups(problem, 4, seed=11, batch_size=4)

    assert _evidence_fingerprint(ga) == _evidence_fingerprint(gb)


# ---- Empty-mode preservation (heaviest emits paired full/empty queries) -----------

def test_empty_mode_variants_stay_empty():
    sel = HeaviestQueryByRole(n_batch_queries=8)
    problem = _make_problem()
    flat = sel.select(problem, 0, seed=2)
    empty_positions = [
        i for i, q in enumerate(flat)
        if all(v is None for v in q.evidence.values())
    ]
    assert empty_positions, "fixture should produce empty-mode queries"

    groups = sel.select_groups(problem, 0, seed=2, batch_size=8)
    # One group per position at B == N; empty positions stay all-None.
    for i in empty_positions:
        for q in groups[i]:
            assert all(v is None for v in q.evidence.values())
