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

from nbn.bench.domains.base import BenchmarkProblem
from nbn.bench.selectors import UniformRandomSelector
from nbn.bench.selectors.heaviest import HeaviestQueryByRole
from nbn.bench.selectors.topological import TopologicalAllocator


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


def _is_empty(q):
    return all(v is None for v in q.evidence.values())


def _positions(flat):
    """(positions with concrete evidence, evidence-free positions)."""
    n_empty = sum(_is_empty(q) for q in flat)
    return len(flat) - n_empty, n_empty


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

    k, k_empty = _positions(sel.select(problem, 4, seed=1))
    groups = sel.select_groups(problem, 4, seed=1, batch_size=512)

    assert k > 0
    # ceil(10240/512) = 20 per position; evidence-free positions are one
    # query (D7).
    assert len(groups) == k * 20 + k_empty
    assert sorted(len(g) for g in groups) == [1] * k_empty + [512] * (k * 20)
    # All queries in any group share (targets, evidence_keys) — §1.3.
    for g in groups:
        assert len({q.targets for q in g}) == 1
        assert len({frozenset(q.evidence.keys()) for q in g}) == 1


@pytest.mark.parametrize("make_sel", range(3), ids=_IDS)
def test_last_chunk_handling_n100_b64(make_sel):
    sel = _selectors(n_batch_queries=100)[make_sel]
    problem = _make_problem(n_train=200)

    flat = sel.select(problem, 4, seed=1)
    k, k_empty = _positions(flat)
    groups = sel.select_groups(problem, 4, seed=1, batch_size=64)

    assert len(groups) == k * 2 + k_empty
    # Per full position: [64, 36], consecutively emitted; empty: [1].
    i = 0
    for q in flat:
        if _is_empty(q):
            assert len(groups[i]) == 1
            i += 1
        else:
            assert [len(groups[i]), len(groups[i + 1])] == [64, 36]
            i += 2


# ---- 4. Sampling with replacement ------------------------------------------------

@pytest.mark.parametrize("make_sel", range(3), ids=_IDS)
def test_replacement_when_train_smaller_than_n(make_sel):
    sel = _selectors(n_batch_queries=50)[make_sel]
    problem = _make_problem(n_train=10)

    k, k_empty = _positions(sel.select(problem, 2, seed=5))
    groups = sel.select_groups(problem, 2, seed=5, batch_size=50)

    assert len(groups) == k + k_empty  # one chunk per position (B == N)
    # no exception, 50 variants (1 for an evidence-free position, D7)
    assert sorted(len(g) for g in groups) == [1] * k_empty + [50] * k


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


def test_empty_mode_position_is_one_query_not_b_copies():
    """D7: an evidence-free position used to expand into n_batch_queries
    identical queries. nbn answered the batch once and divided the time by B
    while pomegranate ran B queries; it is now a single B=1 query for every
    adapter."""
    sel = HeaviestQueryByRole(n_batch_queries=8)
    problem = _make_problem()
    flat = sel.select(problem, 0, seed=2)
    n_empty = sum(all(v is None for v in q.evidence.values()) for q in flat)
    n_full = len(flat) - n_empty
    assert n_empty and n_full

    groups = sel.select_groups(problem, 0, seed=2, batch_size=4)
    empty_groups = [g for g in groups
                    if all(v is None for q in g for v in q.evidence.values())]
    assert [len(g) for g in empty_groups] == [1] * n_empty
    assert sum(len(g) for g in groups) == n_empty + 8 * n_full


def test_query_timer_synchronizes_accelerator(monkeypatch):
    """Per-query time on CUDA must include the kernels the query queued."""
    from nbn.bench.measurements import _batch_dispatch as bd

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: calls.append(d))

    class _A:
        device = "cuda:1"

        def query(self, q):
            return object()

    bd.run_query_groups(_A(), [[object()], [object()]])
    assert calls == ["cuda:1", "cuda:1"]


def test_evidence_free_rows_stay_in_their_pass_batch_size_slice():
    """D7 follow-up: an evidence-free singleton in a B=4 pass was stamped
    batch_size=1, so a sweep pooled every pass's copy into the B=1 slice
    (and TimingOnly stamped the pass rows with the FIRST group's size, 1 when
    the first position was evidence-free). Both now carry the pass's B."""
    from nbn.bench.domains.base import Query
    from nbn.bench.domains.posterior import Posterior
    from nbn.bench.measurements.timing_only import TimingOnly

    problem = _make_problem()
    nodes = list(problem.variables)
    empty = Query(targets=(nodes[0],), evidence={nodes[1]: None}, kind="marginal")
    full = [Query(targets=(nodes[0],), evidence={nodes[1]: torch.tensor(float(v % 2))},
                  kind="marginal") for v in range(4)]

    class _A:
        name = "stub"

        def query(self, q):
            return Posterior(probs=torch.tensor([0.5, 0.5]))

        def query_batch(self, qs):
            return [self.query(q) for q in qs]

    rows = TimingOnly().measure(problem, _A(), [empty, *full],
                                query_groups=[[empty], full],
                                evidence_modes=["empty"] + ["full"] * 4)
    by_mode = {(r.metric, r.evidence_mode): r.batch_size for r in rows}
    assert by_mode[("query_time_s", "empty")] == 4
    assert by_mode[("query_time_s", "full")] == 4
    assert {r.batch_size for r in rows if r.metric in ("fit_time_s", "metrics_time_s")} == {4}
