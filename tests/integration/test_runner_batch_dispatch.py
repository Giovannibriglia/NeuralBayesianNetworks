"""Integration tests for runner batch dispatch (PR 4 stage b, #148).

Exercises the measurement-level group dispatch (the runner path that
cell_worker drives): amortized per-query timing, batch_size stamping,
atomic batch failure statuses, per-group timeout sentinels, and the
n_batch_queries=1 identity path.

Reference: docs/v0.14-batched-queries-design.md §1.6, §4, §7.
"""
from __future__ import annotations

import math
import time

import torch

from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.domains.posterior import Posterior
from benchmarking.measurements.accuracy_timing import AccuracyAndTiming
from benchmarking.measurements.timing_only import TimingOnly


# ---- Fixtures ----------------------------------------------------------------

def _make_problem() -> BenchmarkProblem:
    g = torch.Generator().manual_seed(0)
    nodes = ["X0", "X1", "X2"]
    data = {v: torch.randint(0, 2, (100,), generator=g) for v in nodes}
    return BenchmarkProblem(
        name="dispatch_test",
        dag=[("X0", "X1"), ("X1", "X2")],
        variables=dict.fromkeys(nodes, ("discrete", 2)),
        train_data=data,
        test_data=data,
        queries=[],
        family="discrete",
        problem_id="dispatch_test",
        seed=0,
    )


def _query(v: int = 0) -> Query:
    return Query(
        targets=("X2",),
        evidence={"X0": torch.tensor(v % 2)},
        kind="marginal",
    )


class _MockAdapter:
    """Adapter double: query/query_batch return uniform posteriors."""

    name = "mock-adapter"
    device = "cpu"

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.batch_calls: list[int] = []
        self.single_calls = 0

    def query(self, q: Query) -> Posterior:
        self.single_calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        return Posterior(probs=torch.tensor([0.5, 0.5]))

    def query_batch(self, queries: list[Query]) -> list[Posterior]:
        self.batch_calls.append(len(queries))
        if self.delay_s:
            time.sleep(self.delay_s)
        return [Posterior(probs=torch.tensor([0.5, 0.5])) for _ in queries]


class _OOMAdapter(_MockAdapter):
    def query_batch(self, queries: list[Query]) -> list[Posterior]:
        raise MemoryError("mock allocation failure")


class _ErrorAdapter(_MockAdapter):
    def query_batch(self, queries: list[Query]) -> list[Posterior]:
        raise RuntimeError("mock batch failure")


# ---- 1. Per-query rows with amortized timing -----------------------------------

class TestAmortizedTiming:
    def test_b4_batch_produces_4_rows_with_amortized_time(self):
        problem = _make_problem()
        adapter = _MockAdapter(delay_s=0.05)
        queries = [_query(v) for v in range(4)]

        rows = TimingOnly().measure(
            problem, adapter, queries, query_groups=[queries],
        )

        assert adapter.batch_calls == [4]  # exactly one library call
        assert adapter.single_calls == 0
        qt_rows = [r for r in rows if r.metric == "query_time_s"]
        assert len(qt_rows) == 4
        assert all(r.batch_size == 4 for r in qt_rows)
        assert all(r.status == "ok" for r in qt_rows)
        # Amortized: all four values equal, each = batch_time/4 ≥ delay/4.
        values = [r.value for r in qt_rows]
        assert len(set(values)) == 1
        assert 0.05 / 4 <= values[0] < 0.05  # delay/4 <= t < full delay


# ---- 2-3. Atomic batch failure ---------------------------------------------------

class TestAtomicBatchFailure:
    def test_oom_marks_all_b_rows(self):
        problem = _make_problem()
        queries = [_query(v) for v in range(4)]
        rows = TimingOnly().measure(
            problem, _OOMAdapter(), queries, query_groups=[queries],
        )
        qt_rows = [r for r in rows if r.metric == "query_time_s"]
        assert len(qt_rows) == 4
        assert all(r.status == "oom" for r in qt_rows)
        assert all(math.isnan(r.value) for r in qt_rows)
        assert all(r.batch_size == 4 for r in qt_rows)

    def test_exception_marks_all_b_rows_with_error_msg(self):
        problem = _make_problem()
        queries = [_query(v) for v in range(4)]
        rows = AccuracyAndTiming().measure(
            problem, _ErrorAdapter(), queries, query_groups=[queries],
        )
        per_status = {r.status for r in rows}
        assert per_status == {"error"}
        assert all("mock batch failure" in (r.error_msg or "") for r in rows)
        assert all(r.batch_size == 4 for r in rows)
        # All 4 queries represented: 6 rows each (3 metric + 3 timing).
        assert len(rows) == 4 * 6


# ---- 4. Per-group (K) sentinels on soft timeout -----------------------------------

class TestTimeoutSentinels:
    def test_k_sentinels_not_k_times_n(self):
        """K=3 positions, N=B=4 (one group per position). A slow first
        group exhausts the budget; the 2 unstarted groups produce one
        timeout row-set each — 2 sentinels, not 2×4."""
        problem = _make_problem()
        adapter = _MockAdapter(delay_s=0.2)
        groups = [[_query(v) for v in range(4)] for _ in range(3)]
        flat = [q for g in groups for q in g]

        rows = TimingOnly().measure(
            problem, adapter, flat,
            query_groups=groups, query_budget_s=0.1,
        )

        assert adapter.batch_calls == [4]  # only the first group ran
        timeout_qt = [
            r for r in rows
            if r.metric == "query_time_s" and r.status == "timeout"
        ]
        # One sentinel row-set per unstarted GROUP (2), not per query (8).
        assert len(timeout_qt) == 2
        assert all(r.batch_size == 4 for r in timeout_qt)
        ok_qt = [r for r in rows if r.metric == "query_time_s" and r.status == "ok"]
        assert len(ok_qt) == 4  # the started group's per-query rows


# ---- 5. n_batch_queries=1 identity path ---------------------------------------------

class TestDefaultPathIdentity:
    def test_b1_groups_match_legacy_flat_call(self):
        """Length-1 groups (the n_batch_queries=1 shape) produce
        structurally identical rows to the legacy flat call, and route
        through adapter.query() — the pre-batching code path."""
        problem = _make_problem()
        queries = [_query(v) for v in range(3)]

        a1 = _MockAdapter()
        legacy = TimingOnly().measure(problem, a1, queries)
        a2 = _MockAdapter()
        grouped = TimingOnly().measure(
            problem, a2, queries, query_groups=[[q] for q in queries],
        )

        # Both used the single-query path: no query_batch calls.
        assert a1.batch_calls == a2.batch_calls == []
        assert a1.single_calls == a2.single_calls == 3

        assert len(legacy) == len(grouped)
        for lr, gr in zip(legacy, grouped):
            assert lr.metric == gr.metric
            assert lr.status == gr.status
            assert lr.batch_size == gr.batch_size == 1
            assert lr.query_role == gr.query_role

    def test_accuracy_and_timing_b1_identity(self):
        """Same identity check through AccuracyAndTiming (scoring path)."""
        problem = _make_problem()
        queries = [_query(v) for v in range(2)]

        legacy = AccuracyAndTiming().measure(problem, _MockAdapter(), queries)
        grouped = AccuracyAndTiming().measure(
            problem, _MockAdapter(), queries,
            query_groups=[[q] for q in queries],
        )
        assert [(r.metric, r.status, r.batch_size) for r in legacy] == \
               [(r.metric, r.status, r.batch_size) for r in grouped]
