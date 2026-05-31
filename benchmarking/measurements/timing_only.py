"""TimingOnly — query-time measurement without accuracy scoring.

Used by the scalability benchmark where the selected queries are known
worst-case and accuracy is not meaningful.  Records only ``query_time_s``;
``metrics_time_s`` is always 0.0.

Output shape: one ``CellResult`` per query with ``metric="query_time_s"``
and ``value=query_time_s``.

Reference: docs/v0.13-benchmark-redesign.md §2.2, §3, §4.1
"""
from __future__ import annotations

import time
from typing import Any

from benchmarking.core.results import CellResult
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.measurements.accuracy_timing import _infer_family


class TimingOnly:
    """Measure per-query wall-clock time; no accuracy computation.

    Implements the v0.13 ``Measurement`` protocol.

    For each query, times a single ``adapter.query(q)`` call with
    ``time.perf_counter()``.  No oracle construction, no metric
    computation.

    Output shape
    ------------
    One ``CellResult`` per query with:
      * ``metric = "query_time_s"``
      * ``value = query_time_s``
      * ``metrics_time_s = 0.0`` (no accuracy computation)
      * ``status = "ok"`` on success, ``"error"`` on exception.

    Note: query_time_s varies per row (one timer per adapter.query call).
    fit_time_s and metrics_time_s are recorded identically across all N
    rows for a given (cell, baseline) — they are per-cell costs, repeated
    for query-row joinability in the parquet.  This matches the v0.13
    schema documented in docs/v0.13-benchmark-redesign.md §3.1.
    """

    def measure(
        self,
        problem: BenchmarkProblem,
        adapter: Any,
        queries: list[Query],
        *,
        fit_time_s: float = 0.0,
        benchmark: str = "scalability",
        seed: int = 0,
        query_roles: list[str] | None = None,
        query_budget_s: float = float("inf"),
    ) -> list[CellResult]:
        """Time each query individually; return one row per query.

        Parameters
        ----------
        problem:
            The benchmark problem.  ``true_model`` is not used (no
            accuracy scoring).
        adapter:
            A fitted ``BaselineAdapter``.
        queries:
            Ordered list of queries.
        fit_time_s:
            Wall-clock of ``adapter.fit()``.  Recorded identically in
            every returned ``CellResult`` row.  Default 0.0 for tests.
        benchmark:
            Benchmark name for the v3 schema.  Defaults to
            ``"scalability"`` (the natural home for timing-only runs).
        seed:
            RNG seed for the cell.
        query_roles:
            Role strings parallel to ``queries``.  If ``None``, defaults
            to ``["random"] * len(queries)``.

        Returns
        -------
        list[CellResult]
            One row per query.  ``metric = "query_time_s"``,
            ``value = query_time_s``.  Queries whose cumulative
            ``query_time_s`` exceeds ``query_budget_s`` receive
            ``status="timeout"`` rows with ``query_time_s=NaN``.

        Note: query_time_s varies per row (one timer per adapter.query
        call).  fit_time_s and metrics_time_s are recorded identically
        across all N rows for a given (cell, baseline) — they are
        per-cell costs, repeated for query-row joinability in the parquet.
        This matches the v0.13 schema documented in
        docs/v0.13-benchmark-redesign.md §3.1.
        """
        if query_roles is None:
            query_roles = ["random"] * len(queries)
        if len(query_roles) != len(queries):
            raise ValueError(
                f"query_roles length {len(query_roles)} != queries length {len(queries)}"
            )

        family = problem.family or _infer_family(problem)
        problem_id = problem.problem_id or problem.name
        baseline = adapter.name

        rows: list[CellResult] = []
        cumulative_query_time_s = 0.0
        timed_out_at: int | None = None

        for i, (q, role) in enumerate(zip(queries, query_roles)):
            if cumulative_query_time_s >= query_budget_s:
                timed_out_at = i
                break
            t0 = time.perf_counter()
            try:
                adapter.query(q)
                q_time = time.perf_counter() - t0
                status = "ok"
                err_msg = None
            except Exception as exc:
                q_time = time.perf_counter() - t0
                status = "error"
                err_msg = str(exc)

            cumulative_query_time_s += q_time
            for tk, tv in [
                ("fit_time_s", fit_time_s),
                ("query_time_s", q_time if status == "ok" else float("nan")),
                ("metrics_time_s", 0.0),
            ]:
                rows.append(CellResult(
                    benchmark=benchmark,
                    family=family,
                    problem_id=problem_id,
                    seed=seed,
                    baseline=baseline,
                    query_role=role,
                    metric=tk,
                    value=tv,
                    status=status,
                    fit_time_s=fit_time_s,
                    query_time_s=q_time,
                    metrics_time_s=0.0,
                    error_msg=err_msg,
                ))

        # Timeout rows for queries that never started.
        if timed_out_at is not None:
            for role in query_roles[timed_out_at:]:
                for tk, tv in [
                    ("fit_time_s", fit_time_s),
                    ("query_time_s", float("nan")),
                    ("metrics_time_s", 0.0),
                ]:
                    rows.append(CellResult(
                        benchmark=benchmark,
                        family=family,
                        problem_id=problem_id,
                        seed=seed,
                        baseline=baseline,
                        query_role=role,
                        metric=tk,
                        value=tv,
                        status="timeout",
                        fit_time_s=fit_time_s,
                        query_time_s=float("nan"),
                        metrics_time_s=0.0,
                        error_msg="query budget exceeded",
                    ))

        return rows
