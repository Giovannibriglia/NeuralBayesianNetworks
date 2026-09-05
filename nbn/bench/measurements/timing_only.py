"""TimingOnly — query-time measurement without accuracy scoring.

Used by the scalability benchmark where the selected queries are known
worst-case and accuracy is not meaningful.  Records only ``query_time_s``;
``metrics_time_s`` is always 0.0.

Output shape: one ``CellResult`` per query with ``metric="query_time_s"``
and ``value=query_time_s``.

Reference: docs/v0.13-benchmark-redesign.md §2.2, §3, §4.1
"""
from __future__ import annotations

from typing import Any

from nbn.bench.core.results import CellResult
from nbn.bench.domains.base import BenchmarkProblem, Query
from nbn.bench.measurements.accuracy_timing import _infer_family


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
        query_kinds: list[str] | None = None,
        evidence_strategies: list[str] | None = None,
        evidence_modes: list[str] | None = None,
        query_budget_s: float = float("inf"),
        query_groups: list[list[Query]] | None = None,
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
        if query_kinds is None:
            query_kinds = ["prediction"] * len(queries)
        if len(query_kinds) != len(queries):
            raise ValueError(
                f"query_kinds length {len(query_kinds)} != queries length {len(queries)}"
            )
        if evidence_strategies is None:
            evidence_strategies = ["random"] * len(queries)
        if len(evidence_strategies) != len(queries):
            raise ValueError(
                f"evidence_strategies length {len(evidence_strategies)} "
                f"!= queries length {len(queries)}"
            )
        if evidence_modes is None:
            evidence_modes = ["full"] * len(queries)
        if len(evidence_modes) != len(queries):
            raise ValueError(
                f"evidence_modes length {len(evidence_modes)} "
                f"!= queries length {len(queries)}"
            )

        # v0.14 (#148): grouped dispatch. Without query_groups (legacy
        # callers), every query is its own length-1 group — the dispatch
        # helper routes B=1 through adapter.query(), so behavior is
        # identical to the pre-batching loop.
        if query_groups is None:
            query_groups = [[q] for q in queries]
        n_grouped = sum(len(g) for g in query_groups)
        if n_grouped != len(queries):
            raise ValueError(
                f"query_groups flatten to {n_grouped} queries "
                f"!= queries length {len(queries)}"
            )

        family = problem.family or _infer_family(problem)
        problem_id = problem.problem_id or problem.name
        baseline = adapter.name

        # Group dispatch — atomic batch failure, amortized timing,
        # #127-Stage-4 exception classification all live in the helper.
        from nbn.bench.measurements._batch_dispatch import run_query_groups

        outcomes, unstarted_groups = run_query_groups(
            adapter, query_groups, query_budget_s=query_budget_s,
        )

        rows: list[CellResult] = []
        for o in outcomes:
            i = o.flat_index
            role, qkind = query_roles[i], query_kinds[i]
            estrat, emode = evidence_strategies[i], evidence_modes[i]
            for tk, tv in [
                ("fit_time_s", fit_time_s),
                ("query_time_s",
                 o.query_time_s if o.status == "ok" else float("nan")),
                ("metrics_time_s", 0.0),
            ]:
                rows.append(CellResult(
                    benchmark=benchmark,
                    family=family,
                    problem_id=problem_id,
                    seed=seed,
                    baseline=baseline,
                    query_role=role,
                    query_kind=qkind,
                    evidence_strategy=estrat,
                    evidence_mode=emode,
                    metric=tk,
                    value=tv,
                    status=o.status,
                    fit_time_s=fit_time_s,
                    query_time_s=o.query_time_s,
                    metrics_time_s=0.0,
                    error_msg=o.error_msg,
                    batch_size=o.batch_size,
                ))

        # Timeout rows for groups that never started — one row-set per
        # unstarted GROUP (§4.3 sentinel principle; identical to the
        # per-query emission at B=1).
        for group, flat_start in unstarted_groups:
            role = query_roles[flat_start]
            qkind = query_kinds[flat_start]
            estrat = evidence_strategies[flat_start]
            emode = evidence_modes[flat_start]
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
                    query_kind=qkind,
                    evidence_strategy=estrat,
                    evidence_mode=emode,
                    metric=tk,
                    value=tv,
                    status="timeout",
                    fit_time_s=fit_time_s,
                    query_time_s=float("nan"),
                    metrics_time_s=0.0,
                    error_msg="query budget exceeded",
                    batch_size=len(group),
                ))

        return rows
