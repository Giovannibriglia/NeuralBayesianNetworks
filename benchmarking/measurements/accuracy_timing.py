"""AccuracyAndTiming — per-query accuracy metrics + isolated timing.

Implements the v0.13 ``Measurement`` protocol for benchmarks that require
both distributional accuracy (TV, JSD, W₁) and timing measurements.

Timing isolation (the #107 fix)
-------------------------------
The v0.12 runner mixed inference calls and oracle sampling in a single
loop, making it impossible to separate ``query_time_s`` from oracle
overhead.  This implementation uses two strictly sequential phases:

    Phase 1 — **Query loop** (what the timeout wraps):
        For each query in ``queries``:
            t0 = perf_counter()
            posterior = adapter.query(q)        ← what gets timed
            query_time_s = perf_counter() - t0
        Posteriors are accumulated in memory.

    Phase 2 — **Accuracy scoring** (``metrics_time_s``):
        For each (query, posterior) pair:
            Build oracle (filter_ground_truth or forward_with_clamp)
            Build cpd_pair (discrete) or sample_pair (continuous)
        Call _compute_metrics_per_node(cpd_pairs, sample_pairs)
        metrics_time_s = perf_counter() - t_metrics_start

    ``fit_time_s`` is passed in by the runner (not measured here).

Note: query_time_s varies per row (one timer per adapter.query call).
fit_time_s and metrics_time_s are recorded identically across all N
rows for a given (cell, baseline) — they are per-cell costs, repeated
for query-row joinability in the parquet.  This matches the v0.13
schema documented in docs/v0.13-benchmark-redesign.md §3.1.

Output shape
------------
Returns one ``CellResult`` per (query, metric) combination.  With N
queries and metrics {tv_per_node, jsd_per_node, w1_per_node}:
  * Discrete target: 2 rows per query (tv + jsd; w1 is not meaningful
    on unordered categorical outcomes).
  * Continuous target: up to 3 rows per query (tv + jsd + w1).
  * Failed oracle / not-supported: 1 row per metric with
    ``status="no_result"`` or ``status="not_supported"``.

Reference: docs/v0.13-benchmark-redesign.md §3, §4.1
"""
from __future__ import annotations

import time
from typing import Any

import torch

from benchmarking.core.oracle import filter_ground_truth, forward_with_clamp_posterior_samples
from benchmarking.core.results import CellResult
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.metrics import _compute_metrics_per_node


_METRICS: tuple[str, ...] = ("tv", "jsd", "w1")
_METRIC_KEYS: tuple[str, ...] = ("tv_per_node", "jsd_per_node", "w1_per_node")


class AccuracyAndTiming:
    """Compute per-query TV/JSD/W₁ accuracy and individually timed queries.

    Implements the v0.13 ``Measurement`` protocol.

    Parameters
    ----------
    n_oracle_samples:
        Number of samples drawn from ``problem.true_model`` per query for
        continuous oracle estimation.  Default 2000 matches the v0.12
        ``_compute_inference_metrics`` setting.
    eps_factor:
        Evidence tolerance for discrete ground-truth rejection filter.
        Continuous evidence nodes within ``eps_factor * std`` of the
        observed value are accepted.  Default 0.50 matches v0.12.
    n_eff_min:
        Minimum effective sample size for the discrete rejection filter.
        Queries with fewer than ``n_eff_min`` surviving reference rows are
        skipped with ``status="no_result"``.  Default 10 matches v0.12.
    """

    def __init__(
        self,
        *,
        n_oracle_samples: int = 2000,
        eps_factor: float = 0.50,
        n_eff_min: int = 10,
    ) -> None:
        self.n_oracle_samples = n_oracle_samples
        self.eps_factor = eps_factor
        self.n_eff_min = n_eff_min

    # -----------------------------------------------------------------------
    # Measurement protocol
    # -----------------------------------------------------------------------

    def measure(
        self,
        problem: BenchmarkProblem,
        adapter: Any,
        queries: list[Query],
        *,
        fit_time_s: float = 0.0,
        benchmark: str = "synthetic",
        seed: int = 0,
        query_roles: list[str] | None = None,
    ) -> list[CellResult]:
        """Run timing + accuracy measurement for all queries.

        Parameters
        ----------
        problem:
            The benchmark problem (DAG, data, ground truth, true model).
        adapter:
            A fitted ``BaselineAdapter`` (``fit()`` already called by the
            runner).  ``adapter.name`` populates ``CellResult.baseline``.
        queries:
            Ordered list of queries from the selector.
        fit_time_s:
            Wall-clock of ``adapter.fit()``, passed in by the runner.
            Recorded identically in every returned ``CellResult`` row for
            query-row joinability.  Default 0.0 for tests that measure the
            adapter externally.
        benchmark:
            Benchmark name for the v3 schema (``"synthetic"`` |
            ``"scalability"`` | ``"bnlearn"``).
        seed:
            RNG seed for the cell, recorded in every row.
        query_roles:
            List of role strings parallel to ``queries``.  If ``None``,
            defaults to ``["random"] * len(queries)``.

        Returns
        -------
        list[CellResult]
            One row per (query, metric).  Length is at most
            ``len(queries) * 3`` (tv + jsd + w1); fewer if some metrics
            are not applicable for the family.

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

        # ---- Phase 1: query loop (timed individually) ----
        posteriors: list[Any] = []
        query_times: list[float] = []

        for q in queries:
            t0 = time.perf_counter()
            try:
                posterior = adapter.query(q)
                err_msg = None
            except Exception as exc:
                posterior = None
                err_msg = str(exc)
            query_times.append(time.perf_counter() - t0)
            posteriors.append((posterior, err_msg))

        # ---- Phase 2: accuracy scoring (metrics_time_s) ----
        t_metrics = time.perf_counter()
        per_query_metric_dicts: list[dict | None] = []

        for q, (posterior, _err) in zip(queries, posteriors):
            if posterior is None:
                per_query_metric_dicts.append(None)
                continue
            mdict = self._score_query(problem, q, posterior)
            per_query_metric_dicts.append(mdict)

        metrics_time_s = time.perf_counter() - t_metrics

        # ---- Phase 3: emit CellResult rows ----
        rows: list[CellResult] = []
        for i, (q, (posterior, err_msg), mdict, role) in enumerate(
            zip(queries, posteriors, per_query_metric_dicts, query_roles)
        ):
            q_time = query_times[i]

            if posterior is None or mdict is None:
                # Query failed: one row per metric with error status.
                for mk in _METRIC_KEYS:
                    rows.append(CellResult(
                        benchmark=benchmark,
                        family=family,
                        problem_id=problem_id,
                        seed=seed,
                        baseline=baseline,
                        query_role=role,
                        metric=mk,
                        value=float("nan"),
                        status="error",
                        fit_time_s=fit_time_s,
                        query_time_s=q_time,
                        metrics_time_s=metrics_time_s,
                        error_msg=err_msg or "query failed",
                    ))
                continue

            # Success: one row per metric.
            for mk in _METRIC_KEYS:
                val = mdict.get(mk)
                if val is None:
                    status = "not_supported"
                    value = float("nan")
                elif mdict.get(f"{mk}_no_result"):
                    status = "no_result"
                    value = float("nan")
                else:
                    status = "ok"
                    value = float(val)
                rows.append(CellResult(
                    benchmark=benchmark,
                    family=family,
                    problem_id=problem_id,
                    seed=seed,
                    baseline=baseline,
                    query_role=role,
                    metric=mk,
                    value=value,
                    status=status,
                    fit_time_s=fit_time_s,
                    query_time_s=q_time,
                    metrics_time_s=metrics_time_s,
                    error_msg=None,
                ))

        return rows

    # -----------------------------------------------------------------------
    # Internal: per-query accuracy scoring
    # -----------------------------------------------------------------------

    def _score_query(
        self,
        problem: BenchmarkProblem,
        q: Query,
        posterior: Any,
    ) -> dict[str, float | None]:
        """Return ``{tv_per_node, jsd_per_node, w1_per_node}`` for one query.

        Dispatches on target kind (discrete → cpd_pair; continuous →
        sample_pair).  Returns the ``_compute_metrics_per_node`` dict.
        On oracle failure, returns all-None dict (caller emits no_result).
        """
        target = q.targets[0]
        target_kind = problem.variables[target][0]
        ev_row = dict(q.evidence.items())

        cpd_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        sample_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

        if target_kind == "discrete":
            oracle = filter_ground_truth(
                problem, ev_row, target,
                eps_factor=self.eps_factor,
                n_eff_min=self.n_eff_min,
            )
            if oracle is None:
                return dict.fromkeys(_METRIC_KEYS)

            k = problem.variables[target][1]
            empirical = torch.zeros(k)
            empirical.scatter_add_(
                0,
                oracle.long().clamp(0, k - 1),
                torch.ones_like(oracle, dtype=torch.float),
            )
            empirical = empirical / empirical.sum().clamp_min(1e-12)
            true_logits = torch.log(empirical.clamp_min(1e-12))

            # Posterior.probs is the predicted probability vector.
            if posterior.probs is None:
                return dict.fromkeys(_METRIC_KEYS)
            pred_vec = posterior.probs.cpu().reshape(-1)[:k]
            fit_logits = torch.log(pred_vec.clamp_min(1e-12))
            cpd_pairs.append((true_logits, fit_logits))

        else:
            # Continuous or hybrid-continuous target.
            oracle = forward_with_clamp_posterior_samples(
                problem, [target], ev_row, n_samples=self.n_oracle_samples,
            )
            if oracle is None or oracle.shape[0] < 100:
                return dict.fromkeys(_METRIC_KEYS)

            if posterior.samples is None:
                return dict.fromkeys(_METRIC_KEYS)
            sample_pairs.append((oracle.reshape(-1).cpu(), posterior.samples.cpu().reshape(-1)))

        return _compute_metrics_per_node(
            cpd_pairs=cpd_pairs,
            sample_pairs=sample_pairs,
            metrics=_METRICS,
        )


# ---------------------------------------------------------------------------
# Family inference fallback
# ---------------------------------------------------------------------------

def _infer_family(problem: BenchmarkProblem) -> str:
    """Heuristic family inference from ``problem.variables``.

    Used when ``problem.family`` is empty (e.g. test fixtures that
    construct ``BenchmarkProblem`` manually without a source).

    Note: cannot distinguish ``continuous_lg`` from ``continuous_nongauss``
    — both have all-continuous variables.  Returns ``"continuous_lg"`` as
    the safer default for continuous families.  Problem sources should
    always populate ``problem.family`` explicitly.
    """
    kinds = {k for (k, _) in problem.variables.values()}
    if kinds == {"discrete"}:
        return "discrete"
    if kinds == {"continuous"}:
        return "continuous_lg"
    return "hybrid"
