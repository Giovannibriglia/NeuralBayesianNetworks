"""ParamLearningMeasurement — held-out joint log-likelihood scoring (#109).

Implements the v0.13 ``Measurement`` protocol for parameter-learning (PL)
mode.  Where ``AccuracyAndTiming`` / ``TimingOnly`` are *query-centric* (they
loop ``adapter.query`` over selector queries), PL mode is *data-centric*: a
fitted model is scored ONCE on the held-out ``problem.test_data`` and a single
joint log-likelihood is emitted per cell.

Contract
--------
* ``queries`` (and ``query_groups``) are IGNORED — PL scores ``test_data``,
  not queries.  The selector still runs upstream; its output is simply not
  consumed here.  (Short-circuiting the selector for PL configs is a possible
  follow-up, out of scope for #109 PR 1.)
* Capability gate (mirrors the ``supports_batched_queries`` precedent, see the
  ``BaselineAdapter`` docstring): only adapters that opt in with
  ``supports_scoring = True`` are scored.  Adapters without the flag
  (``getattr(..., "supports_scoring", False) is False``) get a single
  ``status="not_supported"`` row and ``score_data`` is never called on them.
* On success: one ``CellResult`` with ``metric="log_likelihood"``,
  ``value`` = mean held-out joint log-prob (the positive, higher-is-better
  scalar from ``metrics.log_likelihood``), ``status="ok"``.
* On failure (OOM / out-of-range / NotImplementedError raised by
  ``score_data``): one row with the ``_classify_exception`` status and
  ``value=nan`` — same classification the query path uses (#127 Stage 4).

Output shape: exactly ONE row per cell (one ``log_likelihood`` row).  This is
deliberately a single joint score, not a per-node average — the metric name
carries no ``_per_node`` suffix (see ``metrics.log_likelihood``).

PL rows only — this measurement is constructed solely by the ``param-learning``
CLI command and never touches the inference path.

Reference: docs/v0.13-benchmark-redesign.md §3; issue #109.
"""
from __future__ import annotations

import time
from typing import Any

import torch

from benchmarking.core.results import CellResult
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.measurements.accuracy_timing import _infer_family
from benchmarking.metrics import log_likelihood

# Query role stamped on PL rows.  A log_likelihood row is a per-cell, per-model
# score with no query semantics, so it carries a dedicated sentinel role rather
# than borrowing a query's role.  (No CellResult-level validation enforces the
# VALID_QUERY_ROLES_* sets, so this is additive — downstream readers select by
# (metric, baseline), never by query_role, for the LL panel.)
_PL_QUERY_ROLE = "param_learning"


class ParamLearningMeasurement:
    """Score a fitted model's held-out joint log-likelihood (PL mode).

    Implements the v0.13 ``Measurement`` protocol.  Accepts the full
    cell-worker keyword surface (``fit_time_s``, ``benchmark``, ``seed``,
    ``query_roles`` …) for call-site parity with the query-centric
    measurements, but uses only ``problem`` and ``adapter``: ``queries`` /
    ``query_groups`` are ignored (PL scores ``problem.test_data``).
    """

    # Fit-only signal (#109): PL scores via score_data and never queries, so
    # the cell worker builds adapters with require_engine=False — a baseline
    # may omit inference_method (the PL configs do). The query-centric
    # measurements lack this flag (getattr default False -> require_engine=True),
    # so inference cells still reject a missing inference_method early.
    fit_only: bool = True

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
        query_kinds: list[str] | None = None,
        evidence_strategies: list[str] | None = None,
        evidence_modes: list[str] | None = None,
        query_budget_s: float = float("inf"),
        query_groups: list[list[Query]] | None = None,
    ) -> list[CellResult]:
        """Score ``problem.test_data`` once; return a single log_likelihood row.

        Parameters
        ----------
        problem:
            The benchmark problem.  ``problem.test_data`` (dict node →
            tensor ``[N]`` / ``[N, D]``) is the held-out set being scored.
        adapter:
            A fitted ``BaselineAdapter``.  Scored only if it opts in with
            ``supports_scoring = True``.
        fit_time_s:
            Wall-clock of ``adapter.fit()``, passed in by the runner and
            recorded on the row.  Not gated by any timeout.
        queries, query_groups, query_*, query_budget_s:
            Accepted for protocol/call-site parity; IGNORED in PL mode.

        Returns
        -------
        list[CellResult]
            A single-element list: one ``metric="log_likelihood"`` row.
        """
        family = problem.family or _infer_family(problem)
        problem_id = problem.problem_id or problem.name
        baseline = adapter.name

        def _row(*, value: float, status: str,
                 query_time_s: float, metrics_time_s: float,
                 error_msg: str | None) -> CellResult:
            # Single per-cell row.  device / n_parameters / n_nodes /
            # proposal_used are stamped downstream at the cell_worker._emit and
            # runner choke points (same as the other measurements), so they are
            # left at their defaults here.
            return CellResult(
                benchmark=benchmark,
                family=family,
                problem_id=problem_id,
                seed=seed,
                baseline=baseline,
                query_role=_PL_QUERY_ROLE,
                metric="log_likelihood",
                value=value,
                status=status,
                fit_time_s=fit_time_s,
                query_time_s=query_time_s,
                metrics_time_s=metrics_time_s,
                error_msg=error_msg,
            )

        # ---- Capability gate (supports_scoring opt-in flag) ----
        if not getattr(adapter, "supports_scoring", False):
            return [_row(
                value=float("nan"),
                status="not_supported",
                query_time_s=float("nan"),
                metrics_time_s=float("nan"),
                error_msg=(
                    f"{baseline} does not support parameter-learning scoring "
                    f"(supports_scoring is not set)"
                ),
            )]

        # ---- Score held-out test_data once (timed as query_time_s) ----
        from benchmarking.core.runner import _classify_exception

        t0 = time.perf_counter()
        try:
            log_probs = adapter.score_data(problem.test_data)  # [B] per-row joint
        except Exception as exc:
            # Same failure classification as the query path (#127 Stage 4):
            # OOM -> "oom", NotImplementedError/structural -> "not_supported",
            # out-of-range / other -> "error".
            query_time_s = time.perf_counter() - t0
            return [_row(
                value=float("nan"),
                status=_classify_exception(exc),
                query_time_s=query_time_s,
                metrics_time_s=float("nan"),
                error_msg=repr(exc),
            )]
        query_time_s = time.perf_counter() - t0

        # ---- Aggregate to the mean joint log-likelihood (metrics_time_s) ----
        t_metrics = time.perf_counter()
        result = log_likelihood(torch.as_tensor(log_probs).reshape(-1))
        metrics_time_s = time.perf_counter() - t_metrics

        return [_row(
            value=result.value,
            status="ok",
            query_time_s=query_time_s,
            metrics_time_s=metrics_time_s,
            error_msg=None,
        )]
