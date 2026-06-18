"""ParamLearningMeasurement — held-out log-likelihood + parameter recovery (#109).

Implements the v0.13 ``Measurement`` protocol for parameter-learning (PL) mode.
Where ``AccuracyAndTiming`` / ``TimingOnly`` are *query-centric* (they loop
``adapter.query`` over selector queries), PL mode is *data-centric*: a fitted
model is scored ONCE on the held-out ``problem.test_data`` and, where the
adapter and problem support it, its learned discrete CPTs are compared against
the true CPDs.

Per cell this emits UP TO THREE metric rows, each independently gated:
  * ``log_likelihood`` (PR 1) — mean held-out joint log-prob, via
    ``adapter.score_data`` -> ``metrics.log_likelihood``;
  * ``param_recovery_tv`` (PR 2, headline) and ``param_recovery_kl`` (PR 2,
    companion) — frequency-weighted error between learned and true CPTs.

Recovery gating taxonomy (per row):
  * ``ok``            — adapter has ``supports_param_recovery`` AND the problem
                        is fully discrete AND ``true_model`` is present AND the
                        true CPTs pass the row-sum sanity check;
  * ``not_supported`` — adapter lacks ``supports_param_recovery``;
  * ``not_applicable``— supported, but the cell is non-fully-discrete or has no
                        ``true_model`` (or no discrete-with-discrete-parent
                        nodes to recover);
  * ``error``         — the true CPT sanity check failed (a row sum != 1), i.e.
                        the generator/loader is buggy; value=NaN, error_msg
                        names the node and the offending sum.

Timing: held-out scoring time is ``query_time_s`` on the ``log_likelihood``
row only; parameter-recovery work (CPT extraction + the primitives) goes into
``metrics_time_s`` on the recovery rows (``query_time_s`` is NaN there — no
query is performed for them).

Fairness of the frequency weights comes from a DETERMINISTIC sample seed
derived from ``problem.seed`` (every baseline on a problem gets identical
weights), NOT from caching — the per-problem cache is a pure performance
optimization that avoids redundant draws when the measurement instance is
reused in-process.

PL rows only — this measurement is constructed solely by the ``param-learning``
CLI command and never touches the inference path.

Reference: docs/v0.13-benchmark-redesign.md §3; issue #109.
"""
from __future__ import annotations

import time
from typing import Any

import torch

from benchmarking.core.cpt_extraction import extract_discrete_cpts
from benchmarking.core.results import CellResult
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.measurements.accuracy_timing import _infer_family
from benchmarking.metrics import (
    frequency_weights,
    log_likelihood,
    param_recovery_kl,
    param_recovery_tv,
)

# Query role stamped on PL rows — a per-cell, per-model score with no query
# semantics, so a dedicated sentinel rather than a borrowed query role.
_PL_QUERY_ROLE = "param_learning"

# Salt for the deterministic recovery-weight sample seed (golden-ratio constant,
# arbitrary but stable): sample_seed = problem.seed ^ _RECOVERY_SEED_SALT. Keeps
# the weights reproducible and identical across all baselines on a problem.
_RECOVERY_SEED_SALT = 0x9E3779B9

# True-CPT row-sum sanity tolerance.
_CPT_ROW_SUM_TOL = 1e-5


class ParamLearningMeasurement:
    """Score held-out log-likelihood and parameter recovery (PL mode).

    Implements the v0.13 ``Measurement`` protocol. Accepts the full cell-worker
    keyword surface for call-site parity with the query-centric measurements,
    but uses only ``problem`` and ``adapter``: ``queries`` / ``query_groups``
    are ignored (PL scores ``problem.test_data`` and the model parameters).
    """

    # Fit-only signal (#109): PL scores via score_data / parameter extraction
    # and never queries, so the cell worker builds adapters with
    # require_engine=False (a baseline may omit inference_method).
    fit_only: bool = True

    # Number of true-model samples drawn to estimate the parent-config weights.
    # A measurement constant (tunable here), not a config knob.
    N_WEIGHT_SAMPLES: int = 20_000

    def __init__(self) -> None:
        # Per-problem recovery context cache (true CPTs + weights, or a
        # not_applicable / error sentinel). Keyed by problem identity; baseline-
        # independent, so all baselines on a problem share it. Bounded by the
        # number of problems in a run.
        self._recovery_cache: dict[tuple, tuple] = {}

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
        """Emit the log_likelihood row plus the two parameter-recovery rows."""
        family = problem.family or _infer_family(problem)
        problem_id = problem.problem_id or problem.name
        baseline = adapter.name

        def _row(*, metric: str, value: float, status: str,
                 query_time_s: float, metrics_time_s: float,
                 error_msg: str | None) -> CellResult:
            # device / n_parameters / n_nodes / proposal_used are stamped
            # downstream at the cell_worker._emit and runner choke points (same
            # as the other measurements), so they are left at defaults here.
            return CellResult(
                benchmark=benchmark,
                family=family,
                problem_id=problem_id,
                seed=seed,
                baseline=baseline,
                query_role=_PL_QUERY_ROLE,
                metric=metric,
                value=value,
                status=status,
                fit_time_s=fit_time_s,
                query_time_s=query_time_s,
                metrics_time_s=metrics_time_s,
                error_msg=error_msg,
            )

        rows: list[CellResult] = []
        rows.extend(self._log_likelihood_row(problem, adapter, _row))
        rows.extend(self._param_recovery_rows(problem, adapter, _row))
        return rows

    # -----------------------------------------------------------------------
    # log_likelihood (PR 1)
    # -----------------------------------------------------------------------

    def _log_likelihood_row(self, problem, adapter, _row) -> list[CellResult]:
        nan = float("nan")
        if not getattr(adapter, "supports_scoring", False):
            return [_row(
                metric="log_likelihood", value=nan, status="not_supported",
                query_time_s=nan, metrics_time_s=nan,
                error_msg=(
                    f"{adapter.name} does not support parameter-learning "
                    f"scoring (supports_scoring is not set)"
                ),
            )]

        from benchmarking.core.runner import _classify_exception

        t0 = time.perf_counter()
        try:
            log_probs = adapter.score_data(problem.test_data)
        except Exception as exc:
            return [_row(
                metric="log_likelihood", value=nan,
                status=_classify_exception(exc),
                query_time_s=time.perf_counter() - t0, metrics_time_s=nan,
                error_msg=repr(exc),
            )]
        query_time_s = time.perf_counter() - t0

        t_metrics = time.perf_counter()
        result = log_likelihood(torch.as_tensor(log_probs).reshape(-1))
        metrics_time_s = time.perf_counter() - t_metrics

        return [_row(
            metric="log_likelihood", value=result.value, status="ok",
            query_time_s=query_time_s, metrics_time_s=metrics_time_s,
            error_msg=None,
        )]

    # -----------------------------------------------------------------------
    # parameter recovery (PR 2)
    # -----------------------------------------------------------------------

    def _param_recovery_rows(self, problem, adapter, _row) -> list[CellResult]:
        nan = float("nan")

        def _pair(status, tv_val, kl_val, mt, err) -> list[CellResult]:
            return [
                _row(metric="param_recovery_tv", value=tv_val, status=status,
                     query_time_s=nan, metrics_time_s=mt, error_msg=err),
                _row(metric="param_recovery_kl", value=kl_val, status=status,
                     query_time_s=nan, metrics_time_s=mt, error_msg=err),
            ]

        # Gate 1: adapter capability (highest precedence).
        if not getattr(adapter, "supports_param_recovery", False):
            return _pair("not_supported", nan, nan, nan,
                         f"{adapter.name} does not support parameter recovery")

        t0 = time.perf_counter()
        kind, *ctx = self._recovery_context(problem)
        if kind == "not_applicable":
            return _pair("not_applicable", nan, nan,
                         time.perf_counter() - t0, ctx[0])
        if kind == "error":
            return _pair("error", nan, nan,
                         time.perf_counter() - t0, ctx[0])

        # kind == "ok": extract learned CPTs and compute the metrics.
        true_cpts, weights = ctx
        from benchmarking.core.runner import _classify_exception
        try:
            learned = adapter.extract_learned_cpts()
        except Exception as exc:
            return _pair(_classify_exception(exc), nan, nan,
                         time.perf_counter() - t0, repr(exc))

        tc, lc, wt = [], [], []
        for node, true_cpt in true_cpts.items():
            if node not in learned:
                return _pair("error", nan, nan, time.perf_counter() - t0,
                             f"learned CPTs missing node {node!r}")
            if tuple(learned[node].shape) != tuple(true_cpt.shape):
                return _pair(
                    "error", nan, nan, time.perf_counter() - t0,
                    f"CPT shape mismatch for node {node!r}: true "
                    f"{tuple(true_cpt.shape)} vs learned "
                    f"{tuple(learned[node].shape)}",
                )
            tc.append(true_cpt)
            lc.append(learned[node])
            wt.append(weights[node])

        tv = param_recovery_tv(tc, lc, wt)
        kl = param_recovery_kl(tc, lc, wt)
        mt = time.perf_counter() - t0
        return [
            _row(metric="param_recovery_tv", value=tv.value, status="ok",
                 query_time_s=nan, metrics_time_s=mt, error_msg=None),
            _row(metric="param_recovery_kl", value=kl.value, status="ok",
                 query_time_s=nan, metrics_time_s=mt, error_msg=None),
        ]

    # -- per-problem recovery context (cached) -------------------------------

    def _recovery_context(self, problem) -> tuple:
        """Return the baseline-independent recovery context for ``problem``.

        One of:
          ``("ok", true_cpts, weights)`` |
          ``("not_applicable", reason)`` |
          ``("error", message)``

        Cached per problem so the true-CPT extraction + the weight sample run
        once, not once per baseline scored on the problem.
        """
        key = (problem.name, problem.problem_id, problem.seed)
        cached = self._recovery_cache.get(key)
        if cached is not None:
            return cached
        ctx = self._build_recovery_context(problem)
        self._recovery_cache[key] = ctx
        return ctx

    def _build_recovery_context(self, problem) -> tuple:
        if problem.true_model is None:
            return ("not_applicable",
                    "no true_model available for parameter recovery")
        # Fully-discrete check derived from problem.variables (not the dag).
        if not all(kind == "discrete"
                   for kind, _ in problem.variables.values()):
            return ("not_applicable",
                    "parameter recovery is defined for fully-discrete "
                    "networks only")

        true_cpts = extract_discrete_cpts(problem.true_model, problem.variables)
        if not true_cpts:
            return ("not_applicable",
                    "no discrete nodes with discrete parents to recover")

        # Sanity: a true CPT whose rows do not sum to 1 means a buggy
        # generator/loader — surface as error, never silently score it.
        for node, cpt in true_cpts.items():
            row_sums = cpt.sum(dim=-1)
            bad = (row_sums - 1.0).abs() > _CPT_ROW_SUM_TOL
            if bool(bad.any()):
                i = int(bad.nonzero()[0])
                return ("error",
                        f"true CPT for node {node!r} has malformed row {i} "
                        f"(sum={float(row_sums[i]):.6f})")

        weights = self._compute_weights(problem, true_cpts)
        return ("ok", true_cpts, weights)

    def _compute_weights(self, problem, true_cpts) -> dict[str, torch.Tensor]:
        """Empirical parent-config weights per node from a deterministic draw.

        Draws ``N_WEIGHT_SAMPLES`` joint samples from ``true_model`` under a
        seed derived from ``problem.seed`` (so every baseline weights configs
        identically), counts each node's parent configuration in the SAME
        canonical (lex-parent, row-major) index as ``extract_discrete_cpts``,
        and normalizes. ``fork_rng`` isolates the global RNG mutation.
        """
        sample_seed = int(problem.seed) ^ _RECOVERY_SEED_SALT
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(sample_seed)
            sample = problem.true_model.sample(n=self.N_WEIGHT_SAMPLES)

        weights: dict[str, torch.Tensor] = {}
        for node, cpt in true_cpts.items():
            n_configs = int(cpt.shape[0])
            parents = sorted(problem.true_model.dag.parents(node))
            if not parents:
                weights[node] = torch.ones(1, dtype=torch.float64)
                continue
            parent_cards = [int(problem.variables[p][1]) for p in parents]
            # Canonical flat index: first parent slowest (matches the
            # itertools.product order in extract_discrete_cpts).
            idx = torch.zeros(self.N_WEIGHT_SAMPLES, dtype=torch.long)
            stride = 1
            for d in reversed(range(len(parents))):
                vals = sample[parents[d]].reshape(-1).long()
                idx = idx + vals * stride
                stride *= parent_cards[d]
            counts = torch.bincount(idx, minlength=n_configs).to(torch.float64)
            weights[node] = frequency_weights(counts)
        return weights
