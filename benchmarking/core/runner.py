"""v0.13 runner orchestrator.

Wires ProblemSource × QuerySelector × Measurement × BaselineAdapter into a
runnable cell loop.  Single-process, no subprocess isolation.  Soft cumulative
timeout on query_time_s (per the v0.13 doc §3); fit and metrics are excluded
from the timeout budget.

Cell lifecycle per (problem, baseline):
  1. ``build_adapter(spec)``
  2. ``adapter.is_applicable(problem)``   → not_supported sentinel if False
  3. ``selector.select(problem, n_queries, problem.seed)``
  4. ``adapter.fit(problem, **spec.extra_kwargs)``
     → error/oom/timeout sentinel rows on failure or safety-net breach
  5. ``measurement.measure(..., query_budget_s=per_cell_timeout_s)``
     → rows including timeout rows for over-budget queries

JSONL output is written row-by-row (streaming, crash-resilient) via
``JsonlWriter``.

Reference: docs/v0.13-benchmark-redesign.md §3, §4.1, §6
"""
from __future__ import annotations

from time import perf_counter
from typing import Any, Iterator

from benchmarking.core.config import BaselineSpec, RunnerConfig, build_adapter
from benchmarking.core.output import JsonlWriter
from benchmarking.core.results import CellResult

_NAN = float("nan")

# ---------------------------------------------------------------------------
# OOM detection: RuntimeError message substrings from torch's CPU allocator
# and various CUDA allocator error paths.
# ---------------------------------------------------------------------------
_OOM_RUNTIME_MARKERS = (
    "out of memory",
    "cuda oom",
    "alloc_cpu",
    "defaultcpuallocator",
    "cannot allocate memory",
    "can't allocate memory",
)

# ---------------------------------------------------------------------------
# Structural-limit markers for ValueError classification (#117).
# A ValueError whose message contains one of these substrings is a deliberate
# "I don't support this combination" signal from an adapter.  Any other
# ValueError is a genuine programming error → "error" status.
# Ported from benchmarking/_crash_test_utils.py::_STRUCTURAL_LIMIT_MARKERS.
# ---------------------------------------------------------------------------
_STRUCTURAL_LIMIT_MARKERS = (
    "only supports",
    "is discrete-only",
    "cannot condition on",
    "not yet wired",
    "non-Gaussian",
    "not applicable to",
    "refused",
)


def _is_structural_limit(exc: Exception) -> bool:
    """True iff exc is a ValueError raised for structural (not-supported) reasons."""
    if not isinstance(exc, ValueError):
        return False
    msg = str(exc)
    return any(marker in msg for marker in _STRUCTURAL_LIMIT_MARKERS)


# ---------------------------------------------------------------------------
# Exception → status classifier
# ---------------------------------------------------------------------------

def _classify_exception(exc: Exception) -> str:
    """Map an exception to a CellResult status string.

    Returns one of: "oom", "not_supported", "error".

    OOM detection covers:
      - torch.cuda.OutOfMemoryError (explicit CUDA OOM)
      - MemoryError (Python / OS CPU OOM)
      - RuntimeError with OOM-marker substrings (torch CPU allocator)

    Structural-limit detection covers:
      - ImportError (optional dependency missing)
      - NotImplementedError (adapter refuses the combination)
      - ValueError whose message matches _STRUCTURAL_LIMIT_MARKERS (#117)

    ValueError without a structural-limit marker → "error" (real bug).
    All other exceptions → "error".
    """
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return "oom"
    except (ImportError, AttributeError):
        pass

    if isinstance(exc, MemoryError):
        return "oom"

    if isinstance(exc, (ImportError, NotImplementedError)):
        return "not_supported"

    if isinstance(exc, ValueError):
        return "not_supported" if _is_structural_limit(exc) else "error"

    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if any(m in msg for m in _OOM_RUNTIME_MARKERS):
            return "oom"

    return "error"


# ---------------------------------------------------------------------------
# Per-query metadata helpers
# ---------------------------------------------------------------------------

def _evidence_mode_for(q: Any) -> str:
    """Return ``"empty"`` if any evidence value is ``None``, else ``"full"``.

    Phase 3: HeaviestQueryByRole emits paired full/empty queries. A query is
    empty-mode when its evidence variables are structurally specified but
    unobserved (``None`` values, marginalized by the adapter/oracle). Defends
    against mixed dicts by treating any ``None`` as empty-mode.
    """
    return "empty" if any(v is None for v in q.evidence.values()) else "full"


# ---------------------------------------------------------------------------
# Sentinel row helpers
# ---------------------------------------------------------------------------

def _not_supported_sentinel(
    problem: Any, adapter: Any, benchmark: str,
) -> CellResult:
    """Single sentinel row when ``adapter.is_applicable()`` returns False."""
    return CellResult(
        benchmark=benchmark,
        family=problem.family,
        problem_id=problem.problem_id,
        seed=problem.seed,
        baseline=adapter.name,
        query_role="random",
        metric="status",
        value=_NAN,
        status="not_supported",
        fit_time_s=0.0,
        query_time_s=0.0,
        metrics_time_s=0.0,
        error_msg=f"{adapter.name} not applicable to {problem.family}",
    )


def _fit_failure_rows(
    problem: Any,
    adapter: Any,
    queries: list,
    query_roles: list[str],
    benchmark: str,
    *,
    fit_time_s: float,
    status: str,
    error_msg: str,
) -> Iterator[CellResult]:
    """Emit one sentinel row per selected query when fit() fails or breaches
    the safety budget.

    ``fit_time_s`` is NaN if fit raised an exception before completing, or the
    real wall-clock if fit completed but exceeded the multiplier ceiling.
    ``query_time_s`` and ``metrics_time_s`` are always NaN (no queries ran).
    """
    if not queries:
        yield CellResult(
            benchmark=benchmark,
            family=problem.family,
            problem_id=problem.problem_id,
            seed=problem.seed,
            baseline=adapter.name,
            query_role="random",
            metric="status",
            value=_NAN,
            status=status,
            fit_time_s=fit_time_s,
            query_time_s=_NAN,
            metrics_time_s=_NAN,
            error_msg=error_msg,
        )
        return

    for q, role in zip(queries, query_roles):
        yield CellResult(
            benchmark=benchmark,
            family=problem.family,
            problem_id=problem.problem_id,
            seed=problem.seed,
            baseline=adapter.name,
            query_role=role,
            query_kind=getattr(q, "query_kind", "prediction"),
            evidence_strategy=getattr(q, "evidence_strategy", "random"),
            evidence_mode=_evidence_mode_for(q),
            metric="status",
            value=_NAN,
            status=status,
            fit_time_s=fit_time_s,
            query_time_s=_NAN,
            metrics_time_s=_NAN,
            error_msg=error_msg,
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    """Orchestrate the v0.13 cell loop.

    Usage::

        rows = list(Runner().run(cfg))

    Yields ``CellResult`` rows as they are produced.  Simultaneously writes
    each row to ``cfg.jsonl_path`` for crash resilience.

    The runner is stateless; a single ``Runner`` instance can be reused
    across multiple configs.
    """

    def run(self, cfg: RunnerConfig) -> Iterator[CellResult]:
        """Run the full cell loop, yielding CellResult rows as produced.

        Also writes each row to ``cfg.jsonl_path`` immediately (streaming).

        Parameters
        ----------
        cfg:
            A fully-constructed ``RunnerConfig``.

        Yields
        ------
        CellResult
            One per (problem, baseline, query, metric).  Includes
            not_supported, timeout, error, and oom sentinel rows.
        """
        fit_budget_s = (
            cfg.fit_timeout_s
            if cfg.fit_timeout_s is not None
            else cfg.fit_timeout_s_multiplier * cfg.per_cell_timeout_s
        )
        default_role = getattr(cfg.selector, "query_role", "random")

        with JsonlWriter(cfg.jsonl_path) as writer:
            for problem in cfg.problem_source.iter_problems(cfg.source_config):
                for spec in cfg.baselines:
                    yield from self._run_cell(
                        cfg, problem, spec, writer,
                        fit_budget_s=fit_budget_s,
                        default_role=default_role,
                    )

    def _run_cell(
        self,
        cfg: RunnerConfig,
        problem: Any,
        spec: BaselineSpec,
        writer: JsonlWriter,
        *,
        fit_budget_s: float,
        default_role: str,
    ) -> Iterator[CellResult]:
        adapter = build_adapter(spec)

        # --- Applicability gate ---
        if not adapter.is_applicable(problem):
            row = _not_supported_sentinel(problem, adapter, cfg.benchmark)
            writer.write(row)
            yield row
            return

        # --- Query selection (before fit; seed from problem generation) ---
        queries = cfg.selector.select(problem, cfg.n_queries_per_cell, problem.seed)
        # Per-query metadata travels on the Query objects (Phase 2). Selectors
        # that don't set these fields fall back to the dataclass defaults; the
        # selector class-level query_role remains the fallback for query_role.
        query_roles = [getattr(q, "query_role", default_role) for q in queries]
        query_kinds = [getattr(q, "query_kind", "prediction") for q in queries]
        evidence_strategies = [
            getattr(q, "evidence_strategy", "random") for q in queries
        ]
        evidence_modes = [_evidence_mode_for(q) for q in queries]

        # --- Fit ---
        try:
            t0 = perf_counter()
            adapter.fit(problem, **spec.extra_kwargs)
            fit_time_s = perf_counter() - t0
        except Exception as exc:
            status = _classify_exception(exc)
            for row in _fit_failure_rows(
                problem, adapter, queries, query_roles, cfg.benchmark,
                fit_time_s=_NAN, status=status, error_msg=repr(exc),
            ):
                writer.write(row)
                yield row
            return

        # --- Fit safety net ---
        if fit_time_s > fit_budget_s:
            for row in _fit_failure_rows(
                problem, adapter, queries, query_roles, cfg.benchmark,
                fit_time_s=fit_time_s, status="timeout",
                error_msg=f"fit exceeded {fit_budget_s:.0f}s safety budget",
            ):
                writer.write(row)
                yield row
            return

        # --- Measurement (handles per-query cumulative timeout internally) ---
        rows = cfg.measurement.measure(
            problem, adapter, queries,
            fit_time_s=fit_time_s,
            benchmark=cfg.benchmark,
            seed=problem.seed,
            query_roles=query_roles,
            query_kinds=query_kinds,
            evidence_strategies=evidence_strategies,
            evidence_modes=evidence_modes,
            query_budget_s=cfg.per_cell_timeout_s,
        )
        for row in rows:
            writer.write(row)
            yield row
