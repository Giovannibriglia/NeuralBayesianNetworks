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

import dataclasses
import logging
import sys
from typing import Any, Iterator

from tqdm import tqdm

from benchmarking.core._device import resolve_device
from benchmarking.core.config import BaselineSpec, RunnerConfig, build_adapter
from benchmarking.core.output import JsonlWriter
from benchmarking.core.results import CellResult
from benchmarking.domains.base import FailedProblem
from benchmarking.domains._n_parameters import n_parameters_from_problem

logger = logging.getLogger(__name__)

_NAN = float("nan")


def _estimate_total_cells(cfg: RunnerConfig) -> int | None:
    """Best-effort total cell count (problems × baselines) for the progress
    bar. Returns None (indeterminate counter) when the problem grid can't be
    sized cheaply from the source config without materialising the generator."""
    sc = cfg.source_config
    seeds = getattr(sc, "seeds", None)
    if seeds is None:
        return None
    networks = getattr(sc, "networks", None)
    n_problems = None
    if networks is not None:
        # Bnlearn-style source: one cell-group per (network, seed); each
        # network's family is intrinsic, not a grid dimension.
        n_problems = len(networks) * len(seeds)
    else:
        # Synthetic-style source: grid is families × n_nodes × seeds (#155).
        # Sources without a families attribute fall back to a single family.
        families = getattr(sc, "families", None)
        n_families = len(families) if isinstance(families, (list, tuple)) else 1
        for attr in ("n_nodes_values", "n_values", "n_nodes_list", "n_nodes"):
            vals = getattr(sc, attr, None)
            if isinstance(vals, (list, tuple)):
                n_problems = n_families * len(vals) * len(seeds)
                break
    if n_problems is None:
        return None
    try:
        return n_problems * len(cfg.baselines)
    except TypeError:
        return None

# Bug 4 (#127) Stage 2: fixed buffer (seconds) added on top of the
# fit + query budgets to derive the cell-level subprocess hard timeout.
# Absorbs unbounded metrics time so legitimate cells are never killed;
# only genuine hangs (far beyond the internal budgets) trip the backstop.
_CELL_TIMEOUT_BUFFER_S = 120.0

# Field names of CellResult, used to distinguish a fully-formed result row
# (reconstructable into a CellResult) from a synthesized cell_runner row
# (subprocess kill/timeout/no-output: only status/error_msg present).
_CELLRESULT_FIELDS = frozenset(f.name for f in dataclasses.fields(CellResult))

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
        query_role="",
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
            query_role="",
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
# Subprocess result reconstruction (Bug 4 Stage 2)
# ---------------------------------------------------------------------------

def _rows_to_cellresults(
    row_dicts: list[dict],
    problem: Any,
    spec: BaselineSpec,
    benchmark: str,
) -> Iterator[CellResult]:
    """Reconstruct CellResult objects from the subprocess's row dicts.

    A fully-formed row (the worker's ``dataclasses.asdict`` of a
    CellResult) is rebuilt directly.  A synthesized row from
    ``cell_runner`` (subprocess SIGKILL/timeout/no-output — only
    ``status``/``error_msg`` present) is turned into a sentinel
    CellResult carrying the cell's identity so downstream tooling sees a
    well-formed ``oom``/``timeout``/``error`` row.
    """
    baseline_name: str | None = None
    for d in row_dicts:
        if _CELLRESULT_FIELDS.issubset(d.keys()):
            yield CellResult(**{k: d[k] for k in _CELLRESULT_FIELDS})
        else:
            # Synthesized row — fill in the cell identity the parent knows.
            if baseline_name is None:
                baseline_name = build_adapter(spec).name
            yield CellResult(
                benchmark=benchmark,
                family=getattr(problem, "family", "unknown"),
                problem_id=getattr(problem, "problem_id", "unknown"),
                seed=getattr(problem, "seed", 0),
                baseline=baseline_name,
                query_role="",
                metric="status",
                value=_NAN,
                status=d.get("status", "error"),
                fit_time_s=_NAN,
                query_time_s=_NAN,
                metrics_time_s=_NAN,
                error_msg=d.get("error_msg"),
                # No adapter ran (worker died); record the would-be device.
                device=resolve_device(spec.device),
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
        fit_budget_s = cfg.fit_timeout_s
        default_role = getattr(cfg.selector, "query_role", "random")

        # Per-cell progress bar (problem × baseline). Determinate when the
        # source config exposes its problem grid (networks/n-values × seeds);
        # an indeterminate counter otherwise. Disabled when stdout is not a
        # tty (CI / piped / writing to run.log) so no control chars leak.
        total_cells = _estimate_total_cells(cfg)
        pbar = tqdm(
            total=total_cells, disable=not sys.stdout.isatty(),
            ncols=80, unit="cell", desc="cells",
        )
        try:
            with JsonlWriter(cfg.jsonl_path) as writer:
                for problem in cfg.problem_source.iter_problems(cfg.source_config):
                    # A problem source yields this sentinel when a problem
                    # fails to load (e.g. a download 404).  Record one error
                    # row per baseline so the failure is visible in the
                    # parquet, then continue — the source generator stays
                    # alive and advances to the next problem.
                    if isinstance(problem, FailedProblem):
                        logger.warning(
                            "problem %s failed to load (%s); recording %d error "
                            "rows and skipping", problem.problem_id,
                            problem.error_msg, len(cfg.baselines),
                        )
                        for spec in cfg.baselines:
                            error_row = CellResult(
                                benchmark=problem.benchmark,
                                family=problem.family,
                                problem_id=problem.problem_id,
                                seed=-1,
                                baseline=build_adapter(spec).name,
                                query_role="",
                                metric="status",
                                value=float("nan"),
                                status="error",
                                fit_time_s=float("nan"),
                                query_time_s=float("nan"),
                                metrics_time_s=float("nan"),
                                error_msg=f"problem load failed: {problem.error_msg}",
                                query_kind="",
                                evidence_strategy="",
                                evidence_mode="full",
                                n_parameters=None,
                                # Problem never loaded, so no adapter ran;
                                # record the would-be device for this baseline.
                                device=resolve_device(spec.device),
                            )
                            writer.write(error_row)
                            yield error_row
                        pbar.update(len(cfg.baselines))
                        continue
                    for spec in cfg.baselines:
                        name = build_adapter(spec).name
                        pid = getattr(problem, "problem_id", "?")
                        pbar.set_description(f"fitting {name} on {pid}")
                        # Goes to run.log (INFO); console is WARNING-gated.
                        logger.info("fitting %s on %s (seed=%s)",
                                    name, pid, getattr(problem, "seed", 0))
                        yield from self._run_cell(
                            cfg, problem, spec, writer,
                            fit_budget_s=fit_budget_s,
                            default_role=default_role,
                        )
                        pbar.update(1)
        finally:
            pbar.close()

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
        # Bug 4 (#127) Stage 2: the per-cell work (build_adapter →
        # applicability → select → fit → measure) now runs inside a
        # subprocess via cell_worker._run_cell.  A SIGKILL (e.g. the
        # barley 612 GB OOM) therefore takes down only the worker; the
        # parent records a status row and continues.  Same code paths,
        # same row semantics — just isolated.  See
        # docs/v0.13-runner-subprocess-isolation.md §5.
        from benchmarking.core.cell_runner import run_cell_in_subprocess

        ctx = {
            "problem": problem,
            "baseline_spec": spec,
            "seed": problem.seed,
            "selector": cfg.selector,
            "measurement": cfg.measurement,
            "benchmark": cfg.benchmark,
            "n_queries_per_cell": cfg.n_queries_per_cell,
            "per_cell_timeout_s": cfg.per_cell_timeout_s,
            "fit_budget_s": fit_budget_s,
            "default_role": default_role,
        }

        # Cell-level hard timeout is a backstop for hangs the worker
        # cannot self-limit (C-level loop, allocator thrash).  It must be
        # generous enough not to fire on legitimate cells, whose internal
        # budgets are fit_budget_s (fit) + per_cell_timeout_s (queries)
        # plus unbounded metrics time — so we add a fixed buffer on top.
        cell_timeout_s = fit_budget_s + cfg.per_cell_timeout_s + _CELL_TIMEOUT_BUFFER_S

        result = run_cell_in_subprocess(ctx, timeout_s=cell_timeout_s)

        # Persist the cell subprocess's captured stderr to the run.log (INFO,
        # so it stays off the WARNING-gated console). The subprocess is a
        # separate process whose logs would otherwise be lost; this is the
        # only place they re-enter the parent's logging. Bug 4 isolation is
        # untouched — this is the already-captured stream, read post-exit.
        stderr_text = (getattr(result, "stderr", "") or "").strip()
        if stderr_text:
            logger.info("cell subprocess stderr [%s/%s]:\n%s",
                        getattr(problem, "problem_id", "?"),
                        build_adapter(spec).name, stderr_text)

        # #133: enrich every row with the problem's n_parameters (a tighter,
        # cardinality-aware cost proxy than n_nodes; paper figures §5.4b).
        # Computed once per cell from problem.variables + problem.dag and
        # injected here -- the single choke point all rows (measurement rows,
        # every sentinel, subprocess-reconstructed rows) pass through -- rather
        # than threaded through the ~12 CellResult construction sites.
        n_params = n_parameters_from_problem(problem)

        for row in _rows_to_cellresults(result.rows, problem, spec, cfg.benchmark):
            if n_params is not None and row.n_parameters is None:
                row = dataclasses.replace(row, n_parameters=n_params)
            writer.write(row)
            yield row
