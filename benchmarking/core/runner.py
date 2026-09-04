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

import atexit
import dataclasses
import logging
import shutil
import sys
from typing import Any, Iterator

from tqdm import tqdm

from benchmarking.core._device import resolve_device
from benchmarking.core.config import BaselineSpec, RunnerConfig, build_adapter
from benchmarking.core.output import JsonlWriter
from benchmarking.core.results import CellResult
from benchmarking.domains.base import FailedProblem
from benchmarking.domains._n_parameters import (
    n_nodes_from_problem,
    n_parameters_from_problem,
    n_train_from_problem,
)

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
# Speed-benchmark seed-skip (v0.14, #148 PR 2/2)
# ---------------------------------------------------------------------------
# Once a (family, problem_id, baseline, batch_size) config fails on one seed,
# its remaining seeds are skipped — they are recorded as skip sentinels rather
# than re-run. Speed-benchmark-only: gated on cfg.batch_sizes being set (no
# other benchmark sweeps batch sizes, so every accuracy run is untouched).
# Mirrors PR #189's reporting rule ("any failed seed -> config failed"); see
# docs/v0.14-batched-queries-design.md.

_FAILURE_STATUSES = ("oom", "timeout", "error")


def _skip_sentinel(
    problem: Any, baseline_name: str, benchmark: str, batch_size: int,
    status: str, *, device: str | None,
) -> CellResult:
    """One sentinel row for a (baseline, batch_size) seed skipped after the
    config already failed on an earlier seed. Carries the propagated failure
    code (status) and the correct batch_size so PR #189's table/sidecar treat
    it identically to a real failure."""
    return CellResult(
        benchmark=benchmark,
        family=problem.family,
        problem_id=problem.problem_id,
        seed=problem.seed,
        baseline=baseline_name,
        query_role="",
        metric="status",
        value=_NAN,
        status=status,
        fit_time_s=_NAN,
        query_time_s=_NAN,
        metrics_time_s=_NAN,
        error_msg="skipped: config failed on an earlier seed",
        batch_size=int(batch_size),
        device=device,
    )


def _register_failures(
    registry: dict, problem: Any, baseline_name: str,
    attempted_batch_sizes: list[int], rows: list[CellResult],
) -> None:
    """Update the seed-skip registry from one cell's result rows.

    A config ``(family, problem_id, baseline, B)`` is registered as failed (with
    a propagated code) when ``B`` either has a genuine failure row of its own, or
    produced no row at all while the cell did fail somewhere. This covers:
      * per-batch_size query failure  -> that B registered (matches PR #189's
        "any failure row -> config failed", even if B also has some ok rows);
      * fit failure (every B fails)   -> all attempted B registered;
      * wholesale subprocess death    -> one synthesized failure row (B=1) and
        no rows for the others -> all attempted B registered via the fallback.
    A ``not_supported``-only cell has no failure row, so nothing registers —
    later seeds re-emit not_supported normally (and stay '--' in the table)."""
    fail_rows = [r for r in rows if r.status in _FAILURE_STATUSES]
    if not fail_rows:
        return
    code_by_bs: dict[int, str] = {}
    for r in fail_rows:
        code_by_bs.setdefault(r.batch_size, r.status)
    # Batch sizes that produced any normal (ok or failure) row this cell. An
    # attempted B absent here means the subprocess died before emitting it
    # (wholesale OOM/timeout) -> treat as failed via the fallback code.
    covered = {r.batch_size for r in rows
               if r.status == "ok" and r.metric == "query_time_s"}
    covered |= set(code_by_bs)
    fallback = max(set(r.status for r in fail_rows),
                   key=lambda s: sum(r.status == s for r in fail_rows))
    for b in attempted_batch_sizes:
        if b in code_by_bs:
            code = code_by_bs[b]
        elif b not in covered:
            code = fallback
        else:
            continue  # B ran ok and had no failure of its own
        registry[(problem.family, problem.problem_id, baseline_name, b)] = code


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
# Batch-size sweep resolution (v0.14 fit-once query-many, #174)
# ---------------------------------------------------------------------------

def _resolve_batch_sizes(
    cfg: RunnerConfig, spec: BaselineSpec, adapter: Any | None = None,
) -> list[int]:
    """Return the batch_sizes this (cfg, baseline) cell sweeps.

    Design doc §3.2 / §5.4. The cell worker fits once and loops
    ``measure()`` over the returned list, so this is where the
    swept-vs-pinned decision (formerly ``cli._run_cells``) now lives:

    * No top-level ``cfg.batch_sizes`` → ``[spec.batch_size]`` (length 1):
      identity behavior for every non-sweep config (bnlearn, scalability,
      smoke).
    * Explicit YAML pin (``batch_size_pinned``) → ``[spec.batch_size]``:
      a pinned baseline runs once at its pinned value, even if its adapter
      could batch (explicit pin always wins, §5.6).
    * Adapter supports batching and is unpinned → ``cfg.batch_sizes``: the
      baseline is swept across every value, fit once.
    * Adapter does not support batching → ``[spec.batch_size]`` (length 1).

    ``adapter`` may be passed to avoid a redundant ``build_adapter`` when
    the caller already has the instance.
    """
    if not cfg.batch_sizes:
        return [getattr(spec, "batch_size", 1)]
    if spec.batch_size_pinned:
        return [spec.batch_size]
    if adapter is None:
        adapter = build_adapter(spec)
    if getattr(adapter, "supports_batched_queries", False):
        return list(cfg.batch_sizes)
    return [spec.batch_size]


# ---------------------------------------------------------------------------
# Fit-once-save-reload (v0.14, #191 Path 2)
# ---------------------------------------------------------------------------
# Baselines sharing a fit-identity (same library/mechanism/epochs/fit-data,
# differing ONLY in inference_method) fit the base model ONCE: the first writes
# it with torch.save, the rest reload it instead of re-fitting. nbn-only; the
# base model.fit() is bitwise-identical across all nbn engines (Stage-1
# verified). pgmpy/pomegranate/pyro have no shareable NBN model and run
# standalone exactly as before.


def _fit_identity_key(spec: BaselineSpec, problem: Any) -> tuple:
    """The key deciding which baselines share a base fit (LOCKED fields).

    ``(library, mechanism, epochs, batch_size, lr, family, problem_id, seed)``.
    This is the SINGLE source of truth — both grouping (`_assign_fit_roles`)
    and the cache filename (`_fit_cache_filename`) derive from it; never
    recompute inline.

    Excludes:
      * ``inference_method`` — the axis we share across (the whole point);
      * ``n_samples`` — affects only the engine, not ``model.fit()``;
      * ``device`` — reload relocates via ``map_location`` (Stage-1 CPU<->CUDA
        faithful), so two baselines differing only in device still share.
    ``epochs``/``batch_size``/``lr`` are the ``extra_kwargs`` entries that
    affect ``model.fit()`` today; if a future kwarg does, it MUST be added
    here. ``None`` (absent from config) means "mechanism-designed budget" and
    is a DISTINCT identity from any explicit value — do not collapse it to a
    numeric default.
    """
    extra = spec.extra_kwargs or {}
    epochs = extra.get("epochs")
    epochs = int(epochs) if epochs is not None else None
    batch_size = extra.get("batch_size")
    batch_size = int(batch_size) if batch_size is not None else None
    lr = extra.get("lr")
    lr = float(lr) if lr is not None else None
    return (
        spec.library, spec.mechanism, epochs, batch_size, lr,
        getattr(problem, "family", ""),
        getattr(problem, "problem_id", ""),
        getattr(problem, "seed", 0),
    )


def _fit_cache_filename(key: tuple) -> str:
    """Stable hash of the fit-identity key -> cache filename. Derives from the
    same key as grouping, so the fitter's save path and a reloader's load path
    can never drift."""
    import hashlib

    h = hashlib.sha1("|".join(map(str, key)).encode()).hexdigest()[:16]
    return f"fit_{h}.pt"


def _assign_fit_roles(
    cfg: RunnerConfig, problem: Any, failed_configs: dict, seed_skip: bool,
    fitcache_dir,
) -> tuple[list[tuple[str, Any]], dict[int, Any]]:
    """Assign a fit-once-save-reload role to each baseline for this problem.

    Returns ``(roles, delete_after)`` where:
      * ``roles[i] = (role, cache_path)`` for baseline index ``i``; role is
        ``"fit"`` (fit + save), ``"reload"`` (reuse the group's saved base), or
        ``"standalone"`` (fit, no save — singletons and every non-nbn baseline);
        ``cache_path`` is None for standalone.
      * ``delete_after[i] = cache_path`` marks that, after baseline ``i``'s cell
        returns, the parent eagerly deletes that cache file (``i`` is its
        group's LAST live member).

    Grouping is over the POST-seed-skip LIVE nbn baselines: a baseline whose
    every batch_size already failed on an earlier seed is fully skipped (not
    live) and never designated fitter. This is why a seed-skipped fitter cannot
    strand its reloaders — the first LIVE member becomes the fitter.

    Validity across the spec loop: ``failed_configs`` is keyed by baseline NAME,
    and within one problem each baseline has a distinct name, so the failures
    `_register_failures` adds mid-loop only affect the SAME baseline on a LATER
    seed — never another baseline's skip status this problem. So this pre-pass,
    computed once at problem start, stays correct for the whole loop.

    Adjacency is NOT required: live members sharing a key are bucketed wherever
    they sit. Every real config lists them consecutively, so at most one cache
    file exists at a time; an interleaved config would merely hold more than one
    concurrently (still leak-proof via eager delete + the run()-level cleanup).
    """
    n = len(cfg.baselines)
    roles: list[tuple[str, Any]] = [("standalone", None)] * n
    delete_after: dict[int, Any] = {}
    live_by_key: dict[tuple, list[int]] = {}

    for i, spec in enumerate(cfg.baselines):
        if spec.library != "nbn":
            continue  # non-nbn never groups/caches (standalone)
        adapter = build_adapter(spec)
        name = adapter.name
        batch_sizes = _resolve_batch_sizes(cfg, spec, adapter)
        if seed_skip:
            surviving = [
                b for b in batch_sizes
                if (problem.family, problem.problem_id, name, b)
                not in failed_configs
            ]
            if not surviving:
                continue  # fully seed-skipped -> not a live member
        live_by_key.setdefault(_fit_identity_key(spec, problem), []).append(i)

    for key, idxs in live_by_key.items():
        if len(idxs) < 2:
            continue  # singleton group -> standalone, nothing to save/reload
        cache_path = fitcache_dir() / _fit_cache_filename(key)
        roles[idxs[0]] = ("fit", cache_path)
        for j in idxs[1:]:
            roles[j] = ("reload", cache_path)
        delete_after[idxs[-1]] = cache_path

    return roles, delete_after


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

        # Speed-benchmark seed-skip (#148 PR 2/2): only the batch_sizes-sweep
        # (speed) config sets cfg.batch_sizes; every other benchmark leaves it
        # None and is untouched. The registry maps a failed config
        # (family, problem_id, baseline, batch_size) -> propagated failure code.
        seed_skip = bool(cfg.batch_sizes)
        failed_configs: dict[tuple, str] = {}

        # Fit-once-save-reload cache (#191 Path 2). nbn baselines sharing a
        # fit-identity reuse a saved base model. The cache lives under the run
        # dir (cfg.jsonl_path is run_dir/metrics.jsonl), which is timestamp-
        # unique — so a stale cache from a crashed prior run lives under a
        # different dir and can never be reused. Created LAZILY (only when a
        # group of size > 1 first needs it) so non-grouped runs make no empty
        # dir. Each file is deleted eagerly after its group's last live member
        # (below); this atexit + the finally sweep are the leak-proof backstop.
        fitcache_root = cfg.jsonl_path.parent / "_fitcache"
        _fitcache_made = {"done": False}

        def fitcache_dir():
            if not _fitcache_made["done"]:
                fitcache_root.mkdir(parents=True, exist_ok=True)
                _fitcache_made["done"] = True
            return fitcache_root

        def _sweep_fitcache():
            try:
                shutil.rmtree(fitcache_root)
            except (FileNotFoundError, OSError):
                pass

        atexit.register(_sweep_fitcache)

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
                                n_nodes=None,
                                # Problem never loaded, so no adapter ran;
                                # record the would-be device for this baseline.
                                device=resolve_device(spec.device),
                            )
                            writer.write(error_row)
                            yield error_row
                        pbar.update(len(cfg.baselines))
                        continue
                    # Fit-once-save-reload (#191): assign each nbn baseline a
                    # fit / reload / standalone role for this problem. Computed
                    # ONCE here from the post-seed-skip live members; valid for
                    # the whole spec loop (see _assign_fit_roles). non-nbn and
                    # singleton groups -> standalone (unchanged behavior).
                    fit_roles, fit_delete_after = _assign_fit_roles(
                        cfg, problem, failed_configs, seed_skip, fitcache_dir,
                    )
                    for i, spec in enumerate(cfg.baselines):
                        adapter_probe = build_adapter(spec)
                        name = adapter_probe.name
                        # v0.14 fit-once query-many (#174): resolve the
                        # batch_sizes this cell sweeps. The swept-vs-pinned
                        # decision (formerly cli._run_cells) lives here now,
                        # alongside the cell key — reusing the probe adapter
                        # already built for the name so we don't build twice.
                        batch_sizes = _resolve_batch_sizes(cfg, spec, adapter_probe)
                        pid = getattr(problem, "problem_id", "?")

                        # Seed-skip (#148 PR 2/2): for any batch size whose config
                        # already failed on an earlier seed, emit a skip sentinel
                        # (propagated code) instead of re-running it. The cell only
                        # runs the still-live batch sizes; if none remain, the
                        # subprocess is skipped entirely.
                        if seed_skip:
                            to_skip = [
                                b for b in batch_sizes
                                if (problem.family, problem.problem_id, name, b)
                                in failed_configs
                            ]
                            if to_skip:
                                dev = resolve_device(spec.device)
                                n_params = n_parameters_from_problem(problem)
                                n_nodes = n_nodes_from_problem(problem)
                                n_train = n_train_from_problem(problem)
                                for b in to_skip:
                                    code = failed_configs[
                                        (problem.family, problem.problem_id, name, b)
                                    ]
                                    row = _skip_sentinel(
                                        problem, name, cfg.benchmark, b, code,
                                        device=dev,
                                    )
                                    if n_params is not None and row.n_parameters is None:
                                        row = dataclasses.replace(
                                            row, n_parameters=n_params)
                                    if n_nodes is not None and row.n_nodes is None:
                                        row = dataclasses.replace(
                                            row, n_nodes=n_nodes)
                                    if n_train is not None and row.n_train is None:
                                        row = dataclasses.replace(
                                            row, n_train=n_train)
                                    writer.write(row)
                                    yield row
                                logger.info(
                                    "seed-skip %s on %s (seed=%s): skipped B=%s",
                                    name, pid, getattr(problem, "seed", 0), to_skip,
                                )
                                batch_sizes = [b for b in batch_sizes
                                               if b not in to_skip]
                                if not batch_sizes:
                                    pbar.update(1)
                                    continue

                        # Fit-once-save-reload role for this baseline (#191):
                        # "fit" (fit + save base), "reload" (reuse the group's
                        # saved base), or "standalone" (fit, no save).
                        fit_role, cache_path = fit_roles[i]
                        verb = {"reload": "reloading"}.get(fit_role, "fitting")
                        # The resolved device is part of the progress line so a
                        # run.log / console reader can tell at a glance whether
                        # this cell landed on cuda or cpu (per-baseline YAML
                        # pins override --device, so it is not a run-wide fact).
                        dev = resolve_device(spec.device)
                        pbar.set_description(f"{verb} {name} [{dev}] on {pid}")
                        # Goes to run.log (INFO); console is WARNING-gated.
                        logger.info("%s %s [%s] on %s (seed=%s)",
                                    verb, name, dev, pid, getattr(problem, "seed", 0))
                        cell_rows: list[CellResult] = []
                        for row in self._run_cell(
                            cfg, problem, spec, writer,
                            fit_budget_s=fit_budget_s,
                            default_role=default_role,
                            batch_sizes=batch_sizes,
                            fit_role=fit_role,
                            cache_path=cache_path,
                        ):
                            cell_rows.append(row)
                            yield row
                        if seed_skip:
                            _register_failures(
                                failed_configs, problem, name,
                                batch_sizes, cell_rows,
                            )
                        pbar.update(1)
                        # Eager cache deletion (#191): this baseline is its
                        # group's LAST live member, so every reloader has now
                        # reused the saved base — delete it immediately rather
                        # than waiting for the run-end sweep. Keeps at most one
                        # cache file on disk at a time for adjacent configs.
                        if i in fit_delete_after:
                            try:
                                fit_delete_after[i].unlink()
                            except (FileNotFoundError, OSError):
                                pass
        finally:
            pbar.close()
            # Leak-proof backstop: remove the whole cache dir even on crash /
            # early generator close, then drop the atexit hook (the Runner is
            # reusable, so a stale hook from a prior run must not linger).
            _sweep_fitcache()
            atexit.unregister(_sweep_fitcache)

    def _run_cell(
        self,
        cfg: RunnerConfig,
        problem: Any,
        spec: BaselineSpec,
        writer: JsonlWriter,
        *,
        fit_budget_s: float,
        default_role: str,
        batch_sizes: list[int] | None = None,
        fit_role: str = "standalone",
        cache_path: Any | None = None,
    ) -> Iterator[CellResult]:
        # Bug 4 (#127) Stage 2: the per-cell work (build_adapter →
        # applicability → select → fit → measure) now runs inside a
        # subprocess via cell_worker._run_cell.  A SIGKILL (e.g. the
        # barley 612 GB OOM) therefore takes down only the worker; the
        # parent records a status row and continues.  Same code paths,
        # same row semantics — just isolated.  See
        # docs/v0.13-runner-subprocess-isolation.md §5.
        from benchmarking.core.cell_runner import run_cell_in_subprocess

        # v0.14 fit-once query-many (#174): the batch_sizes this cell sweeps.
        # The caller (run()) resolves it per-baseline; fall back to the
        # spec's own batch_size for direct callers (tests) — a length-1
        # sweep, identity behavior.
        if batch_sizes is None:
            batch_sizes = [getattr(spec, "batch_size", 1)]

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
            "batch_sizes": batch_sizes,
            # Fit-once-save-reload (#191): the worker fits + saves ("fit"),
            # reloads the group's saved base ("reload", falling back to fit on
            # a cache miss), or fits without saving ("standalone"). cache_path
            # is None for standalone. str() so the Path pickles cleanly into
            # the subprocess ctx and the worker's os.path checks are simple.
            "fit_role": fit_role,
            "cache_path": str(cache_path) if cache_path is not None else None,
        }

        # Cell-level hard timeout is a backstop for hangs the worker
        # cannot self-limit (C-level loop, allocator thrash).  It must be
        # generous enough not to fire on legitimate cells. The worker fits
        # ONCE (fit_budget_s) then runs one measure() pass per batch_size,
        # each bounded by per_cell_timeout_s (design doc §4.1 / §8.4) — so a
        # swept cell of N batch_sizes needs N × per_cell_timeout_s of query
        # budget. The fixed buffer absorbs unbounded metrics time on top.
        cell_timeout_s = (
            fit_budget_s
            + len(batch_sizes) * cfg.per_cell_timeout_s
            + _CELL_TIMEOUT_BUFFER_S
        )

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
        # than threaded through the ~12 CellResult construction sites. n_nodes
        # (variable count) is injected the same way, in parallel.
        n_params = n_parameters_from_problem(problem)
        n_nodes = n_nodes_from_problem(problem)
        n_train = n_train_from_problem(problem)

        for row in _rows_to_cellresults(result.rows, problem, spec, cfg.benchmark):
            if n_params is not None and row.n_parameters is None:
                row = dataclasses.replace(row, n_parameters=n_params)
            if n_nodes is not None and row.n_nodes is None:
                row = dataclasses.replace(row, n_nodes=n_nodes)
            if n_train is not None and row.n_train is None:
                row = dataclasses.replace(row, n_train=n_train)
            writer.write(row)
            yield row
