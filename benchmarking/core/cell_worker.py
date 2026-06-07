"""Cell worker subprocess (Bug 4 of #127).

Runs one (problem, baseline_spec, seed) cell. Reads pickled input from
argv[1], writes JSONL results to argv[2] with per-row flush so partial
data survives mid-cell death.

This module is imported but NOT YET wired into the runner. Stage 2 of
the Bug 4 PR series replaces the in-process measurement call with a
subprocess invocation of this module.

See docs/v0.13-runner-subprocess-isolation.md §4.2.
"""
from __future__ import annotations

import json
import pickle
import sys
import traceback


def main(input_path: str, output_path: str) -> int:
    """Entry point. Returns 0 on success, 1 on error.

    The exit code itself doesn't carry status semantics for the parent
    — the parent reads the output JSONL to determine per-row status.
    Non-zero exit code combined with no output rows = synthetic error
    row in the parent.
    """
    # Bug 4 (#127): apply the per-cell memory cap before any heavy
    # work. The cap is set by cell_runner via NBN_CELL_MEMORY_LIMIT_BYTES;
    # if absent (e.g., direct test invocation), proceed unbounded.
    _apply_memory_limit()

    try:
        with open(input_path, "rb") as f:
            ctx = pickle.load(f)
    except Exception:
        # Couldn't even read input — minimal error report
        with open(output_path, "w") as out:
            out.write(json.dumps({
                "status": "error",
                "error_msg": f"cell_worker: failed to read input: "
                             f"{traceback.format_exc()[:500]}"
            }) + "\n")
        return 1

    # Open output for line-buffered writing (per-row flush below ensures
    # partial data survives a mid-cell death).
    with open(output_path, "w", buffering=1) as out:
        try:
            rows = _run_cell(ctx)
            for row in rows:
                out.write(json.dumps(row) + "\n")
                out.flush()
            return 0
        except Exception:
            # Catch-all so the subprocess always writes SOMETHING
            out.write(json.dumps({
                "status": "error",
                "error_msg": traceback.format_exc()[:1000]
            }) + "\n")
            out.flush()
            return 1


def _apply_memory_limit() -> None:
    """Apply the per-cell virtual memory cap from the parent.

    Reads NBN_CELL_MEMORY_LIMIT_BYTES (set by cell_runner) and calls
    resource.setrlimit(RLIMIT_AS, ...). Failures are swallowed:
    this is a defense-in-depth measure, not a correctness requirement.
    """
    import os

    raw = os.environ.get("NBN_CELL_MEMORY_LIMIT_BYTES")
    if not raw:
        return

    try:
        limit_bytes = int(raw)
    except ValueError:
        return

    try:
        import resource
        # Both soft and hard limit. Hard limit prevents the subprocess
        # from raising its own cap via setrlimit.
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ImportError, OSError, ValueError):
        # ImportError: not on POSIX (Windows lacks `resource`)
        # OSError: existing limit is already lower, or insufficient
        #   privilege — proceed without raising
        # ValueError: limit is invalid
        return


def _run_cell(ctx: dict) -> list[dict]:
    """Run one (problem, baseline, seed) cell.

    This is the per-cell logic that lived inline in
    ``benchmarking.core.runner.Runner._run_cell`` before Bug 4. Same
    semantics — build adapter, applicability gate, query selection, fit
    (with exception/safety-net classification), measurement with the
    Phase 3 per-query soft timeout — just running inside a subprocess.

    The cell helpers (``_not_supported_sentinel``, ``_fit_failure_rows``,
    ``_classify_exception``, ``_evidence_mode_for``) still live in the
    runner module; they are imported here rather than duplicated.

    Returns a list of plain dicts (``dataclasses.asdict`` of each
    ``CellResult``). The parent reconstructs ``CellResult`` objects.

    See docs/v0.13-runner-subprocess-isolation.md §5.
    """
    import dataclasses
    from time import perf_counter

    from benchmarking.core.config import build_adapter
    from benchmarking.core.runner import (
        _NAN,
        _classify_exception,
        _evidence_mode_for,
        _fit_failure_rows,
        _not_supported_sentinel,
    )

    problem = ctx["problem"]
    spec = ctx["baseline_spec"]
    selector = ctx["selector"]
    measurement = ctx["measurement"]
    benchmark = ctx["benchmark"]
    n_queries_per_cell = ctx["n_queries_per_cell"]
    per_cell_timeout_s = ctx["per_cell_timeout_s"]
    fit_budget_s = ctx["fit_budget_s"]
    default_role = ctx["default_role"]

    def _emit(results) -> list[dict]:
        # Stamp the resolved device on every row at this single choke point
        # (same pattern as the runner's n_parameters injection) rather than
        # threading it through the ~10 CellResult construction sites in the
        # measurements and sentinel helpers. `adapter` is always built before
        # any _emit() call (late-binding closure).
        dev = str(getattr(adapter, "device", "cpu"))
        return [
            dataclasses.asdict(dataclasses.replace(r, device=dev))
            for r in results
        ]

    adapter = build_adapter(spec)

    # --- Applicability gate ---
    if not adapter.is_applicable(problem):
        return _emit([_not_supported_sentinel(problem, adapter, benchmark)])

    # --- Query selection (seed from problem generation) ---
    # v0.14 (#148): grouped selection. select_groups emits
    # list[list[Query]] chunked by the baseline's batch_size; at
    # n_batch_queries=1 (default) every group is one query and the
    # flattened list equals select()'s output exactly. Selectors
    # without select_groups (legacy fixtures) fall back to select().
    batch_size = getattr(spec, "batch_size", 1)
    if hasattr(selector, "select_groups"):
        query_groups = selector.select_groups(
            problem, n_queries_per_cell, problem.seed,
            batch_size=batch_size,
        )
    else:
        query_groups = [
            [q] for q in selector.select(
                problem, n_queries_per_cell, problem.seed,
            )
        ]
    queries = [q for g in query_groups for q in g]
    query_roles = [getattr(q, "query_role", default_role) for q in queries]
    query_kinds = [getattr(q, "query_kind", "prediction") for q in queries]
    evidence_strategies = [
        getattr(q, "evidence_strategy", "random") for q in queries
    ]
    evidence_modes = [_evidence_mode_for(q) for q in queries]

    # Sentinel representatives (§4.3): fit-failure sentinels are emitted
    # per GROUP (one representative query each), not per planned query —
    # at n_batch_queries=1 this is exactly the pre-batching one-sentinel-
    # per-query behavior; at N>1 it avoids K×N sentinel explosion.
    sentinel_queries = [g[0] for g in query_groups]
    sentinel_roles = [
        getattr(q, "query_role", default_role) for q in sentinel_queries
    ]

    # --- Fit ---
    try:
        t0 = perf_counter()
        adapter.fit(problem, **spec.extra_kwargs)
        fit_time_s = perf_counter() - t0
    except Exception as exc:
        status = _classify_exception(exc)
        return _emit(_fit_failure_rows(
            problem, adapter, sentinel_queries, sentinel_roles, benchmark,
            fit_time_s=_NAN, status=status, error_msg=repr(exc),
        ))

    # --- Fit safety net ---
    if fit_time_s > fit_budget_s:
        return _emit(_fit_failure_rows(
            problem, adapter, sentinel_queries, sentinel_roles, benchmark,
            fit_time_s=fit_time_s, status="timeout",
            error_msg=f"fit exceeded {fit_budget_s:.0f}s safety budget",
        ))

    # --- Measurement (handles per-query cumulative timeout internally) ---
    rows = measurement.measure(
        problem, adapter, queries,
        fit_time_s=fit_time_s,
        benchmark=benchmark,
        seed=problem.seed,
        query_roles=query_roles,
        query_kinds=query_kinds,
        evidence_strategies=evidence_strategies,
        evidence_modes=evidence_modes,
        query_budget_s=per_cell_timeout_s,
        query_groups=query_groups,
    )
    return _emit(rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m benchmarking.core.cell_worker INPUT OUTPUT",
              file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
