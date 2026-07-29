"""Cell-level checkpoint/resume + SLURM preemption support.

European HPC clusters cap a single job at 24 h wall-time. The benchmark suite
resumes across jobs at CELL granularity (problem x baseline), reusing two
facts about the existing runner:

  * every CellResult row is already streamed to ``metrics.jsonl`` as it is
    produced (crash-resilient, append mode);
  * each cell runs in an isolated subprocess, so "the cell" is the natural
    atomic unit of work — there is no finer-grained state worth serialising.

This module adds the three missing pieces:

  1. a ``completed_cells.jsonl`` sidecar next to the metrics JSONL — one line
     per FULLY completed cell, written only after every row of that cell has
     been flushed to the metrics JSONL;
  2. startup compaction — on resume, rows belonging to cells NOT recorded as
     complete (i.e. the cell that was in flight when the previous job died)
     are dropped from ``metrics.jsonl`` before appending resumes, so no cell
     ever contributes duplicate rows;
  3. cooperative preemption — SIGUSR1/SIGTERM (SLURM's early-warning and
     wall-time signals) set a flag the runner checks between cells; the run
     stops cleanly and the CLI exits with ``PREEMPTED_EXIT_CODE`` so the next
     job in the SLURM array chain knows to resume.

The cell key is ``(family, problem_id, seed, baseline, n_train)``. ``n_train``
is part of the key because the learning-curves sweep (#109 PR 6) yields
problems that differ ONLY in training-set size — family/problem_id/seed alone
would collide across sweep values. Both the marker and the row-side match use
``n_train_from_problem`` / the row's stamped ``n_train`` column, which the
runner injects at a single choke point, so the two sides cannot drift.

Reference: benchmarking/slurm/README.md
"""
from __future__ import annotations

import json
import logging
import os
import signal
from pathlib import Path
from typing import Any

from benchmarking.domains._n_parameters import n_train_from_problem

logger = logging.getLogger(__name__)

#: Exit code the CLI returns when a run stopped early on SIGUSR1/SIGTERM.
#: 124 mirrors coreutils ``timeout`` ("ran out of time"); the SLURM wrapper
#: treats it as "resume in the next array task", distinct from 0 (all cells
#: done) and any other code (genuine failure).
PREEMPTED_EXIT_CODE = 124

# Statuses that _register_failures treats as genuine failures (mirrored from
# runner._FAILURE_STATUSES; kept as a literal to avoid a circular import).
_FAILURE_STATUSES = ("oom", "timeout", "error")


# ---------------------------------------------------------------------------
# Cooperative preemption flag
# ---------------------------------------------------------------------------

_stop_requested = False


def _handle_stop_signal(signum: int, frame: Any) -> None:
    global _stop_requested
    _stop_requested = True
    try:
        name = signal.Signals(signum).name
    except ValueError:  # pragma: no cover — unknown signum
        name = str(signum)
    logger.warning(
        "received %s — will checkpoint and stop after the current cell", name,
    )


def install_preemption_handlers() -> None:
    """Install SIGUSR1/SIGTERM handlers that request a graceful stop.

    Called by the CLI (main thread) before the cell loop starts. SIGUSR1 is
    SLURM's configurable early warning (``#SBATCH --signal=B:USR1@900``);
    SIGTERM is what SLURM sends at the wall-time limit. Either one only sets
    a flag — the runner finishes the in-flight cell, records it, and returns.
    """
    signal.signal(signal.SIGUSR1, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)


def stop_requested() -> bool:
    """True once a preemption signal has been received."""
    return _stop_requested


def reset_stop() -> None:
    """Clear the flag (tests only)."""
    global _stop_requested
    _stop_requested = False


# ---------------------------------------------------------------------------
# Cell keys and the completed-cells sidecar
# ---------------------------------------------------------------------------

def cell_key(problem: Any, baseline_name: str) -> tuple:
    """The identity of one (problem, baseline) cell across restarts."""
    return (
        getattr(problem, "family", ""),
        str(getattr(problem, "problem_id", "")),
        int(getattr(problem, "seed", 0)),
        baseline_name,
        n_train_from_problem(problem),
    )


def _row_key(row: dict) -> tuple:
    """The same key, computed from a serialised CellResult row dict."""
    n_train = row.get("n_train")
    return (
        row.get("family", ""),
        str(row.get("problem_id", "")),
        int(row.get("seed", 0)),
        row.get("baseline", ""),
        int(n_train) if n_train is not None else None,
    )


def completed_cells_path(jsonl_path: Path) -> Path:
    """The sidecar path: ``completed_cells.jsonl`` next to the metrics JSONL."""
    return Path(jsonl_path).parent / "completed_cells.jsonl"


def append_completed(path: Path, key: tuple) -> None:
    """Record one fully-completed cell (append + flush + fsync).

    Called only after every row of the cell has been written to the metrics
    JSONL, so a crash between the two files can only lose the MARKER (cell
    re-runs, its old rows are compacted away) — never produce a marked cell
    with missing rows.
    """
    family, problem_id, seed, baseline, n_train = key
    line = json.dumps({
        "family": family, "problem_id": problem_id, "seed": seed,
        "baseline": baseline, "n_train": n_train,
    })
    with open(path, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_completed(path: Path) -> set[tuple]:
    """Load the set of completed cell keys (empty set if no sidecar yet)."""
    path = Path(path)
    if not path.exists():
        return set()
    keys: set[tuple] = set()
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            d = json.loads(stripped)
            n_train = d.get("n_train")
            keys.add((
                d["family"], str(d["problem_id"]), int(d["seed"]),
                d["baseline"],
                int(n_train) if n_train is not None else None,
            ))
    return keys


# ---------------------------------------------------------------------------
# Startup compaction (resume)
# ---------------------------------------------------------------------------

def compact_jsonl(jsonl_path: Path, completed: set[tuple]) -> tuple[int, int]:
    """Drop rows of cells NOT recorded as complete; atomic rewrite.

    The previous job may have died mid-cell, leaving that cell's partial rows
    in the metrics JSONL. Re-running the cell would then duplicate them, so on
    resume only rows whose cell key is in ``completed`` survive. Returns
    ``(kept, dropped)`` row counts. A missing JSONL is a no-op ``(0, 0)``.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        return (0, 0)
    kept_lines: list[str] = []
    dropped = 0
    with open(jsonl_path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if _row_key(json.loads(stripped)) in completed:
                kept_lines.append(stripped)
            else:
                dropped += 1
    tmp = jsonl_path.with_suffix(jsonl_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        for line in kept_lines:
            fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, jsonl_path)
    return (len(kept_lines), dropped)


def rebuild_seed_skip_registry(jsonl_path: Path) -> dict[tuple, str]:
    """Rebuild the speed-benchmark seed-skip registry from surviving rows.

    The in-memory ``failed_configs`` registry (#148 PR 2/2) does not survive a
    restart; without it, a resumed speed run would re-attempt configs that
    already failed on an earlier seed. This reconstructs the registry from the
    compacted JSONL: any failure row registers its
    ``(family, problem_id, baseline, batch_size)``. Propagated skip sentinels
    from earlier resumed runs carry the same status code, so propagation
    chains across restarts.

    Approximation vs ``_register_failures``: the "attempted batch size that
    produced no row at all" fallback (wholesale subprocess death) cannot be
    reconstructed from rows alone — such a config is re-attempted once after
    a restart and simply fails again. Safe, just not maximally frugal.
    """
    jsonl_path = Path(jsonl_path)
    registry: dict[tuple, str] = {}
    if not jsonl_path.exists():
        return registry
    with open(jsonl_path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if row.get("status") not in _FAILURE_STATUSES:
                continue
            batch_size = row.get("batch_size")
            if batch_size is None:
                continue
            key = (
                row.get("family", ""), str(row.get("problem_id", "")),
                row.get("baseline", ""), int(batch_size),
            )
            registry.setdefault(key, row["status"])
    return registry
