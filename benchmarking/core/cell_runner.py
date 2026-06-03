"""Cell runner (Bug 4 of #127).

Launches a cell as a subprocess, waits for completion (with timeout
escalation: SIGTERM → grace → SIGKILL), reads the JSONL output, and
classifies the result. A SIGKILLed subprocess produces an `oom`
status row; a timeout-killed subprocess produces a `timeout` row;
an exception in the subprocess produces an `error` row.

This module is imported but NOT YET wired into the runner. Stage 2
replaces Runner.run()'s in-process measurement call with a call to
run_cell_in_subprocess().

See docs/v0.13-runner-subprocess-isolation.md §3-4.
"""
from __future__ import annotations

import json
import os
import pickle
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


# Grace period (seconds) between SIGTERM and SIGKILL escalation
_SIGTERM_GRACE_S = 3.0


# Per-cell memory cap (Bug 4 #127). The subprocess sets RLIMIT_AS on
# itself at startup to this value. Anchored to currently-available
# system memory rather than total, so concurrent processes (the
# parent runner, editor, browser, OS) keep their headroom — SIGKILL
# fires on the runaway subprocess at a threshold survivable for
# the rest of the system, not at OS-distress.
_MEMORY_LIMIT_FRACTION = 0.80
_MEMORY_LIMIT_FLOOR_BYTES = 2 * 1024**3  # 2 GB


def _compute_cell_memory_limit_bytes() -> int:
    """Return the per-cell virtual memory cap for the next subprocess.

    Computed fresh per-call so it adapts to current system state
    (other processes, accumulated runner state). Anchored to
    psutil.virtual_memory().available rather than .total so the
    cap respects what's actually allocatable right now.
    """
    import psutil
    available = psutil.virtual_memory().available
    return max(_MEMORY_LIMIT_FLOOR_BYTES,
               int(available * _MEMORY_LIMIT_FRACTION))


@dataclass
class CellRunResult:
    """Outcome of running one cell in a subprocess.

    rows: list of result rows parsed from subprocess output (each is
      a dict that becomes one JSONL row in the main metrics file).
    exit_code: subprocess exit code (negative for signal kills).
    classification: 'completed' / 'timeout' / 'killed'. The per-row
      status field within rows is independent — this classification
      describes the subprocess lifecycle.
    stderr: subprocess stderr text, captured post-exit. Logging only —
      the result rows still come from the output file, so this does not
      change the isolation/result-passing contract.
    """
    rows: list[dict]
    exit_code: int
    classification: str
    stderr: str = ""


def run_cell_in_subprocess(
    ctx: dict,
    *,
    timeout_s: float | None = None,
    python_executable: str | None = None,
) -> CellRunResult:
    """Run a cell in a subprocess and return the result rows.

    Args:
        ctx: dict to pickle and pass to the subprocess. Must contain
          whatever cell_worker._run_cell expects. Stage 1 worker
          accepts {"problem": ..., "baseline_spec": ..., "seed": ...}
          but is mostly a placeholder.
        timeout_s: subprocess SIGTERMed if it runs longer than this
          many seconds. None = no timeout.
        python_executable: path to python (defaults to sys.executable).

    Returns:
        CellRunResult with parsed rows + classification.
    """
    if python_executable is None:
        python_executable = sys.executable

    # Temp files for input + output
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".pkl", delete=False
    ) as input_f:
        pickle.dump(ctx, input_f)
        input_path = input_f.name

    output_path = input_path.replace(".pkl", ".jsonl")
    Path(output_path).touch()  # ensure exists for partial-read on death

    # Compute and pass the per-cell memory cap to the subprocess. The
    # worker applies RLIMIT_AS to itself on startup. The env-var name
    # NBN_CELL_MEMORY_LIMIT_BYTES is the contract between parent and
    # worker (see cell_worker._apply_memory_limit).
    memory_limit_bytes = _compute_cell_memory_limit_bytes()
    env = os.environ.copy()
    env["NBN_CELL_MEMORY_LIMIT_BYTES"] = str(memory_limit_bytes)

    try:
        # Launch the subprocess
        proc = subprocess.Popen(
            [python_executable, "-m", "benchmarking.core.cell_worker",
             input_path, output_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # New process group so we can SIGTERM the whole group
            start_new_session=True,
            env=env,
        )

        classification = "completed"

        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # Cell exceeded budget — escalate SIGTERM → grace → SIGKILL
            classification = "timeout"
            _terminate_with_grace(proc)

        exit_code = proc.returncode

        # Drain the already-captured stderr now that the process has exited
        # (read is non-blocking on a finished process). Logging only; does not
        # affect the result rows (those come from the output file).
        stderr_text = ""
        if proc.stderr is not None:
            try:
                stderr_text = proc.stderr.read().decode("utf-8", "replace")
            except Exception:
                stderr_text = ""

        # If the subprocess died from a signal, mark as killed
        # (Negative return code on POSIX = killed by signal)
        if exit_code is not None and exit_code < 0 and classification == "completed":
            classification = "killed"
        # Or, on some platforms, signal is in returncode-128:
        # 137 = 128 + 9 (SIGKILL), 143 = 128 + 15 (SIGTERM)
        elif exit_code in (137, 139):  # SIGKILL or SIGSEGV
            if classification == "completed":
                classification = "killed"

        # Read output JSONL (may have partial final line if killed)
        rows = _read_output_jsonl(output_path)

        # If killed and no rows, synthesize one
        if not rows:
            status = _classification_to_status(classification, exit_code)
            rows = [{
                "status": status,
                "error_msg": f"subprocess produced no output rows "
                             f"(classification={classification}, exit_code={exit_code})",
            }]
        else:
            # Promote subprocess-level classification to a synthesized
            # status only if the subprocess didn't write its own. If the
            # subprocess explicitly set status (e.g., status=error with
            # traceback), trust it. If the subprocess was killed mid-write,
            # add a synthesized row to mark the kill.
            if classification in ("timeout", "killed"):
                # Subprocess was killed externally; add a marker row
                # so downstream tooling can see the kill happened.
                status = _classification_to_status(classification, exit_code)
                rows.append({
                    "status": status,
                    "error_msg": f"subprocess {classification} "
                                 f"(exit_code={exit_code})",
                    "_synthesized_by": "cell_runner",
                })

        return CellRunResult(
            rows=rows,
            exit_code=exit_code if exit_code is not None else -1,
            classification=classification,
            stderr=stderr_text,
        )
    finally:
        # Clean up temp files
        for p in (input_path, output_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _terminate_with_grace(proc: subprocess.Popen) -> None:
    """SIGTERM, wait up to _SIGTERM_GRACE_S, then SIGKILL if still alive."""
    try:
        # Send SIGTERM to the whole process group (start_new_session=True)
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return  # already dead

    try:
        proc.wait(timeout=_SIGTERM_GRACE_S)
    except subprocess.TimeoutExpired:
        # SIGTERM didn't take; escalate
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        # Final wait — SIGKILL is uncatchable, should be near-instant
        proc.wait(timeout=2.0)


def _classification_to_status(classification: str, exit_code: int) -> str:
    """Map subprocess lifecycle classification → row status field."""
    if classification == "timeout":
        return "timeout"
    if classification == "killed":
        # K3 convention: any SIGKILL = oom (#127 design decision)
        return "oom"
    return "error"


def _read_output_jsonl(path: str) -> list[dict]:
    """Read JSONL output, tolerating a partial final line.

    The subprocess may have died mid-write, leaving a partial JSON
    object on the final line. We parse all complete lines and discard
    any trailing partial.
    """
    rows: list[dict] = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Partial line — done parsing
                    break
    except FileNotFoundError:
        pass
    return rows
