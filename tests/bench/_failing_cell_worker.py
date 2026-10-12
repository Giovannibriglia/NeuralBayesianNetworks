"""Test-only subprocess that simulates cell failure modes.

Used by the Stage 3 integration tests for Bug 4 (#127). The leading
underscore keeps pytest from collecting this as a test module.

Each invocation runs in one of four modes selected by argv[3]:

- ok:    write one ok row and exit 0
- raise: raise RuntimeError before writing any row (catchable in-process
         failure → non-zero exit, no signal)
- sleep: write one preamble row, then sleep 120s so the parent's
         cell-level timeout SIGTERMs it (→ classification 'timeout')
- oom:   write one preamble row, then SIGKILL itself

Why ``oom`` uses an explicit SIGKILL rather than a real allocation:
``RLIMIT_AS`` + a Python/ctypes allocation raises a *catchable*
``MemoryError`` (exit code, no signal), which the worker's own
try/except handles — that path is already covered by
``test_fit_memory_error_emits_oom_rows``. The architecturally critical
path this stage must prove is the *uncatchable* kill: the production
612 GB OOM tripped the Linux OOM-killer, which sends SIGKILL that no
in-process handler can catch. ``os.kill(getpid(), SIGKILL)`` reproduces
exactly that signal deterministically and without endangering the test
host. The preamble row written + flushed before the kill also proves
that partial output survives a mid-cell death.
"""
from __future__ import annotations

import json
import os
import pickle
import signal
import sys
import time


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: _failing_cell_worker INPUT_PKL OUTPUT_JSONL MODE",
            file=sys.stderr,
        )
        return 2

    input_path, output_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]

    # Verify the input pickle is readable (mirrors the real worker).
    with open(input_path, "rb") as f:
        pickle.load(f)

    if mode == "raise":
        # Catchable in-process failure before any output is written.
        raise RuntimeError("deliberate test failure in cell")

    with open(output_path, "w", buffering=1) as out:
        # Preamble row, flushed immediately, so the partial-data-survives
        # path can be asserted even when the process is killed mid-cell.
        out.write(json.dumps({"status": "ok", "stage": "preamble"}) + "\n")
        out.flush()

        if mode == "ok":
            return 0

        if mode == "sleep":
            time.sleep(120)  # well beyond any test timeout
            return 0

        if mode == "oom":
            # Uncatchable kill — models the OS OOM-killer's SIGKILL.
            os.kill(os.getpid(), signal.SIGKILL)
            return 0  # unreachable

    return 0


if __name__ == "__main__":
    sys.exit(main())
