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


def _run_cell(ctx: dict) -> list[dict]:
    """The actual cell logic.

    Stage 1: minimal placeholder that returns a single ok row.
    Stage 2: full measurement code moves here.
    """
    # Stage 1 placeholder. In Stage 2, this becomes:
    #   - build_adapter(ctx["baseline_spec"])
    #   - adapter.fit(ctx["problem"])
    #   - run queries via measurement code
    #   - yield CellResult rows
    problem = ctx.get("problem")
    baseline_spec = ctx.get("baseline_spec")
    return [{
        "status": "ok",
        "stage": "stage1_placeholder",
        "problem_id": problem.problem_id
            if hasattr(problem, "problem_id")
            else ctx.get("problem_id", "unknown"),
        "baseline": baseline_spec.library
            if hasattr(baseline_spec, "library")
            else "unknown",
        "seed": ctx.get("seed", 0),
    }]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m benchmarking.core.cell_worker INPUT OUTPUT",
              file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
