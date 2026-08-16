"""Classify a run's ``status='oom'`` rows: real OOM, or the cell never started?

Two very different failures both land as ``status='oom'``, and telling them
apart decides whether a run's results are usable.

* **A genuine OOM.** The cell ran, the work was too big, and something
  diagnosed it — the variable-elimination pre-allocation guard reporting the
  peak factor it declined to allocate, a CUDA allocator failure, a Python
  ``MemoryError``.  The row is a true finding about that baseline at that size.

* **A cell that never started.** ``benchmarking/core/cell_runner`` caps each
  subprocess with ``RLIMIT_AS``, which bounds *virtual address space* — and a
  torch process reserves far more of that than it ever resides.  Until
  9b5c6b7 the floor for that cap was 2 GiB, below the ~6 GiB at which torch
  can import at all, so on a loaded host every cell was killed during startup
  and the parent recorded ``oom``.  Those rows say nothing about the baseline;
  they measure how busy the machine was.

The signatures are distinguishable because a cell killed before it wrote
anything gets a row synthesised by the parent (``cell_runner.py``), whose
``error_msg`` reads "subprocess produced no output rows (...)", whereas a
genuine OOM carries the diagnosing component's own message.

Run::

    python -m benchmarking.diagnostics.triage_oom_rows PATH [PATH ...]
    python -m benchmarking.diagnostics.triage_oom_rows            # scans benchmarking/results

``PATH`` may be a ``.parquet``, a ``metrics.jsonl``, or a run directory.
Reads only the columns it needs, so a multi-million-row parquet is cheap.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Iterable

_NEEDED = ("status", "error_msg")

#: (label, predicate-on-lowercased-message, is_spurious)
_SIGNATURES: list[tuple[str, str, bool]] = [
    # The cell died before producing a single row -> it never did any work.
    ("never-started (killed before any output)", "subprocess produced no output rows", True),
    # Killed part-way through; the rows it did write are real, this marks the kill.
    ("killed mid-write", "subprocess killed", False),
    ("VE pre-allocation guard", "pre-allocation guard", False),
    ("CUDA out of memory", "cuda out of memory", False),
    ("Python MemoryError", "memoryerror", False),
    # Not an OOM at all: this cell inherited the status from a sibling that
    # failed earlier.  Whatever caused *that* is the thing to triage; counted
    # separately so it neither inflates the genuine tally nor the spurious one.
    ("propagated skip (inherited from an earlier failure)", "skipped:", False),
]


def _classify(msg: object) -> tuple[str, bool]:
    text = "" if msg is None else str(msg).strip().lower()
    if not text:
        # No message at all is itself suspicious: a diagnosed OOM always says
        # who diagnosed it.
        return "no message", True
    for label, needle, spurious in _SIGNATURES:
        if needle in text:
            return label, spurious
    return "other", False


def _iter_rows(path: str) -> Iterable[dict]:
    """Yield ``{status, error_msg}`` dicts from a parquet or a jsonl."""
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield {k: row.get(k) for k in _NEEDED}
        return

    import pandas as pd
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    cols = [c for c in _NEEDED if c in available]
    if "status" not in cols:
        return
    df = pd.read_parquet(path, columns=cols)
    if "error_msg" not in df.columns:
        df["error_msg"] = None
    for row in df[list(_NEEDED)].to_dict("records"):
        yield row


def _resolve(paths: list[str]) -> list[str]:
    if not paths:
        paths = ["benchmarking/results"]
    out: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True))
            out += sorted(glob.glob(os.path.join(p, "**", "metrics.jsonl"), recursive=True))
        else:
            out.append(p)
    # A run directory usually holds both a parquet and its jsonl; prefer the
    # parquet and drop the redundant jsonl so counts are not doubled.
    parquet_dirs = {os.path.dirname(p) for p in out if p.endswith(".parquet")}
    return [p for p in out if p.endswith(".parquet") or os.path.dirname(p) not in parquet_dirs]


def triage(path: str) -> dict:
    """Return the oom breakdown for one results file."""
    total = 0
    oom: dict[str, int] = {}
    spurious = 0
    for row in _iter_rows(path):
        total += 1
        if row.get("status") != "oom":
            continue
        label, is_spurious = _classify(row.get("error_msg"))
        oom[label] = oom.get(label, 0) + 1
        if is_spurious:
            spurious += 1
    n_oom = sum(oom.values())
    derivative = sum(n for label, n in oom.items() if label.startswith("propagated skip"))
    # The spurious fraction is taken over rows that actually claim an OOM, so
    # a pile of inherited skips cannot dilute it.
    attributable = n_oom - derivative
    return {
        "path": path,
        "rows": total,
        "oom": n_oom,
        "breakdown": oom,
        "spurious": spurious,
        "attributable": attributable,
        "spurious_fraction": (spurious / attributable) if attributable else 0.0,
    }


def _verdict(r: dict) -> str:
    if r["oom"] == 0:
        return "no oom rows — nothing to triage"
    if r["attributable"] == 0:
        return (
            "every oom row is inherited from an earlier failure — triage that "
            "failure instead"
        )
    frac = r["spurious_fraction"]
    if frac >= 0.5:
        return (
            f"{frac:.0%} of the oom rows are cells that never started. This run "
            f"largely measured the per-cell memory cap, not the baselines — "
            f"re-run on a build including 9b5c6b7."
        )
    if frac > 0.0:
        return (
            f"{frac:.0%} of the oom rows are cells that never started; the rest "
            f"are genuine. Partially affected — inspect before using."
        )
    return "all oom rows are genuine, diagnosed OOMs — the fix changes nothing here"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="triage_oom_rows", description=__doc__.split("\n\n")[0],
    )
    ap.add_argument("paths", nargs="*", help="parquet / metrics.jsonl / run directory")
    args = ap.parse_args(argv)

    files = _resolve(args.paths)
    if not files:
        print("no results files found", file=sys.stderr)
        return 1

    worst = 0.0
    for path in files:
        r = triage(path)
        print(f"\n{r['path']}")
        print(f"  rows={r['rows']:,}  oom={r['oom']:,}")
        for label, n in sorted(r["breakdown"].items(), key=lambda kv: -kv[1]):
            print(f"    {n:>8,}  {label}")
        print(f"  -> {_verdict(r)}")
        worst = max(worst, r["spurious_fraction"])

    print(f"\nworst spurious fraction across {len(files)} file(s): {worst:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
