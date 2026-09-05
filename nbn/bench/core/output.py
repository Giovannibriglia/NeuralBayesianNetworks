"""JSONL writer, parquet converter, and path helpers for v0.13 output.

Streaming, line-buffered, crash-resilient JSONL.  NaN serialised as
JSON null (valid JSON; ``pd.read_json`` re-hydrates null → NaN).

Reference: docs/v0.13-benchmark-redesign.md §6
"""
from __future__ import annotations

import dataclasses
import json
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from nbn.bench.core.results import CellResult


class JsonlWriter:
    """Streaming JSONL writer for CellResult rows.

    Opens the file in append mode at construction time, flushes after
    every ``write()`` call, and closes on ``__exit__``/``close()``.

    Usage::

        with JsonlWriter(path) as writer:
            for row in rows:
                writer.write(row)
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._fh = open(path, "a")  # noqa: SIM115 — long-lived; close() in __exit__

    def write(self, row: CellResult) -> None:
        """Serialize one CellResult and write it, flushing immediately."""
        d = dataclasses.asdict(row)
        # NaN is not valid JSON; replace with null for broad parser compatibility.
        d = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}
        self._fh.write(json.dumps(d) + "\n")
        self._fh.flush()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# JSONL → parquet conversion
# ---------------------------------------------------------------------------

# v3 schema integer columns (seed is int; others are float or str).
_INT_COLS = ("seed",)
# v3 schema float columns.
_FLOAT_COLS = ("value", "fit_time_s", "query_time_s", "metrics_time_s")


def jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> None:
    """Convert a v0.13 JSONL sidecar to a v3-schema parquet file.

    Reads the JSONL file written by ``JsonlWriter`` (one CellResult per
    line, NaN as null) and writes a parquet whose column dtypes match
    the v3 schema:
      - ``seed``: int64
      - ``value``, ``fit_time_s``, ``query_time_s``, ``metrics_time_s``: float64
      - ``problem_id``, ``baseline``, etc.: object (str)

    Parameters
    ----------
    jsonl_path:
        Source JSONL file written by JsonlWriter.
    parquet_path:
        Destination parquet path.  Parent directory is created if needed.
    """
    import pandas as pd

    rows = []
    with open(jsonl_path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))

    if not rows:
        warnings.warn(
            f"no rows in {jsonl_path}; parquet not written", stacklevel=2,
        )
        return

    df = pd.DataFrame(rows)

    for col in _INT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("int64")
    for col in _FLOAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    # problem_id is always str in v3 schema.
    if "problem_id" in df.columns:
        df["problem_id"] = df["problem_id"].astype(str)

    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)


# ---------------------------------------------------------------------------
# Path convention helpers
# ---------------------------------------------------------------------------

def _compact_datetime() -> str:
    """Return the current local datetime as a compact string.

    Format: ``YYYYMMDD_HHMMSS``.  Example: ``"20260531_143045"``.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_results_dir(
    benchmark: str,
    config_name: str,
    base: Path = Path("results"),
) -> Path:
    """Create and return the run output directory.

    Path convention::

        <base>/benchmark_<benchmark>_<config_name>_<compact_dt>/

    Example::

        results/benchmark_synthetic_paper_20260531_143045/

    Parameters
    ----------
    benchmark:
        Benchmark name ("synthetic" | "scalability" | "bnlearn").
    config_name:
        Human-readable config label (e.g., "paper", "smoke").
    base:
        Base directory; defaults to ``Path("results")``
        so new runs land alongside the v0.12 archive.

    Returns
    -------
    Path
        Newly created run directory.

    Raises
    ------
    FileExistsError
        If the directory already exists (``exist_ok=False``).
    """
    dt = _compact_datetime()
    run_dir = base / f"benchmark_{benchmark}_{config_name}_{dt}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
