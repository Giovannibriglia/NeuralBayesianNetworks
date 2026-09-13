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

# Rows buffered in memory between parquet row-group writes. 250k rows of
# CellResult dicts is ~0.5 GB of Python objects -- the bound on peak memory
# regardless of how large the JSONL grows.
_CHUNK_ROWS = 250_000


def _cellresult_arrow_schema():
    """Arrow schema derived from the ``CellResult`` dataclass annotations.

    ``int`` -> int64, ``float`` -> float64, everything else -> string;
    ``X | None`` maps to the schema type of ``X`` (nullable, as all Arrow
    fields are). Deriving the schema from the dataclass -- rather than
    inferring it from the data -- is what lets the conversion stream: every
    row group is cast to the same declared types, so a column that is all
    null in one chunk (``ess`` on a VE-only stretch, say) and float in the
    next never produces a schema mismatch between row groups.
    """
    import types
    import typing

    import pyarrow as pa

    _PA = {int: pa.int64(), float: pa.float64(), str: pa.string(), bool: pa.bool_()}
    hints = typing.get_type_hints(CellResult)
    fields = []
    for f in dataclasses.fields(CellResult):
        t = hints[f.name]
        if isinstance(t, types.UnionType) or typing.get_origin(t) is typing.Union:
            args = [a for a in typing.get_args(t) if a is not type(None)]
            t = args[0] if len(args) == 1 else str
        fields.append(pa.field(f.name, _PA.get(t, pa.string())))
    return pa.schema(fields)


def jsonl_to_parquet(
    jsonl_path: Path, parquet_path: Path, *, chunk_rows: int = _CHUNK_ROWS,
) -> None:
    """Convert a v0.13 JSONL sidecar to a v3-schema parquet file, streaming.

    Reads the JSONL file written by ``JsonlWriter`` (one CellResult per
    line, NaN as null) and writes a parquet whose column dtypes match
    the v3 schema:
      - ``seed``, ``n_parameters``, ``n_nodes``, ``n_train``, ``batch_size``: int64
      - ``value``, ``fit_time_s``, ``query_time_s``, ``metrics_time_s`` and
        the other float diagnostics: float64
      - ``problem_id``, ``baseline``, etc.: string

    Memory is bounded by ``chunk_rows``: rows are parsed into per-column
    buffers and flushed to one parquet row group every ``chunk_rows`` rows,
    so a multi-GB JSONL (a paper-scale timing sweep is ~10 GB / ~15M rows)
    converts in well under 1 GB of RSS. The previous list-of-dicts +
    ``pd.DataFrame`` implementation needed ~3-5x the file size in RAM and
    was OOM-killed on exactly that input (2026-09-12 batch_speed run).

    Column set: the ``CellResult`` fields, plus any extra keys present in
    the first chunk (older/newer producers) typed by inference. Rows lacking
    a field get null there. A key first appearing after the first chunk
    cannot be added to an already-open writer and raises ``ValueError``.

    Parameters
    ----------
    jsonl_path:
        Source JSONL file written by JsonlWriter.
    parquet_path:
        Destination parquet path.  Parent directory is created if needed.
        Not created at all when the JSONL has no rows (a warning is issued).
    chunk_rows:
        Rows per parquet row group / in-memory buffer bound.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _cellresult_arrow_schema()
    known = set(schema.names)
    columns: dict[str, list] = {name: [] for name in schema.names}
    extra: dict[str, list] = {}
    n_buffered = 0
    n_total = 0
    writer = None

    def _flush() -> None:
        nonlocal writer, schema, n_buffered
        if n_buffered == 0:
            return
        if writer is None:
            # First chunk: extend the schema with any non-CellResult keys,
            # typed by inference (string when the chunk is all-null).
            for name, values in extra.items():
                inferred = pa.array(values).type
                if pa.types.is_null(inferred):
                    inferred = pa.string()
                schema = schema.append(pa.field(name, inferred))
            Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(parquet_path, schema)
        arrays = []
        for field in schema:
            values = columns.get(field.name)
            if values is None:
                values = extra[field.name]
            if field.name == "problem_id":
                # problem_id is always str in the v3 schema (older JSONL
                # may carry the synthetic n_nodes as an int).
                values = [None if v is None else str(v) for v in values]
            arrays.append(pa.array(values, type=field.type))
        writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
        for buf in columns.values():
            buf.clear()
        for buf in extra.values():
            buf.clear()
        n_buffered = 0

    with open(jsonl_path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            for name in schema.names:
                columns[name].append(row.get(name))
            for key in row.keys() - known:
                if key not in extra:
                    if writer is not None:
                        raise ValueError(
                            f"{jsonl_path}: column {key!r} first appears after "
                            f"row {n_total}; the parquet schema is fixed by the "
                            f"first {chunk_rows} rows"
                        )
                    # Back-fill nulls for the rows already buffered.
                    extra[key] = [None] * n_buffered
                extra[key].append(row[key])
            for key in extra.keys() - row.keys():
                extra[key].append(None)
            n_buffered += 1
            n_total += 1
            if n_buffered >= chunk_rows:
                _flush()

    _flush()
    if writer is None:
        warnings.warn(
            f"no rows in {jsonl_path}; parquet not written", stacklevel=2,
        )
        return
    writer.close()


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
