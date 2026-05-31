"""JSONL writer for v0.13 CellResult rows.

Streaming, line-buffered, crash-resilient.  Each line is a flat JSON
object.  NaN is serialised as JSON null (valid JSON;
``pd.read_json`` re-hydrates null → NaN).

Reference: docs/v0.13-benchmark-redesign.md §6
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from benchmarking.core.results import CellResult


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
