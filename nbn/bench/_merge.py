"""Splice a partial rerun into an earlier run's parquet (``nbn-bench merge``).

A partial rerun (``nbn/bench/configs/*/reruns/``) re-executes only the
cells a fix invalidated — e.g. every continuous-family cell after the oracle
fix (#288), every ``*-avi`` cell after the AVI gate fix (#286).  Plotting the
two parquets together would double-count those cells (``nbn-bench plot``
row-concatenates its inputs), so this replaces them instead.

A **cell** is ``(benchmark, family, problem_id, seed, baseline)``: one fit of
one baseline on one problem, with all of its query / metric / batch-size rows.
Every base row whose cell appears in an update is dropped and the update's
rows for that cell are added; later updates win over earlier ones.  Cells only
in the base are kept untouched.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pandas is imported lazily at call time
    import pandas as pd

logger = logging.getLogger(__name__)

CELL_KEY = ("benchmark", "family", "problem_id", "seed", "baseline")


def merge_frames(base, updates, *, names=()) -> tuple[pd.DataFrame, dict[str, int]]:
    """In-memory core of :func:`merge_runs`: return ``base`` with every cell
    present in any of ``updates`` (DataFrames) replaced, plus the row-count
    stats. ``names`` are optional labels for the log lines (one per update)."""
    import pandas as pd

    merged = base
    missing = [k for k in CELL_KEY if k not in merged.columns]
    if missing:
        raise ValueError(f"not a benchmark parquet (missing {missing})")
    stats = {"kept": 0, "dropped": 0, "added": 0, "new_cells": 0}
    base_cells = set(map(tuple, merged[list(CELL_KEY)].drop_duplicates().to_numpy()))

    for i, upd in enumerate(updates):
        label = names[i] if i < len(names) else f"update {i + 1}"
        cells = upd[list(CELL_KEY)].drop_duplicates()
        cell_set = set(map(tuple, cells.to_numpy()))
        hit = merged.set_index(list(CELL_KEY)).index.isin(list(cell_set))
        stats["dropped"] += int(hit.sum())
        stats["added"] += len(upd)
        stats["new_cells"] += len(cell_set - base_cells)
        for (fam, bl), n in (cells.groupby(["family", "baseline"]).size().items()):
            logger.info("replacing %-28s %-20s %d cell(s) from %s", bl, fam, n, label)
        merged = pd.concat([merged[~hit], upd], ignore_index=True)

    stats["kept"] = len(merged) - stats["added"]
    if stats["new_cells"]:
        logger.warning(
            "%d update cell(s) had no counterpart in the base run (added as new "
            "cells) — check the rerun config matches the original's problems.",
            stats["new_cells"],
        )
    return merged, stats


def load_run(item) -> pd.DataFrame:
    """Read a parquet file or a run directory (``*_metrics.parquet`` inside)."""
    import pandas as pd

    from nbn.bench._paper_figures import _resolve_parquet

    paths = _resolve_parquet([Path(item)])
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def merge_runs(base, updates, out: Path) -> dict[str, int]:
    """Write ``base`` with every cell present in ``updates`` replaced.

    ``base`` / ``updates`` are parquet files or run directories (as accepted
    by ``nbn-bench plot``).  Returns row counts: ``kept``, ``dropped``,
    ``added`` and ``new_cells`` (update cells absent from the base).
    """
    try:
        merged, stats = merge_frames(load_run(base), [load_run(u) for u in updates],
                                     names=[str(u) for u in updates])
    except ValueError as e:
        raise ValueError(f"{base}: {e}") from None
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    logger.info(
        "wrote %s: %d rows (%d kept from base, %d replaced by %d new)",
        out, len(merged), stats["kept"], stats["dropped"], stats["added"],
    )
    return stats
