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

logger = logging.getLogger(__name__)

CELL_KEY = ("benchmark", "family", "problem_id", "seed", "baseline")


def merge_runs(base, updates, out: Path) -> dict[str, int]:
    """Write ``base`` with every cell present in ``updates`` replaced.

    ``base`` / ``updates`` are parquet files or run directories (as accepted
    by ``nbn-bench plot``).  Returns row counts: ``kept``, ``dropped``,
    ``added`` and ``new_cells`` (update cells absent from the base).
    """
    import pandas as pd

    from nbn.bench._paper_figures import _resolve_parquet

    def load(item) -> pd.DataFrame:
        paths = _resolve_parquet([Path(item)])
        return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    merged = load(base)
    missing = [k for k in CELL_KEY if k not in merged.columns]
    if missing:
        raise ValueError(f"{base}: not a benchmark parquet (missing {missing})")
    stats = {"kept": 0, "dropped": 0, "added": 0, "new_cells": 0}
    base_cells = set(map(tuple, merged[list(CELL_KEY)].drop_duplicates().to_numpy()))

    for upd_path in updates:
        upd = load(upd_path)
        cells = upd[list(CELL_KEY)].drop_duplicates()
        cell_set = set(map(tuple, cells.to_numpy()))
        hit = merged.set_index(list(CELL_KEY)).index.isin(list(cell_set))
        stats["dropped"] += int(hit.sum())
        stats["added"] += len(upd)
        stats["new_cells"] += len(cell_set - base_cells)
        for (fam, bl), n in (cells.groupby(["family", "baseline"]).size().items()):
            logger.info("replacing %-28s %-20s %d cell(s) from %s", bl, fam, n, upd_path)
        merged = pd.concat([merged[~hit], upd], ignore_index=True)

    stats["kept"] = len(merged) - stats["added"]
    if stats["new_cells"]:
        logger.warning(
            "%d update cell(s) had no counterpart in the base run (added as new "
            "cells) — check the rerun config matches the original's problems.",
            stats["new_cells"],
        )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    logger.info(
        "wrote %s: %d rows (%d kept from base, %d replaced by %d new)",
        out, len(merged), stats["kept"], stats["dropped"], stats["added"],
    )
    return stats
