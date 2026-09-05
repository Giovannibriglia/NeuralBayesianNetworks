"""v0.13 aggregator: parquet → 4 summary DataFrames.

Consumes a v3-schema metrics parquet (columns: ``family``, ``problem_id``,
``seed``, ``baseline``, ``metric``, ``value``, ``status``, ``fit_time_s``,
``query_time_s``, ``metrics_time_s``, ``error_msg``) and produces:

* ``wide``: pivot table indexed by ``(family, metric, problem_id)`` with
  baselines as columns.  Values are formatted strings ``"mean ± std"``
  (or ``"mean"`` if a single seed) for ok cells; ``"n/a (...)"`` for
  cells that are not applicable, errored, or where the runner failed
  to emit the metric.

* ``long``: tidy long-format DataFrame, one row per
  ``(family, baseline, problem_id, metric)``, with columns ``mean``,
  ``std``, ``n_ok``, ``n_seeds``, ``applicable``, ``formatted``.

* ``status``: per-cell status counts (ok / not_supported / error /
  timeout / oom / no_result), used for figure footers.

* ``pareto``: per ``(family, problem_id)``, the (time, accuracy) Pareto
  frontier — which baselines are non-dominated.

Backward compatibility: v0.12 parquets that have ``n_nodes`` (int) instead
of ``problem_id`` (str) are transparently up-cast at load time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from nbn.bench.core.applicability import is_applicable


_NA_NOT_APPLICABLE = "n/a (not applicable)"
_NA_METRIC_MISSING = "n/a (metric missing)"
_NA_CELL_ERRORED   = "n/a (cell errored)"

# Numeric sort key for problem_id values.
# Synthetic problem_ids are always numeric strings ("5", "10", "100");
# bnlearn problem_ids are names ("asia").  This key sorts numerically
# when all values are digit strings, lexicographically otherwise.
_NUMERIC_SORT_KEY = lambda x: (0, int(x)) if str(x).isdigit() else (1, x)  # noqa: E731


def _fmt_value(mean: float, std: float, n_ok: int) -> str:
    """Format a numeric cell.  Single seed → ``"0.0429"``; multi-seed
    → ``"0.0429 ± 0.005"``."""
    if n_ok == 1:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _classify_cell(
    cell_rows: pd.DataFrame, n_seeds: int, applicable: bool,
) -> tuple[str, float, float, int]:
    """Return ``(formatted_string, mean, std, n_ok)`` for one
    ``(family, baseline, problem_id, metric)`` cell.

    Priority of "n/a" reasons (most informative first):
    1. Not applicable (registry says so).
    2. Cell errored on every seed.
    3. Metric was missing (no rows at all for this metric).
    4. ok rows present → numeric value.
    """
    if not applicable:
        return _NA_NOT_APPLICABLE, np.nan, np.nan, 0
    if len(cell_rows) == 0:
        return _NA_METRIC_MISSING, np.nan, np.nan, 0
    ok_rows = cell_rows[cell_rows["status"] == "ok"]
    if len(ok_rows) == 0:
        return _NA_CELL_ERRORED, np.nan, np.nan, 0
    values = ok_rows["value"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return _NA_METRIC_MISSING, np.nan, np.nan, 0
    mean = float(np.mean(values))
    std = 0.0 if len(values) == 1 else float(np.std(values, ddof=1))
    return _fmt_value(mean, std, len(values)), mean, std, len(values)


def aggregate(parquet_path: str | Path) -> Dict[str, pd.DataFrame]:
    """Aggregate a raw metrics parquet into 4 summary tables.

    See module docstring for output schema.
    """
    df = pd.read_parquet(parquet_path)

    # Backward-compat: v0.12 parquets use n_nodes (int); promote to problem_id (str).
    if "n_nodes" in df.columns and "problem_id" not in df.columns:
        df = df.rename(columns={"n_nodes": "problem_id"})
        df["problem_id"] = df["problem_id"].astype(str)

    families = sorted(df["family"].unique())
    baselines = sorted(df["baseline"].unique())
    problem_id_list = sorted(df["problem_id"].unique(), key=_NUMERIC_SORT_KEY)
    # Exclude the synthetic ``metric='status'`` sentinel rows from the
    # metric listing — they are status markers, not measurement values.
    metrics = sorted(m for m in df["metric"].unique() if m != "status")

    n_seeds = int(df["seed"].nunique())

    long_rows: List[Dict] = []
    wide_index = []
    wide_data: Dict[str, List[str]] = {b: [] for b in baselines}

    for family in families:
        for metric in metrics:
            for pid in problem_id_list:
                wide_index.append((family, metric, pid))
                for baseline in baselines:
                    applicable = is_applicable(baseline, family)
                    cell_rows = df[
                        (df["family"] == family)
                        & (df["baseline"] == baseline)
                        & (df["problem_id"] == pid)
                        & (df["metric"] == metric)
                    ]
                    formatted, mean, std, n_ok = _classify_cell(
                        cell_rows, n_seeds=n_seeds, applicable=applicable,
                    )
                    wide_data[baseline].append(formatted)
                    long_rows.append({
                        "family": family,
                        "baseline": baseline,
                        "problem_id": pid,
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                        "n_ok": n_ok,
                        "n_seeds": n_seeds,
                        "applicable": applicable,
                        "formatted": formatted,
                    })

    wide = pd.DataFrame(
        wide_data,
        index=pd.MultiIndex.from_tuples(
            wide_index, names=["family", "metric", "problem_id"],
        ),
    )
    long = pd.DataFrame(long_rows)

    # Status counts: per (family, baseline, problem_id), how many seeds
    # each status code accumulated.
    status = (
        df.groupby(["family", "baseline", "problem_id", "status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    pareto = _compute_pareto(long)

    return {"wide": wide, "long": long, "status": status, "pareto": pareto}


def _compute_pareto(long: pd.DataFrame) -> pd.DataFrame:
    """For each (family, problem_id), determine which baselines lie on the
    (mean_time, mean_accuracy) Pareto frontier.

    A baseline ``b1`` dominates ``b2`` iff
    ``time(b1) <= time(b2) AND accuracy(b1) <= accuracy(b2)`` and at
    least one inequality is strict.  Pareto-optimal baselines are those
    NOT dominated by any other.
    """
    if long.empty:
        return pd.DataFrame(columns=["family", "baseline", "problem_id",
                                     "is_pareto", "dominated_by"])

    pivoted = long.pivot_table(
        index=["family", "baseline", "problem_id"],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()

    # Coalesce accuracy metrics into a single ``acc`` column.
    acc_col = None
    for cand in ("accuracy", "tv_per_node", "w1_per_node"):
        if cand in pivoted.columns:
            acc_col = cand
            break
    if acc_col is not None:
        pivoted["acc"] = pivoted[acc_col]
        for col in ("tv_per_node", "w1_per_node"):
            if col in pivoted.columns and col != acc_col:
                pivoted["acc"] = pivoted["acc"].fillna(pivoted[col])
    else:
        pivoted["acc"] = np.nan

    time_col = "total_time_s" if "total_time_s" in pivoted.columns else "query_time_s"
    pivoted["time"] = pivoted.get(time_col, pd.Series(np.nan, index=pivoted.index))

    out_rows: List[Dict] = []
    for (family, pid), grp in pivoted.groupby(["family", "problem_id"]):
        finite = grp.dropna(subset=["acc", "time"])
        for _, row in grp.iterrows():
            baseline = row["baseline"]
            t = row.get("time")
            a = row.get("acc")
            if pd.isna(t) or pd.isna(a):
                out_rows.append({
                    "family": family, "baseline": baseline,
                    "problem_id": pid, "is_pareto": False, "dominated_by": (),
                })
                continue
            dominated_by = [
                other["baseline"] for _, other in finite.iterrows()
                if other["baseline"] != baseline
                and other["time"] <= t and other["acc"] <= a
                and (other["time"] < t or other["acc"] < a)
            ]
            out_rows.append({
                "family": family, "baseline": baseline,
                "problem_id": pid,
                "is_pareto": len(dominated_by) == 0,
                "dominated_by": tuple(dominated_by),
            })
    return pd.DataFrame(out_rows)


__all__ = [
    "aggregate",
    "_NA_NOT_APPLICABLE",
    "_NA_METRIC_MISSING",
    "_NA_CELL_ERRORED",
]
