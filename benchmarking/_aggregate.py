"""v0.6c-C-3 — aggregator: parquet → 4 summary DataFrames.

Consumes the raw metrics parquet from a crash-test run (schema:
``family, n_nodes, seed, baseline, metric, value, status, n_skipped,
error_msg``) and produces:

* ``wide``: pivot table indexed by ``(family, metric, n_nodes)`` with
  baselines as columns.  Values are formatted strings ``"mean ± std"``
  (or ``"mean"`` if a single seed) for ok cells; ``"n/a (...)"`` for
  cells that are not applicable, errored, or where the runner failed
  to emit the metric (issue #35 workaround).

* ``long``: tidy long-format dataframe one row per
  ``(family, baseline, n_nodes, metric)``, with columns ``mean``,
  ``std``, ``n_ok``, ``n_seeds``, ``status_code``.

* ``status``: per-cell status counts (ok / not_supported / error /
  timeout / oom / no_result), used for figure footnotes.

* ``pareto``: per ``(family, n_nodes)``, the (time, accuracy) Pareto
  frontier — which baselines are non-dominated.  Used by the plotter
  for highlighting and by the LaTeX writer for boldface.

Issue references:
* #35 — runner emits one metric per cell instead of two.  Aggregator
  reports the missing metric as ``"n/a (metric missing)"``.  Workaround
  only; runtime fix tracked separately.
* #37 — pgmpy-lg-predict W₁ outlier on continuous_lg.  Aggregator
  reports the value honestly; tables and plots show it as-is.
* #30 — HybridRouter cuda assert at n≥10.  Cells with all-error
  status get ``"n/a (cell errored)"``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from benchmarking._baseline_registry import is_applicable


_NA_NOT_APPLICABLE = "n/a (not applicable)"
_NA_METRIC_MISSING = "n/a (metric missing)"
_NA_CELL_ERRORED = "n/a (cell errored)"


def _fmt_value(mean: float, std: float, n_ok: int) -> str:
    """Format a numeric cell.  Single seed → ``"0.0429"``; multi-seed
    → ``"0.0429 ± 0.005"``."""
    if n_ok == 1:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _classify_cell(
    cell_rows: pd.DataFrame, n_seeds: int, applicable: bool,
) -> tuple[str, float, float, int]:
    """Return ``(formatted_string, mean, std, n_ok)`` for a single
    ``(family, baseline, n_nodes, metric)`` cell.

    Priority of "n/a" reasons (most informative first):
    1. Not applicable (registry says so).
    2. Cell errored on every seed.
    3. Metric was missing (no rows at all for this metric — usually
       issue #35).
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
    # Issue #35: the runner sometimes emits status='ok' rows with NaN
    # value (the metric was not populated for that baseline path).
    # Treat as metric-missing rather than formatting "nan" into the
    # output table.
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return _NA_METRIC_MISSING, np.nan, np.nan, 0
    mean = float(np.mean(values))
    if len(values) == 1:
        std = 0.0
    else:
        std = float(np.std(values, ddof=1))
    return _fmt_value(mean, std, len(values)), mean, std, len(values)


def aggregate(parquet_path: str | Path) -> Dict[str, pd.DataFrame]:
    """Aggregate a raw metrics parquet into 4 summary tables.

    See module docstring for output schema.
    """
    df = pd.read_parquet(parquet_path)

    families = sorted(df["family"].unique())
    baselines = sorted(df["baseline"].unique())
    n_nodes_list = sorted(df["n_nodes"].unique())
    # ``status`` and value-bearing metric rows coexist in the parquet.
    # The runner sometimes emits a synthetic ``metric='status'`` row for
    # not_supported / error cells; we exclude these from the metric
    # listing.
    metrics = sorted(m for m in df["metric"].unique() if m != "status")

    n_seeds = int(df["seed"].nunique())

    long_rows: List[Dict] = []
    wide_index = []
    wide_data: Dict[str, List[str]] = {b: [] for b in baselines}

    for family in families:
        for metric in metrics:
            for n in n_nodes_list:
                wide_index.append((family, metric, n))
                for baseline in baselines:
                    applicable = is_applicable(baseline, family)
                    cell_rows = df[
                        (df["family"] == family)
                        & (df["baseline"] == baseline)
                        & (df["n_nodes"] == n)
                        & (df["metric"] == metric)
                    ]
                    formatted, mean, std, n_ok = _classify_cell(
                        cell_rows, n_seeds=n_seeds, applicable=applicable,
                    )
                    wide_data[baseline].append(formatted)
                    long_rows.append({
                        "family": family,
                        "baseline": baseline,
                        "n_nodes": n,
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
            wide_index, names=["family", "metric", "n_nodes"],
        ),
    )
    long = pd.DataFrame(long_rows)

    # Status counts: per (family, baseline, n_nodes), how many seeds
    # each status code accumulated.
    status = (
        df.groupby(["family", "baseline", "n_nodes", "status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Pareto frontier: per (family, n_nodes), find the speed/accuracy
    # frontier across baselines.  Lower is better on both axes.
    pareto = _compute_pareto(long)

    return {"wide": wide, "long": long, "status": status, "pareto": pareto}


def _compute_pareto(long: pd.DataFrame) -> pd.DataFrame:
    """For each (family, n_nodes), determine which baselines lie on the
    (mean_time, mean_accuracy) Pareto frontier.

    A baseline ``b1`` dominates ``b2`` iff
    ``time(b1) <= time(b2) AND accuracy(b1) <= accuracy(b2)`` and at
    least one inequality is strict.  Pareto-optimal baselines are those
    NOT dominated by any other.  Cells with NaN on either axis are
    excluded from the frontier (no comparable point).
    """
    # Pivot to one row per (family, baseline, n_nodes) with two value
    # columns: total_time_s mean and accuracy mean.
    pivoted = long.pivot_table(
        index=["family", "baseline", "n_nodes"],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()
    # ``accuracy`` lives under several metric names depending on the
    # crash test (``accuracy`` for inference; ``tv_per_node`` /
    # ``w1_per_node`` for parameter-learning).  Coalesce to a single
    # ``acc`` column so the Pareto logic is metric-agnostic.
    pivoted["acc"] = pivoted.get(
        "accuracy",
        pivoted.get("tv_per_node", pivoted.get("w1_per_node")),
    )
    if "accuracy" not in pivoted.columns:
        for col in ("tv_per_node", "w1_per_node"):
            if col in pivoted.columns:
                pivoted["acc"] = pivoted["acc"].fillna(pivoted[col])
    pivoted["time"] = pivoted.get("total_time_s")

    out_rows: List[Dict] = []
    for (family, n), grp in pivoted.groupby(["family", "n_nodes"]):
        finite = grp.dropna(subset=["acc", "time"])
        for _, row in grp.iterrows():
            baseline = row["baseline"]
            t = row.get("time")
            a = row.get("acc")
            if pd.isna(t) or pd.isna(a):
                is_pareto = False
                dominated_by: List[str] = []
            else:
                dominated_by = []
                for _, other in finite.iterrows():
                    ot, oa = other["time"], other["acc"]
                    if other["baseline"] == baseline:
                        continue
                    # ``other`` dominates if its time AND acc are <=
                    # ours, with at least one strictly less.
                    if ot <= t and oa <= a and (ot < t or oa < a):
                        dominated_by.append(other["baseline"])
                is_pareto = len(dominated_by) == 0
            out_rows.append({
                "family": family,
                "baseline": baseline,
                "n_nodes": n,
                "is_pareto": is_pareto,
                "dominated_by": tuple(dominated_by),
            })
    return pd.DataFrame(out_rows)


__all__ = [
    "aggregate",
    "_NA_NOT_APPLICABLE",
    "_NA_METRIC_MISSING",
    "_NA_CELL_ERRORED",
]
