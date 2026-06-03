#!/usr/bin/env python3
"""Generate paper figures + LaTeX tables from a benchmark parquet.

Implements the spec in ``docs/v0.13-paper-figures.md`` (commit f70fff7).
Supports the ``iqm_iqr`` (default) and ``mean_std`` aggregation flags,
propagated uniformly to every numeric aggregation (accuracy AND time).

Per (benchmark x family x size) cell it produces:
  - 1 success-rate bar chart (always shown; spec 5.1)
  - accuracy scaling figures: {tv,jsd,w1,log_likelihood}_per_node x
    {n_nodes, n_parameters}; w1 skipped for family==discrete (spec 5.2/5.4)
  - total-query-time scaling: vs {n_nodes, n_parameters} (spec 3.4/5.3)
  - fit_time_s scaling: vs {n_nodes, n_parameters}
and, per (benchmark x family x size):
  - 1 headline LaTeX table (rows=baselines, cols=metric x role cross)
  - per-role and per-kind supplementary tables

n_parameters is read from the parquet if a column exists; otherwise the
corresponding figures are log-skipped (the smoke parquets lack it; the
paper-relaunch parquet is expected to carry it -- see the follow-up issue).

Usage:
    python scripts/make_paper_figures.py \
        --parquet <path> --output-dir <path> \
        [--aggregation iqm_iqr|mean_std] [--benchmark bnlearn|synthetic]

Reference: docs/v0.13-paper-figures.md
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("make_paper_figures")


# --- Constants from the spec --------------------------------------------------

# Accuracy metrics (spec 5.2). log_likelihood is gated on row existence
# (PL-mode only) and is higher-is-better.
ACCURACY_METRICS = ("tv_per_node", "jsd_per_node", "w1_per_node", "log_likelihood")
LOWER_IS_BETTER = frozenset({"tv_per_node", "jsd_per_node", "w1_per_node"})
HIGHER_IS_BETTER = frozenset({"log_likelihood"})
# Pretty labels for tables / axes.
METRIC_LABEL = {
    "tv_per_node": "TV",
    "jsd_per_node": "JSD",
    "w1_per_node": "W1",
    "log_likelihood": "LL",
}

# Families that skip w1_per_node (Wasserstein-1 N/A for discrete posteriors).
DISCRETE_FAMILIES = frozenset({"discrete"})

# Library -> base color (v0.12 convention).
LIBRARY_COLORS = {
    "pgmpy": "tab:blue",
    "nbn": "tab:red",
    "pomegranate": "tab:purple",
    "pyro": "tab:brown",
}
_FALLBACK_COLOR = "tab:gray"

# Collapse the 5 registry size_class buckets to 3 headline buckets (spec 4.1).
SIZE_COLLAPSE = {
    "small": "small",
    "medium": "medium",
    "large": "large+",
    "very_large": "large+",
    "massive": "large+",
}
HEADLINE_SIZES = ("small", "medium", "large+", "overall")

# Provisional synthetic size thresholds on n_nodes (spec 4.2: synthetic
# taxonomy is TBD; this lets the per-size decomposition run until it lands).
def _synthetic_size_bucket(n_nodes: int) -> str:
    if n_nodes < 20:
        return "small"
    if n_nodes < 60:
        return "medium"
    return "large+"


# --- Pure helpers -------------------------------------------------------------

def aggregate(values, method: str) -> tuple[float, float, float]:
    """Return (center, lower_band, upper_band) per the aggregation flag.

    mean_std: center = mean, band = +/-1 std.
    iqm_iqr:  center = interquartile mean (mean of values in [Q1, Q3]),
              band   = +/-(Q3-Q1)/2 around the IQM.
    NaNs are dropped first; empty -> (nan, nan, nan).
    """
    values = np.asarray(list(values), dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if method == "mean_std":
        c = float(np.mean(values))
        s = float(np.std(values, ddof=0))
        return c, c - s, c + s
    if method == "iqm_iqr":
        q1, q3 = np.percentile(values, [25, 75])
        in_range = (values >= q1) & (values <= q3)
        c = float(np.mean(values[in_range])) if in_range.any() else float(np.median(values))
        half = float((q3 - q1) / 2)
        return c, c - half, c + half
    raise ValueError(f"unknown aggregation: {method!r}")


def clip_band(metric_kind: str, lower: float, upper: float) -> tuple[float, float]:
    """Clip a band to the metric's natural range (spec 3.2)."""
    if metric_kind == "time" or metric_kind in LOWER_IS_BETTER:
        return max(0.0, lower), upper          # nonneg
    if metric_kind in {"tv_per_node", "jsd_per_node"}:
        return max(0.0, lower), min(1.0, upper)  # bounded [0,1]
    if metric_kind in HIGHER_IS_BETTER:
        return lower, upper                     # log_likelihood: unbounded
    if metric_kind == "success_rate":
        return max(0.0, lower), min(100.0, upper)
    return lower, upper


def parse_baseline(baseline: str) -> tuple[str, str]:
    """('nbn-cat-ve') -> ('nbn', 'cat-ve'). Library is always the first token;
    the remainder is kept as an opaque variant label (token counts vary across
    libraries, e.g. pgmpy-mle-ve vs pyro-empirical-importance)."""
    parts = baseline.split("-", 1)
    return (parts[0], parts[1] if len(parts) > 1 else "")


def baseline_colors(baselines) -> dict[str, tuple]:
    """Map each baseline -> an RGBA color: the library base color, lightened
    by a distinct factor per baseline within the same library so lines are
    visually separable on white."""
    by_lib: dict[str, list[str]] = {}
    for b in sorted(baselines):
        lib = parse_baseline(b)[0]
        by_lib.setdefault(lib, []).append(b)
    colors: dict[str, tuple] = {}
    for lib, members in by_lib.items():
        base = np.array(matplotlib.colors.to_rgb(LIBRARY_COLORS.get(lib, _FALLBACK_COLOR)))
        n = len(members)
        for i, b in enumerate(members):
            # blend toward white by up to ~0.55 across members
            t = 0.0 if n == 1 else 0.55 * i / (n - 1)
            rgb = base * (1 - t) + np.array([1.0, 1.0, 1.0]) * t
            colors[b] = (*rgb, 1.0)
    return colors


def _log_or_linear(ax, vals, axis: str) -> None:
    """Use log scale if the positive dynamic range exceeds ~1.5 decades."""
    pos = [v for v in vals if v is not None and v > 0 and not np.isnan(v)]
    if len(pos) >= 2 and max(pos) / min(pos) > 30:
        (ax.set_xscale if axis == "x" else ax.set_yscale)("log")


# --- Lookups ------------------------------------------------------------------

def n_nodes_lookup(benchmark: str, problem_ids) -> dict[str, int]:
    if benchmark == "bnlearn":
        from benchmarking.problems.bnlearn import _NETWORKS
        return {p: _NETWORKS[p]["n_nodes"] for p in problem_ids if p in _NETWORKS}
    # synthetic: problem_id is n_nodes as a string
    out = {}
    for p in problem_ids:
        try:
            out[p] = int(p)
        except (TypeError, ValueError):
            pass
    return out


def size_bucket_lookup(benchmark: str, problem_ids, n_nodes: dict[str, int]) -> dict[str, str]:
    if benchmark == "bnlearn":
        from benchmarking.problems.bnlearn import _NETWORKS
        return {
            p: SIZE_COLLAPSE[_NETWORKS[p]["size_class"]]
            for p in problem_ids if p in _NETWORKS
        }
    # synthetic: provisional threshold on n_nodes (spec 4.2 defers this)
    return {p: _synthetic_size_bucket(n_nodes[p]) for p in problem_ids if p in n_nodes}


def n_parameters_lookup(df: pd.DataFrame) -> dict[tuple[str, str], float] | None:
    """Map ``(problem_id, family) -> n_parameters`` from the parquet (#133),
    or None if the column is absent.

    Keyed by ``(problem_id, family)`` rather than ``problem_id`` alone because
    in the synthetic benchmark the same ``problem_id`` (n_nodes) recurs across
    families with different n_parameters (e.g. discrete n=10 -> 256, hybrid -> 72,
    continuous -> 0); keying by problem_id alone would collapse them to whichever
    family sorted first. For bnlearn each problem_id maps to a single family, so
    the extra key is a no-op there (behavior unchanged).
    """
    if "n_parameters" not in df.columns:
        return None
    sub = df[["problem_id", "family", "n_parameters"]].dropna(subset=["n_parameters"])
    if sub.empty:
        return None
    g = sub.groupby(["problem_id", "family"])["n_parameters"].first()
    return {(p, f): float(v) for (p, f), v in g.items()}


# --- Per-query extraction (the melted schema) ---------------------------------
# Each query emits 6 metric rows (tv/jsd/w1_per_node + fit/query/metrics_time_s).
# query_time_s / fit_time_s also appear as dedicated columns (duplicated per row).
# Whole-cell-unsupported baselines emit a single metric=="status" sentinel row.

def per_query_success(df_cell: pd.DataFrame) -> dict[str, float]:
    """Query-level success rate (%) per baseline (spec 5.1).

    A query's execution status is taken from its metric=="query_time_s" row
    (one per executed query); metric=="status" rows are whole-cell unsupported
    units that count as failures.
    """
    out = {}
    for b, g in df_cell.groupby("baseline"):
        executed = g[g["metric"] == "query_time_s"]
        sentinel = g[g["metric"] == "status"]
        total = len(executed) + len(sentinel)
        if total == 0:
            out[b] = 0.0
            continue
        ok = int((executed["status"] == "ok").sum())
        out[b] = 100.0 * ok / total
    return out


def query_time_totals(df_cell: pd.DataFrame) -> pd.DataFrame:
    """Per-(baseline, problem_id, seed) total query time (spec 3.4): sum of
    query_time_s over the metric=="query_time_s", status=="ok" rows."""
    q = df_cell[(df_cell["metric"] == "query_time_s") & (df_cell["status"] == "ok")]
    if q.empty:
        return pd.DataFrame(columns=["baseline", "problem_id", "seed", "total"])
    g = q.groupby(["baseline", "problem_id", "seed"])["value"].sum().reset_index()
    return g.rename(columns={"value": "total"})


def fit_times(df_cell: pd.DataFrame) -> pd.DataFrame:
    """Per-(baseline, problem_id, seed) fit_time_s (one value per cell)."""
    f = df_cell[(df_cell["metric"] == "fit_time_s") & (df_cell["status"] == "ok")]
    if f.empty:
        return pd.DataFrame(columns=["baseline", "problem_id", "seed", "total"])
    g = f.groupby(["baseline", "problem_id", "seed"])["value"].first().reset_index()
    return g.rename(columns={"value": "total"})


# --- Figures ------------------------------------------------------------------

def _savefig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_success_rate(df_cell, out_path: Path, title: str) -> None:
    rates = per_query_success(df_cell)
    if not rates:
        logger.info("skip empty (no baselines): %s", out_path.name)
        return
    baselines = sorted(rates)
    colors = baseline_colors(baselines)
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(baselines)), 4))
    ax.bar(range(len(baselines)), [rates[b] for b in baselines],
           color=[colors[b] for b in baselines])
    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels(baselines, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate (%)")
    ax.set_title(f"{title} — success rate")
    _savefig(fig, out_path)


def _scaling_plot(points_by_baseline, x_label, y_label, title, out_path, metric_kind):
    """points_by_baseline: {baseline: [(x, center, lo, hi), ...]}."""
    if not points_by_baseline:
        logger.info("skip empty (no conditioned data): %s", out_path.name)
        return
    colors = baseline_colors(points_by_baseline.keys())
    fig, ax = plt.subplots(figsize=(6, 4))
    all_x, all_y = [], []
    for b in sorted(points_by_baseline):
        pts = sorted(points_by_baseline[b], key=lambda r: r[0])
        xs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        los = [clip_band(metric_kind, p[2], p[3])[0] for p in pts]
        his = [clip_band(metric_kind, p[2], p[3])[1] for p in pts]
        ax.plot(xs, cs, marker="o", color=colors[b], label=b, markersize=4)
        ax.fill_between(xs, los, his, color=colors[b], alpha=0.2)
        all_x += xs
        all_y += cs + los + his
    _log_or_linear(ax, all_x, "x")
    _log_or_linear(ax, [v for v in all_y if v is not None and v > 0], "y")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    _savefig(fig, out_path)


def fig_accuracy_scaling(df_cell, metric, x_axis, x_lookup, aggregation, out_path, title):
    ok = df_cell[(df_cell["status"] == "ok") & (df_cell["metric"] == metric)]
    if ok.empty:
        logger.info("skip empty (no ok rows for %s): %s", metric, out_path.name)
        return
    success = per_query_success(df_cell)
    points: dict[str, list] = {}
    for b, gb in ok.groupby("baseline"):
        if success.get(b, 0.0) <= 0.0:    # Policy 3: condition on success>0
            continue
        rows = []
        for p, gp in gb.groupby("problem_id"):
            if p not in x_lookup:
                continue
            c, lo, hi = aggregate(gp["value"], aggregation)
            if np.isnan(c):
                continue
            rows.append((x_lookup[p], c, lo, hi))
        if rows:
            points[b] = rows
    direction = "lower better" if metric in LOWER_IS_BETTER else "higher better"
    _scaling_plot(points, x_axis, f"{METRIC_LABEL[metric]} ({direction})",
                  f"{title} — {METRIC_LABEL[metric]} vs {x_axis}", out_path, metric)


def fig_time_scaling(df_cell, time_kind, x_axis, x_lookup, aggregation, out_path, title):
    """time_kind in {'query_total', 'fit'}."""
    totals = query_time_totals(df_cell) if time_kind == "query_total" else fit_times(df_cell)
    if totals.empty:
        logger.info("skip empty (no %s data): %s", time_kind, out_path.name)
        return
    success = per_query_success(df_cell)
    points: dict[str, list] = {}
    for b, gb in totals.groupby("baseline"):
        if success.get(b, 0.0) <= 0.0:
            continue
        rows = []
        for p, gp in gb.groupby("problem_id"):
            if p not in x_lookup:
                continue
            c, lo, hi = aggregate(gp["total"], aggregation)   # across seeds
            if np.isnan(c):
                continue
            rows.append((x_lookup[p], c, lo, hi))
        if rows:
            points[b] = rows
    label = "Total query time (s)" if time_kind == "query_total" else "Fit time (s)"
    _scaling_plot(points, x_axis, label, f"{title} — {label} vs {x_axis}", out_path, "time")


# --- LaTeX tables -------------------------------------------------------------

def _fmt(center: float, lo: float, hi: float) -> str:
    if np.isnan(center):
        return "--"
    half = (hi - lo) / 2
    return f"{center:.3g}$\\pm${half:.2g}"


def _metric_cell(df_cell, baseline, metric, aggregation, role=None, kind=None) -> str:
    sub = df_cell[(df_cell["baseline"] == baseline) & (df_cell["metric"] == metric)
                  & (df_cell["status"] == "ok")]
    if role is not None:
        sub = sub[sub["query_role"] == role]
    if kind is not None:
        sub = sub[sub["query_kind"] == kind]
    if sub.empty:
        return "--"
    return _fmt(*aggregate(sub["value"], aggregation))


def _time_cell(df_cell, baseline, aggregation) -> str:
    t = query_time_totals(df_cell)
    t = t[t["baseline"] == baseline]
    if t.empty:
        return "--"
    return _fmt(*aggregate(t["total"], aggregation))


def _write_table(out_path: Path, header_cols, rows, caption_note=""):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(header_cols)
    lines = [
        "% " + caption_note if caption_note else "% auto-generated; paste into paper",
        "\\begin{tabular}{l" + "r" * (n - 1) + "}",
        "\\toprule",
        " & ".join(header_cols) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out_path.write_text("\n".join(lines))


def _metrics_for_family(family) -> list[str]:
    metrics = list(ACCURACY_METRICS)
    if family in DISCRETE_FAMILIES:
        metrics = [m for m in metrics if m != "w1_per_node"]
    return metrics


def table_headline(df_cell, family, size, aggregation, roles, out_path):
    """Rows=baselines; cols = Success%, then per metric (overall + per role),
    then Total time (spec 6). log_likelihood / w1 columns appear only when data
    exists for that metric in the cell."""
    metrics = [m for m in _metrics_for_family(family)
               if not df_cell[(df_cell["metric"] == m) & (df_cell["status"] == "ok")].empty]
    success = per_query_success(df_cell)
    role_order = ["overall"] + sorted(roles)
    header = ["Method", "Succ.\\%"]
    for m in metrics:
        for r in role_order:
            header.append(f"{METRIC_LABEL[m]} ({r})")
    header.append("Time (s)")
    rows = []
    for b in sorted(success):
        cells = [b.replace("_", "\\_"), f"{success[b]:.0f}"]
        zero = success[b] <= 0.0
        for m in metrics:
            for r in role_order:
                cells.append("--" if zero else
                             _metric_cell(df_cell, b, m, aggregation,
                                          role=None if r == "overall" else r))
        cells.append("--" if zero else _time_cell(df_cell, b, aggregation))
        rows.append(cells)
    _write_table(out_path, header, rows,
                 f"headline {family}/{size}; agg={aggregation}; "
                 f"{len(header)} cols (metric x role cross)")


def _table_scoped(df_cell, family, aggregation, scope_col, scope_val, out_path, note):
    metrics = [m for m in _metrics_for_family(family)
               if not df_cell[(df_cell["metric"] == m) & (df_cell["status"] == "ok")].empty]
    sub = df_cell[df_cell[scope_col] == scope_val]
    success = per_query_success(sub)
    if not success:
        return
    header = ["Method", "Succ.\\%"] + [METRIC_LABEL[m] for m in metrics] + ["Time (s)"]
    rows = []
    for b in sorted(success):
        zero = success[b] <= 0.0
        cells = [b.replace("_", "\\_"), f"{success[b]:.0f}"]
        for m in metrics:
            cells.append("--" if zero else
                         _metric_cell(sub, b, m, aggregation,
                                      **{("role" if scope_col == "query_role" else "kind"): scope_val}))
        cells.append("--" if zero else _time_cell(sub, b, aggregation))
        rows.append(cells)
    _write_table(out_path, header, rows, note)


# --- Orchestration ------------------------------------------------------------

def process_cell(df_cell, benchmark, family, size, aggregation, n_nodes, n_params, cell_dir):
    cell_dir.mkdir(parents=True, exist_ok=True)
    title = f"{benchmark}/{family}/{size}"
    # x-axes available. n_params is keyed by (problem_id, family); scope it to
    # this cell's family and flatten to {problem_id: value} so the scaling
    # helpers can look up by problem_id as for n_nodes.
    axes = [("n_nodes", n_nodes)]
    if n_params is not None:
        n_params_family = {p: v for (p, f), v in n_params.items() if f == family}
        if n_params_family:
            axes.append(("n_parameters", n_params_family))

    fig_success_rate(df_cell, cell_dir / "success_rate.pdf", title)

    for metric in ACCURACY_METRICS:
        if family in DISCRETE_FAMILIES and metric == "w1_per_node":
            continue
        for x_axis, lookup in axes:
            fig_accuracy_scaling(df_cell, metric, x_axis, lookup, aggregation,
                                 cell_dir / f"{metric}_vs_{x_axis}.pdf", title)

    for time_kind, stem in [("query_total", "total_query_time"), ("fit", "fit_time")]:
        for x_axis, lookup in axes:
            fig_time_scaling(df_cell, time_kind, x_axis, lookup, aggregation,
                             cell_dir / f"{stem}_vs_{x_axis}.pdf", title)

    # Tables
    roles = sorted(df_cell["query_role"].dropna().unique())
    kinds = sorted(df_cell["query_kind"].dropna().unique())
    table_headline(df_cell, family, size, aggregation, roles, cell_dir / "table_headline.tex")
    for r in roles:
        _table_scoped(df_cell, family, aggregation, "query_role", r,
                      cell_dir / f"table_role_{r}.tex", f"role={r} {family}/{size}")
    for k in kinds:
        _table_scoped(df_cell, family, aggregation, "query_kind", k,
                      cell_dir / f"table_kind_{k}.tex", f"kind={k} {family}/{size}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--aggregation", choices=["iqm_iqr", "mean_std"], default="iqm_iqr")
    ap.add_argument("--benchmark", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_parquet(args.parquet)
    required = {"benchmark", "family", "problem_id", "seed", "baseline",
                "metric", "value", "status", "query_role", "query_kind"}
    missing = required - set(df.columns)
    if missing:
        logger.error("parquet missing required columns: %s", missing)
        return 1

    n_params_global = n_parameters_lookup(df)
    if n_params_global is None:
        logger.info("n_parameters column absent; skipping *_vs_n_parameters "
                    "figures (will populate from the paper-relaunch parquet)")

    benchmarks = [args.benchmark] if args.benchmark else sorted(df["benchmark"].dropna().unique())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    skipped, produced = [], 0
    for benchmark in benchmarks:
        dfb = df[df["benchmark"] == benchmark].copy()
        if dfb.empty:
            continue
        pids = sorted(dfb["problem_id"].dropna().unique())
        n_nodes = n_nodes_lookup(benchmark, pids)
        size_bucket = size_bucket_lookup(benchmark, pids, n_nodes)
        dfb["__size"] = dfb["problem_id"].map(size_bucket)

        for family in sorted(dfb["family"].dropna().unique()):
            dff = dfb[dfb["family"] == family]
            for size in HEADLINE_SIZES:
                cell = dff if size == "overall" else dff[dff["__size"] == size]
                if cell.empty:
                    skipped.append(f"{benchmark}/{family}/{size}")
                    logger.info("skip empty cell: %s/%s/%s", benchmark, family, size)
                    continue
                process_cell(cell, benchmark, family, size, args.aggregation,
                             n_nodes, n_params_global,
                             args.output_dir / benchmark / family / size)
                produced += 1

    logger.info("done: %d cells produced, %d cells skipped (empty)", produced, len(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
