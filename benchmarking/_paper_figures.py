"""Generate paper figures + LaTeX tables from a benchmark parquet.

Implements the spec in ``docs/v0.13-paper-figures.md``. The public entry point
is :func:`run_plot`, invoked by the ``nbn-bench plot`` subcommand
(``benchmarking/cli.py``); ``scripts/make_paper_figures.py`` remains as a
deprecation shim.

Supports the ``iqm_iqr`` (default) and ``mean_std`` aggregation flags,
propagated uniformly to every numeric aggregation (accuracy AND time).

Output layout is per-FAMILY, with auto-discovered coverage SUBSETS under each
family, at ``<output_dir>/<bench>/<family>/``:

  all/                  every problem, every supported baseline (mixed coverage)
    plots/success_rate.pdf, <metric>_vs_<axis>.pdf,
          {total_query_time,fit_time}_vs_<axis>.pdf
    tables/table_overall.tex, table_kind_<k>.tex, table_role_<r>.tex
  common/               problems solved by the FULL supported baseline set
    methods.txt, problems.txt, plots/ (no success_rate), tables/
  subsetN/              one per auto-discovered solving-baseline set
    methods.txt, problems.txt, plots/, tables/
  _subsets_overview.txt navigation aid

A subset groups the problems whose "solving set" (baselines that are ``ok`` on
every row of the problem) is identical; its plots/tables are restricted to
those problems and baselines. ``<metric>`` in {tv,jsd,w1,log_likelihood}
_per_node (w1 skipped for family==discrete); ``<axis>`` in {n_nodes,
n_parameters}. Filenames carry no ``<family>_`` prefix (the folder names it).

x-axes: ``n_nodes`` is resolved from the parquet column (#195) with a
``_NETWORKS`` / synthetic-int fallback for older parquets (resolve_n_nodes).
``n_parameters`` is read from the parquet if present; the per-family
``*_vs_n_parameters`` file is skipped when that family's n_parameters are
all zero (continuous_gauss) -- a degenerate x=0 axis (decision alpha).

Reference: docs/v0.13-paper-figures.md
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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

# Libraries that cannot batch queries by design (adapter
# ``supports_batched_queries = False``): they are pinned to batch_size=1 in
# configs and process queries sequentially. Source of truth is the adapter
# class flag (benchmarking/adapters/{pyro,pgmpy}_adapter.py); mirrored here as
# a constant so the plotting path stays free of heavy adapter imports.
#
# Detection MUST be config/design-level (the library), NOT observational: a
# *batchable* baseline (e.g. nbn-cat-ve) that OOMs at every batch size above
# B=1 still has only ok data at B=1, but it is NOT a fixed reference — it
# renders as points-with-gaps (Change 1), never a dashed line (#148, Change 2).
_NON_BATCHABLE_LIBRARIES = frozenset({"pyro", "pgmpy"})

# Status -> color for the 100%-stacked status breakdown (spec 5.1). Green for
# the good segment; failure modes warm-progressing-to-distinct. not_supported
# is neutral gray (applicability, not failure). STATUS_ORDER fixes the stacking
# order bottom-up: ok grows from the floor, failures stacked above.
STATUS_COLORS = {
    "ok": "#2ca02c",             # green
    "not_supported": "#7f7f7f",  # gray (neutral — applicability, not failure)
    "timeout": "#ff7f0e",        # orange (over budget)
    "error": "#d62728",          # red (genuine failure)
    "oom": "#8c564b",            # brown (memory failure)
}
STATUS_ORDER = ("ok", "not_supported", "timeout", "error", "oom")

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


def resolve_n_nodes(dfb: pd.DataFrame, benchmark: str) -> dict[str, int]:
    """Map ``problem_id -> n_nodes`` for a benchmark slice.

    Priority:
      1. the parquet ``n_nodes`` column (post-PR-1 #195): used for any
         problem_id with at least one non-null value;
      2. ``n_nodes_lookup`` fallback for the rest -- ``_NETWORKS`` for
         bnlearn, ``int(problem_id)`` for synthetic (older parquets that
         predate the column still plot);
      3. anything still unresolved is omitted, so the scaling helpers
         simply skip those problems rather than crashing.

    Backward-compat diagnostics are gentle (this is "older parquet, here's
    what we did", not an error): one ``info`` when a fallback fired, one
    ``warning`` when some problem_id could not be resolved at all.
    """
    pids = sorted(dfb["problem_id"].dropna().unique())

    from_col: dict[str, int] = {}
    if "n_nodes" in dfb.columns:
        sub = dfb[["problem_id", "n_nodes"]].dropna(subset=["n_nodes"])
        if not sub.empty:
            g = sub.groupby("problem_id")["n_nodes"].first()
            from_col = {str(p): int(v) for p, v in g.items()}

    missing = [p for p in pids if p not in from_col]
    fallback = n_nodes_lookup(benchmark, missing) if missing else {}

    resolved = {**fallback, **from_col}  # column wins over fallback
    out = {p: resolved[p] for p in pids if p in resolved}

    if missing:
        src = "_NETWORKS" if benchmark == "bnlearn" else "problem_id"
        logger.info(
            "n_nodes: %d/%d problem(s) lacked the parquet column (predates "
            "PR-1 #195); resolved via %s fallback", len(missing), len(pids), src,
        )
    unresolved = [p for p in pids if p not in out]
    if unresolved:
        logger.warning(
            "n_nodes unresolved for %d problem(s) (no column, not in %s); "
            "these will be skipped in scaling figures: %s",
            len(unresolved),
            "_NETWORKS" if benchmark == "bnlearn" else "synthetic-int",
            unresolved,
        )
    return out


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


def per_query_status_counts(df_cell: pd.DataFrame) -> pd.DataFrame:
    """Per-baseline status counts over the same unit as :func:`per_query_success`.

    The counting unit is one executed query (``metric=="query_time_s"`` rows,
    one per query, carrying its per-query status: ok / timeout / error / oom /
    not_supported) *plus* the whole-cell sentinel rows (``metric=="status"``,
    one per unsupported/fit-failed unit). Both contribute to the denominator
    exactly as in ``per_query_success`` (total = executed + sentinel), so the
    stacked percentages are the per-query success rate decomposed by status.

    Returns a DataFrame indexed by baseline with one column per status in
    STATUS_ORDER (counts, reindexed with fill 0); empty if no unit rows exist.
    """
    unit = df_cell[df_cell["metric"].isin(["query_time_s", "status"])]
    if unit.empty:
        return pd.DataFrame()
    counts = unit.groupby(["baseline", "status"]).size().unstack(fill_value=0)
    # Surface any status outside the palette rather than silently dropping it.
    unknown = [c for c in counts.columns if c not in STATUS_ORDER]
    if unknown:
        logger.warning("status(es) outside stacked-bar palette dropped: %s", unknown)
    return counts.reindex(columns=list(STATUS_ORDER), fill_value=0)


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


def fig_status_stacked(df_cell, out_path: Path, title: str) -> None:
    """100%-stacked status breakdown per baseline (spec 5.1).

    Each bar sums to 100%; segments show the fraction of each status
    (ok / not_supported / timeout / error / oom) over the per-query unit.
    Replaces the prior single-segment success-rate plot — same artifact name
    (``success_rate.pdf``), richer information: the green segment is exactly the
    old success rate, and the remainder is decomposed by failure mode.
    """
    counts = per_query_status_counts(df_cell)
    totals = counts.sum(axis=1) if not counts.empty else pd.Series(dtype=float)
    counts = counts[totals > 0] if not counts.empty else counts
    if counts.empty:
        logger.info("skip empty (no baselines): %s", out_path.name)
        return
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    baselines = sorted(pct.index)
    pct = pct.loc[baselines]

    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(baselines)), 4))
    bottoms = np.zeros(len(baselines))
    for status in STATUS_ORDER:
        values = pct[status].to_numpy()
        if (values == 0).all():
            continue  # don't add a legend entry for an absent status
        ax.bar(range(len(baselines)), values, bottom=bottoms,
               color=STATUS_COLORS[status], label=status,
               edgecolor="white", linewidth=0.5)
        bottoms += values
    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels(baselines, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of queries")
    ax.set_title(f"{title} — status breakdown")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=7)
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


def _table_slug(value: str) -> str:
    """Sanitize a path component for use inside a ``\\label{tab:...}`` key."""
    return str(value).replace("+", "plus").replace(" ", "_").replace("/", "_")


def _write_table(out_path: Path, header_cols, rows, caption="", label=""):
    """Emit a full ``table`` float (booktabs). ``caption`` is plain text whose
    underscores are escaped here (it is typeset); ``label`` is used verbatim as
    the ``\\label{}`` key (not typeset, so underscores are fine)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(header_cols)
    caption_tex = (caption or "auto-generated; paste into paper").replace("_", "\\_")
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{l" + "r" * (n - 1) + "}",
        "\\toprule",
        " & ".join(header_cols) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption_tex}}}"]
    if label:
        lines.append(f"\\label{{{label}}}")
    lines += ["\\end{table}", ""]
    out_path.write_text("\n".join(lines))


def _metrics_for_family(family) -> list[str]:
    metrics = list(ACCURACY_METRICS)
    if family in DISCRETE_FAMILIES:
        metrics = [m for m in metrics if m != "w1_per_node"]
    return metrics


def table_headline(df_cell, family, size, aggregation, roles, out_path, label=""):
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
                 f"{len(header)} cols (metric x role cross)",
                 label=label)


def _table_scoped(df_cell, family, aggregation, scope_col, scope_val, out_path, note,
                  label=""):
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
    _write_table(out_path, header, rows, note, label=label)


# --- Orchestration ------------------------------------------------------------

def _filter_unsupported_baselines(dff: pd.DataFrame) -> pd.DataFrame:
    """Drop baselines whose every row in this slice is ``not_supported``,
    and drop ``not_supported`` rows from any remaining baseline.

    The semantic: ``not_supported`` means "this baseline cannot handle this
    problem" (e.g. a discrete-only baseline on a continuous family) -- it
    never participated. Showing it as a 100% failure in the success-rate
    figure (or as a row of em-dashes in the tables) is misleading noise.
    Partially-supported baselines stay; their success rate is then computed
    over the SUPPORTED subset, which is the meaningful denominator.
    """
    if "status" not in dff.columns or "baseline" not in dff.columns:
        return dff
    # Baselines with at least one non-not_supported row.
    supported = set(
        dff.loc[dff["status"] != "not_supported", "baseline"].unique()
    )
    # Drop baselines that never had a supported row, AND drop the
    # not_supported rows from baselines that survive.
    return dff[
        dff["baseline"].isin(supported)
        & (dff["status"] != "not_supported")
    ]


def _problem_solving_sets(dff: pd.DataFrame) -> dict[str, frozenset]:
    """Map ``problem_id -> frozenset(baselines that FULLY solved it)``.

    A baseline "fully solves" a problem when every one of its rows for that
    problem is ``status=="ok"`` (no timeout/error/oom). ``dff`` is assumed
    already cleaned of ``not_supported`` rows by
    :func:`_filter_unsupported_baselines`, so the only remaining non-ok
    statuses are genuine failures.
    """
    out: dict[str, frozenset] = {}
    for pid, gp in dff.groupby("problem_id"):
        solvers = {
            bl for bl, gb in gp.groupby("baseline")
            if len(gb) and (gb["status"] == "ok").all()
        }
        out[str(pid)] = frozenset(solvers)
    return out


def _discover_subsets(dff: pd.DataFrame) -> list[dict]:
    """Auto-discover baseline-coverage subsets from a family's data.

    Groups problems by their "solving set" (the baselines that fully succeed
    on every row of the problem). Each unique non-empty solving set with >=1
    problem becomes a subset:
      - ``"common"`` when the solving set equals the full supported set,
      - ``"subset1"``, ``"subset2"``, ... otherwise, numbered by descending
        baseline-set size, then descending problem count, then the
        alphabetically-first baseline (fully deterministic).

    Problems that no baseline fully solves (empty solving set) are skipped.
    Returns dicts ``{name, baselines: sorted list, problems: sorted list}``
    with ``"common"`` first (if present), then the numbered subsets.
    """
    if dff.empty:
        return []
    full = frozenset(dff["baseline"].dropna().unique())
    solving = _problem_solving_sets(dff)

    groups: dict[frozenset, list[str]] = {}
    for pid, sset in solving.items():
        if not sset:
            continue  # no baseline fully solved this problem -> skip
        groups.setdefault(sset, []).append(pid)

    common = groups.pop(full, None)

    # Deterministic ordering for the numbered subsets.
    ordered = sorted(
        groups.items(),
        key=lambda kv: (-len(kv[0]), -len(kv[1]), sorted(kv[0])[0]),
    )

    out: list[dict] = []
    if common is not None:
        out.append(dict(name="common", baselines=sorted(full),
                        problems=sorted(common)))
    for i, (sset, probs) in enumerate(ordered, start=1):
        out.append(dict(name=f"subset{i}", baselines=sorted(sset),
                        problems=sorted(probs)))
    return out


def _write_subset_metadata(subset_dir, name, baselines, problems) -> None:
    subset_dir.mkdir(parents=True, exist_ok=True)
    (subset_dir / "methods.txt").write_text("\n".join(sorted(baselines)) + "\n")
    (subset_dir / "problems.txt").write_text("\n".join(sorted(problems)) + "\n")


def _write_subsets_overview(family_dir, family, subsets) -> None:
    """One-page navigation file listing each subset, its baseline count,
    problem count, and the baselines themselves."""
    lines = [f"Subsets for family={family}", "=" * 60, ""]
    for s in subsets:
        lines.append(f"{s['name']}: {len(s['baselines'])} baselines, "
                     f"{len(s['problems'])} problems")
        lines.append(f"  baselines: {', '.join(sorted(s['baselines']))}")
        lines.append("")
    (family_dir / "_subsets_overview.txt").write_text("\n".join(lines))


def _render_view(dff, benchmark, family, aggregation, n_nodes, n_params,
                 view_dir, *, subset_name, include_success_rate):
    """Render one view (the ``all`` view or one subset) into
    ``view_dir/{plots,tables}/``.

    Filenames carry no ``<family>_`` prefix (the folder path already names the
    family and subset). Table labels are
    ``tab:<bench>_<family>_<subset>_<scope>`` -- unique across subsets.
    ``dff`` is assumed already filtered of ``not_supported`` rows.
    """
    plots_dir = view_dir / "plots"
    tables_dir = view_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    title = f"{benchmark}/{family}/{subset_name}"

    # x-axes. n_params is keyed by (problem_id, family); scope to this family
    # and flatten to {problem_id: value}. Decision alpha: include the
    # n_parameters axis only when at least one problem has a positive value
    # (continuous_gauss is all-zero -> n_nodes axis only).
    axes = [("n_nodes", n_nodes)]
    if n_params is not None:
        n_params_family = {p: v for (p, f), v in n_params.items() if f == family}
        if any(v and v > 0 for v in n_params_family.values()):
            axes.append(("n_parameters", n_params_family))

    if include_success_rate:
        fig_status_stacked(dff, plots_dir / "success_rate.pdf", title)

    for metric in ACCURACY_METRICS:
        if family in DISCRETE_FAMILIES and metric == "w1_per_node":
            continue
        for x_axis, lookup in axes:
            fig_accuracy_scaling(dff, metric, x_axis, lookup, aggregation,
                                 plots_dir / f"{metric}_vs_{x_axis}.pdf", title)

    for time_kind, stem in [("query_total", "total_query_time"), ("fit", "fit_time")]:
        for x_axis, lookup in axes:
            fig_time_scaling(dff, time_kind, x_axis, lookup, aggregation,
                             plots_dir / f"{stem}_vs_{x_axis}.pdf", title)

    # Tables. Labels unique per (family, subset): tab:<bench>_<family>_<subset>_<scope>.
    roles = sorted(dff["query_role"].dropna().unique())
    kinds = sorted(dff["query_kind"].dropna().unique())
    lbl = f"tab:{_table_slug(benchmark)}_{_table_slug(family)}_{_table_slug(subset_name)}"
    scope = f"{family}/{subset_name}"
    table_headline(dff, family, subset_name, aggregation, roles,
                   tables_dir / "table_overall.tex", label=f"{lbl}_overall")
    for r in roles:
        _table_scoped(dff, family, aggregation, "query_role", r,
                      tables_dir / f"table_role_{r}.tex", f"role={r} {scope}",
                      label=f"{lbl}_role_{_table_slug(r)}")
    for k in kinds:
        _table_scoped(dff, family, aggregation, "query_kind", k,
                      tables_dir / f"table_kind_{k}.tex", f"kind={k} {scope}",
                      label=f"{lbl}_kind_{_table_slug(k)}")


def process_family(dff, benchmark, family, aggregation, n_nodes, n_params,
                   family_dir):
    """Per-family orchestrator. Produces, under ``family_dir``:

      - ``all/{plots,tables}/``       always (every problem, every supported
                                      baseline -- mixed-coverage aggregation)
      - ``common/{plots,tables}/``    if any problem is solved by the full
                                      supported baseline set
      - ``subsetN/{plots,tables}/``   one per auto-discovered solving set
      - ``_subsets_overview.txt``     navigation aid

    Each subset is computed over its problems and restricted to its solving
    baselines; ``methods.txt`` / ``problems.txt`` record the membership.
    Subset views omit ``success_rate.pdf`` (100% by construction).
    """
    family_dir.mkdir(parents=True, exist_ok=True)

    # Drop baselines entirely not_supported for this family (and not_supported
    # rows from survivors): a baseline that never participated is noise, not a
    # 100% failure. (PR #197 semantics; the scaling plots filter to ok anyway.)
    dff = _filter_unsupported_baselines(dff)
    if dff.empty:
        logger.info("skip family with no supported baselines: %s", family)
        return

    # 1. The "all" view: every problem, every participating baseline.
    _render_view(dff, benchmark, family, aggregation, n_nodes, n_params,
                 family_dir / "all", subset_name="all", include_success_rate=True)

    # 2. Auto-discover coverage subsets and render each.
    subsets = _discover_subsets(dff)
    _write_subsets_overview(family_dir, family, subsets)
    for s in subsets:
        subset_dir = family_dir / s["name"]
        _write_subset_metadata(subset_dir, s["name"], s["baselines"], s["problems"])
        sub_dff = dff[dff["baseline"].isin(s["baselines"])
                      & dff["problem_id"].isin(s["problems"])]
        if sub_dff.empty:
            continue
        _render_view(sub_dff, benchmark, family, aggregation, n_nodes, n_params,
                     subset_dir, subset_name=s["name"], include_success_rate=False)


def _build_batch_speed_figure(
    df: pd.DataFrame,
    aggregation: str,
    title: str,
):
    """Build the batch-speed figure (v0.14 #148, design doc §6.PR6).

    Log-log batch_size vs amortized per-query time, one line per
    baseline, one facet per family. Returns ``(fig, dnf_lines)`` so
    tests can inspect facet structure / axis scales / annotations;
    :func:`fig_batch_speed` saves and closes it.

    Data contract: ``df`` is a speed-benchmark slice. Plotted points
    use ``status == "ok"`` / ``metric == "query_time_s"`` rows, first
    averaged per (family, baseline, batch_size, seed), then aggregated
    across seeds via :func:`aggregate` for the band. Seed invalidation
    (#148 Change B): a (family, baseline, batch_size) cell with ANY
    failed seed (oom/timeout/error) is dropped entirely — no point, and
    the dashed B=1 reference is suppressed — so a partially-failed config
    never plots a survivor value. DNF annotation (matching the #163
    convention): error/timeout/oom rows are counted per (family,
    baseline, batch_size, status) and reported via a corner note +
    sidecar lines.

    Returns ``(None, [])`` when there is nothing to plot.
    """
    if "batch_size" not in df.columns:
        return None, []
    ok = df[(df["status"] == "ok") & (df["metric"] == "query_time_s")].copy()
    ok = ok[ok["value"].notna() & (ok["value"] > 0)]
    if ok.empty:
        return None, []

    families = sorted(ok["family"].dropna().unique())
    colors = baseline_colors(sorted(ok["baseline"].unique()))

    # Full sweep x-grid — every batch size the run swept (any status, so a size
    # where a baseline FAILED still appears). Batchable lines are laid on this
    # grid with NaN at missing/failed sizes so the line BREAKS at gaps
    # (Change 1) instead of bridging a failed cell.
    grid = sorted(int(b) for b in df["batch_size"].dropna().unique() if b >= 1)

    # Seed invalidation (#148 Change B, speed-only): a (family, baseline,
    # batch_size) cell with ANY failed seed (oom/timeout/error) is fully failed
    # -> no point (NaN gap), and its dashed B=1 reference is suppressed. Built
    # from the full df (failure rows are dropped from `ok`). Seed-count-agnostic.
    fail = df[df["status"].isin(["oom", "timeout", "error"])]
    failed_cells = {
        (f, b, int(bs))
        for f, b, bs in zip(fail["family"], fail["baseline"], fail["batch_size"])
        if pd.notna(bs)
    }

    fig, axes = plt.subplots(
        1, len(families),
        figsize=(4.2 * len(families), 3.6),
        sharey=True, squeeze=False,
    )
    dnf_lines: list[str] = []

    for ax, family in zip(axes[0], families):
        sub = ok[ok["family"] == family]
        for bl in sorted(sub["baseline"].unique()):
            blsub = sub[sub["baseline"] == bl]
            library = parse_baseline(bl)[0]

            # Per-(baseline, batch_size) center+band: per-seed mean of queries
            # first, then seed-aggregate — same two-level pattern as the
            # scaling figures.
            by_bs: dict[int, tuple[float, float, float]] = {}
            for bs, grp in blsub.groupby("batch_size"):
                # Change B: drop any cell with a failed seed (no point/band; the
                # dashed B=1 reference below is suppressed via the same `by_bs`).
                if (family, bl, int(bs)) in failed_cells:
                    continue
                seed_means = grp.groupby("seed")["value"].mean()
                c, lo, hi = aggregate(seed_means, aggregation)
                lo, hi = clip_band("time", lo, hi)
                by_bs[int(bs)] = (c, lo, hi)

            if library in _NON_BATCHABLE_LIBRARIES:
                # Change 2: non-batchable baseline (pinned B=1). If it succeeded
                # at B=1, draw a dashed horizontal reference spanning the full
                # x-axis ("fixed cost, doesn't improve with batching"). If it
                # DNF'd at B=1 there's no ok data here at all -> no line.
                if 1 in by_bs:
                    ax.axhline(by_bs[1][0], ls="--", lw=1.2,
                               color=colors[bl], label=bl)
                continue

            # Change 1: batchable baseline on the full grid; NaN where this
            # baseline has no ok point so the line/band break at the gap.
            ys = [by_bs[b][0] if b in by_bs else float("nan") for b in grid]
            los = [by_bs[b][1] if b in by_bs else float("nan") for b in grid]
            his = [by_bs[b][2] if b in by_bs else float("nan") for b in grid]
            ax.plot(grid, ys, marker="o", markersize=4,
                    color=colors[bl], label=bl)
            ax.fill_between(grid, los, his, color=colors[bl], alpha=0.2)

        # DNF cells for this family — failed rows (error/timeout/oom),
        # deduped to (baseline, batch_size, status). #163 convention:
        # corner count + sidecar detail. Pinned baselines simply absent
        # at swept values are NOT DNF (by-design non-runs, §5.6).
        fam_dnf = df[
            (df["family"] == family)
            & (df["status"].isin(["error", "timeout", "oom"]))
        ]
        n_dnf_before = len(dnf_lines)
        if not fam_dnf.empty:
            grouped = fam_dnf.groupby(
                ["baseline", "batch_size", "status"]
            ).size()
            for (bl, bs, st), _n in grouped.items():
                dnf_lines.append(f"  {family} B={int(bs)} {bl}: {st}")
        n_fam_dnf = len(dnf_lines) - n_dnf_before
        if n_fam_dnf:
            ax.text(0.98, 0.98,
                    f"DNF: {n_fam_dnf} cells (see *_dnf.txt)",
                    fontsize=7, alpha=0.6, ha="right", va="top",
                    transform=ax.transAxes)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch_size")
        ax.set_title(family, fontsize=10)
        ax.legend(fontsize=6, loc="best")

    axes[0][0].set_ylabel("per-query time [s] (amortized)")
    fig.suptitle(f"{title} — per-query time vs batch_size", fontsize=11)
    return fig, dnf_lines


def fig_batch_speed(
    df: pd.DataFrame,
    aggregation: str,
    out_path: Path,
    title: str,
) -> None:
    """Render + save the batch-speed figure (and its DNF sidecar)."""
    fig, dnf_lines = _build_batch_speed_figure(df, aggregation, title)
    if fig is None:
        logger.info("skip empty (no ok batched rows): %s", out_path.name)
        return
    if dnf_lines:
        sidecar = out_path.with_name(out_path.stem + "_dnf.txt")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            "Error/timeout/oom cells for the batch-speed figure:\n"
            + "\n".join(dnf_lines) + "\n",
            encoding="utf-8",
        )
    _savefig(fig, out_path)


# --- Batch-speed per-family LaTeX tables (v0.14 #148, Change 3) ---------------

def _batch_speed_cell(rows: pd.DataFrame, aggregation: str) -> str:
    """One table cell for a (baseline, batch_size) slice.

    Failure-first (speed-only seed invalidation, #148 Change B): ANY row with
    status in {oom, timeout, error} fails the whole cell -> the failure code
    (most frequent). Otherwise ok query-time rows -> ``IQM$\\pm$band`` (same agg
    as the figure); otherwise (not_supported / absent) -> ``--``.

    The "any failure row" rule is seed-count-agnostic: correct whether the
    runner ran every seed or stopped after the first failure (forward-compatible
    with the PR-2 execution-level seed-skip).
    """
    failed = rows[rows["status"].isin(["oom", "timeout", "error"])]
    if not failed.empty:
        return str(failed["status"].mode().iloc[0])
    ok = rows[(rows["status"] == "ok") & (rows["metric"] == "query_time_s")]
    ok = ok[ok["value"].notna() & (ok["value"] > 0)]
    if not ok.empty:
        seed_means = ok.groupby("seed")["value"].mean()
        return _fmt(*aggregate(seed_means, aggregation))
    return "--"


def _write_batch_speed_table(out_path, grid, rows, family, aggregation, label="") -> None:
    """``table`` float (booktabs): rows=methods, cols=batch sizes. The code
    legend lives in ``\\caption`` (hand-built LaTeX — escaped at construction)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["Method"] + [f"$B={b}$" for b in grid]
    ncol = len(header)
    band = "IQM$\\pm$(Q3$-$Q1)/2" if aggregation == "iqm_iqr" else "mean$\\pm$std"
    agg_tex = aggregation.replace("_", "\\_")  # underscore is math-mode in text
    caption = (
        f"Per-query time [s], {band} over seeds (agg={agg_tex}). "
        f"\\texttt{{oom}}/\\texttt{{timeout}}/\\texttt{{error}} = cell DNF at "
        f"that batch size (any failed seed fails the cell); "
        f"-- = not run / non-batchable at $B>1$."
    )
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{l" + "r" * (ncol - 1) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}"]
    if label:
        lines.append(f"\\label{{{label}}}")
    lines += ["\\end{table}", ""]
    out_path.write_text("\n".join(lines))


def batch_speed_tables(df, aggregation, out_dir: Path, bench: str) -> int:
    """One ``batch_speed_table_<family>.tex`` per family present in a swept run.

    Mirrors :func:`fig_batch_speed`'s data contract; reuses :func:`aggregate`
    and :func:`_fmt` so plot and tables agree under ``--aggregation`` (Change 4).
    Returns the number of tables written.
    """
    if "batch_size" not in df.columns:
        return 0
    grid = sorted(int(b) for b in df["batch_size"].dropna().unique() if b >= 1)
    if not grid or max(grid) <= 1:
        return 0

    out_dir = Path(out_dir)
    written = 0
    for family in sorted(df["family"].dropna().unique()):
        fam = df[df["family"] == family]
        rows = []
        for bl in sorted(fam["baseline"].dropna().unique()):
            blsub = fam[fam["baseline"] == bl]
            cells = [
                _batch_speed_cell(blsub[blsub["batch_size"] == b], aggregation)
                for b in grid
            ]
            if all(c == "--" for c in cells):
                continue  # baseline not applicable to this family
            rows.append([bl.replace("_", "\\_")] + cells)
        if not rows:
            continue
        out_path = out_dir / f"batch_speed_table_{family}.tex"
        _write_batch_speed_table(
            out_path, grid, rows, family, aggregation,
            label=f"tab:{_table_slug(bench)}_batch_speed_{_table_slug(family)}",
        )
        written += 1
    return written


def _resolve_parquet(parquet: Path) -> Path:
    """Accept a ``.parquet`` file or a directory; in the latter case find the
    single ``*_metrics.parquet`` inside (the layout written by
    ``nbn-bench inference``)."""
    parquet = Path(parquet)
    if parquet.is_dir():
        matches = sorted(parquet.glob("*_metrics.parquet"))
        if not matches:
            raise FileNotFoundError(
                f"no *_metrics.parquet found in directory {parquet}"
            )
        if len(matches) > 1:
            logger.warning("multiple *_metrics.parquet in %s; using %s",
                           parquet, matches[0].name)
        return matches[0]
    return parquet


def run_plot(
    parquet: Path,
    output_dir: Path,
    aggregation: str = "iqm_iqr",
    benchmark: str | None = None,
) -> int:
    """Generate figures + LaTeX tables from a benchmark parquet.

    Args:
        parquet: path to a ``*_metrics.parquet`` file, or a directory
            containing one (output of ``nbn-bench inference``).
        output_dir: where to write the ``<benchmark>/{plots,tables}/`` tree
            (one set of per-family files across all problems in each family).
        aggregation: ``"iqm_iqr"`` (default) or ``"mean_std"``.
        benchmark: restrict to one benchmark; default processes every
            benchmark present in the parquet.

    Returns:
        Process exit code (0 on success, 1 if the parquet is missing columns).
    """
    parquet = _resolve_parquet(parquet)
    output_dir = Path(output_dir)

    df = pd.read_parquet(parquet)
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

    benchmarks = [benchmark] if benchmark else sorted(df["benchmark"].dropna().unique())
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped, produced = [], 0
    for bench in benchmarks:
        dfb = df[df["benchmark"] == bench].copy()
        if dfb.empty:
            continue
        # n_nodes per problem: parquet column (#195) preferred, _NETWORKS /
        # synthetic-int fallback for older parquets (resolve_n_nodes).
        n_nodes = resolve_n_nodes(dfb, bench)

        for family in sorted(dfb["family"].dropna().unique()):
            dff = dfb[dfb["family"] == family]
            if dff.empty:
                skipped.append(f"{bench}/{family}")
                logger.info("skip empty family: %s/%s", bench, family)
                continue
            process_family(dff, bench, family, aggregation,
                           n_nodes, n_params_global, output_dir / bench / family)
            produced += 1

        # Batch-speed figure (v0.14 #148): auto-detected — rendered when
        # the parquet carries batched rows (batch_size > 1 anywhere),
        # i.e. the output of a batch_sizes-sweep run. One figure per
        # benchmark, faceted by family, at the benchmark level of the
        # output tree.
        if "batch_size" in dfb.columns and (dfb["batch_size"] > 1).any():
            fig_batch_speed(
                dfb, aggregation,
                output_dir / bench / "batch_speed.pdf",
                bench,
            )
            produced += 1
            # Per-family LaTeX tables alongside the figure (#148, Change 3);
            # same aggregation as the figure for consistency (Change 4).
            produced += batch_speed_tables(
                dfb, aggregation, output_dir / bench, bench,
            )

    logger.info("done: %d cells produced, %d cells skipped (empty)", produced, len(skipped))
    return 0
