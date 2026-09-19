"""Generate paper figures + LaTeX tables from a benchmark parquet.

Public entry point: :func:`run_plot`, invoked by ``nbn-bench plot``
(``nbn/bench/cli.py``); ``scripts/make_paper_figures.py`` remains as a
deprecation shim.

Every figure is a **grouped bar plot** over the benchmark's discrete x grid
(``n_nodes``, ``n_train``, ``batch_size`` or bnlearn ``network``); every
figure and table comes in two views (``docs/v0.18-bar-reporting-all-common.md``,
aggregation in :mod:`nbn.bench._paper_agg`):

  all/      each method over the seeds it solved; ``k/n`` annotated when k<n
  common/   each method over the seeds solved by every shown method at that x

Shown methods per x = all non-nbn baselines applicable to the family + the
``top_nbn`` best nbn methods (ranked on the view's own aggregate).

Output layout, at ``<output_dir>/<bench>/<family>/``:

  all/plots/<metric>_vs_<x>.pdf      all/tables/<metric>_vs_<x>.tex
  all/plots/<metric>_vs_<x>_evidence_<mode>.pdf   (+ .tex; one per evidence
                                     mode when the run has >= 2, fit_time excluded)
  all/plots/success_rate.pdf         (status breakdown, diagnostic)
  all/plots/divergence_*.pdf         (calibration-vs-accuracy panel, if both metrics)
  common/plots/<metric>_vs_<x>.pdf   common/tables/<metric>_vs_<x>.tex
  common/common_seeds.txt            C(x) per metric and x
  selection.txt                      shown nbn methods per (view, metric, x)

``<metric>`` in the accuracy set (w1 skipped for discrete families) plus the
timing pseudo-metrics: ``query_time`` (per-query, batch_size sweeps) or
``total_query_time`` + ``fit_time`` (everything else). Runs whose queries mix
evidence modes (``full`` / ``empty``, paired by the heaviest-query selector)
also get each query-derived metric per mode, since the two regimes differ
(empty-evidence batches fall back to sequential queries in nbn).
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nbn.bench._paper_agg import (  # noqa: F401  (re-exported for callers/tests)
    ACCURACY_METRICS,
    CLOSER_TO_VALUE,
    DISCRETE_FAMILIES,
    FAILURE_STATUSES,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    METRIC_LABEL,
    TIME_METRICS,
    Views,
    aggregate,
    assign_x,
    build_views,
    cell_table,
    clip_band,
    is_nbn,
    metric_kind,
    parse_baseline,
    sweep_axis,
    x_order,
)

logger = logging.getLogger(__name__)


# --- Constants ----------------------------------------------------------------

# Divergence panel (#235): metric pairs rendered side-by-side per family when
# both metrics have ok rows (calibration_pit_ks from a PL parquet vs
# w1_per_node from an inference parquet — the PR 9 cross-metric finding).
_DIVERGENCE_PAIRS = (("calibration_pit_ks", "w1_per_node"),)

# Engine suffixes appended to INFERENCE baselines but absent on
# PARAMETER-LEARNING baselines; stripped to align nbn-mdn-lw <-> nbn-mdn.
_ENGINE_SUFFIXES = frozenset({"lw", "ve", "ais", "avi", "router", "predict",
                              "importance"})

# Library -> base color (v0.12 convention).
LIBRARY_COLORS = {
    "pgmpy": "tab:blue",
    "nbn": "tab:red",
    "pomegranate": "tab:purple",
    "pyro": "tab:brown",
}
_FALLBACK_COLOR = "tab:gray"

# Status -> color for the 100%-stacked status breakdown.
STATUS_COLORS = {
    "ok": "#2ca02c",
    "not_supported": "#7f7f7f",
    "not_applicable": "#b0b0b0",
    "timeout": "#ff7f0e",
    "error": "#d62728",
    "oom": "#8c564b",
}
STATUS_ORDER = ("ok", "not_supported", "not_applicable", "timeout", "error", "oom")

# Bar groups per figure before the x grid is split into ``_partK`` files
# (bnlearn has 24 discrete networks; 8 groups x ~7 bars stays legible).
_MAX_GROUPS_PER_FIGURE = 8

_X_LABEL = {"n_nodes": "n_nodes", "n_train": "n_train",
            "batch_size": "batch size $B$", "network": "network"}


# --- Pure helpers -------------------------------------------------------------

def _mechanism_key(baseline: str) -> str:
    """Strip a trailing engine token: nbn-mdn-lw -> nbn-mdn; nbn-cat -> nbn-cat."""
    head, sep, last = baseline.rpartition("-")
    if sep and last in _ENGINE_SUFFIXES:
        return head
    return baseline


def baseline_colors(baselines) -> dict[str, tuple]:
    """Library base color, lightened by a distinct factor per baseline within
    the same library."""
    by_lib: dict[str, list[str]] = {}
    for b in sorted(baselines):
        by_lib.setdefault(parse_baseline(b)[0], []).append(b)
    colors: dict[str, tuple] = {}
    for lib, members in by_lib.items():
        base = np.array(matplotlib.colors.to_rgb(LIBRARY_COLORS.get(lib, _FALLBACK_COLOR)))
        n = len(members)
        for i, b in enumerate(members):
            t = 0.0 if n == 1 else 0.55 * i / (n - 1)
            rgb = base * (1 - t) + np.array([1.0, 1.0, 1.0]) * t
            colors[b] = (*rgb, 1.0)
    return colors


def _log_or_linear(ax, vals, axis: str) -> None:
    """Use log scale if the positive dynamic range exceeds ~1.5 decades."""
    pos = [v for v in vals if v is not None and v > 0 and not np.isnan(v)]
    if len(pos) >= 2 and max(pos) / min(pos) > 30:
        (ax.set_xscale if axis == "x" else ax.set_yscale)("log")


def _direction(metric) -> str:
    if metric in TIME_METRICS or metric in LOWER_IS_BETTER:
        return "lower better"
    if metric in CLOSER_TO_VALUE:
        return f"closer to {CLOSER_TO_VALUE[metric]:g} better"
    return "higher better"


# --- Lookups ------------------------------------------------------------------

def n_nodes_lookup(benchmark: str, problem_ids) -> dict[str, int]:
    if benchmark == "bnlearn":
        from nbn.bench.problems.bnlearn import _NETWORKS
        return {p: _NETWORKS[p]["n_nodes"] for p in problem_ids if p in _NETWORKS}
    out = {}
    for p in problem_ids:
        try:
            out[p] = int(p)
        except (TypeError, ValueError):
            pass
    return out


def resolve_n_nodes(dfb: pd.DataFrame, benchmark: str) -> dict[str, int]:
    """Map ``problem_id -> n_nodes``: parquet column first (#195), then the
    ``_NETWORKS`` / synthetic-int fallback; unresolved problems are omitted."""
    pids = sorted(dfb["problem_id"].dropna().unique())
    from_col: dict[str, int] = {}
    if "n_nodes" in dfb.columns:
        sub = dfb[["problem_id", "n_nodes"]].dropna(subset=["n_nodes"])
        if not sub.empty:
            g = sub.groupby("problem_id")["n_nodes"].first()
            from_col = {str(p): int(v) for p, v in g.items()}
    missing = [p for p in pids if p not in from_col]
    fallback = n_nodes_lookup(benchmark, missing) if missing else {}
    resolved = {**fallback, **from_col}
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
    """Map ``(problem_id, family) -> n_parameters`` (#133), or None if absent."""
    if "n_parameters" not in df.columns:
        return None
    sub = df[["problem_id", "family", "n_parameters"]].dropna(subset=["n_parameters"])
    if sub.empty:
        return None
    g = sub.groupby(["problem_id", "family"])["n_parameters"].first()
    return {(p, f): float(v) for (p, f), v in g.items()}


# --- Per-query status extraction (success_rate.pdf) ---------------------------

def per_query_success(df_cell: pd.DataFrame) -> dict[str, float]:
    """Query-level success rate (%) per baseline; PL-mode cells (no per-query
    rows) are binary on the presence of an ok accuracy-metric row."""
    out = {}
    for b, g in df_cell.groupby("baseline"):
        executed = g[g["metric"] == "query_time_s"]
        sentinel = g[g["metric"] == "status"]
        total = len(executed) + len(sentinel)
        if total == 0:
            pl = g[g["metric"].isin(ACCURACY_METRICS)]
            out[b] = 100.0 if bool((pl["status"] == "ok").any()) else 0.0
            continue
        ok = int((executed["status"] == "ok").sum())
        out[b] = 100.0 * ok / total
    return out


def per_query_status_counts(df_cell: pd.DataFrame) -> pd.DataFrame:
    """Per-baseline status counts over the per-query unit (query_time_s rows +
    whole-cell sentinels); PL-mode fallback counts accuracy-metric rows."""
    unit = df_cell[df_cell["metric"].isin(["query_time_s", "status"])]
    if unit.empty:
        unit = df_cell[df_cell["metric"].isin(ACCURACY_METRICS)]
    if unit.empty:
        return pd.DataFrame()
    counts = unit.groupby(["baseline", "status"]).size().unstack(fill_value=0)
    unknown = [c for c in counts.columns if c not in STATUS_ORDER]
    if unknown:
        logger.warning("status(es) outside stacked-bar palette dropped: %s", unknown)
    return counts.reindex(columns=list(STATUS_ORDER), fill_value=0)


# --- Figures ------------------------------------------------------------------

def _savefig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_status_stacked(df_cell, out_path: Path, title: str) -> None:
    """100%-stacked status breakdown per baseline (``success_rate.pdf``)."""
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
            continue
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


def fig_divergence(df_cell, metric_a, metric_b, family, aggregation, out_path, title):
    """Two-panel divergence figure (#235): metric_a over metric_b, one bar per
    mechanism, x-order ranked by metric_a."""
    def _agg_by_mech(metric):
        ok = df_cell[(df_cell["family"] == family) & (df_cell["metric"] == metric)
                     & (df_cell["status"] == "ok")].copy()
        if ok.empty:
            return {}
        ok["_mech"] = ok["baseline"].map(_mechanism_key)
        out = {}
        for mech, g in ok.groupby("_mech"):
            c, lo, hi = aggregate(g["value"], aggregation)
            if not np.isnan(c):
                lo, hi = clip_band(metric, lo, hi)
                out[mech] = (c, lo, hi)
        return out

    a, b = _agg_by_mech(metric_a), _agg_by_mech(metric_b)
    if not a or not b:
        logger.info("skip divergence (a metric has no ok rows): %s", out_path.name)
        return
    order = sorted(set(a) | set(b), key=lambda m: (a.get(m, (float("inf"),))[0], m))
    xs = list(range(len(order)))
    fig, (ax_a, ax_b) = plt.subplots(2, 1, sharex=True,
                                     figsize=(max(5, 0.9 * len(order)), 6))
    for ax, vals, metric in ((ax_a, a, metric_a), (ax_b, b, metric_b)):
        heights = [vals[m][0] if m in vals else np.nan for m in order]
        lo_err = [vals[m][0] - vals[m][1] if m in vals else 0.0 for m in order]
        hi_err = [vals[m][2] - vals[m][0] if m in vals else 0.0 for m in order]
        ax.bar(xs, heights, yerr=[lo_err, hi_err], color="tab:blue",
               alpha=0.8, capsize=3, edgecolor="white")
        ax.set_ylabel(f"{METRIC_LABEL[metric]} ({_direction(metric)})")
        ax.grid(True, axis="y", alpha=0.3)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax_a.set_title(
        f"{title} — {METRIC_LABEL[metric_a]} vs {METRIC_LABEL[metric_b]} divergence\n"
        f"(x ordered by {METRIC_LABEL[metric_a]}; out-of-order bars below = "
        f"ranking disagreement)", fontsize=10)
    _savefig(fig, out_path)


def _slot_order(view: pd.DataFrame) -> list[str]:
    """Bar slots: non-nbn methods first (sorted), then the nbn methods shown
    anywhere in this figure (sorted). Fixed across groups so a method keeps
    its position and colour."""
    shown = view[view["shown"]]["method"].unique()
    return (sorted([m for m in shown if not is_nbn(m)])
            + sorted([m for m in shown if is_nbn(m)]))


def _tick(x, x_axis: str) -> str:
    return str(x) if x_axis == "network" else f"{int(x):d}"


def fig_bars(view: pd.DataFrame, xs, x_axis: str, metric: str, view_name: str,
             out_path: Path, title: str, common_sets: dict | None = None) -> list[Path]:
    """Grouped bar plot: one group per x, one bar per shown method, error bar =
    aggregation band. Annotations: ``k/n`` above a bar whose method solved
    fewer than all seeds (``all`` view); the failure code in a slot whose
    method solved none; ``+∞`` for an infinite center. ``common`` groups carry
    ``|C|=k`` under the x label. The x grid is split into ``_partK`` files
    beyond :data:`_MAX_GROUPS_PER_FIGURE` groups.

    Returns the written paths (empty when nothing is drawable)."""
    view = view[view["shown"]]
    if view.empty or not ((view["k"] > 0) & view["center"].notna()).any():
        logger.info("skip empty (no solved seeds): %s", out_path.name)
        return []
    slots = _slot_order(view)
    colors = baseline_colors(slots)
    kind = metric_kind(metric)
    xs = list(xs)
    chunks = [xs[i:i + _MAX_GROUPS_PER_FIGURE]
              for i in range(0, len(xs), _MAX_GROUPS_PER_FIGURE)]
    by_key = {(r.method, r.x): r for r in view.itertuples(index=False)}
    written = []
    for ci, chunk in enumerate(chunks):
        n_x, n_s = len(chunk), len(slots)
        width = 0.8 / max(n_s, 1)
        fig, ax = plt.subplots(figsize=(max(6.0, n_x * (0.3 * n_s + 0.6)), 4.2))
        finite_vals, notes = [], []   # notes: (xpos, kind, text, color, y)
        for si, m in enumerate(slots):
            centers, los, his, poss = [], [], [], []
            for xi, x in enumerate(chunk):
                r = by_key.get((m, x))
                pos = xi + (si - (n_s - 1) / 2) * width
                if r is None:
                    continue
                if r.k == 0:
                    if r.code:
                        notes.append((pos, "code", r.code, colors[m], None))
                    continue
                if np.isposinf(r.center):
                    notes.append((pos, "inf", "+∞", colors[m], None))
                    continue
                if np.isnan(r.center):
                    continue
                lo, hi = clip_band(kind, r.lo, r.hi)
                if np.isnan(lo) or np.isnan(hi):
                    lo, hi = r.center, r.center
                centers.append(r.center)
                los.append(lo)
                his.append(hi)
                poss.append(pos)
                if view_name == "all" and r.k < r.n:
                    notes.append((pos, "kn", f"{r.k}/{r.n}", colors[m], hi))
            if not poss:
                # no bar in this chunk (DNF / +inf everywhere): keep the legend
                # entry so the failure-code / +∞ annotation is attributable.
                ax.bar([0.0], [np.nan], color=colors[m], label=m)
                continue
            yerr = [np.array(centers) - np.array(los), np.array(his) - np.array(centers)]
            ax.bar(poss, centers, width=width * 0.95, yerr=yerr, capsize=2,
                   color=colors[m], edgecolor="white", linewidth=0.5, label=m,
                   error_kw=dict(lw=0.8))
            finite_vals += centers + his
        positives = [v for v in finite_vals if v > 0]
        if kind == "time" and positives and len(positives) == len(finite_vals):
            ax.set_yscale("log")
        else:
            _log_or_linear(ax, finite_vals, "y")
        y0, y1 = ax.get_ylim()
        for pos, nk, text, col, y in notes:
            if nk == "kn":
                ax.text(pos, y, text, ha="center", va="bottom", fontsize=5.5,
                        color="black", rotation=90)
            elif nk == "code":
                ax.text(pos, y0, text, ha="center", va="bottom", fontsize=5.5,
                        color=col, rotation=90, alpha=0.9)
            else:
                ax.text(pos, y1, text, ha="center", va="top", fontsize=8, color=col)
        ax.set_xticks(range(n_x))
        labels = [_tick(x, x_axis) for x in chunk]
        if view_name == "common" and common_sets is not None:
            labels = [f"{lab}\n|C|={len(common_sets.get(x, []))}"
                      for lab, x in zip(labels, chunk)]
        is_net = x_axis == "network"
        ax.set_xticklabels(labels, rotation=30 if is_net else 0,
                           ha="right" if is_net else "center", fontsize=8)
        ax.set_xlabel(_X_LABEL.get(x_axis, x_axis))
        ax.set_ylabel(f"{METRIC_LABEL.get(metric, metric)} ({_direction(metric)})")
        part = f" (part {ci + 1}/{len(chunks)})" if len(chunks) > 1 else ""
        ax.set_title(f"{title} — {METRIC_LABEL.get(metric, metric)} vs {x_axis} "
                     f"[{view_name}]{part}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=min(4, max(1, n_s)), frameon=False)
        path = out_path if len(chunks) == 1 else out_path.with_name(
            f"{out_path.stem}_part{ci + 1}{out_path.suffix}")
        _savefig(fig, path)
        written.append(path)
    return written


# --- LaTeX tables -------------------------------------------------------------

def _fmt(center: float, lo: float, hi: float) -> str:
    if np.isnan(center):
        return "--"
    if np.isposinf(center):
        return "$+\\infty$"
    half = (hi - lo) / 2
    if np.isnan(half):
        return f"{center:.3g}"
    return f"{center:.3g}$\\pm${half:.2g}"


def _bold_best(centrals: dict, criterion: str) -> set:
    """Baselines tied-best on ``criterion`` (a metric name or ``"time"``) to
    .3g display precision. None / NaN / +-inf are excluded."""
    finite = {b: c for b, c in centrals.items()
              if c is not None and np.isfinite(c)}
    if not finite:
        return set()
    if criterion in CLOSER_TO_VALUE:
        keyed = {b: abs(c - CLOSER_TO_VALUE[criterion]) for b, c in finite.items()}
        higher = False
    else:
        keyed = dict(finite)
        higher = criterion in HIGHER_IS_BETTER
    best = (max if higher else min)(keyed.values())
    return {b for b, k in keyed.items() if f"{k:.3g}" == f"{best:.3g}"}


def _bold(cell: str) -> str:
    return cell if cell == "--" else f"\\textbf{{{cell}}}"


def _table_slug(value: str) -> str:
    return str(value).replace("+", "plus").replace(" ", "_").replace("/", "_")


_PL_METRICS = frozenset({"param_recovery_tv", "param_recovery_kl",
                         "calibration_pit_ks", "calibration_sd_ratio",
                         "log_likelihood"})
_INFERENCE_METRICS = frozenset({"tv_per_node", "jsd_per_node", "w1_per_node"})


def _benchmark_caption(df_view) -> str:
    """INFERENCE SPEED / SAMPLE EFFICIENCY PARAMETER LEARNING /
    <BENCH> INFERENCE|PARAMETER LEARNING, from the parquet contents (#241)."""
    cols = df_view.columns
    if "batch_size" in cols and (df_view["batch_size"] > 1).any():
        return "INFERENCE SPEED"
    ok_metrics = set(df_view.loc[df_view["status"] == "ok", "metric"]) \
        if "status" in cols else set(df_view["metric"])
    has_pl = bool(ok_metrics & _PL_METRICS)
    has_inf = bool(ok_metrics & _INFERENCE_METRICS)
    if ("n_train" in cols and df_view["n_train"].dropna().nunique() > 1 and has_pl):
        return "SAMPLE EFFICIENCY PARAMETER LEARNING"
    benchmark = (df_view["benchmark"].dropna().iloc[0]
                 if "benchmark" in cols and not df_view["benchmark"].dropna().empty
                 else "synthetic")
    bench = "BNLEARN" if str(benchmark).lower() == "bnlearn" else "SYNTHETIC"
    if has_pl and not has_inf:
        return f"{bench} PARAMETER LEARNING"
    return f"{bench} INFERENCE"


def _write_table(out_path: Path, header_cols, rows, caption="", label="",
                 footer_rows=()):
    """Emit a full ``table`` float (booktabs). ``footer_rows`` go under a
    second ``\\midrule``."""
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
    if footer_rows:
        lines.append("\\midrule")
        lines += [" & ".join(r) + " \\\\" for r in footer_rows]
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


def _x_header(x, x_axis: str) -> str:
    if x_axis == "network":
        return str(x).replace("_", "\\_")
    sym = {"n_nodes": "n", "n_train": "n_{tr}", "batch_size": "B"}[x_axis]
    return f"${sym}={int(x)}$"


def write_view_table(view: pd.DataFrame, xs, x_axis: str, metric: str, view_name: str,
                     out_path: Path, caption_prefix: str, label: str,
                     common_sets: dict | None = None, top_nbn: int = 2) -> bool:
    """Rows = shown methods (nbn rows marked $^\\dagger$), columns = x values.

    ``all``:    ``c±h (k/n)`` with the ``(k/n)`` only when k < n; failure code
                when no seed was solved; ``--`` when the method did not run or
                is not among the top-N nbn at that column.
    ``common``: ``c±h`` over C(x); a footer row gives |C(x)| per column.
    Best per column in bold (direction-aware). Returns False if nothing to
    write."""
    view = view[view["shown"]]
    if view.empty or not ((view["k"] > 0) & view["center"].notna()).any():
        return False
    slots = _slot_order(view)
    xs = list(xs)
    by_key = {(r.method, r.x): r for r in view.itertuples(index=False)}
    kind = metric_kind(metric)
    cells, centrals = {}, {}
    for m in slots:
        cells[m], centrals[m] = {}, {}
        for x in xs:
            r = by_key.get((m, x))
            if r is None:
                cells[m][x], centrals[m][x] = "--", None
                continue
            if r.k == 0:
                cells[m][x] = (r.code or "--").replace("_", "\\_")
                centrals[m][x] = None
                continue
            s = _fmt(r.center, r.lo, r.hi)
            if view_name == "all" and r.k < r.n:
                s += f" ({r.k}/{r.n})"
            cells[m][x] = s
            centrals[m][x] = r.center if np.isfinite(r.center) else None
    bold = {x: _bold_best({m: centrals[m][x] for m in slots}, kind) for x in xs}
    header = ["Method"] + [_x_header(x, x_axis) for x in xs]
    rows = []
    for m in slots:
        name = m.replace("_", "\\_") + ("$^\\dagger$" if is_nbn(m) else "")
        row = [name]
        for x in xs:
            c = cells[m][x]
            row.append(_bold(c) if m in bold[x] else c)
        rows.append(row)
    footer = ()
    if view_name == "common" and common_sets is not None:
        footer = (["$|C|$ (common seeds)"]
                  + [str(len(common_sets.get(x, []))) for x in xs],)
    if view_name == "all":
        rule = ("(k/n) = seeds solved / seeds run, shown when k<n; a failure "
                "code = no seed solved; -- = not run, or an nbn method outside "
                "the top-N at that column")
    else:
        rule = ("each value over the seeds solved by every shown method at "
                "that column (|C| row); methods with no solved seed are DNF "
                "and do not constrain C; -- = not run, or an nbn method "
                "outside the top-N at that column")
    caption = (f"{caption_prefix}. {METRIC_LABEL.get(metric, metric)} "
               f"({_direction(metric)}) vs {x_axis}, view={view_name}. "
               f"Center$\\pm$band across seeds; $\\dagger$ = top-{top_nbn} nbn "
               f"per column; {rule}.")
    _write_table(out_path, header, rows, caption, label=label, footer_rows=footer)
    return True


# --- Orchestration ------------------------------------------------------------

def _filter_unsupported_baselines(dff: pd.DataFrame, family: str = "") -> pd.DataFrame:
    """Drop baselines whose every row in this slice is ``not_supported``, and
    drop ``not_supported`` rows from any remaining baseline: a baseline that
    never participated is applicability, not a 100% failure.

    Says which baselines it dropped: a WARNING when the rows carry the
    runner's broken-library marker (the baseline should have run; its
    library was missing / broken in that run), INFO otherwise (genuine
    applicability)."""
    if "status" not in dff.columns or "baseline" not in dff.columns:
        return dff
    supported = set(
        dff.loc[dff["status"] != "not_supported", "baseline"].unique()
    )
    from nbn.bench.core.runner import LIBRARY_BROKEN_PREFIX
    where = f" in {family}" if family else ""
    for name in sorted(set(dff["baseline"].unique()) - supported):
        msgs = dff.loc[dff["baseline"] == name, "error_msg"].dropna().astype(str) \
            if "error_msg" in dff.columns else pd.Series([], dtype=str)
        broken = msgs[msgs.str.contains(LIBRARY_BROKEN_PREFIX, regex=False)]
        if not broken.empty:
            m = broken.iloc[0]
            m = m[m.find(LIBRARY_BROKEN_PREFIX) + len(LIBRARY_BROKEN_PREFIX):]
            logger.warning(
                "omitting %s%s: its library was not usable in that run "
                "(every cell not_supported) -- %s", name, where, m)
        else:
            logger.info("omitting %s%s: not applicable to any problem", name, where)
    return dff[
        dff["baseline"].isin(supported)
        & (dff["status"] != "not_supported")
    ]


def _family_metrics(dff: pd.DataFrame, family: str, x_axis: str) -> list[str]:
    """Accuracy metrics with >= 1 ok row, then the timing pseudo-metrics that
    make sense for the sweep."""
    metrics = [m for m in _metrics_for_family(family)
               if not dff[(dff["metric"] == m) & (dff["status"] == "ok")].empty]
    has_query = not dff[(dff["metric"] == "query_time_s") & (dff["status"] == "ok")].empty
    if x_axis == "batch_size":
        if has_query:
            metrics.append("query_time")
        return metrics
    if has_query:
        metrics.append("total_query_time")
    has_fit_rows = not dff[(dff["metric"] == "fit_time_s") & (dff["status"] == "ok")].empty
    has_fit_col = ("fit_time_s" in dff.columns
                   and not dff[dff["metric"].isin(ACCURACY_METRICS)
                               & (dff["status"] == "ok")].empty)
    if has_fit_rows or has_fit_col:
        metrics.append("fit_time")
    return metrics


def _evidence_modes(dfx: pd.DataFrame) -> list[str]:
    """Evidence modes present on the per-query rows, or ``[]`` when there is
    fewer than two: a single mode is the whole run, so a per-mode figure would
    duplicate the combined one."""
    if "evidence_mode" not in dfx.columns or "query_role" not in dfx.columns:
        return []
    per_query = dfx["query_role"].fillna("") != ""
    modes = sorted(dfx.loc[per_query, "evidence_mode"].dropna().unique())
    return modes if len(modes) >= 2 else []


def _evidence_slice(dfx: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Rows of one evidence mode plus every cell-level row (fit / metrics time,
    whole-cell sentinels — ``query_role == ""``): those apply to the cell
    whatever the query's evidence, and the sentinels keep a failed cell
    unsolved in every mode. Per-query status rows follow their own mode.

    Why per mode at all: batched nbn queries with empty evidence fall back to
    the sequential path (``nbn_adapter.query_batch``), so the combined
    ``query_time`` figure of a batch_size sweep sits halfway between the two
    regimes (2026-09-16 run: full-evidence 41x, empty 1x, combined 2x)."""
    cell_level = dfx["query_role"].fillna("") == ""
    return dfx[cell_level | (dfx["evidence_mode"] == mode)]


def process_family(dff, benchmark, family, aggregation, n_nodes, n_params,
                   family_dir, top_nbn: int = 2) -> int:
    """Per-family orchestrator: the ``all`` and ``common`` trees under
    ``family_dir`` (see the module docstring). Returns the number of figures
    written."""
    family_dir = Path(family_dir)
    family_dir.mkdir(parents=True, exist_ok=True)
    dff = _filter_unsupported_baselines(dff, family)
    if dff.empty:
        logger.info("skip family with no supported baselines: %s", family)
        return 0

    x_axis = sweep_axis(dff, benchmark)
    dfx = assign_x(dff, x_axis, n_nodes)
    if dfx.empty:
        logger.warning("skip family %s: no row resolved an x value (%s)", family, x_axis)
        return 0
    xs = x_order(dfx, x_axis, n_nodes)
    title = f"{benchmark}/{family}"
    caption_prefix = f"{_benchmark_caption(dff)}. {family}"
    lbl = f"tab:{_table_slug(benchmark)}_{_table_slug(family)}"

    produced = 0
    selection_lines = [f"Shown nbn methods per (view, metric, {x_axis}) — top-{top_nbn}", ""]
    common_lines = [f"Common seeds C(x) per metric and {x_axis}", ""]
    modes = _evidence_modes(dfx)
    for metric in _family_metrics(dff, family, x_axis):
        stem = f"{metric}_vs_{x_axis}"
        # (rows, file stem, title/caption suffix) per rendered slice: the
        # combined figure plus one per evidence mode when the run has >= 2.
        slices = [(dfx, stem, "")]
        if metric != "fit_time":
            slices += [(_evidence_slice(dfx, mode), f"{stem}_evidence_{mode}",
                        f" [evidence={mode}]") for mode in modes]
        for dfs, s_stem, suffix in slices:
            cells, n_total = cell_table(dfs, metric)
            if cells.empty or not cells["solved"].any():
                logger.info("skip %s/%s%s: no solved cell", family, metric, suffix)
                continue
            views = build_views(cells, n_total, aggregation, metric, xs, top_n=top_nbn)
            tag = f"{metric}{suffix}"
            for view_name, vdf in (("all", views.all), ("common", views.common)):
                vdir = family_dir / view_name
                paths = fig_bars(vdf, xs, x_axis, metric, view_name,
                                 vdir / "plots" / f"{s_stem}.pdf", title + suffix,
                                 common_sets=views.common_sets)
                produced += len(paths)
                write_view_table(vdf, xs, x_axis, metric, view_name,
                                 vdir / "tables" / f"{s_stem}.tex",
                                 caption_prefix + suffix.replace("[", "(").replace("]", ")"),
                                 label=f"{lbl}_{view_name}_{_table_slug(s_stem)}",
                                 common_sets=views.common_sets, top_nbn=top_nbn)
                sel = views.selection[view_name]
                for x in xs:
                    if x in sel:
                        selection_lines.append(
                            f"{view_name}\t{tag}\t{x_axis}={_tick(x, x_axis)}\t"
                            + (", ".join(sel[x]) if sel[x] else "(none)"))
            for x in xs:
                insts = views.common_sets.get(x, [])
                common_lines.append(
                    f"{tag}\t{x_axis}={_tick(x, x_axis)}\t|C|={len(insts)}\t"
                    + (", ".join(insts) if insts else "(empty)"))

    all_plots = family_dir / "all" / "plots"
    fig_status_stacked(dff, all_plots / "success_rate.pdf", title)
    for metric_a, metric_b in _DIVERGENCE_PAIRS:
        have_a = not dff[(dff["metric"] == metric_a) & (dff["status"] == "ok")].empty
        have_b = not dff[(dff["metric"] == metric_b) & (dff["status"] == "ok")].empty
        if have_a and have_b:
            fig_divergence(dff, metric_a, metric_b, family, aggregation,
                           all_plots / f"divergence_{metric_a}_vs_{metric_b}.pdf",
                           title)
            produced += 1
    (family_dir / "selection.txt").write_text("\n".join(selection_lines) + "\n")
    (family_dir / "common").mkdir(parents=True, exist_ok=True)
    (family_dir / "common" / "common_seeds.txt").write_text(
        "\n".join(common_lines) + "\n")
    return produced


def _resolve_parquet(parquet) -> list[Path]:
    """A ``.parquet`` file, a run directory (single ``*_metrics.parquet``
    inside), or a list of those -> ``list[Path]``."""
    items = parquet if isinstance(parquet, (list, tuple)) else [parquet]
    resolved: list[Path] = []
    for item in items:
        item = Path(item)
        if item.is_dir():
            matches = sorted(item.glob("*_metrics.parquet"))
            if not matches:
                raise FileNotFoundError(
                    f"no *_metrics.parquet found in directory {item}"
                )
            if len(matches) > 1:
                logger.warning("multiple *_metrics.parquet in %s; using %s",
                               item, matches[0].name)
            resolved.append(matches[0])
        else:
            resolved.append(item)
    return resolved


def run_plot(
    parquet: Path,
    output_dir: Path,
    aggregation: str = "iqm_iqr",
    benchmark: str | None = None,
    top_nbn: int = 2,
) -> int:
    """Generate figures + LaTeX tables from one or more benchmark parquets.

    Args:
        parquet: a ``*_metrics.parquet`` file, a run directory, or a list of
            those (row-concatenated, #235).
        output_dir: root of the ``<benchmark>/<family>/{all,common}/`` tree.
        aggregation: ``"iqm_iqr"`` (default) or ``"mean_std"``.
        benchmark: restrict to one benchmark; default = every one present.
        top_nbn: how many nbn methods to show per x (default 2).

    Returns:
        Process exit code (0 on success, 1 if the parquet is missing columns).
    """
    paths = _resolve_parquet(parquet)
    output_dir = Path(output_dir)
    frames = [pd.read_parquet(p) for p in sorted(paths, key=str)]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    required = {"benchmark", "family", "problem_id", "seed", "baseline",
                "metric", "value", "status", "query_role", "query_kind"}
    missing = required - set(df.columns)
    if missing:
        logger.error("parquet missing required columns: %s", missing)
        return 1

    n_params_global = n_parameters_lookup(df)
    if n_params_global is None:
        logger.info("n_parameters column absent")

    benchmarks = [benchmark] if benchmark else sorted(df["benchmark"].dropna().unique())
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped, produced = [], 0
    for bench in benchmarks:
        dfb = df[df["benchmark"] == bench].copy()
        if dfb.empty:
            continue
        n_nodes = resolve_n_nodes(dfb, bench)
        for family in sorted(dfb["family"].dropna().unique()):
            dff = dfb[dfb["family"] == family]
            if dff.empty:
                skipped.append(f"{bench}/{family}")
                continue
            produced += process_family(dff, bench, family, aggregation, n_nodes,
                                       n_params_global, output_dir / bench / family,
                                       top_nbn=top_nbn)

    logger.info("done: %d figures produced, %d families skipped (empty)",
                produced, len(skipped))
    return 0
