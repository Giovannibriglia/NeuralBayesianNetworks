"""Publication figures for the paper: ``nbn-bench paper``.

One PDF per (benchmark, metric), a single row of panels with one panel per
data family, all families sharing one legend. Aggregation is the seed-level
``all`` / ``common`` contract of :mod:`nbn.bench._paper_agg`
(``docs/v0.18-bar-reporting-all-common.md``); this module only decides
**which methods are shown** and how the panels are laid out.

Shown methods, per (benchmark, family): every non-nbn baseline applicable to
the family, plus the best **parametric** nbn method (cat, neuralcat, lg, mdn,
flow) and the best **non-parametric** one (kde, knn, flexcode, smoothed).
The two nbn methods are chosen once per family on the family's headline
metric (TV for discrete, W1 for continuous / hybrid; per-query time when the
run has no accuracy metric), ranked on the ``all`` view by (i) the number of
x values at which the method solved at least one seed and (ii) the mean
direction-aware center over those x values. When a category has no
applicable method in the family (discrete runs have no non-parametric nbn),
the slot goes to the runner-up of the other category, so two nbn methods are
shown whenever two exist. The same two methods appear in every panel and
table of that family; ``selection.txt`` records the ranking.

Inputs are run directories (or parquets); each benchmark group takes a base
run followed by any number of partial reruns, spliced in with
:func:`nbn.bench._merge.merge_frames` (cells in a rerun replace the base's).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from nbn.bench._merge import load_run, merge_frames
from nbn.bench._paper_agg import (
    METRIC_LABEL,
    TIME_METRICS,
    Views,
    all_view,
    assign_x,
    cell_table,
    clip_band,
    common_sets,
    common_view,
    is_nbn,
    metric_kind,
    parse_baseline,
    rank_key,
    sweep_axis,
    x_order,
)
from nbn.bench._paper_figures import (
    _direction,
    _evidence_slice,
    _filter_unsupported_baselines,
    resolve_n_nodes,
    write_view_table,
)

logger = logging.getLogger(__name__)

# --- Conventions --------------------------------------------------------------

GROUPS = ("inference", "scalability", "speed", "bnlearn",
          "param_learning", "learning_curves")

FAMILY_ORDER = ("discrete", "continuous_lg", "continuous_nongauss", "hybrid",
                "continuous_gauss", "clg")
FAMILY_TITLE = {
    "discrete": "discrete",
    "continuous_lg": "continuous (Gaussian)",
    "continuous_nongauss": "continuous (non-Gaussian)",
    "hybrid": "hybrid",
    "continuous_gauss": "Gaussian",
    "clg": "conditional linear Gaussian",
}

PARAMETRIC_MECHS = frozenset({"cat", "neuralcat", "lg", "mdn", "flow"})
NONPARAMETRIC_MECHS = frozenset({"kde", "knn", "flexcode", "smoothed"})

# Headline metric candidates per family kind, first present wins.
_HEADLINE_DISCRETE = ("tv_per_node", "param_recovery_tv", "log_likelihood",
                      "query_time")
_HEADLINE_CONTINUOUS = ("w1_per_node", "calibration_pit_ks", "log_likelihood",
                        "query_time")

# Metrics rendered per group; "accuracy" = the family's headline accuracy
# metric (may differ across panels: TV vs W1). Each entry: (file tag, metric
# or "accuracy", evidence mode or None for the combined slice).
_PLAN: dict[str, list[tuple[str, str, str | None]]] = {
    "inference": [("accuracy", "accuracy", None),
                  ("total_query_time", "total_query_time", None),
                  ("fit_time", "fit_time", None)],
    "scalability": [("accuracy", "accuracy", None),
                    ("accuracy_full", "accuracy", "full"),
                    ("query_time", "query_time", None),
                    ("query_time_full", "query_time", "full")],
    "speed": [("query_time_full", "query_time", "full"),
              ("query_time_empty", "query_time", "empty")],
    "bnlearn": [("accuracy", "accuracy", None),
                ("total_query_time", "total_query_time", None)],
    "param_learning": [("log_likelihood", "log_likelihood", None),
                       ("param_recovery_tv", "param_recovery_tv", None),
                       ("calibration_pit_ks", "calibration_pit_ks", None),
                       ("fit_time", "fit_time", None)],
    "learning_curves": [("log_likelihood", "log_likelihood", None),
                        ("param_recovery_tv", "param_recovery_tv", None),
                        ("calibration_pit_ks", "calibration_pit_ks", None),
                        ("fit_time", "fit_time", None)],
}

_X_LABEL = {"n_nodes": "$n$ (nodes)", "n_train": r"$n_{\mathrm{train}}$",
            "batch_size": "$B$ (queries per batch)", "network": ""}
_Y_LABEL = dict(METRIC_LABEL)
_Y_LABEL.update({"query_time": "Time per query (s)",
                 "total_query_time": "Query time (s)",
                 "fit_time": "Fit time (s)"})

_MAX_GROUPS_PER_PANEL = 8       # x values per panel before a family is chunked
_MAX_GROUPS_PER_ROW = 20        # bar groups per figure row
_MAX_PANELS_PER_ROW = 4
_LEGEND_COLS = 6

_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

# Colours are fixed per method so a method looks the same in every figure:
# one colour per non-nbn baseline; nbn methods take a hue per mechanism
# (reds / magentas for parametric, oranges for non-parametric) lightened by
# engine (VE darkest, then LW, AIS, AVI).
_FIXED_COLORS = {
    "pgmpy-mle-ve": "#1f77b4",
    "pgmpy-bayes-ve": "#7fbde0",
    "pgmpy-lg-predict": "#08306b",
    "pomegranate-discrete-ve": "#9467bd",
    "pyro-empirical-importance": "#8c564b",
}
_MECH_HUE = {
    "cat": "#b2182b", "neuralcat": "#d6604d", "lg": "#c2185b", "mdn": "#7f0a13",
    "flow": "#e7298a", "hybrid": "#a50f15",
    "kde": "#e6550d", "knn": "#fd8d3c", "flexcode": "#f7b267", "smoothed": "#fdd0a2",
}
_ENGINE_TINT = {"ve": 0.0, "lw": 0.18, "ais": 0.36, "avi": 0.52, "router": 0.0}
_FALLBACK_COLOR = "#7f7f7f"


def method_color(baseline: str) -> str:
    """Hex colour of a method (see the palette comment above)."""
    if baseline in _FIXED_COLORS:
        return _FIXED_COLORS[baseline]
    if not is_nbn(baseline):
        return _FALLBACK_COLOR
    parts = parse_baseline(baseline)[1].split("-")
    hue = _MECH_HUE.get(parts[0], _FALLBACK_COLOR)
    tint = _ENGINE_TINT.get(parts[1], 0.0) if len(parts) > 1 else 0.0
    rgb = np.array(matplotlib.colors.to_rgb(hue))
    rgb = rgb * (1 - tint) + np.ones(3) * tint
    return matplotlib.colors.to_hex(rgb)


_MECH_LABEL = {"cat": "cat", "neuralcat": "neural-cat", "lg": "LG", "mdn": "MDN",
               "flow": "flow", "kde": "KDE", "knn": "kNN", "flexcode": "FlexCode",
               "smoothed": "smoothed-cat", "hybrid": "hybrid"}
_ENGINE_LABEL = {"ve": "VE", "lw": "LW", "ais": "AIS", "avi": "AVI",
                 "router": "router"}
_BASELINE_LABEL = {
    "pgmpy-mle-ve": "pgmpy MLE+VE",
    "pgmpy-bayes-ve": "pgmpy Bayes+VE",
    "pgmpy-lg-predict": "pgmpy LG",
    "pomegranate-discrete-ve": "pomegranate",
    "pyro-empirical-importance": "pyro IS",
}


# --- Pure helpers -------------------------------------------------------------

def mechanism(baseline: str) -> str:
    """'nbn-mdn-lw' -> 'mdn'; 'nbn-cat' -> 'cat'; 'pgmpy-mle-ve' -> 'mle'."""
    rest = parse_baseline(baseline)[1]
    return rest.split("-", 1)[0] if rest else ""


def nbn_category(baseline: str) -> str | None:
    """'parametric' / 'nonparametric' for an nbn baseline, else None."""
    if not is_nbn(baseline):
        return None
    m = mechanism(baseline)
    if m in PARAMETRIC_MECHS:
        return "parametric"
    if m in NONPARAMETRIC_MECHS:
        return "nonparametric"
    return None


def short_label(baseline: str) -> str:
    """Legend label: 'nbn-mdn-lw' -> 'NBN MDN+LW', 'pgmpy-mle-ve' -> 'pgmpy MLE+VE'."""
    if baseline in _BASELINE_LABEL:
        return _BASELINE_LABEL[baseline]
    lib, rest = parse_baseline(baseline)
    if lib == "nbn" and rest:
        parts = rest.split("-")
        mech = _MECH_LABEL.get(parts[0], parts[0])
        if len(parts) > 1:
            return f"NBN {mech}+{_ENGINE_LABEL.get(parts[1], parts[1].upper())}"
        return f"NBN {mech}"
    return baseline


def headline_metric(family: str, available: set[str]) -> str | None:
    """The family's headline metric among ``available`` (metric names with
    >= 1 ok row, plus the timing pseudo-metrics), or None."""
    cands = _HEADLINE_DISCRETE if family == "discrete" else _HEADLINE_CONTINUOUS
    for m in cands:
        if m in available:
            return m
    return None


# --- Selection ----------------------------------------------------------------

@dataclass
class Ranking:
    method: str
    category: str | None
    n_x_solved: int
    score: float          # mean rank_key over solved x (lower is better)


def rank_nbn(view: pd.DataFrame, kind: str) -> list[Ranking]:
    """Rank the nbn methods of an ``all`` view: more x values with >= 1 solved
    seed first, then lower mean direction-aware center, then name."""
    out = []
    for m, g in view.groupby("method"):
        if not is_nbn(m):
            continue
        solved = g[g["k"] > 0]
        keys = [rank_key(c, kind) for c in solved["center"]]
        keys = [k for k in keys if np.isfinite(k)]
        score = float(np.mean(keys)) if keys else float("inf")
        out.append(Ranking(m, nbn_category(m), int(len(solved)), score))
    return sorted(out, key=lambda r: (-r.n_x_solved, r.score, r.method))


def select_nbn(ranking: list[Ranking]) -> list[str]:
    """Best parametric + best non-parametric; a missing category is filled by
    the runner-up of the other so two methods are shown when two exist.
    Methods that solved nothing are never selected."""
    ranking = [r for r in ranking if r.n_x_solved > 0]
    chosen: list[str] = []
    for cat in ("parametric", "nonparametric"):
        for r in ranking:
            if r.category == cat and r.method not in chosen:
                chosen.append(r.method)
                break
    for r in ranking:
        if len(chosen) >= 2:
            break
        if r.method not in chosen:
            chosen.append(r.method)
    return chosen[:2]


def views_for_methods(cells: pd.DataFrame, n_total: dict, aggregation: str,
                      metric: str, xs, methods) -> Views:
    """``all`` / ``common`` views over a FIXED shown set (no per-x top-N)."""
    kind = metric_kind(metric)
    va = all_view(cells, n_total, aggregation)
    va["shown"] = va["method"].isin(methods)
    sets = common_sets(cells, methods, xs)
    vc = common_view(cells, n_total, aggregation, methods, xs, sets)
    vc["shown"] = True
    nbn = [m for m in methods if is_nbn(m)]
    return Views(metric=metric, kind=kind, xs=list(xs), all=va, common=vc,
                 common_sets={x: sorted(s) for x, s in sets.items()},
                 selection={"all": dict.fromkeys(xs, nbn),
                            "common": dict.fromkeys(xs, nbn)})


# --- Per-family preparation ---------------------------------------------------

@dataclass
class FamilyData:
    group: str
    benchmark: str
    family: str
    dfx: pd.DataFrame            # rows with _x / _inst assigned
    x_axis: str
    xs: list
    headline: str | None
    shown: list[str]             # non-nbn + selected nbn
    ranking: list[Ranking] = field(default_factory=list)


def _available_metrics(dff: pd.DataFrame) -> set[str]:
    ok = dff[dff["status"] == "ok"]
    out = set(ok["metric"].unique())
    if "query_time_s" in out:
        out |= {"query_time", "total_query_time"}
    if "fit_time_s" in out or ("fit_time_s" in dff.columns and not ok.empty):
        out.add("fit_time")
    return out


def prepare_family(group: str, benchmark: str, family: str, dff: pd.DataFrame,
                   aggregation: str, n_nodes: dict) -> FamilyData | None:
    dff = _filter_unsupported_baselines(dff, family)
    if dff.empty:
        return None
    x_axis = sweep_axis(dff, benchmark)
    dfx = assign_x(dff, x_axis, n_nodes)
    if dfx.empty:
        logger.warning("%s/%s: no row resolved an x value (%s)", group, family, x_axis)
        return None
    xs = x_order(dfx, x_axis, n_nodes)
    headline = headline_metric(family, _available_metrics(dfx))
    methods = sorted(dfx["baseline"].unique())
    non_nbn = [m for m in methods if not is_nbn(m)]
    ranking: list[Ranking] = []
    chosen: list[str] = []
    if headline is not None:
        cells, n_total = cell_table(dfx, headline)
        if not cells.empty:
            va = all_view(cells, n_total, aggregation)
            ranking = rank_nbn(va, metric_kind(headline))
            chosen = select_nbn(ranking)
    return FamilyData(group, benchmark, family, dfx, x_axis, xs, headline,
                      non_nbn + chosen, ranking)


def _family_sort_key(f: str) -> tuple[int, str]:
    return (FAMILY_ORDER.index(f) if f in FAMILY_ORDER else len(FAMILY_ORDER), f)


# --- Drawing ------------------------------------------------------------------

def _tick(x, x_axis: str) -> str:
    return str(x) if x_axis == "network" else f"{int(x):d}"


def _draw_panel(ax, view: pd.DataFrame, xs, x_axis: str, metric: str,
                methods: list[str], view_name: str, common: dict | None) -> bool:
    """Grouped bars for one family on ``ax``. Returns False when nothing was
    drawable (the panel is then annotated as empty)."""
    view = view[view["shown"] & view["method"].isin(methods)]
    by_key = {(r.method, r.x): r for r in view.itertuples(index=False)}
    kind = metric_kind(metric)
    n_x, n_s = len(xs), len(methods)
    width = 0.82 / max(n_s, 1)
    finite, notes = [], []
    drew = False
    for si, m in enumerate(methods):
        poss, centers, los, his = [], [], [], []
        for xi, x in enumerate(xs):
            r = by_key.get((m, x))
            pos = xi + (si - (n_s - 1) / 2) * width
            if r is None:
                continue
            if r.k == 0:
                if r.code:
                    notes.append((pos, "code", r.code, method_color(m), None))
                continue
            if np.isposinf(r.center):
                notes.append((pos, "inf", "+∞", method_color(m), None))
                continue
            if np.isnan(r.center):
                continue
            lo, hi = clip_band(kind, r.lo, r.hi)
            if np.isnan(lo) or np.isnan(hi):
                lo, hi = r.center, r.center
            poss.append(pos)
            centers.append(r.center)
            los.append(lo)
            his.append(hi)
            if view_name == "all" and r.k < r.n:
                notes.append((pos, "kn", f"{r.k}/{r.n}", "black", hi))
        if not poss:
            continue
        drew = True
        yerr = [np.array(centers) - np.array(los), np.array(his) - np.array(centers)]
        ax.bar(poss, centers, width=width * 0.92, yerr=yerr, capsize=1.2,
               color=method_color(m), edgecolor="black" if is_nbn(m) else "none",
               linewidth=0.35, error_kw=dict(lw=0.5, capthick=0.5))
        finite += centers + his
    positives = [v for v in finite if v > 0]
    wide = len(positives) >= 2 and max(positives) / min(positives) > 30
    if (kind == "time" and positives) or wide:
        ax.set_yscale("log")
    y0, y1 = ax.get_ylim()
    for pos, nk, text, col, y in notes:
        if nk == "kn":
            ax.text(pos, y, text, ha="center", va="bottom", fontsize=4.2,
                    color=col, rotation=90)
        elif nk == "code":
            ax.text(pos, y0, text, ha="center", va="bottom", fontsize=4.2,
                    color=col, rotation=90, alpha=0.95)
        else:
            ax.text(pos, y1, text, ha="center", va="top", fontsize=6, color=col)
    ax.set_xticks(range(n_x))
    labels = [_tick(x, x_axis) for x in xs]
    if view_name == "common" and common is not None:
        labels = [f"{lab}\n|C|={len(common.get(x, []))}" for lab, x in zip(labels, xs)]
    is_net = x_axis == "network"
    ax.set_xticklabels(labels, rotation=35 if is_net else 0,
                       ha="right" if is_net else "center")
    ax.set_xlim(-0.6, n_x - 0.4)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    return drew


def _chunk(xs: list, size: int) -> list[list]:
    return [xs[i:i + size] for i in range(0, len(xs), size)]


def _layout_rows(panels: list[tuple]) -> list[list[tuple]]:
    """Pack panels (each carrying its x chunk) into rows of at most
    ``_MAX_PANELS_PER_ROW`` panels and ``_MAX_GROUPS_PER_ROW`` bar groups."""
    rows, cur, cur_groups = [], [], 0
    for p in panels:
        n = len(p[2])
        if cur and (len(cur) >= _MAX_PANELS_PER_ROW or cur_groups + n > _MAX_GROUPS_PER_ROW):
            rows.append(cur)
            cur, cur_groups = [], 0
        cur.append(p)
        cur_groups += n
    if cur:
        rows.append(cur)
    return rows


def fig_family_row(panels: list[tuple], metric_of: dict[str, str], out_path: Path,
                   view_name: str) -> bool:
    """Write one figure: ``panels`` = [(family, view_df, xs, x_axis, methods,
    common_sets)], ``metric_of[family]`` = the metric drawn in that panel.
    Families with more than ``_MAX_GROUPS_PER_PANEL`` x values are split into
    consecutive panels; panels are packed into rows of width 7 in."""
    split = []
    for fam, vdf, xs, x_axis, methods, common in panels:
        chunks = _chunk(list(xs), _MAX_GROUPS_PER_PANEL)
        for ci, chunk in enumerate(chunks):
            title = FAMILY_TITLE.get(fam, fam)
            if len(chunks) > 1:
                title += f" ({ci + 1}/{len(chunks)})"
            split.append((fam, vdf, chunk, x_axis, methods, common, title))
    rows = _layout_rows(split)
    if not rows:
        return False
    all_methods: list[str] = []
    for p in split:
        for m in p[4]:
            if m not in all_methods:
                all_methods.append(m)
    all_methods = ([m for m in all_methods if not is_nbn(m)]
                   + [m for m in all_methods if is_nbn(m)])
    ncol = min(len(all_methods), _LEGEND_COLS)
    legend_rows = int(np.ceil(len(all_methods) / max(ncol, 1)))
    head = 0.16 + 0.13 * legend_rows          # inches reserved for the legend

    with plt.rc_context(_STYLE):
        n_rows = len(rows)
        height = 1.9 * n_rows + head
        fig = plt.figure(figsize=(7.0, height))
        outer = fig.add_gridspec(n_rows, 1, hspace=0.55, top=1 - head / height,
                                 bottom=0.02)
        drew_any = False
        for ri, row in enumerate(rows):
            ratios = [max(len(p[2]), 2) for p in row]
            gs = outer[ri].subgridspec(1, len(row), width_ratios=ratios, wspace=0.28)
            for pi, (fam, vdf, chunk, x_axis, methods, common, title) in enumerate(row):
                ax = fig.add_subplot(gs[0, pi])
                metric = metric_of[fam]
                drew = _draw_panel(ax, vdf, chunk, x_axis, metric, methods,
                                   view_name, common)
                drew_any |= drew
                if not drew:
                    ax.text(0.5, 0.5, "no solved cell", ha="center", va="center",
                            transform=ax.transAxes, fontsize=6, color="gray")
                ax.set_title(title, pad=2)
                ax.set_xlabel(_X_LABEL.get(x_axis, x_axis), labelpad=1)
                # y label per panel when metrics differ, else first panel only
                ylab = f"{_Y_LABEL.get(metric, metric)}"
                if len(set(metric_of.values())) > 1 or pi == 0:
                    if metric not in TIME_METRICS:
                        ylab += f" ({_direction(metric).replace(' better', ' is better')})"
                    ax.set_ylabel(ylab, labelpad=2)
        handles = [Patch(facecolor=method_color(m), edgecolor="black" if is_nbn(m) else "none",
                         linewidth=0.35, label=short_label(m)) for m in all_methods]
        fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 8),
                   frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.0,
                   handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return drew_any


# --- Orchestration ------------------------------------------------------------

def load_group(paths) -> pd.DataFrame:
    """Base run + reruns -> one merged frame."""
    paths = [Path(p) for p in paths]
    base = load_run(paths[0])
    if len(paths) == 1:
        return base
    merged, stats = merge_frames(base, [load_run(p) for p in paths[1:]],
                                 names=[str(p) for p in paths[1:]])
    logger.info("%s: merged %d rerun(s): %d rows replaced by %d",
                paths[0].name, len(paths) - 1, stats["dropped"], stats["added"])
    return merged


def _present_modes(dfx: pd.DataFrame) -> set[str]:
    """Evidence modes on the per-query rows (any count)."""
    if "evidence_mode" not in dfx.columns or "query_role" not in dfx.columns:
        return set()
    per_query = dfx["query_role"].fillna("") != ""
    return set(dfx.loc[per_query, "evidence_mode"].dropna().unique())


def _families(df: pd.DataFrame) -> list[str]:
    return sorted(df["family"].dropna().unique(), key=_family_sort_key)


def _prepare_group(group: str, df: pd.DataFrame, aggregation: str) -> list[FamilyData]:
    out = []
    for bench in sorted(df["benchmark"].dropna().unique()):
        dfb = df[df["benchmark"] == bench]
        n_nodes = resolve_n_nodes(dfb, bench)
        for fam in _families(dfb):
            fd = prepare_family(group, bench, fam, dfb[dfb["family"] == fam],
                                aggregation, n_nodes)
            if fd is not None:
                out.append(fd)
    return out


def _selection_report(fds: list[FamilyData]) -> str:
    lines = ["nbn method selection per (benchmark group, family)",
             "rule: all non-nbn baselines + best parametric nbn + best non-parametric nbn,",
             "ranked on the headline metric (all view) by #x solved, then mean center.", ""]
    for fd in fds:
        nbn_shown = [m for m in fd.shown if is_nbn(m)]
        lines.append(f"[{fd.group}] {fd.benchmark}/{fd.family}  headline={fd.headline}  "
                     f"x={fd.x_axis}")
        lines.append(f"  shown: {', '.join(fd.shown) if fd.shown else '(none)'}")
        lines.append(f"  selected nbn: {', '.join(nbn_shown) if nbn_shown else '(none)'}")
        if fd.ranking:
            lines.append(f"  {'rank':<4} {'method':<26} {'category':<14} {'#x':>3} {'score':>10}")
            for i, r in enumerate(fd.ranking, 1):
                lines.append(f"  {i:<4} {r.method:<26} {str(r.category):<14} "
                             f"{r.n_x_solved:>3} {r.score:>10.4g}")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_group(fds: list[FamilyData], group: str, out_dir: Path, aggregation: str,
                 view_name: str) -> list[Path]:
    """All figures + tables of one benchmark group."""
    written: list[Path] = []
    tables = out_dir / "tables"
    for tag, metric_spec, mode in _PLAN[group]:
        panels, metric_of = [], {}
        for fd in fds:
            metric = fd.headline if metric_spec == "accuracy" else metric_spec
            if metric is None or (metric_spec == "accuracy" and metric in TIME_METRICS):
                continue   # no accuracy metric in this run (e.g. batch speed)
            dfx = fd.dfx
            if mode is not None:
                if mode not in _present_modes(dfx):
                    continue
                dfx = _evidence_slice(dfx, mode)
            cells, n_total = cell_table(dfx, metric)
            if cells.empty or not cells["solved"].any():
                logger.info("%s/%s/%s%s: no solved cell", group, fd.family, metric,
                            f"[{mode}]" if mode else "")
                continue
            views = views_for_methods(cells, n_total, aggregation, metric, fd.xs, fd.shown)
            vdf = views.all if view_name == "all" else views.common
            panels.append((fd.family, vdf, fd.xs, fd.x_axis, fd.shown, views.common_sets))
            metric_of[fd.family] = metric
            suffix = f"_{mode}" if mode else ""
            tex = tables / f"{group}_{fd.family}_{metric}{suffix}.tex"
            write_view_table(
                vdf, fd.xs, fd.x_axis, metric, view_name, tex,
                caption_prefix=f"{group.replace('_', ' ')}, {FAMILY_TITLE.get(fd.family, fd.family)}"
                               + (f", evidence={mode}" if mode else ""),
                label=f"tab:{group}_{fd.family}_{metric}{suffix}",
                common_sets=views.common_sets,
                dagger_note="best parametric / best non-parametric nbn method of the family")
        if not panels:
            continue
        path = out_dir / f"{group}_{tag}.pdf"
        if fig_family_row(panels, metric_of, path, view_name):
            written.append(path)
            logger.info("wrote %s", path)
    return written


def run_paper(groups: dict[str, list], output_dir: Path, aggregation: str = "iqm_iqr",
              view: str = "all") -> int:
    """Entry point of ``nbn-bench paper``.

    ``groups``: ``{group_name: [base_run, rerun, ...]}`` for any subset of
    :data:`GROUPS`. Writes ``<output_dir>/<group>_<metric>.pdf``,
    ``<output_dir>/tables/<group>_<family>_<metric>.tex`` and
    ``<output_dir>/selection.txt``. Returns a process exit code."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unknown = set(groups) - set(GROUPS)
    if unknown:
        logger.error("unknown benchmark group(s): %s", sorted(unknown))
        return 1
    if not groups:
        logger.error("no benchmark group given")
        return 1
    logging.getLogger("fontTools").setLevel(logging.WARNING)  # pdf font subsetting chatter
    all_fds: list[FamilyData] = []
    written: list[Path] = []
    for group in GROUPS:
        if group not in groups or not groups[group]:
            continue
        df = load_group(groups[group])
        fds = _prepare_group(group, df, aggregation)
        if not fds:
            logger.warning("%s: nothing to plot", group)
            continue
        all_fds += fds
        written += render_group(fds, group, output_dir, aggregation, view)
    (output_dir / "selection.txt").write_text(_selection_report(all_fds))
    logger.info("done: %d figure(s) in %s", len(written), output_dir)
    return 0
