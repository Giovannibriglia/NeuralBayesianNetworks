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

``exclude`` (baseline globs such as ``nbn-flow-*`` or ``nbn-*-ais``) and
``exclude_families`` remove methods / data families from every group before
the selection, so a paper can leave out methods it does not discuss without
touching the parquets. ``always_show`` (``[group/]baseline-glob``) adds a
method to every panel of a family where it is applicable, after the two
selected ones, e.g. ``scalability/nbn-lg-lw`` to keep the one mechanism that
reaches the largest networks visible even when it is not the most accurate.
"""
from __future__ import annotations

import fnmatch
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
    "clg": "CLG",
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
# or "accuracy" or a tuple of those for a stacked figure with one row per
# metric and a single shared legend, evidence mode or None for the combined
# slice).
_PLAN: dict[str, list[tuple[str, str | tuple[str, ...], str | None]]] = {
    "inference": [("accuracy", "accuracy", None),
                  ("total_query_time", "total_query_time", None),
                  ("accuracy_time", ("accuracy", "total_query_time"), None),
                  ("fit_time", "fit_time", None)],
    "scalability": [("accuracy", "accuracy", None),
                    ("accuracy_full", "accuracy", "full"),
                    ("query_time", "query_time", None),
                    ("query_time_full", "query_time", "full")],
    "speed": [("query_time_full", "query_time", "full"),
              ("query_time_empty", "query_time", "empty")],
    "bnlearn": [("accuracy", "accuracy", None),
                ("total_query_time", "total_query_time", None),
                ("accuracy_time", ("accuracy", "total_query_time"), None)],
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
_ROW_HEIGHT_IN = 1.9            # default height of one row of panels, inches
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
    solved_xs: list               # x values with >= 1 solved seed
    candidate: bool               # coverage >= half of the x grid (or the relaxed rule)
    score: float                  # mean rank_key over X* (lower is better)

    @property
    def n_x_solved(self) -> int:
        return len(self.solved_xs)


@dataclass
class CategoryRanking:
    category: str
    x_star: list                  # common solved x of the candidates
    ranked: list[Ranking]         # candidates first (by score), then the rest


def _solved_xs(view: pd.DataFrame, method: str, xs) -> list:
    """x values (in grid order) with >= 1 solved seed for ``method``."""
    g = view[(view["method"] == method) & (view["k"] > 0) & view["center"].notna()]
    have = set(g["x"].unique())
    return [x for x in xs if x in have]


def rank_category(view: pd.DataFrame, kind: str, xs, methods) -> CategoryRanking | None:
    """Rank the nbn ``methods`` of one category on an ``all`` view.

    1. candidates = methods that solved >= 1 seed at >= ceil(|xs| / 2) x
       values; if none, every method with >= 1 solved x.
    2. X* = intersection of the candidates' solved-x sets.
    3. score = mean direction-aware center (:func:`rank_key`) over X*;
       rank by score, then more solved x, then name. Non-candidates follow,
       scored on X* ∩ their solved x (∞ when empty).
    Returns None when no method of the category solved anything."""
    solved = {m: _solved_xs(view, m, xs) for m in methods}
    solved = {m: v for m, v in solved.items() if v}
    if not solved:
        return None
    half = int(np.ceil(len(xs) / 2))
    cands = [m for m, v in solved.items() if len(v) >= half] or list(solved)
    x_star = set.intersection(*(set(solved[m]) for m in cands))
    centers = {(r.method, r.x): r.center for r in view.itertuples(index=False)}

    def score(m):
        keys = [rank_key(centers[(m, x)], kind) for x in x_star if x in solved[m]]
        keys = [k for k in keys if np.isfinite(k)]
        return float(np.mean(keys)) if keys else float("inf")

    cat = nbn_category(next(iter(methods)))
    ranked = [Ranking(m, cat, solved[m], m in cands, score(m)) for m in solved]
    ranked.sort(key=lambda r: (not r.candidate, r.score, -r.n_x_solved, r.method))
    return CategoryRanking(cat or "", [x for x in xs if x in x_star], ranked)


def rank_nbn(view: pd.DataFrame, kind: str, xs) -> dict[str, CategoryRanking]:
    """``{category: CategoryRanking}`` for the nbn methods of an ``all`` view."""
    out = {}
    for cat in ("parametric", "nonparametric"):
        methods = sorted(m for m in view["method"].unique() if nbn_category(m) == cat)
        if methods:
            cr = rank_category(view, kind, xs, methods)
            if cr is not None:
                out[cat] = cr
    return out


def select_nbn(rankings: dict[str, CategoryRanking]) -> list[str]:
    """Best parametric + best non-parametric (the first candidate of each
    category); a missing category is filled by the runner-up of the other so
    two methods are shown when two exist."""
    chosen: list[str] = []
    for cat in ("parametric", "nonparametric"):
        cr = rankings.get(cat)
        if cr is not None and cr.ranked:
            chosen.append(cr.ranked[0].method)
    if len(chosen) < 2:
        for cat in ("parametric", "nonparametric"):
            cr = rankings.get(cat)
            if cr is None:
                continue
            for r in cr.ranked:
                if len(chosen) >= 2:
                    break
                if r.method not in chosen and np.isfinite(r.score):
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
    shown: list[str]             # non-nbn + selected nbn + forced
    ranking: dict[str, CategoryRanking] = field(default_factory=dict)
    forced: list[str] = field(default_factory=list)   # always_show methods added


def _available_metrics(dff: pd.DataFrame) -> set[str]:
    ok = dff[dff["status"] == "ok"]
    out = set(ok["metric"].unique())
    if "query_time_s" in out:
        out |= {"query_time", "total_query_time"}
    if "fit_time_s" in out or ("fit_time_s" in dff.columns and not ok.empty):
        out.add("fit_time")
    return out


def forced_methods(group: str, methods, always_show=()) -> list[str]:
    """The ``methods`` (applicable baselines of one family) matched by an
    ``always_show`` pattern, which is ``glob`` or ``group/glob``."""
    out: list[str] = []
    for pat in always_show:
        g, _, glob = pat.rpartition("/")
        if g and g != group:
            continue
        out += [m for m in methods if fnmatch.fnmatchcase(m, glob) and m not in out]
    return out


def prepare_family(group: str, benchmark: str, family: str, dff: pd.DataFrame,
                   aggregation: str, n_nodes: dict, always_show=()) -> FamilyData | None:
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
    ranking: dict[str, CategoryRanking] = {}
    chosen: list[str] = []
    if headline is not None:
        cells, n_total = cell_table(dfx, headline)
        if not cells.empty:
            va = all_view(cells, n_total, aggregation)
            ranking = rank_nbn(va, metric_kind(headline), xs)
            chosen = select_nbn(ranking)
    forced = [m for m in forced_methods(group, methods, always_show) if m not in chosen]
    return FamilyData(group, benchmark, family, dfx, x_axis, xs, headline,
                      non_nbn + chosen + forced, ranking, forced)


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
                   view_name: str, row_height: float = _ROW_HEIGHT_IN) -> bool:
    """Write one figure: ``panels`` = [(family, view_df, xs, x_axis, methods,
    common_sets)], ``metric_of[family]`` = the metric drawn in that panel.
    Families with more than ``_MAX_GROUPS_PER_PANEL`` x values are split into
    consecutive panels; panels are packed into rows of width 7 in."""
    return fig_family_rows([(panels, metric_of)], out_path, view_name, row_height)


def fig_family_rows(panel_rows: list[tuple[list[tuple], dict[str, str]]],
                    out_path: Path, view_name: str,
                    row_height: float = _ROW_HEIGHT_IN) -> bool:
    """Write one figure with one block of panel rows per ``(panels,
    metric_of)`` entry (e.g. accuracy above query time), all rows sharing a
    single legend at the top; each row of panels is ``row_height`` inches
    tall. See :func:`fig_family_row` for the panel tuple."""
    rows: list[list[tuple]] = []
    split_all: list[tuple] = []
    for panels, metric_of in panel_rows:
        split = []
        for fam, vdf, xs, x_axis, methods, common in panels:
            chunks = _chunk(list(xs), _MAX_GROUPS_PER_PANEL)
            for ci, chunk in enumerate(chunks):
                title = FAMILY_TITLE.get(fam, fam)
                if len(chunks) > 1:
                    title += f" ({ci + 1}/{len(chunks)})"
                split.append((fam, vdf, chunk, x_axis, methods, common, title,
                              metric_of))
        rows += _layout_rows(split)
        split_all += split
    if not rows:
        return False
    all_methods: list[str] = []
    for p in split_all:
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
        height = row_height * n_rows + head
        fig = plt.figure(figsize=(7.0, height))
        outer = fig.add_gridspec(n_rows, 1, hspace=0.42, top=1 - head / height,
                                 bottom=0.02)
        drew_any = False
        for ri, row in enumerate(rows):
            ratios = [max(len(p[2]), 2) for p in row]
            gs = outer[ri].subgridspec(1, len(row), width_ratios=ratios, wspace=0.28)
            for pi, (fam, vdf, chunk, x_axis, methods, common, title,
                     metric_of) in enumerate(row):
                ax = fig.add_subplot(gs[0, pi])
                metric = metric_of[fam]
                drew = _draw_panel(ax, vdf, chunk, x_axis, metric, methods,
                                   view_name, common)
                drew_any |= drew
                if not drew:
                    ax.text(0.5, 0.5, "no solved cell", ha="center", va="center",
                            transform=ax.transAxes, fontsize=6, color="gray")
                # panel titles only on the first block of a stacked figure
                if metric_of is panel_rows[0][1]:
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


def drop_excluded(df: pd.DataFrame, exclude=(), exclude_families=()) -> pd.DataFrame:
    """``df`` without the baselines matching any ``exclude`` glob
    (``fnmatch`` on the baseline name) and without ``exclude_families``."""
    keep = pd.Series(True, index=df.index)
    if exclude:
        names = df["baseline"].dropna().unique()
        gone = {b for b in names if any(fnmatch.fnmatchcase(b, pat) for pat in exclude)}
        keep &= ~df["baseline"].isin(gone)
    if exclude_families:
        keep &= ~df["family"].isin(set(exclude_families))
    return df[keep]


def _present_modes(dfx: pd.DataFrame) -> set[str]:
    """Evidence modes on the per-query rows (any count)."""
    if "evidence_mode" not in dfx.columns or "query_role" not in dfx.columns:
        return set()
    per_query = dfx["query_role"].fillna("") != ""
    return set(dfx.loc[per_query, "evidence_mode"].dropna().unique())


def _families(df: pd.DataFrame) -> list[str]:
    return sorted(df["family"].dropna().unique(), key=_family_sort_key)


def _prepare_group(group: str, df: pd.DataFrame, aggregation: str,
                   always_show=()) -> list[FamilyData]:
    out = []
    for bench in sorted(df["benchmark"].dropna().unique()):
        dfb = df[df["benchmark"] == bench]
        n_nodes = resolve_n_nodes(dfb, bench)
        for fam in _families(dfb):
            fd = prepare_family(group, bench, fam, dfb[dfb["family"] == fam],
                                aggregation, n_nodes, always_show)
            if fd is not None:
                out.append(fd)
    return out


def _selection_report(fds: list[FamilyData]) -> str:
    lines = [
        "nbn method selection per (benchmark group, family)",
        "rule: all non-nbn baselines + best parametric nbn + best non-parametric nbn.",
        "per category: candidates = methods that solved >= 1 seed at >= half of the",
        "x grid (else every method with >= 1 solved x); X* = common solved x of the",
        "candidates; score = mean direction-aware center over X* (lower is better);",
        "rank by score, then coverage, then name. Non-candidates are listed after,",
        "scored on X* ∩ their solved x. An empty category is filled by the other's",
        "runner-up.", ""]
    for fd in fds:
        nbn_shown = [m for m in fd.shown if is_nbn(m) and m not in fd.forced]
        lines.append(f"[{fd.group}] {fd.benchmark}/{fd.family}  headline={fd.headline}  "
                     f"x={fd.x_axis}  grid={[_tick(x, fd.x_axis) for x in fd.xs]}")
        lines.append(f"  shown: {', '.join(fd.shown) if fd.shown else '(none)'}")
        lines.append(f"  selected nbn: {', '.join(nbn_shown) if nbn_shown else '(none)'}")
        if fd.forced:
            lines.append(f"  always shown (not selected): {', '.join(fd.forced)}")
        for cat, cr in fd.ranking.items():
            lines.append(f"  {cat}: X* = {[_tick(x, fd.x_axis) for x in cr.x_star]}")
            lines.append(f"    {'rank':<4} {'method':<26} {'cand':<5} {'#x':>3} "
                         f"{'score on X*':>12}  solved x")
            for i, r in enumerate(cr.ranked, 1):
                lines.append(f"    {i:<4} {r.method:<26} {'yes' if r.candidate else 'no':<5} "
                             f"{r.n_x_solved:>3} {r.score:>12.4g}  "
                             f"{[_tick(x, fd.x_axis) for x in r.solved_xs]}")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_group(fds: list[FamilyData], group: str, out_dir: Path, aggregation: str,
                 view_name: str, row_height: float = _ROW_HEIGHT_IN) -> list[Path]:
    """All figures + tables of one benchmark group."""
    written: list[Path] = []
    tables = out_dir / "tables"
    for tag, spec, mode in _PLAN[group]:
        specs = spec if isinstance(spec, tuple) else (spec,)
        panel_rows = []
        for metric_spec in specs:
            panels, metric_of = _metric_panels(fds, group, metric_spec, mode, aggregation,
                                               view_name, tables, write_tables=not
                                               isinstance(spec, tuple))
            if panels:
                panel_rows.append((panels, metric_of))
        if len(panel_rows) < len(specs):
            continue   # a stacked figure needs every metric row
        path = out_dir / f"{group}_{tag}.pdf"
        if fig_family_rows(panel_rows, path, view_name, row_height):
            written.append(path)
            logger.info("wrote %s", path)
    return written


def _metric_panels(fds: list[FamilyData], group: str, metric_spec: str, mode,
                   aggregation: str, view_name: str, tables: Path,
                   write_tables: bool = True) -> tuple[list[tuple], dict[str, str]]:
    """The panel tuples (one per family) of one metric spec, plus the metric
    drawn per family; writes the matching LaTeX tables unless told not to
    (stacked figures reuse the tables of their single-metric siblings)."""
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
        if not write_tables:
            continue
        suffix = f"_{mode}" if mode else ""
        tex = tables / f"{group}_{fd.family}_{metric}{suffix}.tex"
        write_view_table(
            vdf, fd.xs, fd.x_axis, metric, view_name, tex,
            caption_prefix=f"{group.replace('_', ' ')}, {FAMILY_TITLE.get(fd.family, fd.family)}"
                           + (f", evidence={mode}" if mode else ""),
            label=f"tab:{group}_{fd.family}_{metric}{suffix}",
            common_sets=views.common_sets,
            dagger_note="best parametric / best non-parametric nbn method of the family")
    return panels, metric_of


def run_paper(groups: dict[str, list], output_dir: Path, aggregation: str = "iqm_iqr",
              view: str = "all", exclude=(), exclude_families=(),
              row_height: float = _ROW_HEIGHT_IN, always_show=()) -> int:
    """Entry point of ``nbn-bench paper``.

    ``groups``: ``{group_name: [base_run, rerun, ...]}`` for any subset of
    :data:`GROUPS`. Writes ``<output_dir>/<group>_<metric>.pdf``,
    ``<output_dir>/tables/<group>_<family>_<metric>.tex`` and
    ``<output_dir>/selection.txt``. ``exclude`` / ``exclude_families``: see
    :func:`drop_excluded`; ``always_show``: see :func:`forced_methods`;
    ``row_height``: inches per row of panels. Returns a process exit code."""
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
        df = drop_excluded(load_group(groups[group]), exclude, exclude_families)
        fds = _prepare_group(group, df, aggregation, always_show)
        if not fds:
            logger.warning("%s: nothing to plot", group)
            continue
        all_fds += fds
        written += render_group(fds, group, output_dir, aggregation, view, row_height)
    report = _selection_report(all_fds)
    if exclude or exclude_families or always_show:
        report = (f"excluded baselines: {', '.join(exclude) or '(none)'}\n"
                  f"excluded families: {', '.join(exclude_families) or '(none)'}\n"
                  f"always shown: {', '.join(always_show) or '(none)'}\n\n"
                  + report)
    (output_dir / "selection.txt").write_text(report)
    logger.info("done: %d figure(s) in %s", len(written), output_dir)
    return 0
