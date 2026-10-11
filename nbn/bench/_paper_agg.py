"""Seed-level aggregation behind the paper figures: the ``all`` / ``common`` views.

Every benchmark figure is a grouped bar plot over a discrete x grid (n_nodes,
n_train, batch_size or network name). The unit of "solved / not solved" is the
**seed** (more precisely the cell ``(problem_id, seed)`` with the x column
factored out); a method solves a seed for a metric when every row of that
metric in the cell is ``ok`` and no failure sentinel (``metric == "status"``
with timeout / oom / error) was emitted for the cell.

Two views share one contract (``docs/v0.18-bar-reporting-all-common.md``):

* ``all``    — each method is aggregated over the seeds *it* solved. The
               table / bar annotation reports ``f/n`` (failed / total seeds at
               that x) whenever ``f > 0``; a method with ``k == 0`` shows its
               failure code instead of a value.
* ``common`` — each method is aggregated over ``C(x)``, the seeds solved by
               **every shown method** at that x (methods with ``k == 0`` are
               DNF and do not constrain the intersection, otherwise a single
               failing baseline would empty the column).

Shown methods at each x = every non-nbn baseline applicable to the family plus
the ``top_n`` nbn methods, ranked on the ``all`` view (direction-aware on the
metric center). The ``common`` view reuses exactly that selection, so both
views always show the same methods and ``C(x)`` is the intersection over
that fixed set.

Aggregation across seeds is ``iqm_iqr`` (interquartile mean ± IQR/2 in the
tables; the bars draw the min / max of the interquartile values, see
:func:`plot_band`) or ``mean_std``; the per-seed value is the per-cell reduction (mean over queries
for accuracy and per-query time, sum for total query time, first for fit time).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- Metric conventions --------------------------------------------------------

ACCURACY_METRICS = (
    "tv_per_node", "jsd_per_node", "w1_per_node", "ks_per_node", "log_likelihood",
    "param_recovery_tv", "param_recovery_kl",
    "calibration_pit_ks", "calibration_sd_ratio",
)
LOWER_IS_BETTER = frozenset({
    "tv_per_node", "jsd_per_node", "w1_per_node", "ks_per_node",
    "param_recovery_tv", "param_recovery_kl", "calibration_pit_ks",
    "nll_per_node",
})
HIGHER_IS_BETTER = frozenset({"log_likelihood"})
CLOSER_TO_VALUE = {"calibration_sd_ratio": 1.0}
METRIC_LABEL = {
    "tv_per_node": "TV",
    "jsd_per_node": "JSD",
    "w1_per_node": "W1",
    "ks_per_node": "KS",
    "log_likelihood": "LL",
    "nll_per_node": "NLL per node",
    "param_recovery_tv": "TV (recovery)",
    "param_recovery_kl": "KL (recovery)",
    "calibration_pit_ks": "PIT-KS",
    "calibration_sd_ratio": "SD-ratio",
    "query_time": "Per-query time (s)",
    "total_query_time": "Total query time (s)",
    "fit_time": "Fit time (s)",
}
TIME_METRICS = ("query_time", "total_query_time", "fit_time")
DISCRETE_FAMILIES = frozenset({"discrete"})
# Families with no discrete node: their tv_per_node / jsd_per_node rows can
# only come from the deprecated 256-bin histogram estimate on continuous
# samples (removed in D2; noise-dominated, identical distributions -> TV 0.2).
# Paper views skip them; continuous accuracy is W1 / KS.
CONTINUOUS_FAMILIES = frozenset({
    "continuous_lg", "continuous_nongauss", "continuous_gauss",
})
DEPRECATED_CONTINUOUS_METRICS = frozenset({"tv_per_node", "jsd_per_node"})
FAILURE_STATUSES = frozenset({"timeout", "oom", "error"})


# Renamed baseline labels: old result files are read under the new name so a
# base run and its reruns group as one method. pomegranate's predict_proba is
# loopy belief propagation, never variable elimination (D10).
BASELINE_ALIASES = {"pomegranate-discrete-ve": "pomegranate-discrete-lbp"}


def normalize_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite legacy baseline labels (:data:`BASELINE_ALIASES`) in place of
    a copy; frames without a ``baseline`` column pass through."""
    if "baseline" not in df.columns or not df["baseline"].isin(BASELINE_ALIASES).any():
        return df
    out = df.copy()
    out["baseline"] = out["baseline"].replace(BASELINE_ALIASES)
    return out


def parse_baseline(baseline: str) -> tuple[str, str]:
    """('nbn-cat-ve') -> ('nbn', 'cat-ve')."""
    parts = str(baseline).split("-", 1)
    return (parts[0], parts[1] if len(parts) > 1 else "")


def is_nbn(baseline: str) -> bool:
    return parse_baseline(baseline)[0] == "nbn"


# --- Method style (shared by every figure) ---------------------------------------
# One colour per approach, the same in every figure, from colour-blind-safe
# palettes (Okabe-Ito + Paul Tol "muted"): a colour per external baseline and
# per nbn mechanism. The inference engine of an nbn method is a hatch, so
# "cat + VE" and "cat + LW" share the colour of the cat mechanism.
_APPROACH_COLOR = {
    "pgmpy-mle": "#0072B2", "pgmpy-bayes": "#56B4E9", "pgmpy-lg": "#332288",
    "pomegranate": "#AA4499", "pyro": "#999999",
    "nbn-cat": "#D55E00", "nbn-cat-bayes": "#E69F00", "nbn-neuralcat": "#F0E442",
    "nbn-lg": "#CC6677", "nbn-mdn": "#882255", "nbn-flow": "#117733",
    "nbn-kde": "#44AA99", "nbn-knn": "#999933", "nbn-flexcode": "#DDCC77",
    "nbn-smoothed": "#661100", "nbn-hybrid": "#000000",
}
_ENGINE_HATCH = {"ve": "", "lw": "/////", "ais": ".....", "avi": "xxxxx"}
_STYLE_FALLBACK = "#BBBBBB"


def approach(baseline: str) -> tuple[str, str]:
    """(approach key, engine) of a baseline: 'nbn-cat-bayes' -> ('nbn-cat-bayes',
    ''), 'nbn-mdn-lw' -> ('nbn-mdn', 'lw'), 'pgmpy-mle-ve' -> ('pgmpy-mle', ''),
    'pyro-empirical-importance' -> ('pyro', '')."""
    lib, rest = parse_baseline(baseline)
    parts = rest.split("-") if rest else []
    if lib != "nbn":
        if lib == "pgmpy" and parts:
            return f"pgmpy-{parts[0]}", ""
        return lib, ""
    engine = parts[-1] if parts and parts[-1] in _ENGINE_HATCH else ""
    mech = parts[:-1] if engine else parts
    if len(mech) > 1 and mech[1] == "router":
        mech = mech[:1]
    return "nbn-" + "-".join(mech), engine


def method_style(baseline: str) -> tuple[str, str]:
    """(hex colour, hatch) of a method, identical in every figure."""
    key, engine = approach(baseline)
    return _APPROACH_COLOR.get(key, _STYLE_FALLBACK), _ENGINE_HATCH.get(engine, "")


def log_axis_ok(metric: str) -> bool:
    """Figures use a log y axis except for metrics that take negative values."""
    return metric not in HIGHER_IS_BETTER   # log-likelihood


def metric_kind(metric: str) -> str:
    """``"time"`` for the timing pseudo-metrics, else the metric name itself
    (the key used by :func:`clip_band` / :func:`rank_key`)."""
    return "time" if metric in TIME_METRICS else metric


def aggregate(values, method: str) -> tuple[float, float, float]:
    """Return (center, lower_band, upper_band) per the aggregation flag.

    mean_std: center = mean, band = +/-1 std.
    iqm_iqr:  center = interquartile mean (mean of values in [Q1, Q3]),
              band   = +/-(Q3-Q1)/2 around the IQM.
    NaNs are dropped first; empty -> (nan, nan, nan). Any +inf -> (inf, nan,
    nan): the unsmoothed-MLE recovery-KL sentinel must not be trimmed away.
    """
    values = np.asarray(list(values), dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if np.isposinf(values).any():
        return float("inf"), float("nan"), float("nan")
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


def plot_band(values, method: str) -> tuple[float, float]:
    """(lower, upper) of the error bar drawn around :func:`aggregate`'s center.

    iqm_iqr:  the smallest and the largest observed value among those that
              enter the interquartile mean (the values in [Q1, Q3]), so the
              bar spans real per-seed results and always contains the IQM.
    mean_std: mean -/+ 1 std (the table band).
    Empty or any +inf -> (nan, nan)."""
    values = np.asarray(list(values), dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0 or np.isposinf(values).any():
        return float("nan"), float("nan")
    if method == "iqm_iqr":
        q1, q3 = np.percentile(values, [25, 75])
        mid = values[(values >= q1) & (values <= q3)]
        if mid.size == 0:
            mid = values
        return float(mid.min()), float(mid.max())
    _, lo, hi = aggregate(values, method)
    return lo, hi


def clip_band(kind: str, lower: float, upper: float) -> tuple[float, float]:
    """Clip a band to the metric's natural range."""
    if kind == "time" or kind in LOWER_IS_BETTER:
        lower = max(0.0, lower)
        if kind in {"tv_per_node", "jsd_per_node", "ks_per_node"}:
            upper = min(1.0, upper)
        return lower, upper
    if kind in CLOSER_TO_VALUE:
        return max(0.0, lower), upper
    if kind == "success_rate":
        return max(0.0, lower), min(100.0, upper)
    return lower, upper


def rank_key(center: float, kind: str) -> float:
    """Lower is better after this transform; NaN / +-inf sort last."""
    if center is None or not np.isfinite(center):
        return float("inf")
    if kind in CLOSER_TO_VALUE:
        return abs(center - CLOSER_TO_VALUE[kind])
    if kind in HIGHER_IS_BETTER:
        return -center
    return center


# --- x axis / instance assignment ---------------------------------------------

def is_n_train_sweep(dff: pd.DataFrame) -> bool:
    """True when some problem was fit at >= 2 distinct ``n_train`` values
    (a learning-curves sweep). A parquet where each problem has its own
    single ``n_train`` (bnlearn sizes the training set per network) is not
    a sweep, even though ``n_train`` varies across problems."""
    if "n_train" not in dff.columns:
        return False
    n_train = dff["n_train"].dropna()
    if n_train.nunique() <= 1:
        return False
    if "problem_id" not in dff.columns:
        return True
    per_problem = dff.loc[n_train.index].groupby("problem_id")["n_train"].nunique()
    return bool((per_problem > 1).any())


def sweep_axis(dff: pd.DataFrame, benchmark: str) -> str:
    """Which column is the x grid for this family slice.

    ``batch_size`` when the parquet carries batched rows (a batch_sizes sweep),
    ``n_train`` when some problem was fit at >= 2 values (learning curves),
    ``network`` for bnlearn (one bar group per real network), else ``n_nodes``.
    """
    if "batch_size" in dff.columns and (dff["batch_size"].fillna(0) > 1).any():
        return "batch_size"
    if is_n_train_sweep(dff):
        return "n_train"
    if str(benchmark).lower() == "bnlearn":
        return "network"
    return "n_nodes"


def assign_x(dff: pd.DataFrame, x_axis: str, n_nodes: dict) -> pd.DataFrame:
    """Return a copy with ``_x`` (the x value) and ``_inst`` (the seed-level
    instance key: what is left of ``(problem_id, seed)`` once the x column
    is factored out). Rows whose x cannot be resolved are dropped."""
    out = dff.copy()
    pid = out["problem_id"].astype(str)
    seed = out["seed"].astype(str)
    if x_axis == "batch_size":
        out["_x"] = out["batch_size"].fillna(1).astype(int)
        out["_inst"] = pid + "/s" + seed
    elif x_axis == "n_train":
        out = out[out["n_train"].notna()].copy()
        out["_x"] = out["n_train"].astype(int)
        out["_inst"] = out["problem_id"].astype(str) + "/s" + out["seed"].astype(str)
    elif x_axis == "network":
        out["_x"] = pid
        out["_inst"] = "s" + seed
    else:  # n_nodes
        out["_x"] = pid.map(lambda p: n_nodes.get(p))
        out = out[out["_x"].notna()].copy()
        out["_x"] = out["_x"].astype(int)
        out["_inst"] = "s" + out["seed"].astype(str)
    return out


def x_order(dfx: pd.DataFrame, x_axis: str, n_nodes: dict) -> list:
    """The x grid in display order: numeric ascending, or networks by size."""
    xs = list(dfx["_x"].dropna().unique())
    if x_axis == "network":
        return sorted(xs, key=lambda p: (n_nodes.get(str(p), 0), str(p)))
    return sorted(int(x) for x in xs)


# --- Per-cell (seed) table ----------------------------------------------------

def _metric_rows(dfx: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str, str]:
    """(rows, value_column, reducer) for a metric name.

    Accuracy metrics and per-query / total query time read ``metric`` rows;
    fit time reads ``metric == "fit_time_s"`` rows when the parquet has them
    (inference mode) and falls back to the ``fit_time_s`` column over the
    cell's accuracy rows (parameter-learning mode, one row per metric).
    ``nll_per_node`` is derived: the held-out ``log_likelihood`` rows, negated
    and divided by the number of nodes (positive, lower is better)."""
    if metric == "nll_per_node":
        rows = dfx[dfx["metric"] == "log_likelihood"].copy()
        n = pd.to_numeric(rows["n_nodes"], errors="coerce") if "n_nodes" in rows else np.nan
        rows["value"] = -pd.to_numeric(rows["value"], errors="coerce") / n
        return rows, "value", "mean"
    if metric == "query_time":
        return dfx[dfx["metric"] == "query_time_s"], "value", "mean"
    if metric == "total_query_time":
        return dfx[dfx["metric"] == "query_time_s"], "value", "sum"
    if metric == "fit_time":
        rows = dfx[dfx["metric"] == "fit_time_s"]
        if not rows.empty:
            return _with_shared_base_fit(rows, "value"), "value", "first"
        if "fit_time_s" in dfx.columns:
            rows = dfx[dfx["metric"].isin(ACCURACY_METRICS)]
            return _with_shared_base_fit(rows, "fit_time_s"), "fit_time_s", "first"
        return dfx.iloc[0:0], "value", "first"
    return dfx[dfx["metric"] == metric], "value", "mean"


def _with_shared_base_fit(rows: pd.DataFrame, vcol: str) -> pd.DataFrame:
    """Fit time of a fit-once group member = shared base fit + its own attach.

    In a fit-once group (#191) only the fitter runs ``model.fit()``; the
    reloaders' ``fit_time_s`` is just reload + engine attach, so reporting it
    as "Fit time" understated their cost by the whole base fit. The runner
    stamps the fitter's ``base_fit_time_s`` on every member and each member's
    own ``attach_time_s``: when both are present the reported fit time is
    their sum (for the fitter this equals its ``fit_time_s`` up to timer
    overhead). Rows without the split (non-nbn baselines, legacy parquets)
    keep ``fit_time_s``.
    """
    if "base_fit_time_s" not in rows.columns or "attach_time_s" not in rows.columns:
        return rows
    base = pd.to_numeric(rows["base_fit_time_s"], errors="coerce")
    attach = pd.to_numeric(rows["attach_time_s"], errors="coerce")
    both = base.notna() & attach.notna()
    if not both.any():
        return rows
    rows = rows.copy()
    rows[vcol] = pd.to_numeric(rows[vcol], errors="coerce").astype(float)
    rows.loc[both, vcol] = (base + attach)[both]
    return rows


def _query_ids(rows: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Per-row query identifier within its cell: the ``query_idx`` column
    (stamped since D5) or, for legacy parquets, the row's position among the
    cell's rows of the metric (measurements emit queries in selector order,
    which is deterministic per seed and identical across baselines)."""
    pos = rows.groupby(keys, sort=False).cumcount()
    if "query_idx" in rows.columns:
        qi = pd.to_numeric(rows["query_idx"], errors="coerce")
        return qi.where(qi.notna(), pos).astype(int)
    return pos.astype(int)


def cell_table(dfx: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict]:
    """One row per attempted (method, x, inst) for ``metric``.

    Columns: ``method, x, inst, solved, value, code``. ``solved`` is True when
    the cell has >= 1 ok row for the metric, no failure row, no failure
    sentinel and no ok row with a non-finite value; ``value`` is the per-cell
    reduction (NaN when not solved); ``code`` is the most frequent failure
    status (``"nan"`` for an ok row without a usable number; None when
    solved). Cells whose rows are all applicability / no-reference statuses
    (not_supported / not_applicable / no_result) were never attempted and are
    omitted.

    **Same queries for every method (D5).** For per-query reductions (mean,
    sum) the value of a solved cell is computed over the queries that are
    ``ok`` for *every* method that solved that ``(x, inst)``, so methods are
    never averaged over different query subsets (e.g. one method's
    ``not_supported`` / ``no_result`` query silently leaving its mean over an
    easier subset). A solved cell sharing no query with the others becomes
    unsolved with code ``"no_common_query"``. ``fit_time`` (one value per
    cell) is not restricted.

    Also returns ``n_total``: ``{x: number of distinct instances at x}`` over
    every row of ``dfx`` (any metric / status, sentinels included), the
    denominator of the ``k/n`` annotation.
    """
    n_total = {x: int(g["_inst"].nunique()) for x, g in dfx.groupby("_x")}
    keys = ["baseline", "_x", "_inst"]

    sentinel = dfx[(dfx["metric"] == "status") & dfx["status"].isin(FAILURE_STATUSES)]
    sent_code = {k: g["status"].mode().iloc[0] for k, g in sentinel.groupby(keys)}

    rows, vcol, reducer = _metric_rows(dfx, metric)
    rows = rows.copy()
    rows["_q"] = _query_ids(rows, keys) if not rows.empty else pd.Series(dtype=int)
    rows["_v"] = pd.to_numeric(rows[vcol], errors="coerce") if not rows.empty \
        else pd.Series(dtype=float)

    status_of: dict = {}
    ok_queries: dict = {}
    ok_rows: dict = {}
    for k, g in rows.groupby(keys):
        st = g["status"]
        okm = st == "ok"
        fail = st.isin(FAILURE_STATUSES)
        s_code = sent_code.get(k)
        if not bool(okm.any() or fail.any() or s_code is not None):
            continue
        ok_rows[k] = g[okm]
        if okm.any() and not fail.any() and s_code is None:
            if ok_rows[k]["_v"].isna().any():
                # An ok status without a usable number (e.g. pomegranate's NaN
                # posteriors, #198/#218) is not a result; it used to be dropped
                # silently when other rows of the cell were finite. (+inf is a
                # value: the unsmoothed-MLE recovery-KL sentinel.)
                status_of[k] = (False, "nan")
            else:
                status_of[k] = (True, None)
                ok_queries[k] = set(ok_rows[k]["_q"])
        else:
            code = (g.loc[fail, "status"].mode().iloc[0] if fail.any() else s_code)
            status_of[k] = (False, code)

    # Common query set per (x, inst) over the methods that solved it.
    common: dict = {}
    if reducer != "first":
        for k, qs in ok_queries.items():
            xi = (k[1], k[2])
            common[xi] = qs if xi not in common else common[xi] & qs

    records = []
    for k, (solved, code) in status_of.items():
        value = float("nan")
        if solved:
            g = ok_rows[k]
            if reducer != "first":
                g = g[g["_q"].isin(common.get((k[1], k[2]), set()))]
            vals = g["_v"]
            if vals.empty:
                solved, code = False, "no_common_query"
            elif reducer == "sum":
                value = float(vals.sum())
            elif reducer == "first":
                value = float(vals.iloc[0])
            else:
                value = float(vals.mean())
        records.append(dict(method=k[0], x=k[1], inst=k[2], solved=solved,
                            value=value, code=code))
    # Sentinel-only cells (fit failure / seed-skip / worker death): no metric
    # row at all, but the cell was attempted and failed.
    for k, code in sent_code.items():
        if k not in status_of:
            records.append(dict(method=k[0], x=k[1], inst=k[2], solved=False,
                                value=float("nan"), code=code))
    cols = ["method", "x", "inst", "solved", "value", "code"]
    return pd.DataFrame.from_records(records, columns=cols), n_total


# --- Views --------------------------------------------------------------------

@dataclass
class Views:
    """Result of :func:`build_views` for one (family, metric)."""
    metric: str
    kind: str
    xs: list
    all: pd.DataFrame            # method, x, center, lo, hi, plo, phi, k, n, code, shown
    common: pd.DataFrame         # same columns
    common_sets: dict = field(default_factory=dict)   # x -> sorted list of inst
    selection: dict = field(default_factory=dict)     # view -> {x: [nbn methods]}


def _agg_cell(vals, aggregation: str) -> tuple[float, float, float]:
    return aggregate(vals, aggregation)


def _fail_code(g: pd.DataFrame):
    codes = g.loc[~g["solved"], "code"].dropna()
    return str(codes.mode().iloc[0]) if len(codes) else None


def _view_frame(recs) -> pd.DataFrame:
    cols = ["method", "x", "center", "lo", "hi", "plo", "phi", "k", "n", "code"]
    df = pd.DataFrame.from_records(recs, columns=cols)
    # keep ``code`` a plain str-or-None column (pandas would coerce None to
    # NaN, which is truthy and would print as "nan" in a table cell).
    df["code"] = df["code"].astype(object).where(df["code"].notna(), None)
    return df


def all_view(cells: pd.DataFrame, n_total: dict, aggregation: str) -> pd.DataFrame:
    """Per (method, x): aggregate over the seeds the method solved."""
    recs = []
    for (m, x), g in cells.groupby(["method", "x"]):
        solved = g[g["solved"]]
        c, lo, hi = _agg_cell(solved["value"], aggregation)
        plo, phi = plot_band(solved["value"], aggregation)
        recs.append(dict(method=m, x=x, center=c, lo=lo, hi=hi, plo=plo, phi=phi,
                         k=int(len(solved)),
                         n=int(n_total.get(x, len(g))), code=_fail_code(g)))
    return _view_frame(recs)


def common_sets(cells: pd.DataFrame, methods, xs) -> dict:
    """``{x: set(inst)}`` solved by every method in ``methods`` that solved at
    least one seed at x. Methods with no attempted cell at x do not
    participate (e.g. a non-batchable baseline at B > 1)."""
    out = {}
    for x in xs:
        cx = cells[cells["x"] == x]
        sets = []
        for m in methods:
            g = cx[cx["method"] == m]
            if g.empty:
                continue
            solved = set(g.loc[g["solved"], "inst"])
            if solved:
                sets.append(solved)
        out[x] = set.intersection(*sets) if sets else set()
    return out


def common_view(cells: pd.DataFrame, n_total: dict, aggregation: str,
                methods, xs, sets: dict) -> pd.DataFrame:
    """Per (method, x) for ``methods``: aggregate over ``sets[x]`` only.
    ``k`` is the number of common seeds the method contributes (== |C(x)| when
    it solved them all, 0 when it is DNF at x)."""
    recs = []
    for x in xs:
        cx = cells[cells["x"] == x]
        C = sets.get(x, set())
        for m in methods:
            g = cx[cx["method"] == m]
            if g.empty:
                continue
            solved = g[g["solved"] & g["inst"].isin(C)]
            c, lo, hi = _agg_cell(solved["value"], aggregation)
            plo, phi = plot_band(solved["value"], aggregation)
            recs.append(dict(method=m, x=x, center=c, lo=lo, hi=hi, plo=plo, phi=phi,
                             k=int(len(solved)), n=int(len(C)), code=_fail_code(g)))
    return _view_frame(recs)


def select_top(view: pd.DataFrame, candidates, kind: str, top_n: int) -> dict:
    """``{x: [method, ...]}`` — the ``top_n`` candidates at each x, ranked on
    the view's center (direction-aware), ties broken by more solved seeds then
    name. Candidates with no solved seed at x rank after every solved one, so
    a DNF nbn method is shown (with its failure code) only when fewer than
    ``top_n`` nbn methods solved anything at that x."""
    out = {}
    for x, g in view.groupby("x"):
        g = g[g["method"].isin(candidates)]
        ranked = sorted(
            g.itertuples(index=False),
            key=lambda r: (r.k == 0, rank_key(r.center, kind), -r.k, str(r.method)),
        )
        out[x] = [r.method for r in ranked[:top_n]]
    return out


def build_views(cells: pd.DataFrame, n_total: dict, aggregation: str,
                metric: str, xs, top_n: int = 2) -> Views:
    """Assemble the ``all`` and ``common`` views with the nbn selection.

    ``shown`` marks the rows that make it into the figure / table: every
    non-nbn method attempted at x, plus the nbn methods selected on the
    ``all`` view at x (the same set in both views).
    """
    kind = metric_kind(metric)
    methods = sorted(cells["method"].unique())
    non_nbn = [m for m in methods if not is_nbn(m)]
    nbn = [m for m in methods if is_nbn(m)]

    va = all_view(cells, n_total, aggregation)
    sel_all = select_top(va, nbn, kind, top_n)
    va["shown"] = [
        (not is_nbn(r.method)) or (r.method in sel_all.get(r.x, []))
        for r in va.itertuples(index=False)
    ]

    # common: the SAME nbn selection as ``all`` (chosen once, on the all view),
    # so the two views always show the same methods and are directly
    # comparable; C(x) is the intersection over that shown set.
    sel_common = {x: list(sel_all.get(x, [])) for x in xs}
    shown_by_x = {x: non_nbn + sel_common[x] for x in xs}
    sets = {}
    recs = []
    for x in xs:
        sx = common_sets(cells, shown_by_x[x], [x])[x]
        sets[x] = sx
        recs.append(common_view(cells, n_total, aggregation, shown_by_x[x], [x], {x: sx}))
    vc = (pd.concat(recs, ignore_index=True) if recs
          else _view_frame([]))
    vc["shown"] = True
    return Views(
        metric=metric, kind=kind, xs=list(xs), all=va, common=vc,
        common_sets={x: sorted(s) for x, s in sets.items()},
        selection={"all": sel_all, "common": sel_common},
    )
