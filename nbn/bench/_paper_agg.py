"""Seed-level aggregation behind the paper figures: the ``all`` / ``common`` views.

Every benchmark figure is a grouped bar plot over a discrete x grid (n_nodes,
n_train, batch_size or network name). The unit of "solved / not solved" is the
**seed** (more precisely the cell ``(problem_id, seed)`` with the x column
factored out); a method solves a seed for a metric when every row of that
metric in the cell is ``ok`` and no failure sentinel (``metric == "status"``
with timeout / oom / error) was emitted for the cell.

Two views share one contract (``docs/v0.18-bar-reporting-all-common.md``):

* ``all``    — each method is aggregated over the seeds *it* solved. The
               table / bar annotation reports ``k/n`` (solved / total seeds at
               that x) whenever ``k < n``; a method with ``k == 0`` shows its
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

Aggregation across seeds is ``iqm_iqr`` (interquartile mean ± IQR/2) or
``mean_std``; the per-seed value is the per-cell reduction (mean over queries
for accuracy and per-query time, sum for total query time, first for fit time).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- Metric conventions --------------------------------------------------------

ACCURACY_METRICS = (
    "tv_per_node", "jsd_per_node", "w1_per_node", "log_likelihood",
    "param_recovery_tv", "param_recovery_kl",
    "calibration_pit_ks", "calibration_sd_ratio",
)
LOWER_IS_BETTER = frozenset({
    "tv_per_node", "jsd_per_node", "w1_per_node",
    "param_recovery_tv", "param_recovery_kl", "calibration_pit_ks",
})
HIGHER_IS_BETTER = frozenset({"log_likelihood"})
CLOSER_TO_VALUE = {"calibration_sd_ratio": 1.0}
METRIC_LABEL = {
    "tv_per_node": "TV",
    "jsd_per_node": "JSD",
    "w1_per_node": "W1",
    "log_likelihood": "LL",
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
FAILURE_STATUSES = frozenset({"timeout", "oom", "error"})


def parse_baseline(baseline: str) -> tuple[str, str]:
    """('nbn-cat-ve') -> ('nbn', 'cat-ve')."""
    parts = str(baseline).split("-", 1)
    return (parts[0], parts[1] if len(parts) > 1 else "")


def is_nbn(baseline: str) -> bool:
    return parse_baseline(baseline)[0] == "nbn"


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


def clip_band(kind: str, lower: float, upper: float) -> tuple[float, float]:
    """Clip a band to the metric's natural range."""
    if kind == "time" or kind in LOWER_IS_BETTER:
        lower = max(0.0, lower)
        if kind in {"tv_per_node", "jsd_per_node"}:
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
    cell's accuracy rows (parameter-learning mode, one row per metric)."""
    if metric == "query_time":
        return dfx[dfx["metric"] == "query_time_s"], "value", "mean"
    if metric == "total_query_time":
        return dfx[dfx["metric"] == "query_time_s"], "value", "sum"
    if metric == "fit_time":
        rows = dfx[dfx["metric"] == "fit_time_s"]
        if not rows.empty:
            return rows, "value", "first"
        if "fit_time_s" in dfx.columns:
            rows = dfx[dfx["metric"].isin(ACCURACY_METRICS)]
            return rows, "fit_time_s", "first"
        return dfx.iloc[0:0], "value", "first"
    return dfx[dfx["metric"] == metric], "value", "mean"


def cell_table(dfx: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict]:
    """One row per attempted (method, x, inst) for ``metric``.

    Columns: ``method, x, inst, solved, value, code``. ``solved`` is True when
    the cell has >= 1 ok row for the metric, no failure row, and no failure
    sentinel; ``value`` is the per-cell reduction over ok rows (NaN when not
    solved); ``code`` is the most frequent failure status (None when solved).
    Cells whose rows are all applicability statuses (not_supported /
    not_applicable) were never attempted and are omitted. An ok cell whose
    reduced value is NaN counts as unsolved with code ``"nan"``.

    Also returns ``n_total``: ``{x: number of distinct instances at x}`` over
    every row of ``dfx`` (any metric / status, sentinels included), the
    denominator of the ``k/n`` annotation.
    """
    n_total = {x: int(g["_inst"].nunique()) for x, g in dfx.groupby("_x")}
    keys = ["baseline", "_x", "_inst"]

    sentinel = dfx[(dfx["metric"] == "status") & dfx["status"].isin(FAILURE_STATUSES)]
    sent_code = {k: g["status"].mode().iloc[0] for k, g in sentinel.groupby(keys)}

    rows, vcol, reducer = _metric_rows(dfx, metric)
    records = []
    seen = set()
    for k, g in rows.groupby(keys):
        st = g["status"]
        okm = st == "ok"
        fail = st.isin(FAILURE_STATUSES)
        s_code = sent_code.get(k)
        attempted = bool(okm.any() or fail.any() or s_code is not None)
        if not attempted:
            continue
        seen.add(k)
        solved = bool(okm.any() and not fail.any() and s_code is None)
        if solved:
            vals = pd.to_numeric(g.loc[okm, vcol], errors="coerce").dropna()
            if reducer == "sum":
                value = float(vals.sum()) if len(vals) else float("nan")
            elif reducer == "first":
                value = float(vals.iloc[0]) if len(vals) else float("nan")
            else:
                value = float(vals.mean()) if len(vals) else float("nan")
            code = None
            if np.isnan(value):
                # ok status but no usable number (e.g. pomegranate's NaN
                # posteriors, #218): not a result, so not solved.
                solved, code = False, "nan"
        else:
            value = float("nan")
            code = (g.loc[fail, "status"].mode().iloc[0] if fail.any() else s_code)
        records.append(dict(method=k[0], x=k[1], inst=k[2], solved=solved,
                            value=value, code=code))
    # Sentinel-only cells (fit failure / seed-skip / worker death): no metric
    # row at all, but the cell was attempted and failed.
    for k, code in sent_code.items():
        if k not in seen:
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
    all: pd.DataFrame            # method, x, center, lo, hi, k, n, code, shown
    common: pd.DataFrame         # same columns
    common_sets: dict = field(default_factory=dict)   # x -> sorted list of inst
    selection: dict = field(default_factory=dict)     # view -> {x: [nbn methods]}


def _agg_cell(vals, aggregation: str) -> tuple[float, float, float]:
    return aggregate(vals, aggregation)


def _fail_code(g: pd.DataFrame):
    codes = g.loc[~g["solved"], "code"].dropna()
    return str(codes.mode().iloc[0]) if len(codes) else None


def _view_frame(recs) -> pd.DataFrame:
    cols = ["method", "x", "center", "lo", "hi", "k", "n", "code"]
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
        recs.append(dict(method=m, x=x, center=c, lo=lo, hi=hi, k=int(len(solved)),
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
            recs.append(dict(method=m, x=x, center=c, lo=lo, hi=hi,
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
