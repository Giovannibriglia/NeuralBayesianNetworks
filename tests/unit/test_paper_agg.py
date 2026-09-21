"""Unit tests for the seed-level ``all`` / ``common`` aggregation
(``nbn/bench/_paper_agg.py``, docs/v0.18-bar-reporting-all-common.md)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nbn.bench._paper_agg import (
    all_view,
    assign_x,
    build_views,
    cell_table,
    common_sets,
    common_view,
    rank_key,
    select_top,
    sweep_axis,
    x_order,
)


def _row(baseline, seed, metric="query_time_s", value=0.01, status="ok",
         batch_size=1, problem_id="50", **extra):
    return {
        "benchmark": "synthetic", "family": "discrete", "problem_id": problem_id,
        "seed": seed, "baseline": baseline, "query_role": "hub",
        "query_kind": "prediction", "metric": metric, "value": value,
        "status": status, "fit_time_s": 1.0, "query_time_s": value,
        "metrics_time_s": 0.0, "error_msg": None, "batch_size": batch_size,
        **extra,
    }


def _batch_speed_pgmpy_case() -> pd.DataFrame:
    """The 2026-09-11 batch_speed discrete pgmpy scenario, in miniature:

    * nbn-cat-ve, nbn-cat-lw: 5 seeds ok at B in {1, 4} (two queries each).
    * pgmpy-mle-ve (pinned B=1): seeds 0,1 ok; seed 2 partial (one ok query,
      one timeout); seeds 3,4 seed-skipped (a single ``status`` sentinel).
    * pyro (pinned B=1): every seed timed out.
    """
    rows = []
    for b, v in (("nbn-cat-ve", 0.001), ("nbn-cat-lw", 0.01)):
        for bs in (1, 4):
            for seed in range(5):
                for q in range(2):
                    rows.append(_row(b, seed, value=v * (1 + 0.1 * seed) / bs,
                                     batch_size=bs))
    for seed in (0, 1):
        for q in range(2):
            rows.append(_row("pgmpy-mle-ve", seed, value=0.003 * (1 + seed)))
    rows.append(_row("pgmpy-mle-ve", 2, value=0.9))
    rows.append(_row("pgmpy-mle-ve", 2, value=float("nan"), status="timeout"))
    for seed in (3, 4):
        rows.append(_row("pgmpy-mle-ve", seed, metric="status", value=float("nan"),
                         status="timeout"))
    for seed in range(5):
        rows.append(_row("pyro-empirical-importance", seed, value=float("nan"),
                         status="timeout"))
    return pd.DataFrame(rows)


def _prep(df, metric="query_time"):
    x_axis = sweep_axis(df, "synthetic")
    dfx = assign_x(df, x_axis, {"50": 50})
    xs = x_order(dfx, x_axis, {})
    cells, n_total = cell_table(dfx, metric)
    return x_axis, xs, cells, n_total


# --- x axis / instances --------------------------------------------------------

def test_sweep_axis_detection():
    df = _batch_speed_pgmpy_case()
    assert sweep_axis(df, "synthetic") == "batch_size"
    plain = df[df["batch_size"] == 1]
    assert sweep_axis(plain, "synthetic") == "n_nodes"
    assert sweep_axis(plain, "bnlearn") == "network"
    lc = plain.assign(n_train=[4096 if i % 2 else 8192 for i in range(len(plain))])
    assert sweep_axis(lc, "synthetic") == "n_train"


def test_sweep_axis_per_problem_n_train_is_not_a_sweep():
    # bnlearn sizes n_train per network (asia 10240, barley 81920, ...): n_train
    # varies across problems but is constant within each one, so the x grid is
    # still the network, not n_train.
    df = _batch_speed_pgmpy_case()
    plain = df[df["batch_size"] == 1].copy()
    half = len(plain) // 2
    plain["problem_id"] = ["asia"] * half + ["barley"] * (len(plain) - half)
    plain["n_train"] = [10240] * half + [81920] * (len(plain) - half)
    assert sweep_axis(plain, "bnlearn") == "network"
    assert sweep_axis(plain, "synthetic") == "n_nodes"
    # ... while a real sweep (n_train varying inside a problem) still wins.
    lc = plain.assign(n_train=[4096 if i % 2 else 8192 for i in range(len(plain))])
    assert sweep_axis(lc, "bnlearn") == "n_train"


def test_assign_x_instances_factor_out_the_axis():
    df = _batch_speed_pgmpy_case()
    dfx = assign_x(df, "batch_size", {})
    assert set(dfx["_x"]) == {1, 4}
    assert set(dfx["_inst"]) == {f"50/s{s}" for s in range(5)}
    dfn = assign_x(df[df["batch_size"] == 1], "n_nodes", {"50": 50})
    assert set(dfn["_x"]) == {50}
    assert set(dfn["_inst"]) == {f"s{s}" for s in range(5)}
    # unresolvable problem ids are dropped, not crashed on
    assert assign_x(df, "n_nodes", {}).empty


def test_x_order_networks_by_size():
    dfx = pd.DataFrame({"_x": ["alarm", "asia", "asia"]})
    assert x_order(dfx, "network", {"asia": 8, "alarm": 37}) == ["asia", "alarm"]


# --- cell table ---------------------------------------------------------------

def test_cell_table_pgmpy_case():
    _, xs, cells, n_total = _prep(_batch_speed_pgmpy_case())
    assert n_total == {1: 5, 4: 5}
    pg = cells[(cells["method"] == "pgmpy-mle-ve") & (cells["x"] == 1)].set_index("inst")
    assert pg.loc["50/s0", "solved"] and pg.loc["50/s1", "solved"]
    assert pg.loc["50/s0", "value"] == pytest.approx(0.003)
    # partial seed: one ok query + one timeout -> NOT solved (seed-level rule)
    assert not pg.loc["50/s2", "solved"] and pg.loc["50/s2", "code"] == "timeout"
    # seed-skipped sentinels: attempted, failed, code propagated
    assert not pg.loc["50/s3", "solved"] and pg.loc["50/s3", "code"] == "timeout"
    # pinned baseline has no cell at B=4
    assert cells[(cells["method"] == "pgmpy-mle-ve") & (cells["x"] == 4)].empty
    # per-query time = mean over the cell's queries
    ve = cells[(cells["method"] == "nbn-cat-ve") & (cells["x"] == 4)].set_index("inst")
    assert ve.loc["50/s0", "value"] == pytest.approx(0.001 / 4)


def test_cell_table_total_query_time_sums_and_fit_time_first():
    df = _batch_speed_pgmpy_case()
    df = df[df["batch_size"] == 1]
    df = pd.concat([df, pd.DataFrame([_row("nbn-cat-ve", s, metric="fit_time_s", value=2.5)
                                      for s in range(5)])], ignore_index=True)
    dfx = assign_x(df, "n_nodes", {"50": 50})
    tot, _ = cell_table(dfx, "total_query_time")
    ve = tot[tot["method"] == "nbn-cat-ve"].set_index("inst")
    assert ve.loc["s0", "value"] == pytest.approx(0.002)
    fit, _ = cell_table(dfx, "fit_time")
    assert fit[fit["method"] == "nbn-cat-ve"]["value"].tolist() == [2.5] * 5


def test_cell_table_fit_time_from_column_in_pl_mode():
    """PL parquets have no fit_time_s metric rows; the column on the accuracy
    rows is used instead."""
    rows = [{"benchmark": "synthetic", "family": "discrete", "problem_id": "6",
             "seed": s, "baseline": b, "query_role": "", "query_kind": "prediction",
             "metric": "log_likelihood", "value": -1.0, "status": "ok",
             "fit_time_s": 3.0 + s, "query_time_s": float("nan"),
             "metrics_time_s": 0.0, "error_msg": None}
            for s in range(2) for b in ("nbn-cat", "pgmpy-mle")]
    dfx = assign_x(pd.DataFrame(rows), "n_nodes", {"6": 6})
    fit, n_total = cell_table(dfx, "fit_time")
    assert n_total == {6: 2}
    assert sorted(fit[fit["method"] == "nbn-cat"]["value"]) == [3.0, 4.0]


def test_cell_table_nan_ok_value_is_not_solved():
    """status ok with a NaN metric (pomegranate posteriors, #218) is not a
    result: unsolved with code 'nan'."""
    df = pd.DataFrame([_row("pomegranate-discrete-ve", s, metric="tv_per_node",
                            value=float("nan")) for s in range(3)])
    dfx = assign_x(df, "n_nodes", {"50": 50})
    cells, _ = cell_table(dfx, "tv_per_node")
    assert (~cells["solved"]).all() and set(cells["code"]) == {"nan"}


def test_cell_table_applicability_rows_are_not_attempted():
    df = pd.DataFrame([_row("nbn-lg", s, metric="calibration_pit_ks",
                            value=float("nan"), status="not_applicable")
                       for s in range(2)])
    dfx = assign_x(df, "n_nodes", {"50": 50})
    cells, _ = cell_table(dfx, "calibration_pit_ks")
    assert cells.empty


# --- views --------------------------------------------------------------------

def test_all_view_k_over_n_and_failure_code():
    _, xs, cells, n_total = _prep(_batch_speed_pgmpy_case())
    va = all_view(cells, n_total, "iqm_iqr").set_index(["method", "x"])
    pg = va.loc[("pgmpy-mle-ve", 1)]
    assert (pg["k"], pg["n"]) == (2, 5)
    # aggregated over the two solved seeds only: values 0.003 and 0.006
    assert pg["center"] == pytest.approx(0.0045)
    assert pg["code"] == "timeout"
    py = va.loc[("pyro-empirical-importance", 1)]
    assert py["k"] == 0 and py["code"] == "timeout" and math.isnan(py["center"])
    ve = va.loc[("nbn-cat-ve", 4)]
    assert (ve["k"], ve["n"]) == (5, 5) and ve["code"] is None
    assert va["code"].map(lambda c: c is None or isinstance(c, str)).all()


def test_common_sets_ignore_dnf_and_absent_methods():
    _, xs, cells, n_total = _prep(_batch_speed_pgmpy_case())
    methods = ["pgmpy-mle-ve", "pyro-empirical-importance", "nbn-cat-ve", "nbn-cat-lw"]
    sets = common_sets(cells, methods, xs)
    # B=1: pgmpy's two seeds; pyro (k=0) does not empty the intersection
    assert sets[1] == {"50/s0", "50/s1"}
    # B=4: pgmpy/pyro have no cell -> all five seeds
    assert sets[4] == {f"50/s{s}" for s in range(5)}


def test_common_view_aggregates_over_common_seeds_only():
    _, xs, cells, n_total = _prep(_batch_speed_pgmpy_case())
    methods = ["pgmpy-mle-ve", "nbn-cat-ve"]
    sets = common_sets(cells, methods, xs)
    vc = common_view(cells, n_total, "mean_std", methods, xs, sets).set_index(["method", "x"])
    ve = vc.loc[("nbn-cat-ve", 1)]
    # seeds 0,1 only: 0.001 and 0.0011 -> mean 0.00105
    assert ve["center"] == pytest.approx(0.00105)
    assert (ve["k"], ve["n"]) == (2, 2)
    assert vc.loc[("nbn-cat-ve", 4)]["k"] == 5


def test_rank_key_directions():
    assert rank_key(0.1, "tv_per_node") < rank_key(0.2, "tv_per_node")
    assert rank_key(-3.0, "log_likelihood") < rank_key(-10.0, "log_likelihood")
    assert rank_key(1.05, "calibration_sd_ratio") < rank_key(0.8, "calibration_sd_ratio")
    assert rank_key(float("inf"), "param_recovery_kl") == float("inf")
    assert rank_key(float("nan"), "time") == float("inf")


def test_select_top_ranks_on_center_and_dnf_last():
    view = pd.DataFrame([
        dict(method="nbn-a", x=1, center=0.5, k=5, n=5),
        dict(method="nbn-b", x=1, center=0.1, k=1, n=5),
        dict(method="nbn-c", x=1, center=0.3, k=5, n=5),
        dict(method="nbn-d", x=1, center=float("nan"), k=0, n=5),
    ])
    sel = select_top(view, ["nbn-a", "nbn-b", "nbn-c", "nbn-d"], "tv_per_node", 2)
    assert sel[1] == ["nbn-b", "nbn-c"]
    sel1 = select_top(view, ["nbn-a", "nbn-b", "nbn-c"], "log_likelihood", 1)
    assert sel1[1] == ["nbn-a"]
    # a DNF method fills a slot only when the solved ones run out
    assert select_top(view, ["nbn-a", "nbn-d"], "tv_per_node", 2)[1] == ["nbn-a", "nbn-d"]
    assert select_top(view, ["nbn-d"], "tv_per_node", 1)[1] == ["nbn-d"]


def test_build_views_end_to_end_pgmpy_case():
    x_axis, xs, cells, n_total = _prep(_batch_speed_pgmpy_case())
    v = build_views(cells, n_total, "iqm_iqr", "query_time", xs, top_n=1)
    # all: non-nbn always shown, only the best nbn (ve) at each x
    shown = v.all[v.all["shown"]]
    assert set(shown[shown["x"] == 1]["method"]) == {
        "pgmpy-mle-ve", "pyro-empirical-importance", "nbn-cat-ve"}
    assert set(shown[shown["x"] == 4]["method"]) == {"nbn-cat-ve"}
    assert v.selection["all"] == {1: ["nbn-cat-ve"], 4: ["nbn-cat-ve"]}
    # common: C(1) = pgmpy's two seeds, C(4) = all five
    assert v.common_sets == {1: ["50/s0", "50/s1"],
                             4: [f"50/s{s}" for s in range(5)]}
    vc = v.common.set_index(["method", "x"])
    assert vc.loc[("pgmpy-mle-ve", 1)]["center"] == pytest.approx(0.0045)
    assert vc.loc[("nbn-cat-ve", 1)]["k"] == 2
    assert vc.loc[("pyro-empirical-importance", 1)]["k"] == 0


def test_build_views_common_reuses_all_selection():
    """The nbn set is chosen once on the all view and kept in common, even if
    a different nbn method would win on the common seeds."""
    rows = [_row("nbn-a", s, metric="tv_per_node", value=v)
            for s, v in ((0, 0.05), (1, 0.05), (2, 0.05), (3, 0.05), (4, 0.9))]
    rows += [_row("nbn-b", s, metric="tv_per_node", value=0.2) for s in range(5)]
    rows += [_row("pgmpy-mle", s, metric="tv_per_node", value=0.1) for s in (3, 4)]
    rows += [_row("pgmpy-mle", s, metric="tv_per_node", value=float("nan"), status="error")
             for s in (0, 1, 2)]
    dfx = assign_x(pd.DataFrame(rows), "n_nodes", {"50": 50})
    cells, n_total = cell_table(dfx, "tv_per_node")
    v = build_views(cells, n_total, "mean_std", "tv_per_node", [50], top_n=1)
    # all view: nbn-a wins (mean 0.22 < 0.2? no: 0.22 > 0.2 -> nbn-b wins)
    assert v.selection["all"][50] == ["nbn-b"]
    assert v.selection["common"][50] == ["nbn-b"]
    assert v.common_sets[50] == ["s3", "s4"]
    vc = v.common.set_index("method")
    assert set(vc.index) == {"pgmpy-mle", "nbn-b"}
    assert vc.loc["nbn-b", "k"] == 2


def test_build_views_plus_inf_center_survives():
    rows = [_row("pgmpy-mle", s, metric="param_recovery_kl", value=np.inf) for s in range(3)]
    rows += [_row("nbn-cat", s, metric="param_recovery_kl", value=0.2) for s in range(3)]
    dfx = assign_x(pd.DataFrame(rows), "n_nodes", {"50": 50})
    cells, n_total = cell_table(dfx, "param_recovery_kl")
    v = build_views(cells, n_total, "iqm_iqr", "param_recovery_kl", [50])
    va = v.all.set_index("method")
    assert np.isposinf(va.loc["pgmpy-mle", "center"]) and va.loc["pgmpy-mle", "k"] == 3
