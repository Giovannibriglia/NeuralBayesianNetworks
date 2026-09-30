"""``nbn-bench paper``: multi-panel publication figures (nbn/bench/_paper_panels.py).

Covers the method-selection rule (best parametric + best non-parametric nbn
per family, chosen once on the headline metric; fallback to the runner-up of
the other category), the legend labels / colours, rerun splicing, and one
end-to-end render each for an inference-mode and a parameter-learning-mode
miniature parquet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nbn.bench._paper_panels import (
    drop_excluded,
    headline_metric,
    method_color,
    nbn_category,
    rank_nbn,
    run_paper,
    select_nbn,
    short_label,
)


# --- Pure helpers -------------------------------------------------------------

class TestLabelsAndCategories:
    @pytest.mark.parametrize("baseline, label", [
        ("pgmpy-mle-ve", "pgmpy MLE+VE"),
        ("pgmpy-bayes-ve", "pgmpy Bayes+VE"),
        ("pgmpy-lg-predict", "pgmpy LG"),
        ("pomegranate-discrete-ve", "pomegranate"),
        ("pyro-empirical-importance", "pyro IS"),
        ("nbn-cat-ve", "NBN cat+VE"),
        ("nbn-mdn-lw", "NBN MDN+LW"),
        ("nbn-kde-lw", "NBN KDE+LW"),
        ("nbn-knn-lw", "NBN kNN+LW"),
        ("nbn-flexcode-lw", "NBN FlexCode+LW"),
        ("nbn-lg-avi", "NBN LG+AVI"),
        ("nbn-mdn", "NBN MDN"),          # parameter-learning baseline, no engine
    ])
    def test_short_label(self, baseline, label):
        assert short_label(baseline) == label

    def test_categories(self):
        assert nbn_category("nbn-cat-ve") == "parametric"
        assert nbn_category("nbn-flow-avi") == "parametric"
        assert nbn_category("nbn-kde-lw") == "nonparametric"
        assert nbn_category("nbn-flexcode") == "nonparametric"
        assert nbn_category("pgmpy-mle-ve") is None

    def test_colors_are_deterministic_and_distinct_per_engine(self):
        assert method_color("nbn-cat-ve") == method_color("nbn-cat-ve")
        engines = {method_color(f"nbn-cat-{e}") for e in ("ve", "lw", "ais", "avi")}
        assert len(engines) == 4
        assert method_color("pgmpy-mle-ve") != method_color("pgmpy-bayes-ve")

    def test_headline_metric(self):
        assert headline_metric("discrete", {"tv_per_node", "w1_per_node"}) == "tv_per_node"
        assert headline_metric("continuous_lg", {"tv_per_node", "w1_per_node"}) == "w1_per_node"
        assert headline_metric("hybrid", {"query_time"}) == "query_time"
        assert headline_metric("discrete", {"param_recovery_tv", "log_likelihood"}) \
            == "param_recovery_tv"
        assert headline_metric("discrete", set()) is None


def _view(rows):
    """rows: (method, x, center, k) -> an ``all`` view frame."""
    v = pd.DataFrame(rows, columns=["method", "x", "center", "k"])
    v["lo"] = v["hi"] = v["center"]
    v["n"] = 5
    v["code"] = None
    return v


class TestSelection:
    def test_candidates_need_half_coverage_and_rank_on_common_x(self):
        # MDN better than LG where both ran, but DNF at n >= 500 (fit budget).
        # Coverage 3/5 >= ceil(5/2)=3 -> candidate; X* = {10, 50, 100};
        # ranked on X*, MDN wins.
        v = _view([
            ("nbn-lg-lw", 10, 0.13, 5), ("nbn-lg-lw", 50, 0.22, 5),
            ("nbn-lg-lw", 100, 0.09, 5), ("nbn-lg-lw", 500, 0.10, 5),
            ("nbn-lg-lw", 1000, 0.15, 5),
            ("nbn-mdn-lw", 10, 0.05, 5), ("nbn-mdn-lw", 50, 0.06, 5),
            ("nbn-mdn-lw", 100, 0.06, 5), ("nbn-mdn-lw", 500, float("nan"), 0),
            ("nbn-kde-lw", 10, 0.04, 5), ("nbn-kde-lw", 50, float("nan"), 0),
            ("nbn-knn-lw", 10, 0.07, 5), ("nbn-knn-lw", 50, 0.10, 5),
            ("nbn-knn-lw", 100, 0.07, 5),
            ("pgmpy-lg-predict", 10, 0.1, 5),
        ])
        xs = [10, 50, 100, 500, 1000]
        rk = rank_nbn(v, "w1_per_node", xs)
        par = rk["parametric"]
        assert par.x_star == [10, 50, 100]
        assert [r.method for r in par.ranked] == ["nbn-mdn-lw", "nbn-lg-lw"]
        assert par.ranked[0].score == pytest.approx((0.05 + 0.06 + 0.06) / 3)
        assert par.ranked[1].score == pytest.approx((0.13 + 0.22 + 0.09) / 3)
        non = rk["nonparametric"]
        # kde solved 1/5 x < 3 -> not a candidate; knn is the only candidate
        assert non.x_star == [10, 50, 100]
        assert [(r.method, r.candidate) for r in non.ranked] == \
            [("nbn-knn-lw", True), ("nbn-kde-lw", False)]
        assert select_nbn(rk) == ["nbn-mdn-lw", "nbn-knn-lw"]

    def test_relaxed_rule_when_nobody_reaches_half_coverage(self):
        v = _view([
            ("nbn-cat-ve", 10, 0.05, 5),
            ("nbn-cat-lw", 10, 0.01, 5), ("nbn-cat-lw", 50, 0.02, 5),
        ])
        rk = rank_nbn(v, "tv_per_node", [10, 50, 100, 500, 1000, 5000])
        par = rk["parametric"]
        assert all(r.candidate for r in par.ranked)
        assert par.x_star == [10]
        assert [r.method for r in par.ranked] == ["nbn-cat-lw", "nbn-cat-ve"]

    def test_select_fills_missing_category_with_runner_up(self):
        v = _view([
            ("nbn-cat-ve", 10, 0.1, 5), ("nbn-cat-ve", 50, 0.1, 5),
            ("nbn-cat-lw", 10, 0.2, 5), ("nbn-cat-lw", 50, 0.2, 5),
            ("nbn-cat-avi", 10, 0.3, 5), ("nbn-cat-avi", 50, 0.3, 5),
        ])
        rk = rank_nbn(v, "tv_per_node", [10, 50])
        assert "nonparametric" not in rk
        assert select_nbn(rk) == ["nbn-cat-ve", "nbn-cat-lw"]

    def test_methods_that_solved_nothing_are_never_ranked(self):
        v = _view([
            ("nbn-lg-lw", 10, 0.2, 5),
            ("nbn-kde-lw", 10, float("nan"), 0),
        ])
        rk = rank_nbn(v, "w1_per_node", [10])
        assert "nonparametric" not in rk
        assert select_nbn(rk) == ["nbn-lg-lw"]


# --- End-to-end -----------------------------------------------------------------

def _inf_row(*, family, baseline, seed, problem_id, metric, value, status="ok",
             query_role="hub", evidence_mode="full", batch_size=1):
    return {
        "benchmark": "synthetic", "family": family, "problem_id": problem_id,
        "seed": seed, "baseline": baseline, "query_role": query_role,
        "query_kind": "prediction", "metric": metric, "value": value,
        "status": status, "fit_time_s": 1.0, "query_time_s": 0.01,
        "metrics_time_s": 0.0, "error_msg": None, "batch_size": batch_size,
        "evidence_mode": evidence_mode, "n_nodes": int(problem_id),
    }


def _inference_df() -> pd.DataFrame:
    """discrete: pgmpy + 3 nbn-cat engines (no non-parametric -> fallback);
    continuous_lg: pgmpy-lg + nbn-lg-lw / nbn-mdn-lw / nbn-kde-lw."""
    rows = []
    acc = {"nbn-cat-ve": 0.01, "nbn-cat-lw": 0.02, "nbn-cat-ais": 0.03,
           "pgmpy-mle-ve": 0.01}
    for pid in ("10", "50"):
        for seed in (0, 1):
            for b, v in acc.items():
                for m in ("tv_per_node", "jsd_per_node"):
                    rows.append(_inf_row(family="discrete", baseline=b, seed=seed,
                                         problem_id=pid, metric=m, value=v))
                rows.append(_inf_row(family="discrete", baseline=b, seed=seed,
                                     problem_id=pid, metric="query_time_s", value=0.01))
                rows.append(_inf_row(family="discrete", baseline=b, seed=seed,
                                     problem_id=pid, metric="fit_time_s", value=1.0,
                                     query_role=""))
    cont = {"nbn-lg-lw": 0.10, "nbn-mdn-lw": 0.05, "nbn-kde-lw": 0.07,
            "pgmpy-lg-predict": 0.2}
    for pid in ("10", "50"):
        for seed in (0, 1):
            for b, v in cont.items():
                rows.append(_inf_row(family="continuous_lg", baseline=b, seed=seed,
                                     problem_id=pid, metric="w1_per_node", value=v))
                rows.append(_inf_row(family="continuous_lg", baseline=b, seed=seed,
                                     problem_id=pid, metric="query_time_s", value=0.02))
                rows.append(_inf_row(family="continuous_lg", baseline=b, seed=seed,
                                     problem_id=pid, metric="fit_time_s", value=2.0,
                                     query_role=""))
    # a not_supported baseline in continuous_lg must not appear
    rows.append(_inf_row(family="continuous_lg", baseline="pgmpy-mle-ve", seed=0,
                         problem_id="10", metric="status", value=float("nan"),
                         status="not_supported", query_role=""))
    return pd.DataFrame(rows)


def _pl_row(*, family, baseline, seed, problem_id, metric, value, n_train=None):
    return {
        "benchmark": "synthetic", "family": family, "problem_id": problem_id,
        "seed": seed, "baseline": baseline, "query_role": "", "query_kind": "",
        "metric": metric, "value": value, "status": "ok", "fit_time_s": 3.0,
        "query_time_s": None, "metrics_time_s": 0.0, "error_msg": None,
        "n_nodes": int(problem_id), "n_train": n_train,
    }


def _pl_df(sweep: bool = False) -> pd.DataFrame:
    rows = []
    n_trains = (64, 256) if sweep else (None,)
    for pid in ("10", "20"):
        for nt in n_trains:
            for seed in (0, 1):
                for b, tv in (("nbn-cat", 0.02), ("nbn-smoothed", 0.03),
                              ("pgmpy-mle", 0.02)):
                    rows.append(_pl_row(family="discrete", baseline=b, seed=seed,
                                        problem_id=pid, metric="param_recovery_tv",
                                        value=tv, n_train=nt))
                    rows.append(_pl_row(family="discrete", baseline=b, seed=seed,
                                        problem_id=pid, metric="log_likelihood",
                                        value=-10.0, n_train=nt))
                for b, ks in (("nbn-lg", 0.05), ("nbn-mdn", 0.04), ("nbn-kde", 0.06)):
                    rows.append(_pl_row(family="continuous_lg", baseline=b, seed=seed,
                                        problem_id=pid, metric="calibration_pit_ks",
                                        value=ks, n_train=nt))
                    rows.append(_pl_row(family="continuous_lg", baseline=b, seed=seed,
                                        problem_id=pid, metric="log_likelihood",
                                        value=-5.0, n_train=nt))
    return pd.DataFrame(rows)


def _run_dir(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    d = tmp_path / name
    d.mkdir()
    df.to_parquet(d / f"{name}_metrics.parquet", index=False)
    return d


class TestEndToEnd:
    def test_inference_group_renders_and_selects(self, tmp_path):
        run = _run_dir(tmp_path, "complete", _inference_df())
        out = tmp_path / "figs"
        assert run_paper({"inference": [run]}, out) == 0
        for f in ("inference_accuracy.pdf", "inference_total_query_time.pdf",
                  "inference_accuracy_time.pdf", "inference_fit_time.pdf"):
            assert (out / f).exists(), f
        sel = (out / "selection.txt").read_text()
        # discrete: no non-parametric nbn -> two best parametric engines
        assert "synthetic/discrete" in sel
        assert "selected nbn: nbn-cat-ve, nbn-cat-lw" in sel
        # continuous: best parametric (mdn) + best non-parametric (kde)
        assert "selected nbn: nbn-mdn-lw, nbn-kde-lw" in sel
        # tables: shown methods only, nbn rows daggered, not_supported dropped
        tex = (out / "tables" / "inference_continuous_lg_w1_per_node.tex").read_text()
        assert "nbn-mdn-lw$^\\dagger$" in tex and "nbn-kde-lw$^\\dagger$" in tex
        assert "nbn-lg-lw" not in tex and "pgmpy-mle-ve" not in tex
        assert "pgmpy-lg-predict" in tex
        assert "best parametric / best non-parametric" in tex

    def test_common_view(self, tmp_path):
        run = _run_dir(tmp_path, "complete", _inference_df())
        out = tmp_path / "figs"
        assert run_paper({"inference": [run]}, out, view="common") == 0
        tex = (out / "tables" / "inference_discrete_tv_per_node.tex").read_text()
        assert "$|C|$ (common seeds)" in tex

    def test_rerun_is_spliced_into_base(self, tmp_path):
        base = _inference_df()
        rerun = base[(base["family"] == "continuous_lg")
                     & (base["baseline"] == "nbn-kde-lw")].copy()
        rerun.loc[rerun["metric"] == "w1_per_node", "value"] = 0.01   # now the best
        b = _run_dir(tmp_path, "complete", base)
        r = _run_dir(tmp_path, "rerun", rerun)
        out = tmp_path / "figs"
        assert run_paper({"inference": [b, r]}, out) == 0
        tex = (out / "tables" / "inference_continuous_lg_w1_per_node.tex").read_text()
        assert "nbn-kde-lw$^\\dagger$ & \\textbf{0.01$\\pm$0}" in tex   # rerun replaced 0.07

    def test_param_learning_group(self, tmp_path):
        run = _run_dir(tmp_path, "param_learning", _pl_df())
        out = tmp_path / "figs"
        assert run_paper({"param_learning": [run]}, out) == 0
        for f in ("param_learning_log_likelihood.pdf", "param_learning_param_recovery_tv.pdf",
                  "param_learning_calibration_pit_ks.pdf", "param_learning_fit_time.pdf"):
            assert (out / f).exists(), f
        sel = (out / "selection.txt").read_text()
        assert "headline=param_recovery_tv" in sel
        assert "selected nbn: nbn-cat, nbn-smoothed" in sel
        assert "headline=calibration_pit_ks" in sel
        assert "selected nbn: nbn-mdn, nbn-kde" in sel

    def test_learning_curves_group_uses_n_train_axis(self, tmp_path):
        run = _run_dir(tmp_path, "learning_curves", _pl_df(sweep=True))
        out = tmp_path / "figs"
        assert run_paper({"learning_curves": [run]}, out) == 0
        assert (out / "learning_curves_param_recovery_tv.pdf").exists()
        assert "x=n_train" in (out / "selection.txt").read_text()
        tex = (out / "tables" / "learning_curves_discrete_param_recovery_tv.tex").read_text()
        assert "$n_{tr}=64$" in tex and "$n_{tr}=256$" in tex

    def test_exclude_removes_methods_and_families_before_selection(self, tmp_path):
        run = _run_dir(tmp_path, "complete", _inference_df())
        out = tmp_path / "figs"
        assert run_paper({"inference": [run]}, out, exclude=["nbn-kde-*", "nbn-*-lw"],
                         exclude_families=["discrete"]) == 0
        sel = (out / "selection.txt").read_text()
        assert sel.startswith("excluded baselines: nbn-kde-*, nbn-*-lw\n"
                              "excluded families: discrete\n")
        assert "synthetic/discrete" not in sel
        assert "nbn-kde-lw" not in sel and "nbn-mdn-lw" not in sel
        assert not (out / "tables" / "inference_discrete_tv_per_node.tex").exists()
        # every nbn method of the family was excluded: only the external baseline is left
        tex = (out / "tables" / "inference_continuous_lg_w1_per_node.tex").read_text()
        assert "pgmpy-lg-predict" in tex and "nbn-" not in tex

    def test_drop_excluded_is_a_glob_on_the_baseline_name(self):
        df = _inference_df()
        kept = drop_excluded(df, exclude=["nbn-*-ais"])
        assert "nbn-cat-ais" not in set(kept["baseline"])
        assert {"nbn-cat-ve", "nbn-cat-lw", "pgmpy-mle-ve"} <= set(kept["baseline"])
        assert len(drop_excluded(df)) == len(df)

    def test_row_height_sets_the_figure_height(self, tmp_path):
        pymupdf = pytest.importorskip("pymupdf")
        run = _run_dir(tmp_path, "complete", _inference_df())
        heights = {}
        for rh in (1.9, 1.2):
            out = tmp_path / f"figs_{rh}"
            assert run_paper({"inference": [run]}, out, row_height=rh) == 0
            with pymupdf.open(out / "inference_accuracy_time.pdf") as doc:
                heights[rh] = doc[0].rect.height
        # two stacked rows of panels: 2 * (1.9 - 1.2) in = 100.8 pt shorter, up to the
        # tight-bbox crop
        assert heights[1.9] - heights[1.2] == pytest.approx(2 * 0.7 * 72, abs=12)

    def test_unknown_group_is_an_error(self, tmp_path):
        assert run_paper({"bogus": [tmp_path]}, tmp_path / "o") == 1
        assert run_paper({}, tmp_path / "o") == 1
