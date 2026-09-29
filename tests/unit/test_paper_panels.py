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
    Ranking,
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


class TestSelection:
    def test_rank_more_x_solved_wins_then_lower_score(self):
        view = pd.DataFrame([
            # method, x, center, k
            dict(method="nbn-cat-ve", x=10, center=0.05, k=5),
            dict(method="nbn-cat-ve", x=50, center=0.05, k=5),
            dict(method="nbn-cat-lw", x=10, center=0.01, k=5),   # better but 1 x only
            dict(method="nbn-cat-lw", x=50, center=float("nan"), k=0),
            dict(method="nbn-cat-ais", x=10, center=0.06, k=5),
            dict(method="nbn-cat-ais", x=50, center=0.06, k=5),
            dict(method="pgmpy-mle-ve", x=10, center=0.0, k=5),
        ])
        view["lo"] = view["hi"] = view["center"]
        view["n"] = 5
        view["code"] = None
        ranked = rank_nbn(view, "tv_per_node")
        assert [r.method for r in ranked] == ["nbn-cat-ve", "nbn-cat-ais", "nbn-cat-lw"]
        assert ranked[0].n_x_solved == 2 and ranked[2].n_x_solved == 1

    def test_select_best_parametric_and_best_nonparametric(self):
        ranking = [
            Ranking("nbn-lg-lw", "parametric", 5, 0.2),
            Ranking("nbn-mdn-lw", "parametric", 5, 0.3),
            Ranking("nbn-knn-lw", "nonparametric", 5, 0.4),
            Ranking("nbn-kde-lw", "nonparametric", 5, 0.5),
        ]
        assert select_nbn(ranking) == ["nbn-lg-lw", "nbn-knn-lw"]

    def test_select_fills_missing_category_with_runner_up(self):
        ranking = [
            Ranking("nbn-cat-ve", "parametric", 5, 0.1),
            Ranking("nbn-cat-lw", "parametric", 5, 0.2),
            Ranking("nbn-cat-avi", "parametric", 5, 0.3),
        ]
        assert select_nbn(ranking) == ["nbn-cat-ve", "nbn-cat-lw"]

    def test_select_skips_methods_that_solved_nothing(self):
        ranking = [
            Ranking("nbn-lg-lw", "parametric", 5, 0.2),
            Ranking("nbn-kde-lw", "nonparametric", 0, float("inf")),
        ]
        assert select_nbn(ranking) == ["nbn-lg-lw"]


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
                  "inference_fit_time.pdf"):
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

    def test_unknown_group_is_an_error(self, tmp_path):
        assert run_paper({"bogus": [tmp_path]}, tmp_path / "o") == 1
        assert run_paper({}, tmp_path / "o") == 1
