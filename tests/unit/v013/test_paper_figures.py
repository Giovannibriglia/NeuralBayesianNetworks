"""Tests for the `nbn-bench plot` subcommand (nbn/bench/_paper_figures.py).

Covers, on tiny synthetic parquets:
1. The CLI pipeline runs end-to-end for both aggregation flags
2. The all/common bar layout under <bench>/<family>/{all,common}/{plots,tables}
3. Lookups (n_nodes resolution, n_parameters), applicability filtering
4. Learning-curve (n_train) and bnlearn (network) x axes
5. Divergence panel, bold-best, benchmark-named captions
6. The deprecated scripts/make_paper_figures.py shim still works + warns
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

_SHIM = Path(__file__).resolve().parents[3] / "scripts" / "make_paper_figures.py"


def _make_minimal_parquet(tmp_path: Path) -> Path:
    """Tiny DataFrame matching the melted v3 parquet schema (bnlearn/discrete,
    two real networks, two seeds, two baselines)."""
    rows = []
    for baseline in ["nbn-cat-ve", "pgmpy-mle-ve"]:
        for problem_id in ["asia", "alarm"]:        # both real bnlearn networks
            for seed in [0, 1]:
                for role in ["hub", "cut", "random", "terminal"]:
                    for kind in ["diagnosis", "prediction"]:
                        for metric, value in [
                            ("tv_per_node", 0.05),
                            ("jsd_per_node", 0.01),
                            ("w1_per_node", float("nan")),   # N/A for discrete
                            ("fit_time_s", 0.5),
                            ("query_time_s", 0.01),
                            ("metrics_time_s", 0.001),
                        ]:
                            rows.append({
                                "benchmark": "bnlearn",
                                "family": "discrete",
                                "problem_id": problem_id,
                                "seed": seed,
                                "baseline": baseline,
                                "query_role": role,
                                "query_kind": kind,
                                "evidence_strategy": "random",
                                "evidence_mode": "full",
                                "metric": metric,
                                "value": value,
                                "status": "not_supported" if metric == "w1_per_node" else "ok",
                                "fit_time_s": 0.5,
                                "query_time_s": 0.01,
                                "metrics_time_s": 0.001,
                                "error_msg": None,
                            })
    out = tmp_path / "test_metrics.parquet"
    pd.DataFrame(rows).to_parquet(out)
    return out


def _run(parquet: Path, out_dir: Path, aggregation: str, *extra):
    """Invoke the `nbn-bench plot` subcommand (positional parquet)."""
    return subprocess.run(
        [sys.executable, "-m", "nbn.bench.cli", "plot", str(parquet),
         "--output-dir", str(out_dir), "--aggregation", aggregation, *extra],
        capture_output=True, text=True,
    )


@pytest.mark.parametrize("aggregation", ["iqm_iqr", "mean_std"])
def test_pipeline_runs(tmp_path, aggregation):
    parquet = _make_minimal_parquet(tmp_path)
    out_dir = tmp_path / f"figures_{aggregation}"
    result = _run(parquet, out_dir, aggregation)
    assert result.returncode == 0, f"script failed: {result.stderr[-800:]}"
    assert list(out_dir.rglob("*.pdf")), f"no PDFs in {out_dir}"
    assert list(out_dir.rglob("*.tex")), f"no LaTeX tables in {out_dir}"


def test_w1_skipped_for_discrete(tmp_path):
    """family==discrete must not emit w1_per_node figures."""
    parquet = _make_minimal_parquet(tmp_path)
    out_dir = tmp_path / "figures"
    assert _run(parquet, out_dir, "iqm_iqr").returncode == 0
    assert not list(out_dir.rglob("*w1_per_node*")), "w1 figures should be skipped for discrete"


def test_n_parameters_is_not_an_axis(tmp_path):
    """The n_parameters axis was dropped with the bar layout: only the
    benchmark's x grid is rendered (network for bnlearn)."""
    parquet = _make_minimal_parquet(tmp_path)
    out_dir = tmp_path / "figures"
    result = _run(parquet, out_dir, "iqm_iqr")
    assert result.returncode == 0
    assert not list(out_dir.rglob("*_vs_n_parameters*"))
    assert list(out_dir.rglob("*_vs_network*"))


def test_n_parameters_lookup_keyed_by_problem_and_family():
    """#133 regression: the same problem_id across two families with different
    n_parameters must be retrieved independently."""
    from nbn.bench._paper_figures import n_parameters_lookup

    df = pd.DataFrame([
        {"problem_id": "10", "family": "discrete", "n_parameters": 256.0},
        {"problem_id": "10", "family": "hybrid", "n_parameters": 72.0},
        {"problem_id": "10", "family": "continuous_lg", "n_parameters": 0.0},
        {"problem_id": "5", "family": "discrete", "n_parameters": 44.0},
    ])
    lut = n_parameters_lookup(df)
    assert lut[("10", "discrete")] == 256.0
    assert lut[("10", "hybrid")] == 72.0
    assert lut[("10", "continuous_lg")] == 0.0
    assert lut[("5", "discrete")] == 44.0


def test_n_parameters_lookup_absent_returns_none():
    df = pd.DataFrame([{"problem_id": "5", "family": "discrete", "value": 1.0}])
    from nbn.bench._paper_figures import n_parameters_lookup
    assert n_parameters_lookup(df) is None


def test_run_plot_direct_call(tmp_path):
    from nbn.bench._paper_figures import run_plot

    parquet = _make_minimal_parquet(tmp_path)
    out_dir = tmp_path / "figures_direct"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="iqm_iqr") == 0
    assert list(out_dir.rglob("*.pdf"))
    assert list(out_dir.rglob("*.tex"))


def test_run_plot_accepts_directory(tmp_path):
    """run_plot resolves a directory to its *_metrics.parquet."""
    from nbn.bench._paper_figures import run_plot

    parquet = _make_minimal_parquet(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "smoke_metrics.parquet").write_bytes(parquet.read_bytes())
    out_dir = tmp_path / "figures_from_dir"
    assert run_plot(parquet=results_dir, output_dir=out_dir, aggregation="iqm_iqr") == 0
    assert list(out_dir.rglob("*.pdf"))


def test_all_common_output_layout(tmp_path):
    """The bar layout: <bench>/<family>/{all,common}/{plots,tables}/<metric>_vs_<x>
    plus success_rate.pdf (all only), selection.txt and common_seeds.txt."""
    from nbn.bench._paper_figures import run_plot

    parquet = _make_minimal_parquet(tmp_path)   # bnlearn / discrete / asia,alarm
    out_dir = tmp_path / "figs"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="iqm_iqr") == 0

    fam = out_dir / "bnlearn" / "discrete"
    for view in ("all", "common"):
        plots, tables = fam / view / "plots", fam / view / "tables"
        for stem in ("tv_per_node_vs_network", "jsd_per_node_vs_network",
                     "total_query_time_vs_network", "fit_time_vs_network"):
            assert (plots / f"{stem}.pdf").exists(), f"{view}/{stem}.pdf"
            assert (tables / f"{stem}.tex").exists(), f"{view}/{stem}.tex"
    assert (fam / "all" / "plots" / "success_rate.pdf").exists()
    assert not (fam / "common" / "plots" / "success_rate.pdf").exists()
    assert (fam / "selection.txt").exists()
    assert (fam / "common" / "common_seeds.txt").exists()
    # the pre-v0.18 artefacts are gone
    assert not (fam / "_subsets_overview.txt").exists()
    assert not list(fam.rglob("table_overall.tex"))
    assert not list(fam.rglob("table_role_*.tex"))
    assert not [d for d in fam.iterdir() if d.is_dir() and d.name.startswith("subset")]
    # table contract: float wrapper, unique label per (view, metric), network columns
    tex = (fam / "all" / "tables" / "tv_per_node_vs_network.tex").read_text()
    assert "\\begin{table}[t]" in tex
    assert "\\label{tab:bnlearn_discrete_all_tv_per_node_vs_network}" in tex
    assert "Method & asia & alarm \\\\" in tex          # sorted by n_nodes (8 < 37)
    assert "nbn-cat-ve$^\\dagger$" in tex and "pgmpy-mle-ve &" in tex
    common = (fam / "common" / "tables" / "tv_per_node_vs_network.tex").read_text()
    assert "$|C|$ (common seeds) & 2 & 2 \\\\" in common
    assert "\\label{tab:bnlearn_discrete_common_tv_per_node_vs_network}" in common


def _make_allzero_nparams_parquet(tmp_path: Path) -> Path:
    """A continuous_gauss-like family with an n_nodes column and all-zero
    n_parameters (the bnlearn gaussian case)."""
    rows = []
    for baseline in ["nbn-flow-lw", "pyro-mle-lw"]:
        for problem_id, n_nodes in [("ecoli70", 46), ("arth150", 107)]:
            for seed in [0, 1]:
                for kind in ["diagnosis", "prediction"]:
                    for metric, value in [
                        ("tv_per_node", 0.05),
                        ("w1_per_node", 0.2),
                        ("fit_time_s", 0.5),
                        ("query_time_s", 0.01),
                        ("metrics_time_s", 0.001),
                    ]:
                        rows.append({
                            "benchmark": "bnlearn",
                            "family": "continuous_gauss",
                            "problem_id": problem_id,
                            "seed": seed,
                            "baseline": baseline,
                            "query_role": "random",
                            "query_kind": kind,
                            "evidence_strategy": "random",
                            "evidence_mode": "full",
                            "metric": metric,
                            "value": value,
                            "status": "ok",
                            "fit_time_s": 0.5,
                            "query_time_s": 0.01,
                            "metrics_time_s": 0.001,
                            "error_msg": None,
                            "n_nodes": n_nodes,
                            "n_parameters": 0,
                        })
    out = tmp_path / "allzero_metrics.parquet"
    pd.DataFrame(rows).to_parquet(out)
    return out


def test_continuous_family_renders_w1_on_network_axis(tmp_path):
    from nbn.bench._paper_figures import run_plot

    parquet = _make_allzero_nparams_parquet(tmp_path)
    out_dir = tmp_path / "figs_allzero"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="iqm_iqr") == 0
    plots = out_dir / "bnlearn" / "continuous_gauss" / "all" / "plots"
    assert (plots / "tv_per_node_vs_network.pdf").exists()
    assert (plots / "w1_per_node_vs_network.pdf").exists()
    assert not list((out_dir / "bnlearn").rglob("*_vs_n_parameters*"))


def test_resolve_n_nodes_column_wins(tmp_path, caplog):
    from nbn.bench._paper_figures import resolve_n_nodes

    df = pd.DataFrame([
        {"benchmark": "bnlearn", "problem_id": "asia", "n_nodes": 99},
        {"benchmark": "bnlearn", "problem_id": "asia", "n_nodes": 99},
    ])
    assert resolve_n_nodes(df, "bnlearn") == {"asia": 99}


def test_resolve_n_nodes_falls_back_to_networks(tmp_path, caplog):
    import logging

    from nbn.bench._paper_figures import resolve_n_nodes

    df = pd.DataFrame([
        {"benchmark": "bnlearn", "problem_id": "asia"},
        {"benchmark": "bnlearn", "problem_id": "alarm"},
    ])
    with caplog.at_level(logging.INFO):
        out = resolve_n_nodes(df, "bnlearn")
    assert out == {"asia": 8, "alarm": 37}        # from _NETWORKS
    assert any("predates PR-1" in r.message for r in caplog.records)


def test_resolve_n_nodes_synthetic_numeric_fallback():
    from nbn.bench._paper_figures import resolve_n_nodes

    df = pd.DataFrame([{"benchmark": "synthetic", "problem_id": "100"}])
    assert resolve_n_nodes(df, "synthetic") == {"100": 100}


def test_resolve_n_nodes_drops_unknown(caplog):
    import logging

    from nbn.bench._paper_figures import resolve_n_nodes

    df = pd.DataFrame([{"benchmark": "bnlearn", "problem_id": "not_a_real_net"}])
    with caplog.at_level(logging.WARNING):
        out = resolve_n_nodes(df, "bnlearn")
    assert out == {}
    assert any("unresolved" in r.message for r in caplog.records)


def test_not_supported_baselines_excluded_entirely(tmp_path):
    """A baseline whose every row is not_supported is dropped from the tables;
    a partially-supported baseline survives with its solved seeds."""
    from nbn.bench._paper_figures import run_plot

    rows = []
    for baseline, mapping in [
        ("nbn-cat-ve", {"asia": "ok", "alarm": "ok"}),
        ("pgmpy-lg-predict", {"asia": "not_supported", "alarm": "not_supported"}),
        ("nbn-flow-lw", {"asia": "ok", "alarm": "not_supported"}),
        ("pgmpy-mle-ve", {"asia": "ok", "alarm": "ok"}),
    ]:
        for problem_id, status in mapping.items():
            for seed in [0, 1]:
                for kind in ["diagnosis", "prediction"]:
                    for metric in ["tv_per_node", "fit_time_s",
                                   "query_time_s", "metrics_time_s"]:
                        rows.append({
                            "benchmark": "bnlearn",
                            "family": "discrete",
                            "problem_id": problem_id,
                            "seed": seed,
                            "baseline": baseline,
                            "query_role": "random",
                            "query_kind": kind,
                            "evidence_strategy": "random",
                            "evidence_mode": "full",
                            "metric": metric,
                            "value": 0.05 if status == "ok" else None,
                            "status": status,
                            "fit_time_s": 0.5 if status == "ok" else None,
                            "query_time_s": 0.01 if status == "ok" else None,
                            "metrics_time_s": 0.001 if status == "ok" else None,
                            "error_msg": None,
                            "n_nodes": 8 if problem_id == "asia" else 37,
                            "n_parameters": 36 if problem_id == "asia" else 752,
                        })
    parquet = tmp_path / "test_metrics.parquet"
    pd.DataFrame(rows).to_parquet(parquet)
    out_dir = tmp_path / "figs"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="iqm_iqr") == 0

    tex = (out_dir / "bnlearn" / "discrete" / "all" / "tables"
           / "tv_per_node_vs_network.tex").read_text()
    assert "pgmpy-lg-predict" not in tex        # fully not_supported: gone
    assert "pgmpy-mle-ve" in tex                 # non-nbn: always shown
    assert "nbn-cat-ve" in tex and "nbn-flow-lw" in tex   # both nbn fit in top-2
    flow = [ln for ln in tex.splitlines() if ln.startswith("nbn-flow-lw")][0]
    assert flow.split("&")[2].strip().startswith("--")    # alarm: not run


def test_filter_unsupported_baselines_unit():
    from nbn.bench._paper_figures import _filter_unsupported_baselines

    df = pd.DataFrame([
        {"baseline": "A", "status": "ok"},
        {"baseline": "A", "status": "ok"},
        {"baseline": "B", "status": "not_supported"},
        {"baseline": "B", "status": "not_supported"},
        {"baseline": "C", "status": "ok"},
        {"baseline": "C", "status": "not_supported"},
        {"baseline": "D", "status": "timeout"},
    ])
    out = _filter_unsupported_baselines(df)
    assert set(out["baseline"]) == {"A", "C", "D"}
    assert (out["status"] != "not_supported").all()
    assert len(out[out["baseline"] == "C"]) == 1


def test_filter_unsupported_baselines_reports_broken_library(caplog):
    import logging

    from nbn.bench._paper_figures import _filter_unsupported_baselines
    from nbn.bench.core.runner import LIBRARY_BROKEN_PREFIX

    df = pd.DataFrame([
        {"baseline": "A", "status": "ok", "error_msg": None},
        {"baseline": "pgmpy-mle-ve", "status": "not_supported",
         "error_msg": LIBRARY_BROKEN_PREFIX + "ImportError('sklearn too old')"},
        {"baseline": "nbn-lg-lw", "status": "not_supported",
         "error_msg": "nbn-lg-lw not applicable to discrete"},
    ])
    with caplog.at_level(logging.INFO, logger="nbn.bench._paper_figures"):
        out = _filter_unsupported_baselines(df, "discrete")
    assert set(out["baseline"]) == {"A"}
    warn = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn) == 1
    assert "pgmpy-mle-ve" in warn[0].getMessage() and "sklearn too old" in warn[0].getMessage()
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("nbn-lg-lw" in m and "not applicable" in m for m in info)


def test_deprecated_shim_still_works_and_warns(tmp_path):
    parquet = _make_minimal_parquet(tmp_path)
    out_dir = tmp_path / "figures_shim"
    result = subprocess.run(
        [sys.executable, str(_SHIM), "--parquet", str(parquet),
         "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"shim failed: {result.stderr[-800:]}"
    assert "deprecat" in (result.stdout + result.stderr).lower()
    assert list(out_dir.rglob("*.pdf"))


# --- n_train learning-curve axis ---------------------------------------------

def _make_learning_curve_parquet(tmp_path: Path, n_trains=(50, 200, 800),
                                 seeds=(0, 1)) -> Path:
    """A PL-mode learning-curve parquet: one synthetic discrete problem,
    n_train swept, per-cell metric rows (no query_time_s / status sentinels).
    param_recovery_tv is ok and decreasing in n_train; log_likelihood ok;
    calibration_pit_ks not_applicable (discrete)."""
    rows = []
    tv_by_n = {n: round(0.3 / (i + 1), 4) for i, n in enumerate(sorted(n_trains))}
    for baseline in ["nbn-cat", "pgmpy-bayes"]:
        for n_train in n_trains:
            for seed in seeds:
                for metric, value, status in [
                    ("param_recovery_tv", tv_by_n[n_train] + 0.01 * seed, "ok"),
                    ("log_likelihood", -10.0 + tv_by_n[n_train], "ok"),
                    ("calibration_pit_ks", float("nan"), "not_applicable"),
                ]:
                    rows.append({
                        "benchmark": "synthetic", "family": "discrete",
                        "problem_id": "6", "seed": seed, "baseline": baseline,
                        "query_role": "", "query_kind": "prediction",
                        "evidence_strategy": "random", "evidence_mode": "full",
                        "metric": metric, "value": value, "status": status,
                        "fit_time_s": 1.5, "query_time_s": float("nan"),
                        "metrics_time_s": 0.01, "error_msg": None,
                        "n_nodes": 6, "n_train": n_train,
                    })
    out = tmp_path / "lc_metrics.parquet"
    pd.DataFrame(rows).to_parquet(out)
    return out


def test_n_train_axis_when_swept(tmp_path):
    """An n_train sweep uses n_train as the x grid (columns = n_train values,
    aggregation across seeds), calibration is skipped for discrete, and fit
    time comes from the column (PL mode has no fit_time_s rows)."""
    from nbn.bench._paper_figures import run_plot

    parquet = _make_learning_curve_parquet(tmp_path)
    out_dir = tmp_path / "figs_lc"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="mean_std") == 0
    fam = out_dir / "synthetic" / "discrete"
    for view in ("all", "common"):
        assert (fam / view / "plots" / "param_recovery_tv_vs_n_train.pdf").exists()
        assert (fam / view / "plots" / "log_likelihood_vs_n_train.pdf").exists()
        assert (fam / view / "plots" / "fit_time_vs_n_train.pdf").exists()
    assert not list(fam.rglob("calibration_*"))
    assert not list(fam.rglob("*_vs_n_nodes*"))
    tex = (fam / "all" / "tables" / "param_recovery_tv_vs_n_train.tex").read_text()
    assert "Method & $n_{tr}=50$ & $n_{tr}=200$ & $n_{tr}=800$ \\\\" in tex
    cat = [ln for ln in tex.splitlines() if ln.startswith("nbn-cat")][0]
    # seeds 0.3 / 0.31 -> mean 0.305 ± 0.005, all seeds solved -> no (k/n)
    assert "0.305$\\pm$0.005" in cat and "(" not in cat.replace("$^\\dagger$", "")
    assert "SAMPLE EFFICIENCY PARAMETER LEARNING" in tex


def test_n_train_axis_skipped_without_sweep(tmp_path):
    from nbn.bench._paper_figures import run_plot

    parquet = _make_learning_curve_parquet(tmp_path, n_trains=(200,))
    out_dir = tmp_path / "figs_lc1"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="mean_std") == 0
    fam = out_dir / "synthetic" / "discrete"
    assert not list(fam.rglob("*_vs_n_train*"))
    assert (fam / "all" / "plots" / "param_recovery_tv_vs_n_nodes.pdf").exists()


def test_status_counts_pl_mode_per_metric(tmp_path):
    """PL parquets have no per-query rows: the stacked status figure counts
    accuracy-metric rows (ok + not_applicable)."""
    from nbn.bench._paper_figures import per_query_status_counts

    df = pd.read_parquet(_make_learning_curve_parquet(tmp_path))
    counts = per_query_status_counts(df)
    assert counts.loc["nbn-cat", "ok"] == 12          # 3 n_train x 2 seeds x 2 ok metrics
    assert counts.loc["nbn-cat", "not_applicable"] == 6


def test_status_counts_inference_path(tmp_path):
    from nbn.bench._paper_figures import per_query_status_counts

    df = pd.read_parquet(_make_minimal_parquet(tmp_path))
    counts = per_query_status_counts(df)
    assert counts.loc["nbn-cat-ve", "ok"] == 32       # 2 problems x 2 seeds x 8 queries


# --- multi-parquet (#235) -----------------------------------------------------

def test_single_parquet_list_of_one_unchanged(tmp_path):
    from nbn.bench._paper_figures import run_plot

    parquet = _make_minimal_parquet(tmp_path)
    out_dir = tmp_path / "figs_list1"
    assert run_plot(parquet=[parquet], output_dir=out_dir, aggregation="iqm_iqr") == 0
    assert list(out_dir.rglob("*.pdf"))


def test_multi_parquet_concat_renders_both(tmp_path):
    from nbn.bench._paper_figures import run_plot

    inf = _make_minimal_parquet(tmp_path)                 # bnlearn, tv/jsd
    pl = _make_learning_curve_parquet(tmp_path)           # synthetic, recovery/LL
    out_dir = tmp_path / "figs_multi"
    assert run_plot(parquet=[inf, pl], output_dir=out_dir, aggregation="mean_std") == 0
    assert list((out_dir / "bnlearn").rglob("tv_per_node_vs_*.pdf"))
    assert list((out_dir / "synthetic").rglob("param_recovery_tv_vs_*.pdf"))


def test_multi_parquet_cli_nargs(tmp_path):
    inf = _make_minimal_parquet(tmp_path)
    pl = _make_learning_curve_parquet(tmp_path)
    out_dir = tmp_path / "figs_cli_multi"
    result = subprocess.run(
        [sys.executable, "-m", "nbn.bench.cli", "plot", str(inf), str(pl),
         "--output-dir", str(out_dir), "--aggregation", "mean_std", "--top-nbn", "1"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"script failed: {result.stderr[-800:]}"
    assert list((out_dir / "bnlearn").rglob("*.pdf"))
    assert list((out_dir / "synthetic").rglob("*.pdf"))


# --- divergence panel (#235) --------------------------------------------------

def test_mechanism_key_suffix_aware():
    from nbn.bench._paper_figures import _mechanism_key

    assert _mechanism_key("nbn-mdn-lw") == "nbn-mdn"
    assert _mechanism_key("nbn-mdn") == "nbn-mdn"
    assert _mechanism_key("nbn-cat") == "nbn-cat"
    assert _mechanism_key("pgmpy-mle") == "pgmpy-mle"
    assert _mechanism_key("nbn-cat-ve") == "nbn-cat"
    assert _mechanism_key("pyro-empirical-importance") == "pyro-empirical"


def _make_divergence_parquet(tmp_path: Path) -> Path:
    rows = []
    pit = {"nbn-mdn": 0.13, "nbn-kde": 0.06}
    w1 = {"nbn-mdn-lw": 0.07, "nbn-kde-lw": 0.09}
    common = dict(benchmark="synthetic", family="continuous_nongauss",
                  problem_id="5", seed=0, query_role="", query_kind="prediction",
                  evidence_strategy="random", evidence_mode="full",
                  fit_time_s=float("nan"), query_time_s=float("nan"),
                  metrics_time_s=0.01, error_msg=None, n_nodes=5, n_train=None)
    for b, v in pit.items():
        rows.append({**common, "baseline": b, "metric": "calibration_pit_ks",
                     "value": v, "status": "ok"})
    for b, v in w1.items():
        rows.append({**common, "baseline": b, "metric": "w1_per_node",
                     "value": v, "status": "ok"})
    out = tmp_path / "div_metrics.parquet"
    pd.DataFrame(rows).to_parquet(out)
    return out


def test_divergence_panel_renders_and_aligns_mechanisms(tmp_path):
    from nbn.bench._paper_figures import _mechanism_key, run_plot

    parquet = _make_divergence_parquet(tmp_path)
    out_dir = tmp_path / "figs_div"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="mean_std") == 0
    panel = (out_dir / "synthetic" / "continuous_nongauss" / "all" / "plots"
             / "divergence_calibration_pit_ks_vs_w1_per_node.pdf")
    assert panel.exists()
    df = pd.read_parquet(parquet)
    assert {_mechanism_key(b) for b in df.baseline.unique()} == {"nbn-mdn", "nbn-kde"}


def test_divergence_panel_skipped_without_both_metrics(tmp_path):
    from nbn.bench._paper_figures import run_plot

    parquet = _make_learning_curve_parquet(tmp_path)
    out_dir = tmp_path / "figs_nodiv"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="mean_std") == 0
    assert not list(out_dir.rglob("divergence_*.pdf"))


# --- partial-timeout cells (scaling walls) -----------------------------------

def _make_walls_parquet(tmp_path: Path) -> Path:
    """nbn-kde-lw times out on some queries at n=500 (query-budget wall);
    nbn-knn-lw is ok throughout; pgmpy-lg-predict ok throughout."""
    rows = []
    for baseline in ["nbn-kde-lw", "nbn-knn-lw", "pgmpy-lg-predict"]:
        for n_nodes in (10, 500):
            for seed in (0, 1):
                for q in range(4):
                    wall = baseline == "nbn-kde-lw" and n_nodes == 500 and q >= 2
                    status = "timeout" if wall else "ok"
                    for metric, value in [("w1_per_node", 0.1 + 0.01 * q),
                                          ("query_time_s", 0.5 * (n_nodes / 10)),
                                          ("fit_time_s", 1.0)]:
                        rows.append({
                            "benchmark": "synthetic", "family": "continuous_lg",
                            "problem_id": str(n_nodes), "seed": seed,
                            "baseline": baseline, "query_role": "hub",
                            "query_kind": "prediction", "evidence_strategy": "random",
                            "evidence_mode": "full", "metric": metric,
                            "value": float("nan") if wall else value, "status": status,
                            "fit_time_s": 1.0, "query_time_s": value,
                            "metrics_time_s": 0.0, "error_msg": None, "n_nodes": n_nodes,
                        })
    out = tmp_path / "walls_metrics.parquet"
    pd.DataFrame(rows).to_parquet(out)
    return out


def test_partial_timeout_seed_is_unsolved(tmp_path):
    """A seed with any timed-out query is unsolved for every metric of that
    cell: kde reads 'timeout' at n=500 in both views, and does not constrain
    the common set there."""
    from nbn.bench._paper_figures import run_plot

    parquet = _make_walls_parquet(tmp_path)
    out_dir = tmp_path / "figs_walls"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="iqm_iqr") == 0
    fam = out_dir / "synthetic" / "continuous_lg"
    for stem in ("w1_per_node_vs_n_nodes", "total_query_time_vs_n_nodes"):
        tex = (fam / "all" / "tables" / f"{stem}.tex").read_text()
        assert "Method & $n=10$ & $n=500$ \\\\" in tex
        kde = [ln for ln in tex.splitlines() if ln.startswith("nbn-kde-lw")][0]
        assert kde.split("&")[2].strip().rstrip("\\").strip() == "timeout"
        knn = [ln for ln in tex.splitlines() if ln.startswith("nbn-knn-lw")][0]
        assert "timeout" not in knn
    common = (fam / "common" / "tables" / "w1_per_node_vs_n_nodes.tex").read_text()
    assert "$|C|$ (common seeds) & 2 & 2 \\\\" in common


# --- bold-best + captions -----------------------------------------------------

def test_bold_best_directions():
    from nbn.bench._paper_figures import _bold_best

    assert _bold_best({"a": 0.05, "b": 0.02, "c": 0.09}, "tv_per_node") == {"b"}
    assert _bold_best({"a": -10.0, "b": -3.0}, "log_likelihood") == {"b"}
    assert _bold_best({"a": 0.9, "b": 1.1, "c": 1.4}, "calibration_sd_ratio") == {"a", "b"}
    assert _bold_best({"a": 5.0, "b": 1.0}, "time") == {"b"}


def test_bold_best_excludes_nan_and_inf():
    from nbn.bench._paper_figures import _bold_best

    assert _bold_best({"a": float("inf"), "b": 0.5, "c": None}, "param_recovery_kl") == {"b"}
    assert _bold_best({"a": float("inf"), "b": None}, "param_recovery_kl") == set()


def test_bold_best_ties_to_display_precision():
    from nbn.bench._paper_figures import _bold_best

    assert _bold_best({"a": 0.02961, "b": 0.02964, "c": 0.10}, "tv_per_node") == {"a", "b"}


def test_table_bolds_best_per_column(tmp_path):
    from nbn.bench._paper_figures import run_plot

    parquet = _make_learning_curve_parquet(tmp_path)
    out_dir = tmp_path / "figs_bold"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="mean_std") == 0
    body = (out_dir / "synthetic" / "discrete" / "all" / "tables"
            / "log_likelihood_vs_n_train.tex").read_text()
    # identical LL for both baselines -> both bold in every column
    assert body.count("\\textbf{") == 6


def _mini(metrics, *, benchmark="synthetic", batch_size=None, n_train=None):
    rows = []
    for m in metrics:
        row = {"benchmark": benchmark, "family": "discrete", "metric": m,
               "status": "ok"}
        if batch_size is not None:
            row["batch_size"] = batch_size
        if n_train is not None:
            row["n_train"] = n_train
        rows.append(row)
    return pd.DataFrame(rows)


def test_benchmark_caption_four_buckets():
    from nbn.bench._paper_figures import _benchmark_caption

    assert _benchmark_caption(_mini(["query_time_s"], batch_size=4)) == "INFERENCE SPEED"
    se = pd.concat([_mini(["param_recovery_tv"], n_train=30),
                    _mini(["param_recovery_tv"], n_train=300)], ignore_index=True)
    assert _benchmark_caption(se) == "SAMPLE EFFICIENCY PARAMETER LEARNING"
    assert _benchmark_caption(_mini(["tv_per_node"], benchmark="bnlearn")) == "BNLEARN INFERENCE"
    assert _benchmark_caption(
        _mini(["param_recovery_tv"], benchmark="bnlearn")) == "BNLEARN PARAMETER LEARNING"
    assert _benchmark_caption(_mini(["tv_per_node"])) == "SYNTHETIC INFERENCE"
    assert _benchmark_caption(_mini(["log_likelihood"])) == "SYNTHETIC PARAMETER LEARNING"


def test_benchmark_caption_single_n_train_is_not_sample_efficiency():
    from nbn.bench._paper_figures import _benchmark_caption

    assert _benchmark_caption(
        _mini(["param_recovery_tv"], n_train=300)) == "SYNTHETIC PARAMETER LEARNING"


# --- many networks: figure splitting -----------------------------------------

def test_many_networks_split_into_parts(tmp_path):
    """More than 8 bar groups split the figure into _partK files."""
    from nbn.bench._paper_figures import run_plot
    from nbn.bench.problems.bnlearn import _NETWORKS

    nets = sorted(_NETWORKS)[:10]
    rows = []
    for baseline in ["nbn-cat-ve", "pgmpy-mle-ve"]:
        for net in nets:
            for metric, value in [("tv_per_node", 0.05), ("query_time_s", 0.01),
                                  ("fit_time_s", 0.5)]:
                rows.append({
                    "benchmark": "bnlearn", "family": "discrete", "problem_id": net,
                    "seed": 1, "baseline": baseline, "query_role": "hub",
                    "query_kind": "prediction", "evidence_strategy": "random",
                    "evidence_mode": "full", "metric": metric, "value": value,
                    "status": "ok", "fit_time_s": 0.5, "query_time_s": 0.01,
                    "metrics_time_s": 0.0, "error_msg": None,
                })
    parquet = tmp_path / "many_metrics.parquet"
    pd.DataFrame(rows).to_parquet(parquet)
    out_dir = tmp_path / "figs_many"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="iqm_iqr") == 0
    plots = out_dir / "bnlearn" / "discrete" / "all" / "plots"
    assert (plots / "tv_per_node_vs_network_part1.pdf").exists()
    assert (plots / "tv_per_node_vs_network_part2.pdf").exists()
    assert not (plots / "tv_per_node_vs_network.pdf").exists()
    # single seed: table has a value without a band
    tex = (out_dir / "bnlearn" / "discrete" / "all" / "tables"
           / "tv_per_node_vs_network.tex").read_text()
    assert "0.05$\\pm$0" in tex
