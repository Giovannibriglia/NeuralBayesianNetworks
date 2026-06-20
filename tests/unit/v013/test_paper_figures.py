"""Smoke test for the `nbn-bench plot` subcommand (benchmarking/_paper_figures.py).

Doesn't verify figure content (manual review). Verifies:
1. Pipeline runs without exception against a representative DataFrame
2. Both aggregation flags work
3. At least one PDF and one .tex are emitted
4. n_parameters absence is logged and the *_vs_n_parameters figures skipped
5. The deprecated scripts/make_paper_figures.py shim still works + warns

Reference: docs/v0.13-paper-figures.md
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

_SHIM = Path(__file__).resolve().parents[3] / "scripts" / "make_paper_figures.py"


def _make_minimal_parquet(tmp_path: Path) -> Path:
    """Tiny DataFrame matching the melted v3 parquet schema."""
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


def _run(parquet: Path, out_dir: Path, aggregation: str):
    """Invoke the `nbn-bench plot` subcommand (positional parquet)."""
    return subprocess.run(
        [sys.executable, "-m", "benchmarking.cli", "plot", str(parquet),
         "--output-dir", str(out_dir), "--aggregation", aggregation],
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


def test_n_parameters_absent_logged(tmp_path):
    """Missing n_parameters column -> info log + skipped *_vs_n_parameters figures."""
    parquet = _make_minimal_parquet(tmp_path)
    out_dir = tmp_path / "figures"
    result = _run(parquet, out_dir, "iqm_iqr")
    assert result.returncode == 0
    log = result.stdout + result.stderr
    assert "n_parameters" in log and "absent" in log.lower()
    assert not list(out_dir.rglob("*_vs_n_parameters*")), "n_parameters figures should be skipped"
    assert list(out_dir.rglob("*_vs_n_nodes*")), "n_nodes figures should still be produced"


def test_n_parameters_lookup_keyed_by_problem_and_family():
    """#133 regression: the same problem_id across two families with different
    n_parameters must be retrieved independently (synthetic case), not collapsed
    to whichever family sorts first."""
    from benchmarking._paper_figures import n_parameters_lookup

    df = pd.DataFrame([
        {"problem_id": "10", "family": "discrete", "n_parameters": 256.0},
        {"problem_id": "10", "family": "hybrid", "n_parameters": 72.0},
        {"problem_id": "10", "family": "continuous_lg", "n_parameters": 0.0},
        {"problem_id": "5", "family": "discrete", "n_parameters": 44.0},
    ])
    lut = n_parameters_lookup(df)
    assert lut[("10", "discrete")] == 256.0
    assert lut[("10", "hybrid")] == 72.0          # NOT collapsed to 256
    assert lut[("10", "continuous_lg")] == 0.0
    assert lut[("5", "discrete")] == 44.0
    # family-scoped flatten (mirrors process_family) yields the right x-lookup
    hybrid = {p: v for (p, f), v in lut.items() if f == "hybrid"}
    assert hybrid == {"10": 72.0}


def test_n_parameters_lookup_absent_returns_none():
    df = pd.DataFrame([{"problem_id": "5", "family": "discrete", "value": 1.0}])
    from benchmarking._paper_figures import n_parameters_lookup
    assert n_parameters_lookup(df) is None


def test_run_plot_direct_call(tmp_path):
    """The module entry point run_plot() works without the CLI/subprocess."""
    from benchmarking._paper_figures import run_plot

    parquet = _make_minimal_parquet(tmp_path)
    out_dir = tmp_path / "figures_direct"
    rc = run_plot(parquet=parquet, output_dir=out_dir, aggregation="iqm_iqr")
    assert rc == 0
    assert list(out_dir.rglob("*.pdf"))
    assert list(out_dir.rglob("*.tex"))


def test_run_plot_accepts_directory(tmp_path):
    """run_plot resolves a directory to its *_metrics.parquet."""
    from benchmarking._paper_figures import run_plot

    # Lay the parquet out as `<dir>/<name>_metrics.parquet`.
    parquet = _make_minimal_parquet(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    target = results_dir / "smoke_metrics.parquet"
    target.write_bytes(parquet.read_bytes())

    out_dir = tmp_path / "figures_from_dir"
    rc = run_plot(parquet=results_dir, output_dir=out_dir, aggregation="iqm_iqr")
    assert rc == 0
    assert list(out_dir.rglob("*.pdf"))


def test_per_family_output_layout(tmp_path):
    """run_plot produces the nested per-family/subset layout under
    <bench>/<family>/<subset>/{plots,tables}/ with no <family>_ prefix (PR-4)."""
    from benchmarking._paper_figures import run_plot

    parquet = _make_minimal_parquet(tmp_path)   # bnlearn / discrete / asia,alarm
    out_dir = tmp_path / "figs"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="iqm_iqr") == 0

    all_dir = out_dir / "bnlearn" / "discrete" / "all"
    plots = all_dir / "plots"
    tables = all_dir / "tables"
    assert (plots / "tv_per_node_vs_n_nodes.pdf").exists()
    assert (plots / "jsd_per_node_vs_n_nodes.pdf").exists()
    assert (plots / "success_rate.pdf").exists()
    assert (plots / "total_query_time_vs_n_nodes.pdf").exists()
    assert (plots / "fit_time_vs_n_nodes.pdf").exists()
    # Tables: overall + per-kind + per-role (decision beta keeps per-role).
    assert (tables / "table_overall.tex").exists()
    assert (tables / "table_kind_diagnosis.tex").exists()
    assert (tables / "table_kind_prediction.tex").exists()
    assert (tables / "table_role_hub.tex").exists()
    # Float wrapper + label present (PR #189 universal table format).
    overall = (tables / "table_overall.tex").read_text()
    assert "\\pm" in overall and "\\label{tab:bnlearn_discrete_all_overall}" in overall
    # No old flat <family>_ files at the bench level.
    assert not (out_dir / "bnlearn" / "plots").exists()
    assert not list(plots.glob("*discrete_*"))
    # discrete skips w1 entirely.
    assert not list(plots.glob("*w1_per_node*"))
    # Subsets auto-discovered: overview + at least one subset dir.
    assert (out_dir / "bnlearn" / "discrete" / "_subsets_overview.txt").exists()


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
                            "n_parameters": 0,      # gaussian: all-zero (decision alpha)
                        })
    out = tmp_path / "allzero_metrics.parquet"
    pd.DataFrame(rows).to_parquet(out)
    return out


def test_all_zero_n_parameters_skips_that_axis(tmp_path):
    """A family whose n_parameters are all zero (continuous_gauss) still gets
    the n_nodes axis, but the degenerate *_vs_n_parameters file is skipped
    (decision alpha)."""
    from benchmarking._paper_figures import run_plot

    parquet = _make_allzero_nparams_parquet(tmp_path)
    out_dir = tmp_path / "figs_allzero"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="iqm_iqr") == 0

    plots = out_dir / "bnlearn" / "continuous_gauss" / "all" / "plots"
    assert (plots / "tv_per_node_vs_n_nodes.pdf").exists()
    # skipped everywhere in the family tree, not just the all view
    assert not list((out_dir / "bnlearn" / "continuous_gauss").rglob("*_vs_n_parameters*")), (
        "all-zero n_parameters family must skip the degenerate n_parameters axis"
    )


def test_resolve_n_nodes_column_wins(tmp_path, caplog):
    """resolve_n_nodes prefers the parquet column over the fallback."""
    from benchmarking._paper_figures import resolve_n_nodes

    df = pd.DataFrame([
        # asia is 8 in _NETWORKS, but the column says 99 → column wins.
        {"benchmark": "bnlearn", "problem_id": "asia", "n_nodes": 99},
        {"benchmark": "bnlearn", "problem_id": "asia", "n_nodes": 99},
    ])
    assert resolve_n_nodes(df, "bnlearn") == {"asia": 99}


def test_resolve_n_nodes_falls_back_to_networks(tmp_path, caplog):
    """No n_nodes column → _NETWORKS fallback for bnlearn, with an info log."""
    import logging

    from benchmarking._paper_figures import resolve_n_nodes

    df = pd.DataFrame([
        {"benchmark": "bnlearn", "problem_id": "asia"},
        {"benchmark": "bnlearn", "problem_id": "alarm"},
    ])
    with caplog.at_level(logging.INFO):
        out = resolve_n_nodes(df, "bnlearn")
    assert out == {"asia": 8, "alarm": 37}        # from _NETWORKS
    assert any("predates PR-1" in r.message for r in caplog.records)


def test_resolve_n_nodes_synthetic_numeric_fallback():
    """Synthetic problem_id is n_nodes as a string."""
    from benchmarking._paper_figures import resolve_n_nodes

    df = pd.DataFrame([{"benchmark": "synthetic", "problem_id": "100"}])
    assert resolve_n_nodes(df, "synthetic") == {"100": 100}


def test_resolve_n_nodes_drops_unknown(caplog):
    """An unknown bnlearn network (no column, not in _NETWORKS) is omitted and
    a warning is emitted."""
    import logging

    from benchmarking._paper_figures import resolve_n_nodes

    df = pd.DataFrame([{"benchmark": "bnlearn", "problem_id": "not_a_real_net"}])
    with caplog.at_level(logging.WARNING):
        out = resolve_n_nodes(df, "bnlearn")
    assert out == {}
    assert any("unresolved" in r.message for r in caplog.records)


def test_not_supported_baselines_excluded_entirely(tmp_path):
    """A baseline whose every row is not_supported is dropped from the
    per-family success-rate figure AND the per-family tables.
    Partially-supported baselines survive."""
    from benchmarking._paper_figures import run_plot

    rows = []
    # baseline A: ok on every problem (fully supported)
    # baseline B: not_supported on every problem (must vanish)
    # baseline C: ok on asia, not_supported on alarm (partial — keeps)
    for baseline, mapping in [
        ("nbn-cat-ve", {"asia": "ok", "alarm": "ok"}),
        ("pgmpy-lg-predict", {"asia": "not_supported", "alarm": "not_supported"}),
        ("nbn-flow-lw", {"asia": "ok", "alarm": "not_supported"}),
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
    assert run_plot(parquet=parquet, output_dir=out_dir,
                    aggregation="iqm_iqr") == 0

    overall_tex = (out_dir / "bnlearn" / "discrete" / "all" / "tables"
                   / "table_overall.tex").read_text()
    # Fully not_supported baseline: gone everywhere.
    assert "pgmpy-lg-predict" not in overall_tex
    # Fully supported: stays.
    assert "nbn-cat-ve" in overall_tex
    # Partially supported: stays (success rate computed over supported subset).
    assert "nbn-flow-lw" in overall_tex


def test_filter_unsupported_baselines_unit():
    """Direct unit test for the helper."""
    from benchmarking._paper_figures import _filter_unsupported_baselines

    df = pd.DataFrame([
        {"baseline": "A", "status": "ok"},
        {"baseline": "A", "status": "timeout"},
        {"baseline": "B", "status": "not_supported"},
        {"baseline": "B", "status": "not_supported"},
        {"baseline": "C", "status": "ok"},
        {"baseline": "C", "status": "not_supported"},
    ])
    out = _filter_unsupported_baselines(df)
    # B is dropped entirely; C survives but its not_supported row drops.
    assert set(out["baseline"]) == {"A", "C"}
    assert (out["status"] != "not_supported").all()
    assert len(out) == 3   # A:2 ok+timeout, C:1 ok


def _coverage_df(spec: dict, family: str = "discrete") -> pd.DataFrame:
    """Build a melted-schema DataFrame from spec {(problem, baseline): status}.

    'ok' rows carry real values; any other status carries NaN. n_nodes is
    looked up from _NETWORKS by problem name when possible.
    """
    from benchmarking.problems.bnlearn import _NETWORKS
    metrics = [("tv_per_node", 0.05), ("jsd_per_node", 0.01),
               ("fit_time_s", 0.5), ("query_time_s", 0.01),
               ("metrics_time_s", 0.001)]
    rows = []
    for (pid, bl), status in spec.items():
        nn = _NETWORKS.get(pid, {}).get("n_nodes", 10)
        ok = status == "ok"
        for seed in [0, 1]:
            for kind in ["diagnosis", "prediction"]:
                for m, v in metrics:
                    rows.append({
                        "benchmark": "bnlearn", "family": family,
                        "problem_id": pid, "seed": seed, "baseline": bl,
                        "query_role": "random", "query_kind": kind,
                        "evidence_strategy": "random", "evidence_mode": "full",
                        "metric": m, "value": (v if ok else float("nan")),
                        "status": status,
                        "fit_time_s": 0.5, "query_time_s": 0.01,
                        "metrics_time_s": 0.001, "error_msg": None,
                        "n_nodes": nn, "n_parameters": 100,
                    })
    return pd.DataFrame(rows)


def test_discover_subsets_basic():
    """3 baselines A/B/C, 4 problems with distinct solving sets."""
    from benchmarking._paper_figures import _discover_subsets
    spec = {}
    for bl, st in [("A", "ok"), ("B", "ok"), ("C", "ok")]:
        spec[("asia", bl)] = st                         # P1: common {A,B,C}
    for bl, st in [("A", "ok"), ("B", "ok"), ("C", "error")]:
        spec[("alarm", bl)] = st                        # P2: {A,B}
        spec[("child", bl)] = st                        # P3: {A,B}
    for bl, st in [("A", "ok"), ("B", "error"), ("C", "error")]:
        spec[("sachs", bl)] = st                        # P4: {A}
    subs = _discover_subsets(_coverage_df(spec))
    names = [s["name"] for s in subs]
    assert names == ["common", "subset1", "subset2"]
    by = {s["name"]: s for s in subs}
    assert by["common"]["baselines"] == ["A", "B", "C"]
    assert by["common"]["problems"] == ["asia"]
    assert by["subset1"]["baselines"] == ["A", "B"]
    assert by["subset1"]["problems"] == ["alarm", "child"]
    assert by["subset2"]["baselines"] == ["A"]
    assert by["subset2"]["problems"] == ["sachs"]


def test_subset_filenames_and_metadata(tmp_path):
    from benchmarking._paper_figures import run_plot
    spec = {}
    for bl, st in [("nbn-cat-ve", "ok"), ("pgmpy-mle-ve", "ok")]:
        spec[("asia", bl)] = st                         # common {both}
    spec[("alarm", "nbn-cat-ve")] = "ok"
    spec[("alarm", "pgmpy-mle-ve")] = "error"           # subset1 {nbn-cat-ve}
    parquet = tmp_path / "cov_metrics.parquet"
    _coverage_df(spec).to_parquet(parquet)
    out = tmp_path / "figs"
    assert run_plot(parquet=parquet, output_dir=out, aggregation="iqm_iqr") == 0

    fam = out / "bnlearn" / "discrete"
    assert (fam / "all" / "plots" / "tv_per_node_vs_n_nodes.pdf").exists()
    assert (fam / "all" / "tables" / "table_overall.tex").exists()
    assert (fam / "all" / "plots" / "success_rate.pdf").exists()
    # common: both baselines on asia; no success_rate in subset views.
    assert (fam / "common" / "methods.txt").read_text().split() == [
        "nbn-cat-ve", "pgmpy-mle-ve"]
    assert (fam / "common" / "problems.txt").read_text().split() == ["asia"]
    assert not (fam / "common" / "plots" / "success_rate.pdf").exists()
    assert (fam / "subset1" / "methods.txt").read_text().split() == ["nbn-cat-ve"]
    assert (fam / "subset1" / "problems.txt").read_text().split() == ["alarm"]
    assert (fam / "_subsets_overview.txt").exists()
    overview = (fam / "_subsets_overview.txt").read_text()
    assert "common" in overview and "subset1" in overview


def test_subset_table_labels_unique(tmp_path):
    from benchmarking._paper_figures import run_plot
    import re
    spec = {}
    for bl, st in [("nbn-cat-ve", "ok"), ("pgmpy-mle-ve", "ok")]:
        spec[("asia", bl)] = st
    spec[("alarm", "nbn-cat-ve")] = "ok"
    spec[("alarm", "pgmpy-mle-ve")] = "error"
    parquet = tmp_path / "cov_metrics.parquet"
    _coverage_df(spec).to_parquet(parquet)
    out = tmp_path / "figs"
    assert run_plot(parquet=parquet, output_dir=out, aggregation="iqm_iqr") == 0
    labels = []
    for tex in out.rglob("*.tex"):
        labels += re.findall(r"\\label\{(tab:[^}]+)\}", tex.read_text())
    assert labels, "no labels found"
    assert len(labels) == len(set(labels)), \
        f"duplicate labels: {sorted({lbl for lbl in labels if labels.count(lbl) > 1})}"


def test_problem_with_no_solver_excluded(tmp_path):
    from benchmarking._paper_figures import run_plot
    spec = {}
    for bl, st in [("nbn-cat-ve", "ok"), ("pgmpy-mle-ve", "ok")]:
        spec[("asia", bl)] = st                          # normal common problem
    # P_hard: every baseline fails -> empty solving set -> excluded from subsets
    spec[("alarm", "nbn-cat-ve")] = "error"
    spec[("alarm", "pgmpy-mle-ve")] = "error"
    parquet = tmp_path / "cov_metrics.parquet"
    _coverage_df(spec).to_parquet(parquet)
    out = tmp_path / "figs"
    assert run_plot(parquet=parquet, output_dir=out, aggregation="iqm_iqr") == 0
    fam = out / "bnlearn" / "discrete"
    # 'alarm' must not appear in ANY subset's problems.txt
    for probs in fam.rglob("problems.txt"):
        assert "alarm" not in probs.read_text().split(), \
            f"unsolved problem leaked into {probs}"


@pytest.mark.slow
def test_real_bnlearn_parquet_no_crash(tmp_path):
    """The on-disk bnlearn_complete parquet (no n_nodes column, diverse
    coverage) renders without crashing and yields >=1 subset per family."""
    from benchmarking._paper_figures import run_plot
    rundir = Path(__file__).resolve().parents[3] / (
        "benchmarking/results/benchmark_bnlearn_bnlearn_complete_20260612_182220")
    if not rundir.exists():
        pytest.skip("bnlearn_complete parquet not present")
    out = tmp_path / "figs"
    assert run_plot(parquet=rundir, output_dir=out, aggregation="iqm_iqr") == 0
    for fam in ["discrete", "clg", "continuous_gauss"]:
        fam_dir = out / "bnlearn" / fam
        assert (fam_dir / "all").is_dir()
        assert (fam_dir / "_subsets_overview.txt").exists()
        subset_dirs = [d for d in fam_dir.iterdir()
                       if d.is_dir() and d.name not in {"all"}]
        assert subset_dirs, f"{fam}: no subset dirs"


def test_deprecated_shim_still_works_and_warns(tmp_path):
    """scripts/make_paper_figures.py delegates to run_plot and emits a
    DeprecationWarning (backward compat for existing callers)."""
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


# --- n_train learning-curve axis (PR 13) --------------------------------------

def _make_learning_curve_parquet(tmp_path: Path, n_trains=(50, 200, 800)) -> Path:
    """A PL-mode learning-curve parquet: one synthetic discrete problem,
    n_train swept, per-cell metric rows (no query_time_s / status sentinels).
    param_recovery_tv is ok and decreasing in n_train; log_likelihood ok;
    calibration_pit_ks not_applicable (discrete) -> must skip its n_train fig."""
    rows = []
    # monotonically-decreasing recovery TV per n_train (a real learning curve).
    tv_by_n = {n: round(0.3 / (i + 1), 4) for i, n in enumerate(sorted(n_trains))}
    for baseline in ["nbn-cat", "pgmpy-bayes"]:
        for n_train in n_trains:
            for metric, value, status in [
                ("param_recovery_tv", tv_by_n[n_train], "ok"),
                ("log_likelihood", -10.0 + tv_by_n[n_train], "ok"),
                ("calibration_pit_ks", float("nan"), "not_applicable"),
            ]:
                rows.append({
                    "benchmark": "synthetic", "family": "discrete",
                    "problem_id": "6", "seed": 0, "baseline": baseline,
                    "query_role": "", "query_kind": "prediction",
                    "evidence_strategy": "random", "evidence_mode": "full",
                    "metric": metric, "value": value, "status": status,
                    "fit_time_s": float("nan"), "query_time_s": float("nan"),
                    "metrics_time_s": 0.01, "error_msg": None,
                    "n_nodes": 6, "n_train": n_train,
                })
    out = tmp_path / "lc_metrics.parquet"
    pd.DataFrame(rows).to_parquet(out)
    return out


def test_n_train_curve_uses_within_problem_grouping(tmp_path):
    """fig_accuracy_vs_n_train groups by n_train (within-problem), so a single
    problem_id with N swept n_train values yields N distinct x-points per
    baseline -- not one collapsed point (the n_nodes-axis behaviour)."""
    from benchmarking import _paper_figures as pf

    df = pd.read_parquet(_make_learning_curve_parquet(tmp_path, n_trains=(50, 200, 800)))
    captured = {}
    orig = pf._scaling_plot
    pf._scaling_plot = lambda points, *a, **k: captured.update(points)
    try:
        pf.fig_accuracy_vs_n_train(df, "param_recovery_tv", "mean_std",
                                   tmp_path / "tv_vs_n_train.pdf", "t")
    finally:
        pf._scaling_plot = orig

    xs = sorted(p[0] for p in captured["nbn-cat"])
    assert xs == [50.0, 200.0, 800.0]          # 3 distinct x-points, from n_train
    ys = [p[1] for p in sorted(captured["nbn-cat"])]
    assert all(ys[i] > ys[i + 1] for i in range(len(ys) - 1))   # decreasing curve


def test_n_train_figure_rendered_and_calibration_skipped(tmp_path):
    """run_plot emits <metric>_vs_n_train.pdf for metrics with ok rows and
    skips those without (no degenerate single-point curves)."""
    from benchmarking._paper_figures import run_plot

    parquet = _make_learning_curve_parquet(tmp_path)
    out_dir = tmp_path / "figs"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="mean_std") == 0
    plots = out_dir / "synthetic" / "discrete" / "all" / "plots"
    assert (plots / "param_recovery_tv_vs_n_train.pdf").exists()
    assert (plots / "log_likelihood_vs_n_train.pdf").exists()
    # calibration is not_applicable on discrete -> no ok rows -> no n_train fig.
    assert not (plots / "calibration_pit_ks_vs_n_train.pdf").exists()


def test_n_train_axis_skipped_without_sweep(tmp_path):
    """A single n_train value is not a sweep: no n_train figure is emitted."""
    from benchmarking._paper_figures import run_plot

    parquet = _make_learning_curve_parquet(tmp_path, n_trains=(200,))
    out_dir = tmp_path / "figs_nosweep"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="mean_std") == 0
    assert not list(out_dir.rglob("*_vs_n_train.pdf"))


# --- PL-mode status counting (#236) -------------------------------------------

def test_status_counts_pl_mode_per_metric(tmp_path):
    """On a PL parquet (no query_time_s / status rows), per_query_status_counts
    counts accuracy-metric rows by status. Each baseline here has, per n_train,
    1 recovery-ok + 1 LL-ok + 1 calibration-not_applicable -> 2/3 ok, 1/3 NA."""
    from benchmarking import _paper_figures as pf

    df = pd.read_parquet(_make_learning_curve_parquet(tmp_path, n_trains=(50, 200, 800)))
    counts = pf.per_query_status_counts(df)
    assert set(counts.index) == {"nbn-cat", "pgmpy-bayes"}
    for b in counts.index:
        assert counts.loc[b, "ok"] == 6              # 3 recovery + 3 LL
        assert counts.loc[b, "not_applicable"] == 3  # 3 calibration
        # not_supported never appears in PL bars (no such rows here).
        assert counts.loc[b, "not_supported"] == 0


def test_status_counts_inference_path_byte_identical(tmp_path):
    """An inference parquet (query_time_s rows present) routes through the
    original unit count — the PL fallback does NOT fire, so the w1
    not_supported accuracy-metric rows are not counted."""
    from benchmarking import _paper_figures as pf

    df = pd.read_parquet(_make_minimal_parquet(tmp_path))
    counts = pf.per_query_status_counts(df)
    # 2 problems x 2 seeds x 4 roles x 2 kinds = 32 ok query_time_s rows/baseline.
    for b in counts.index:
        assert counts.loc[b, "ok"] == 32
        assert counts.loc[b, "not_supported"] == 0   # w1 rows NOT counted (fallback inert)
        assert counts.loc[b, "not_applicable"] == 0


# --- multi-parquet input (#235) -----------------------------------------------

def test_single_parquet_list_of_one_unchanged(tmp_path):
    """run_plot with a one-element list behaves like the single-path call."""
    from benchmarking._paper_figures import run_plot

    parquet = _make_minimal_parquet(tmp_path)
    out_dir = tmp_path / "figs_list1"
    assert run_plot(parquet=[parquet], output_dir=out_dir, aggregation="iqm_iqr") == 0
    assert list(out_dir.rglob("*.pdf"))


def test_multi_parquet_concat_renders_both(tmp_path):
    """Two parquets (a bnlearn inference parquet + a synthetic PL parquet) are
    row-concatenated; run_plot processes both benchmarks and renders the
    metrics each contributed."""
    from benchmarking._paper_figures import run_plot

    inf = _make_minimal_parquet(tmp_path)                 # bnlearn, tv/jsd
    pl = _make_learning_curve_parquet(tmp_path)           # synthetic, recovery/LL
    out_dir = tmp_path / "figs_multi"
    assert run_plot(parquet=[inf, pl], output_dir=out_dir, aggregation="mean_std") == 0
    # inference metric from the bnlearn parquet
    assert list((out_dir / "bnlearn").rglob("tv_per_node_vs_*.pdf"))
    # PL metric from the synthetic parquet
    assert list((out_dir / "synthetic").rglob("param_recovery_tv_vs_*.pdf"))


def test_multi_parquet_cli_nargs(tmp_path):
    """The `nbn-bench plot` positional accepts multiple parquets (nargs=+)."""
    inf = _make_minimal_parquet(tmp_path)
    pl = _make_learning_curve_parquet(tmp_path)
    out_dir = tmp_path / "figs_cli_multi"
    result = subprocess.run(
        [sys.executable, "-m", "benchmarking.cli", "plot", str(inf), str(pl),
         "--output-dir", str(out_dir), "--aggregation", "mean_std"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"script failed: {result.stderr[-800:]}"
    assert list((out_dir / "bnlearn").rglob("*.pdf"))
    assert list((out_dir / "synthetic").rglob("*.pdf"))


# --- divergence panel (#235) --------------------------------------------------

def test_mechanism_key_suffix_aware():
    """_mechanism_key strips a trailing engine suffix to align PL and inference
    baselines for the same mechanism, and preserves everything else."""
    from benchmarking._paper_figures import _mechanism_key

    assert _mechanism_key("nbn-mdn-lw") == "nbn-mdn"      # inference -> mechanism
    assert _mechanism_key("nbn-mdn") == "nbn-mdn"         # PL form preserved
    assert _mechanism_key("nbn-cat") == "nbn-cat"         # no engine suffix
    assert _mechanism_key("pgmpy-mle") == "pgmpy-mle"     # param-method, not engine
    assert _mechanism_key("nbn-cat-ve") == "nbn-cat"      # ve is an engine
    assert _mechanism_key("pyro-empirical-importance") == "pyro-empirical"


def _make_divergence_parquet(tmp_path: Path) -> Path:
    """A nongauss union parquet: calibration_pit_ks rows (PL-form baselines,
    no engine suffix) + w1_per_node rows (inference-form baselines, -lw suffix)
    for the SAME two mechanisms, so the panel must align them by mechanism key."""
    rows = []
    pit = {"nbn-mdn": 0.13, "nbn-kde": 0.06}      # PIT-KS: kde best, mdn worst
    w1 = {"nbn-mdn-lw": 0.07, "nbn-kde-lw": 0.09}  # w1: mdn best (the disagreement)
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
    """run_plot emits the divergence panel on a family with both metrics, and
    the panel aligns nbn-mdn (PL) with nbn-mdn-lw (inference) as one mechanism."""
    from benchmarking._paper_figures import run_plot, fig_divergence, _mechanism_key

    parquet = _make_divergence_parquet(tmp_path)
    out_dir = tmp_path / "figs_div"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="mean_std") == 0
    panel = (out_dir / "synthetic" / "continuous_nongauss" / "all" / "plots"
             / "divergence_calibration_pit_ks_vs_w1_per_node.pdf")
    assert panel.exists()
    # The suffix-mismatched baselines collapse to the same two mechanism keys.
    df = pd.read_parquet(parquet)
    keys = {_mechanism_key(b) for b in df.baseline.unique()}
    assert keys == {"nbn-mdn", "nbn-kde"}


def test_divergence_panel_skipped_without_both_metrics(tmp_path):
    """A family with only one of the pair's metrics renders no divergence panel."""
    from benchmarking._paper_figures import run_plot

    # _make_learning_curve_parquet has calibration (not_applicable) + recovery,
    # but no w1_per_node ok rows -> the (pit_ks, w1) pair is incomplete.
    parquet = _make_learning_curve_parquet(tmp_path)
    out_dir = tmp_path / "figs_nodiv"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="mean_std") == 0
    assert not list(out_dir.rglob("divergence_*.pdf"))


# --- scaling-wall partial-timeout rendering (#233) ----------------------------

def _make_walls_parquet(tmp_path: Path) -> Path:
    """A scalability-shaped parquet: nbn-kde-lw has partial timeouts at the
    upper n_nodes (the query-budget wall — some queries finish, some time out);
    nbn-knn-lw is ok throughout (the asymmetric wall). Per-query rows replicate
    the attempt's status across query_time_s / fit_time_s / w1_per_node."""
    rows = []
    common = dict(benchmark="synthetic", family="continuous_lg",
                  query_kind="prediction", evidence_strategy="random",
                  evidence_mode="full", error_msg=None, n_train=None)
    # (baseline, n_nodes) -> list of per-query statuses
    grid = {
        ("nbn-kde-lw", 100): ["ok", "ok"],
        ("nbn-kde-lw", 200): ["ok", "timeout"],          # partial wall
        ("nbn-knn-lw", 100): ["ok", "ok"],
        ("nbn-knn-lw", 200): ["ok", "ok"],               # no wall
    }
    for (baseline, n_nodes), statuses in grid.items():
        for qi, st in enumerate(statuses):
            for metric in ("query_time_s", "fit_time_s", "w1_per_node"):
                val = 0.5 if st == "ok" else float("nan")
                rows.append({**common, "problem_id": str(n_nodes), "n_nodes": n_nodes,
                             "seed": 0, "baseline": baseline, "query_role": "random",
                             "metric": metric, "value": val, "status": st,
                             "fit_time_s": 0.5, "query_time_s": val,
                             "metrics_time_s": 0.01})
    out = tmp_path / "walls_metrics.parquet"
    pd.DataFrame(rows).to_parquet(out)
    return out


def test_dnf_cells_detects_partial_walls(tmp_path):
    """_dnf_cells flags (baseline, x) cells with BOTH ok and timeout rows, and
    skips baselines that are ok throughout (the asymmetric-wall signal)."""
    from benchmarking._paper_figures import _dnf_cells

    df = pd.read_parquet(_make_walls_parquet(tmp_path))
    dnf = _dnf_cells(df, "query_time_s", {"100": 100, "200": 200})
    assert set(dnf) == {"nbn-kde-lw"}                 # knn ok throughout -> absent
    assert set(dnf["nbn-kde-lw"]) == {200}            # wall only at n_nodes=200
    assert dnf["nbn-kde-lw"][200] == {"timeout": 1}


def test_scaling_wall_sidecar_and_query_only(tmp_path):
    """run_plot writes a *_dnf.txt sidecar for the query-time + accuracy scaling
    figures (kde's wall), but NOT the fit-time figure (a query timeout's status
    is replicated onto fit_time_s rows; marking the fit plot would mislabel a
    query-budget wall as a fit wall)."""
    from benchmarking._paper_figures import run_plot

    parquet = _make_walls_parquet(tmp_path)
    out_dir = tmp_path / "figs_walls"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="mean_std") == 0
    plots = out_dir / "synthetic" / "continuous_lg" / "all" / "plots"
    assert (plots / "total_query_time_vs_n_nodes_dnf.txt").exists()
    assert (plots / "w1_per_node_vs_n_nodes_dnf.txt").exists()
    assert not (plots / "fit_time_vs_n_nodes_dnf.txt").exists()   # query-wall, not fit
    body = (plots / "total_query_time_vs_n_nodes_dnf.txt").read_text()
    assert "nbn-kde-lw" in body and "x=200" in body and "nbn-knn-lw" not in body


def test_all_ok_parquet_writes_no_dnf_sidecar(tmp_path):
    """An all-ok parquet (no partial timeouts) writes no DNF sidecar — the
    marker path is dormant and rendering is unchanged."""
    from benchmarking._paper_figures import run_plot

    parquet = _make_minimal_parquet(tmp_path)
    out_dir = tmp_path / "figs_allok"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="iqm_iqr") == 0
    assert not list(out_dir.rglob("*_dnf.txt"))


# --- bold-best per metric column (#241) ---------------------------------------

def test_bold_best_directions():
    """_bold_best picks the winner per metric direction, excluding NaN/+inf."""
    from benchmarking._paper_figures import _bold_best

    # lower-better distance: smallest wins.
    assert _bold_best({"a": 0.05, "b": 0.02, "c": 0.09}, "tv_per_node") == {"b"}
    # higher-better: largest wins.
    assert _bold_best({"a": -10.0, "b": -3.0}, "log_likelihood") == {"b"}
    # closer-to-1: |c-1| minimized (0.9 and 1.1 are equidistant -> tie).
    assert _bold_best({"a": 0.9, "b": 1.1, "c": 1.4}, "calibration_sd_ratio") == {"a", "b"}
    # time column (lower-better).
    assert _bold_best({"a": 5.0, "b": 1.0}, "time") == {"b"}


def test_bold_best_excludes_nan_and_inf():
    """+inf (unsmoothed KL) and None never win; an all-excluded column bolds none."""
    from benchmarking._paper_figures import _bold_best

    assert _bold_best({"a": float("inf"), "b": 0.5, "c": None}, "param_recovery_kl") == {"b"}
    assert _bold_best({"a": float("inf"), "b": None}, "param_recovery_kl") == set()


def test_bold_best_ties_to_display_precision():
    """Two centrals that round to the same .3g string bold together."""
    from benchmarking._paper_figures import _bold_best

    # 0.02961 and 0.02964 both render as "0.0296" at .3g -> tied; 0.10 does not.
    assert _bold_best({"a": 0.02961, "b": 0.02964, "c": 0.10}, "tv_per_node") == {"a", "b"}


def test_headline_table_bolds_best_cell(tmp_path):
    """A rendered headline table has \\textbf on the best baseline per metric."""
    from benchmarking._paper_figures import run_plot

    parquet = _make_learning_curve_parquet(tmp_path)   # discrete PL: recovery + LL
    out_dir = tmp_path / "figs_bold"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="mean_std") == 0
    body = (out_dir / "synthetic" / "discrete" / "all" / "tables"
            / "table_overall.tex").read_text()
    assert "\\textbf{" in body
