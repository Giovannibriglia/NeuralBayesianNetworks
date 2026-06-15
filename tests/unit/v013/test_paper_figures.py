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
    """run_plot produces the flat per-family layout under <bench>/{plots,tables}/
    with <family>_* filenames (PR-2)."""
    from benchmarking._paper_figures import run_plot

    parquet = _make_minimal_parquet(tmp_path)   # bnlearn / discrete / asia,alarm
    out_dir = tmp_path / "figs"
    assert run_plot(parquet=parquet, output_dir=out_dir, aggregation="iqm_iqr") == 0

    plots = out_dir / "bnlearn" / "plots"
    tables = out_dir / "bnlearn" / "tables"
    assert (plots / "discrete_tv_per_node_vs_n_nodes.pdf").exists()
    assert (plots / "discrete_jsd_per_node_vs_n_nodes.pdf").exists()
    assert (plots / "discrete_success_rate.pdf").exists()
    assert (plots / "discrete_total_query_time_vs_n_nodes.pdf").exists()
    assert (plots / "discrete_fit_time_vs_n_nodes.pdf").exists()
    # Tables: overall + per-kind + per-role (decision beta keeps per-role).
    assert (tables / "discrete_table_overall.tex").exists()
    assert (tables / "discrete_table_kind_diagnosis.tex").exists()
    assert (tables / "discrete_table_kind_prediction.tex").exists()
    assert (tables / "discrete_table_role_hub.tex").exists()
    # Float wrapper + label present (PR #189 universal table format).
    overall = (tables / "discrete_table_overall.tex").read_text()
    assert "\\pm" in overall and "\\label{tab:bnlearn_discrete_overall}" in overall
    # No old per-(family, size) cell tree.
    assert not (out_dir / "bnlearn" / "discrete").exists()
    # discrete skips w1 entirely.
    assert not list(plots.glob("*w1_per_node*"))


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

    plots = out_dir / "bnlearn" / "plots"
    assert (plots / "continuous_gauss_tv_per_node_vs_n_nodes.pdf").exists()
    assert not list(plots.glob("*_vs_n_parameters*")), (
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

    overall_tex = (out_dir / "bnlearn" / "tables"
                   / "discrete_table_overall.tex").read_text()
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
