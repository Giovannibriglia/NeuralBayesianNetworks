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
    # family-scoped flatten (mirrors process_cell) yields the right x-lookup
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
