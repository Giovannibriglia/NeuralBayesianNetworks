"""Slow integration test: nbn-bench inference must produce all four output
types after a successful run (JSONL + parquet + figures + tables).

Regression for the gap discovered during PR #119 manual verification:
the v0.13 CLI was draining the Runner iterator and writing JSONL but not
invoking the post-run pipeline (jsonl_to_parquet, aggregate, write_all,
render_figures).  v0.12 produced all four automatically.

Marked @pytest.mark.slow — excluded from the fast CI gate.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Minimal inference config — faster than inference_smoke.yaml (1 family,
# 1 n_node size, 1 seed, 2 baselines, 2 queries, 30s timeout).
# Produces enough rows for all post-run steps to succeed.
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG = {
    "version": "v0.13",
    "benchmark": "synthetic",
    "config_name": "cli_test",
    "metrics": "all",
    "selector": "uniform_random",
    "source": {
        "families": ["discrete"],
        "n_nodes_list": [5],
        "seeds": [0],
        "n_train": 200,
        "n_test": 50,
        "n_reference": 500,
        "edge_density": 0.20,
        "max_in_degree": 2,
        "cardinality": 4,
        "fraction_continuous": 0.0,
    },
    "baselines": [
        {
            "library": "pgmpy",
            "mechanism": "discrete",
            "param_method": "mle",
            "inference_method": "ve",
        },
        {
            "library": "nbn",
            "mechanism": "cat",
            "param_method": "mle",
            "inference_method": "ve",
        },
    ],
    "n_queries_per_cell": 2,
    "per_cell_timeout_s": 30.0,
}


@pytest.mark.slow
def test_cli_inference_produces_all_outputs(tmp_path: Path) -> None:
    """nbn-bench inference must produce JSONL + parquet + figures + tables.

    Regression for PR #119 gap: CLI only wrote JSONL; no parquet, no figures,
    no tables.  This test pins the full post-run pipeline.
    """
    # Write the minimal config to a temp file.
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_MINIMAL_CONFIG))

    result = subprocess.run(
        [sys.executable, "-m", "benchmarking.cli", "inference",
         "--config", str(cfg_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, (
        f"CLI exited {result.returncode}\n"
        f"stdout: {result.stdout[-1000:]}\n"
        f"stderr: {result.stderr[-1000:]}"
    )

    # Find the results directory (compact_datetime suffix makes name unpredictable).
    results_root = tmp_path / "results"
    assert results_root.exists(), "results/ directory was not created"

    run_dirs = list(results_root.glob("benchmark_synthetic_cli_test_*"))
    assert len(run_dirs) == 1, (
        f"expected exactly 1 run dir, found: {[d.name for d in run_dirs]}"
    )
    run_dir = run_dirs[0]

    # 1. JSONL
    jsonl_files = list(run_dir.glob("*.jsonl"))
    assert jsonl_files, f"no *.jsonl in {run_dir}; contents: {list(run_dir.iterdir())}"

    # 2. Parquet
    parquet_files = list(run_dir.glob("*.parquet"))
    assert parquet_files, (
        f"no *.parquet in {run_dir}; post-run step 1 (jsonl_to_parquet) "
        f"did not run or failed.  stderr: {result.stderr[-500:]}"
    )

    # 3. Tables
    tables_dir = run_dir / "tables"
    assert tables_dir.exists(), (
        f"tables/ dir absent in {run_dir}; post-run step 2 (write_all) "
        f"did not run or failed."
    )
    csv_files = list(tables_dir.glob("*.csv"))
    assert csv_files, f"no *.csv in {tables_dir}"

    # 4. Figures
    figures_dir = run_dir / "figures"
    assert figures_dir.exists(), (
        f"figures/ dir absent in {run_dir}; post-run step 3 (render_figures) "
        f"did not run or failed."
    )
    png_files = list(figures_dir.glob("*.png"))
    assert png_files, f"no *.png in {figures_dir}"


@pytest.mark.slow
def test_cli_param_learning_stub_exits_zero(tmp_path: Path) -> None:
    """param-learning stub must exit 0 (informational, not an error)."""
    cfg_path = tmp_path / "cfg.yaml"
    d = dict(_MINIMAL_CONFIG)
    d["config_name"] = "pl_stub_test"
    cfg_path.write_text(yaml.safe_dump(d))

    result = subprocess.run(
        [sys.executable, "-m", "benchmarking.cli", "param-learning",
         "--config", str(cfg_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, (
        f"param-learning stub must exit 0; got {result.returncode}\n"
        f"stderr: {result.stderr}"
    )
    assert "not yet implemented" in result.stderr.lower(), (
        "expected 'not yet implemented' message in stderr; "
        f"got: {result.stderr!r}"
    )
