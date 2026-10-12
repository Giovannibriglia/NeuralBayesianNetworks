"""Slow integration test: nbn-bench inference must produce the canonical run
artifacts after a successful run (JSONL + parquet).

The post-run pipeline writes JSONL → parquet only. Paper figures + LaTeX
tables are produced on demand by the separate `nbn-bench plot` command; the
old auto-generated post-run figures/ + tables/ dirs were stale (old-schema)
and were removed in v0.14 (PR-2). This test pins JSONL + parquet presence.

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
    """nbn-bench inference must produce JSONL + parquet (the canonical run
    artifacts).

    Regression for PR #119 gap: CLI only wrote JSONL; no parquet.  Figures +
    tables are NOT produced post-run anymore (v0.14 PR-2) — they come from
    `nbn-bench plot`; this test asserts they are ABSENT from the run dir.
    """
    # Write the minimal config to a temp file.
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_MINIMAL_CONFIG))

    result = subprocess.run(
        [sys.executable, "-m", "nbn.bench.cli", "inference",
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
        f"no *.parquet in {run_dir}; post-run step (jsonl_to_parquet) "
        f"did not run or failed.  stderr: {result.stderr[-500:]}"
    )

    # 3. No stale figures/ or tables/ — removed in v0.14 (PR-2); these are
    #    produced on demand by `nbn-bench plot`, not auto-generated post-run.
    assert not (run_dir / "tables").exists(), (
        f"tables/ dir should NOT be created post-run (v0.14 PR-2); found in {run_dir}"
    )
    assert not (run_dir / "figures").exists(), (
        f"figures/ dir should NOT be created post-run (v0.14 PR-2); found in {run_dir}"
    )


@pytest.mark.slow
def test_cli_param_learning_produces_parquet(tmp_path: Path) -> None:
    """nbn-bench param-learning runs end-to-end and writes the parquet (#109).

    Un-stubs the former informational stub: the command now constructs
    ParamLearningMeasurement and drives the same JSONL → parquet pipeline as
    inference. Asserts a clean exit + a parquet carrying log_likelihood rows
    with at least one NBN row status="ok" and a finite value, and the pgmpy
    baseline status="not_supported".

    PL-realistic config: the baselines OMIT inference_method (PL never queries),
    exercising the fit-only build path (require_engine=False). The earlier
    version inherited _MINIMAL_CONFIG's inference_method and so missed the
    build_adapter regression that CI caught.
    """
    import math

    import pandas as pd

    cfg_path = tmp_path / "cfg.yaml"
    d = dict(_MINIMAL_CONFIG)
    d["config_name"] = "pl_run_test"
    d["metrics"] = "log_likelihood"   # required by the param-learning command
    # Drop inference_method from every baseline — PL specs declare none.
    d["baselines"] = [
        {k: v for k, v in b.items() if k != "inference_method"}
        for b in _MINIMAL_CONFIG["baselines"]
    ]
    cfg_path.write_text(yaml.safe_dump(d))

    result = subprocess.run(
        [sys.executable, "-m", "nbn.bench.cli", "param-learning",
         "--config", str(cfg_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, (
        f"param-learning exited {result.returncode}\n"
        f"stdout: {result.stdout[-1000:]}\n"
        f"stderr: {result.stderr[-1000:]}"
    )

    results_root = tmp_path / "results"
    run_dirs = list(results_root.glob("benchmark_synthetic_pl_run_test_*"))
    assert len(run_dirs) == 1, (
        f"expected exactly 1 run dir, found: {[d.name for d in run_dirs]}"
    )
    parquet_files = list(run_dirs[0].glob("*.parquet"))
    assert parquet_files, (
        f"no *.parquet in {run_dirs[0]}; stderr: {result.stderr[-500:]}"
    )

    df = pd.read_parquet(parquet_files[0])
    ll = df[df["metric"] == "log_likelihood"]
    assert not ll.empty, (
        f"no log_likelihood rows in parquet; metrics present: "
        f"{sorted(df['metric'].unique())}"
    )

    # NBN implements score_data -> at least one ok row with a finite value.
    nbn_ok = ll[(ll["baseline"].str.startswith("nbn-")) & (ll["status"] == "ok")]
    assert not nbn_ok.empty, (
        f"expected an NBN log_likelihood ok row; got:\n"
        f"{ll[['baseline', 'status', 'value']].to_string(index=False)}"
    )
    assert all(math.isfinite(v) for v in nbn_ok["value"]), nbn_ok["value"].tolist()

    # pgmpy does not implement score_data yet -> not_supported.
    pgmpy_rows = ll[ll["baseline"].str.startswith("pgmpy-")]
    assert not pgmpy_rows.empty
    assert (pgmpy_rows["status"] == "not_supported").all(), (
        f"pgmpy should be not_supported in PR 1; got:\n"
        f"{pgmpy_rows[['baseline', 'status']].to_string(index=False)}"
    )
