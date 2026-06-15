"""Integration tests for the batch_speed benchmark sweep (PR 5, #148).

Drives the CLI's sweep dispatch (`_run_cells`) end-to-end on the speed
smoke config and asserts the design-doc contract:

  - swept baselines (nbn-cat-ve) produce rows at every batch_sizes value
  - pinned baselines (pgmpy-mle-ve, batch_size: 1) run exactly once
  - batch_size column stamped with the sweep value per row
  - no errors / timeouts on the tiny fixture
  - n_batch_queries plumbing: K positions × 16 variants per
    (baseline, batch_size) combination

Reference: docs/v0.14-batched-queries-design.md §1.7, §5.4-5.6, §7.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarking.cli import _run_cells
from benchmarking.core.yaml_config import load_runner_config

_SMOKE = (
    "benchmarking/configs/synthetic/speed/inference_speed_smoke.yaml"
)


@pytest.fixture(scope="module")
def sweep_df(tmp_path_factory) -> pd.DataFrame:
    """Run the speed smoke once through the sweep dispatch; load rows."""
    jsonl = tmp_path_factory.mktemp("speed_smoke") / "metrics.jsonl"
    cfg = load_runner_config(_SMOKE, device_override="cpu", jsonl_path=jsonl)
    _run_cells(cfg)
    rows = [
        json.loads(line)
        for line in Path(jsonl).read_text().splitlines() if line.strip()
    ]
    return pd.DataFrame(rows)


@pytest.mark.slow
class TestSweepDispatch:
    def test_swept_baseline_runs_at_every_batch_size(self, sweep_df):
        qt = sweep_df[sweep_df.metric == "query_time_s"]
        nbn = qt[qt.baseline == "nbn-cat-ve"]
        assert sorted(nbn["batch_size"].unique()) == [1, 4]

    def test_pinned_baseline_runs_once_at_1(self, sweep_df):
        qt = sweep_df[sweep_df.metric == "query_time_s"]
        pgmpy = qt[qt.baseline == "pgmpy-mle-ve"]
        assert sorted(pgmpy["batch_size"].unique()) == [1]

    def test_batch_size_column_stamped(self, sweep_df):
        qt = sweep_df[sweep_df.metric == "query_time_s"]
        by = qt.groupby(["baseline", "batch_size"]).size()
        # 160 = K positions × n_batch_queries=16 variants; K=10 for the
        # 8-node fixture DAG (heaviest emits 10 positions on it).
        n_per_combo = by.iloc[0]
        assert (by == n_per_combo).all()  # same count for every combo
        assert n_per_combo % 16 == 0      # K × 16 (n_batch_queries plumbed)
        assert set(by.index) == {
            ("nbn-cat-ve", 1), ("nbn-cat-ve", 4), ("pgmpy-mle-ve", 1),
        }

    def test_status_distribution_clean(self, sweep_df):
        assert set(sweep_df["status"].unique()) == {"ok"}

    def test_output_schema(self, sweep_df):
        expected = {
            "benchmark", "family", "problem_id", "seed", "baseline",
            "query_role", "metric", "value", "status", "fit_time_s",
            "query_time_s", "metrics_time_s", "error_msg", "query_kind",
            "evidence_strategy", "evidence_mode", "n_parameters",
            "n_nodes", "device", "batch_size",
        }
        assert expected.issubset(set(sweep_df.columns))

    def test_n_nodes_populated_on_ok_rows(self, sweep_df):
        """n_nodes is injected per-problem (parallel to n_parameters) and is
        populated for every ok row — 8 for the 8-node fixture DAG."""
        ok = sweep_df[sweep_df.status == "ok"]
        assert ok["n_nodes"].notna().all()
        assert set(ok["n_nodes"].unique()) == {8}

    def test_n_batch_queries_row_count(self, sweep_df):
        """K×16 query rows per (problem, seed, baseline, batch_size)."""
        qt = sweep_df[sweep_df.metric == "query_time_s"]
        for (_, _, _, bs), grp in qt.groupby(
            ["problem_id", "seed", "baseline", "batch_size"]
        ):
            k = len(grp) / 16
            assert k == int(k) and k >= 1, (
                f"expected K×16 rows, got {len(grp)} (batch_size={bs})"
            )


@pytest.mark.slow
class TestNoSweepPathUnchanged:
    def test_config_without_batch_sizes_takes_single_pass(self, tmp_path):
        """A config without batch_sizes goes through the pre-sweep path
        (cfg.batch_sizes is None → single Runner pass)."""
        cfg = load_runner_config(
            "benchmarking/configs/bnlearn/smoke_tests/inference_smoke.yaml",
            device_override="cpu",
            jsonl_path=tmp_path / "unused.jsonl",
        )
        assert cfg.batch_sizes is None
