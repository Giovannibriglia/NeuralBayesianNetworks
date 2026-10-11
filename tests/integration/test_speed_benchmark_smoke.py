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

from nbn.bench.cli import _run_cells
from nbn.bench.core.yaml_config import load_runner_config

_SMOKE = (
    "nbn/bench/configs/synthetic/speed/inference_speed_smoke.yaml"
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
        """Every (baseline, batch_size) slice holds the same queries.

        The heaviest selector emits K positions on the 8-node fixture, half of
        them evidence-free (empty mode). Since D7 an evidence-free position is
        ONE query in every pass (it used to be n_batch_queries=16 identical
        copies that nbn answered once while dividing the time by 16). Its rows
        are stamped with the pass's batch size, so each slice holds
        K_full x 16 full-mode rows + K_empty empty-mode rows -- fewer rows
        than before (K x 16), and still the same count for every combo.
        """
        qt = sweep_df[sweep_df.metric == "query_time_s"]
        by = qt.groupby(["baseline", "batch_size"]).size()
        n_per_combo = by.iloc[0]
        assert (by == n_per_combo).all()  # same count for every combo
        full = qt[qt.evidence_mode == "full"].groupby(["baseline", "batch_size"]).size()
        assert (full % 16 == 0).all()     # K_full x 16 (n_batch_queries plumbed)
        empty = qt[qt.evidence_mode == "empty"].groupby(["baseline", "batch_size"]).size()
        assert (empty < 16).all()         # one row per evidence-free position
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
        """Per (problem, seed, baseline, batch_size): K_full x 16 full-mode
        query rows (n_batch_queries variants per position) and one row per
        evidence-free position (D7: a single query, not 16 copies; stamped
        with the pass's batch size, so a B=4 pass's evidence-free rows are not
        pooled into the B=1 slice)."""
        qt = sweep_df[sweep_df.metric == "query_time_s"]
        keys = ["problem_id", "seed", "baseline", "batch_size"]
        k_empty = None
        for key, grp in qt.groupby(keys):
            n_full = int((grp.evidence_mode == "full").sum())
            n_empty = int((grp.evidence_mode == "empty").sum())
            k = n_full / 16
            assert k == int(k) and k >= 1, (
                f"expected K_full x 16 full-mode rows, got {n_full} ({key})")
            k_empty = n_empty if k_empty is None else k_empty
            assert n_empty == k_empty, f"evidence-free rows differ across slices ({key})"


@pytest.mark.slow
class TestNoSweepPathUnchanged:
    def test_config_without_batch_sizes_takes_single_pass(self, tmp_path):
        """A config without batch_sizes goes through the pre-sweep path
        (cfg.batch_sizes is None → single Runner pass)."""
        cfg = load_runner_config(
            "nbn/bench/configs/bnlearn/smoke_tests/inference_smoke.yaml",
            device_override="cpu",
            jsonl_path=tmp_path / "unused.jsonl",
        )
        assert cfg.batch_sizes is None
