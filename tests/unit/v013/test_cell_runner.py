"""Tests for cell_runner subprocess isolation infrastructure (Bug 4 of #127).

Covers the infrastructure (output parsing, exit-code mapping) plus, as of
Stage 2, the real cell path: cell_worker._run_cell now builds the adapter,
selects queries, fits, and measures inside the subprocess. The slow test
exercises a full synthetic cell end-to-end via run_cell_in_subprocess.
"""
from __future__ import annotations

import pytest

from benchmarking.core.cell_runner import (
    _classification_to_status,
    _read_output_jsonl,
    run_cell_in_subprocess,
)


class TestOutputParsing:
    """_read_output_jsonl tolerates partial final lines."""

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert _read_output_jsonl(str(p)) == []

    def test_missing_file(self, tmp_path):
        assert _read_output_jsonl(str(tmp_path / "nonexistent.jsonl")) == []

    def test_two_complete_rows(self, tmp_path):
        p = tmp_path / "two.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n')
        rows = _read_output_jsonl(str(p))
        assert rows == [{"a": 1}, {"b": 2}]

    def test_partial_final_line(self, tmp_path):
        """Subprocess died mid-write — final line is partial JSON."""
        p = tmp_path / "partial.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n{"c":')  # truncated
        rows = _read_output_jsonl(str(p))
        assert rows == [{"a": 1}, {"b": 2}]

    def test_only_partial(self, tmp_path):
        """Subprocess died before any complete line."""
        p = tmp_path / "only_partial.jsonl"
        p.write_text('{"a":')
        assert _read_output_jsonl(str(p)) == []


class TestStatusMapping:
    """_classification_to_status maps lifecycle → row status."""

    def test_completed_normal(self):
        # Completed subprocess: classification status only applies
        # when subprocess didn't write its own. The mapper is for
        # synthetic rows on kill.
        assert _classification_to_status("completed", 0) == "error"
        # (subprocess wrote no rows + completed = error, by design)

    def test_timeout(self):
        assert _classification_to_status("timeout", -15) == "timeout"

    def test_killed(self):
        assert _classification_to_status("killed", -9) == "oom"
        # K3 convention: SIGKILL = oom regardless of true cause

    def test_killed_137(self):
        # Return code 137 = 128 + 9 (SIGKILL on some platforms)
        assert _classification_to_status("killed", 137) == "oom"


class TestSubprocessLaunch:
    """Integration tests for the parent-subprocess interaction."""

    def test_malformed_ctx_produces_error_row(self):
        """A ctx missing required keys is handled gracefully.

        The worker's _run_cell raises KeyError on the missing 'problem'
        key; the worker's catch-all writes a status=error row rather than
        crashing, and the parent returns it. The subprocess exits 1 but
        is not signal-killed, so classification stays 'completed'.
        """
        result = run_cell_in_subprocess({"not": "a real ctx"}, timeout_s=30)

        assert result.classification == "completed"
        assert result.exit_code == 1
        assert len(result.rows) == 1
        assert result.rows[0]["status"] == "error"

    def test_timeout_kills_subprocess(self):
        """A subprocess that takes too long is SIGTERMed → marked timeout."""
        # Needs a deliberately-hanging worker; added in Stage 3 alongside
        # the deliberate-OOM and deliberate-exception failure-mode tests.
        pytest.skip("Timeout-killing tested in Stage 3 (needs sleepy worker)")

    @pytest.mark.slow
    def test_real_cell_completes_ok(self):
        """Full real cell runs in the subprocess and returns ok rows.

        Exercises the Stage 2 path end-to-end: build_adapter → fit →
        select → measure, all inside the worker, with real picklable
        objects (synthetic discrete problem, nbn-cat-ve, topological
        selector, AccuracyAndTiming).
        """
        from benchmarking.core.config import BaselineSpec
        from benchmarking.measurements.accuracy_timing import AccuracyAndTiming
        from benchmarking.problems.synthetic import (
            SyntheticConfig,
            SyntheticProblemSource,
        )
        from benchmarking.selectors.topological import TopologicalAllocator

        scfg = SyntheticConfig(
            families=["discrete"], n_nodes_list=[8], seeds=[0],
            n_train=200, n_test=50, n_reference=100,
            edge_density=0.20, max_in_degree=3, cardinality=3,
            fraction_continuous=0.5,
        )
        problem = next(SyntheticProblemSource().iter_problems(scfg))
        spec = BaselineSpec(
            library="nbn", mechanism="cat",
            param_method="mle", inference_method="ve",
        )

        ctx = {
            "problem": problem,
            "baseline_spec": spec,
            "seed": problem.seed,
            "selector": TopologicalAllocator(),
            "measurement": AccuracyAndTiming(),
            "benchmark": "synthetic",
            "n_queries_per_cell": 4,
            "per_cell_timeout_s": 60.0,
            "fit_budget_s": 600.0,
            "default_role": "random",
        }

        result = run_cell_in_subprocess(ctx, timeout_s=120)

        assert result.classification == "completed"
        assert result.exit_code == 0
        assert len(result.rows) >= 1
        # All rows carry the cell identity and a valid status.
        for row in result.rows:
            assert row["problem_id"] == problem.problem_id
            assert row["baseline"] == "nbn-cat-ve"
            assert row["status"] in {"ok", "error", "timeout", "oom", "not_supported"}
        assert any(r["status"] == "ok" for r in result.rows)
