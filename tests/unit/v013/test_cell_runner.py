"""Tests for cell_runner subprocess isolation infrastructure (Bug 4 of #127).

Stage 1: tests for the infrastructure itself (subprocess launch,
output parsing, exit-code mapping). Real measurement code lands in
Stage 2 — these tests use the Stage 1 placeholder _run_cell.
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile

import pytest

from benchmarking.core.cell_runner import (
    _classification_to_status,
    _read_output_jsonl,
    run_cell_in_subprocess,
)


# Module-level fakes so they pickle cleanly (pickle cannot serialize
# classes defined inside a function/method). Real BaselineSpec and
# BenchmarkProblem are module-level too, so this mirrors actual usage.
class _FakeSpec:
    library = "fake-baseline"


class _FakeProblem:
    problem_id = "fake-problem"


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

    def test_normal_completion(self):
        """A well-behaved subprocess produces ok rows + exits cleanly."""
        # The Stage 1 placeholder doesn't use a real BaselineSpec; the
        # module-level fakes provide the .library / .problem_id attributes
        # the placeholder reads, and pickle cleanly.
        ctx = {
            "problem": _FakeProblem(),
            "baseline_spec": _FakeSpec(),
            "seed": 42,
        }

        result = run_cell_in_subprocess(ctx, timeout_s=30)

        assert result.exit_code == 0
        assert result.classification == "completed"
        assert len(result.rows) >= 1
        assert result.rows[0]["status"] == "ok"
        assert result.rows[0]["stage"] == "stage1_placeholder"

    def test_timeout_kills_subprocess(self):
        """A subprocess that takes too long is SIGTERMed → marked timeout."""
        # Inject a sleeping cell. Stage 1's placeholder returns quickly,
        # so we need a different mechanism. Use a small inline script:
        #   _SLEEPY_CTX = {"_sleep_for": 30}  # special hook
        # Or test this in Stage 2 when we have real cells.
        # For Stage 1: skip this test, mark it as expected in Stage 2/3.
        pytest.skip("Timeout-killing tested in Stage 3 (needs sleepy worker)")

    def test_pickle_round_trip(self):
        """The input pickle survives the round trip via the subprocess."""
        # Just confirm that loading pickle in subprocess works
        ctx = {"key": "value", "number": 42}

        # Manually invoke the worker (no timeout, no kill)
        import benchmarking.core.cell_worker as cw

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pkl", delete=False
        ) as f:
            pickle.dump(ctx, f)
            input_path = f.name

        output_path = input_path.replace(".pkl", ".jsonl")

        try:
            cw.main(input_path, output_path)

            # Read output
            with open(output_path) as f:
                rows = [json.loads(line) for line in f if line.strip()]

            assert len(rows) == 1
            # Placeholder uses ctx.get("problem", {}).problem_id or fallback to ctx.get("problem_id")
            # ctx has no "problem" key, so fallback applies. But ctx also has no "problem_id".
            # So default "unknown" applies. Good enough for round-trip.
            assert rows[0]["status"] == "ok"
        finally:
            for p in (input_path, output_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
