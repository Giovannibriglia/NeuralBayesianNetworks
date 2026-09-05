"""Tests for cell_runner subprocess isolation infrastructure (Bug 4 of #127).

Covers the infrastructure (output parsing, exit-code mapping) plus, as of
Stage 2, the real cell path: cell_worker._run_cell now builds the adapter,
selects queries, fits, and measures inside the subprocess. The slow test
exercises a full synthetic cell end-to-end via run_cell_in_subprocess.
"""
from __future__ import annotations

import pytest

from nbn.bench.core.cell_runner import (
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

    # Note: deliberate timeout / OOM / exception failure modes are covered
    # by TestFailureModes (Stage 3) below.

    @pytest.mark.slow
    def test_real_cell_completes_ok(self):
        """Full real cell runs in the subprocess and returns ok rows.

        Exercises the Stage 2 path end-to-end: build_adapter → fit →
        select → measure, all inside the worker, with real picklable
        objects (synthetic discrete problem, nbn-cat-ve, topological
        selector, AccuracyAndTiming).
        """
        from nbn.bench.core.config import BaselineSpec
        from nbn.bench.measurements.accuracy_timing import AccuracyAndTiming
        from nbn.bench.problems.synthetic import (
            SyntheticConfig,
            SyntheticProblemSource,
        )
        from nbn.bench.selectors.topological import TopologicalAllocator

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


@pytest.mark.slow
class TestFailureModes:
    """Stage 3: empirical proof that the runner survives each failure mode.

    Each test launches a subprocess (the dedicated ``_failing_cell_worker``)
    that fails in a specific way, then drives it through the *same*
    lifecycle the production parent uses — SIGTERM→grace→SIGKILL escalation
    on timeout, and the real ``cell_runner`` classification/parsing helpers
    (``_classification_to_status``, ``_read_output_jsonl``). This is the
    empirical verification of the PR's architectural claim: an exception,
    a hang, or an uncatchable kill in a cell is contained — the parent
    records a status and survives.

    See docs/v0.13-runner-subprocess-isolation.md §6.2.
    """

    @staticmethod
    def _launch_failing_worker(mode: str, *, timeout_s: float = 10.0):
        """Run _failing_cell_worker in ``mode`` and classify the outcome.

        Returns (classification, exit_code, rows). The process-lifecycle
        management mirrors run_cell_in_subprocess; classification and
        output parsing reuse the production cell_runner helpers.
        """
        import os
        import pickle
        import signal
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        worker = Path(__file__).parent / "_failing_cell_worker.py"
        assert worker.exists()

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pkl", delete=False
        ) as f:
            pickle.dump({"test": True}, f)
            input_path = f.name
        output_path = input_path.replace(".pkl", ".jsonl")
        Path(output_path).touch()

        try:
            proc = subprocess.Popen(
                [sys.executable, str(worker), input_path, output_path, mode],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            classification = "completed"
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                classification = "timeout"
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=2.0)

            exit_code = proc.returncode if proc.returncode is not None else -1
            # Signal death (negative returncode, or 128+sig) → killed.
            if classification == "completed" and (
                exit_code < 0 or exit_code in (137, 139)
            ):
                classification = "killed"

            rows = _read_output_jsonl(output_path)
            return classification, exit_code, rows
        finally:
            import os as _os
            for p in (input_path, output_path):
                try:
                    _os.unlink(p)
                except OSError:
                    pass

    def test_ok_mode_completes_normally(self):
        classification, exit_code, rows = self._launch_failing_worker("ok")
        assert classification == "completed"
        assert exit_code == 0
        assert any(r.get("status") == "ok" for r in rows)

    def test_raise_mode_produces_nonzero_exit(self):
        """An exception in the cell → non-zero exit, runner not crashed."""
        classification, exit_code, rows = self._launch_failing_worker("raise")
        assert classification == "completed"
        assert exit_code != 0
        # No rows were written before the exception fired; the parent maps
        # a completed-but-no-output subprocess to a synthesized error row.
        assert _classification_to_status(classification, exit_code) == "error"

    def test_sleep_mode_killed_via_timeout(self):
        """A hanging cell is SIGTERMed by the parent → timeout, partial data kept."""
        classification, exit_code, rows = self._launch_failing_worker(
            "sleep", timeout_s=2.0,
        )
        assert classification == "timeout"
        assert _classification_to_status(classification, exit_code) == "timeout"
        # The preamble row written+flushed before the sleep must survive.
        assert any(r.get("stage") == "preamble" for r in rows)

    def test_oom_mode_uncatchable_kill(self):
        """An uncatchable SIGKILL (models the OS OOM-killer) → oom, runner survives.

        This is the architecturally critical case: the production 612 GB
        OOM tripped the kernel OOM-killer, whose SIGKILL no in-process
        handler can catch. The worker SIGKILLs itself; the parent must
        detect the signal death, classify it as oom, preserve the
        already-flushed preamble row, and (critically) keep running.
        """
        classification, exit_code, rows = self._launch_failing_worker("oom")
        assert classification == "killed", (
            f"expected signal death; got classification={classification}, "
            f"exit_code={exit_code}, rows={rows}"
        )
        assert exit_code < 0 or exit_code == 137
        # K3 convention: any SIGKILL → oom.
        assert _classification_to_status(classification, exit_code) == "oom"
        # Partial output (flushed before the kill) survives.
        assert any(r.get("stage") == "preamble" for r in rows)

    def test_memory_limit_enforced(self):
        """RLIMIT_AS in the worker bounds subprocess allocation.

        Sets a low per-cell limit (300 MB) via the env var contract and
        launches a worker that attempts to allocate well beyond it.
        The kernel must SIGKILL the subprocess; the parent classifies
        it as killed → oom (K3 convention).

        This is the safety guarantee that lets benchmark runs not
        destabilize the host system when a baseline misbehaves.
        """
        import os
        import pickle
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        # Pickle a minimal but valid ctx — won't matter, we'll OOM before
        # building the adapter
        ctx = {"deliberate": "memory test"}

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pkl", delete=False,
        ) as f:
            pickle.dump(ctx, f)
            input_path = f.name
        output_path = input_path.replace(".pkl", ".jsonl")
        Path(output_path).touch()

        # Test worker that tries to allocate way more than the limit
        worker_script = Path(tempfile.mkdtemp()) / "mem_test_worker.py"
        worker_script.write_text(
            "import sys\n"
            "from nbn.bench.core.cell_worker import _apply_memory_limit\n"
            "_apply_memory_limit()\n"
            "# Now attempt 1 GB allocation. With a 300 MB cap, this OOMs.\n"
            "import ctypes\n"
            "buf = ctypes.create_string_buffer(1024 * 1024 * 1024)\n"
            "print('REACHED END — cap not enforced')\n"
        )

        env = os.environ.copy()
        env["NBN_CELL_MEMORY_LIMIT_BYTES"] = str(300 * 1024 * 1024)

        try:
            proc = subprocess.Popen(
                [sys.executable, str(worker_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=env,
            )
            # 10s was too tight: the worker imports
            # nbn.bench.core.cell_worker (and torch beneath it), which
            # under a parallel pytest run can take longer than that, turning
            # a passing assertion into a TimeoutExpired.  The claim under
            # test is that the cap is *enforced*, not that it is enforced
            # quickly, so the budget is generous.
            proc.wait(timeout=120)
            exit_code = proc.returncode

            # Expected: signal death (negative returncode or 137)
            # OR MemoryError exit (non-zero) — either is acceptable; the
            # point is that the cap was enforced, the subprocess did not
            # consume 1 GB on this system.
            # Use the classification helper to confirm interpretation:
            if exit_code < 0 or exit_code == 137:
                # Signal kill — K3 → oom
                assert _classification_to_status("killed", exit_code) == "oom"
            else:
                # Python caught MemoryError before alloc — also valid
                # (cap was enforced, just earlier in the call chain)
                assert exit_code != 0
        finally:
            for p in (input_path, output_path, str(worker_script)):
                try:
                    os.unlink(p)
                except OSError:
                    pass
