"""Tests for cell-level checkpoint/resume (benchmarking/core/checkpoint.py).

Covers:
  TestCellKey            — key identity, n_train inclusion (learning-curves)
  TestSidecar            — append/load roundtrip of completed_cells.jsonl
  TestCompaction         — partial-row compaction, seed-skip registry rebuild
  TestRunnerResume       — runner-level skip / marker / preemption behavior
                           (Runner._run_cell monkeypatched — no subprocesses)
  TestCliResultsDir      — --results-dir / --resume argument resolution

Reference: benchmarking/slurm/README.md
"""
from __future__ import annotations

import json
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
import torch

from benchmarking.core import checkpoint
from benchmarking.core.config import BaselineSpec, RunnerConfig
from benchmarking.core.results import CellResult
from benchmarking.core.runner import Runner
from benchmarking.domains._n_parameters import n_train_from_problem
from benchmarking.domains.base import BenchmarkProblem, GroundTruth
from benchmarking.measurements import TimingOnly
from benchmarking.selectors.uniform import UniformRandomSelector


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _FixedProblemSource:
    """Yields a single pre-built problem; source_config IS the problem."""

    def iter_problems(self, problem: BenchmarkProblem) -> Iterator[BenchmarkProblem]:
        yield problem


def _discrete_problem(n_nodes: int = 3, seed: int = 0, n_train: int = 100) -> BenchmarkProblem:
    nodes = [f"X{i}" for i in range(n_nodes)]
    return BenchmarkProblem(
        name=f"discrete_n{n_nodes}_seed{seed}",
        dag=[(nodes[i], nodes[i + 1]) for i in range(n_nodes - 1)],
        variables=dict.fromkeys(nodes, ("discrete", 4)),
        train_data={n: torch.randint(0, 4, (n_train,)).float() for n in nodes},
        test_data={n: torch.randint(0, 4, (20,)).float() for n in nodes},
        queries=[],
        ground_truth=GroundTruth(),
        family="discrete",
        problem_id=str(n_nodes),
        seed=seed,
    )


def _row(problem: BenchmarkProblem, baseline: str, **overrides) -> CellResult:
    """One well-formed ok row for (problem, baseline), n_train stamped the way
    the real runner choke point stamps it."""
    defaults = dict(
        benchmark="synthetic",
        family=problem.family,
        problem_id=problem.problem_id,
        seed=problem.seed,
        baseline=baseline,
        query_role="random",
        metric="query_time_s",
        value=0.01,
        status="ok",
        fit_time_s=0.1,
        query_time_s=0.01,
        metrics_time_s=0.0,
        n_train=n_train_from_problem(problem),
    )
    defaults.update(overrides)
    return CellResult(**defaults)


def _runner_cfg(problem: BenchmarkProblem, specs: list[BaselineSpec],
                tmp_path: Path, *, resume: bool = False) -> RunnerConfig:
    return RunnerConfig(
        benchmark="synthetic",
        config_name="test",
        problem_source=_FixedProblemSource(),
        source_config=problem,
        selector=UniformRandomSelector(),
        measurement=TimingOnly(),
        baselines=specs,
        n_queries_per_cell=2,
        per_cell_timeout_s=60.0,
        jsonl_path=tmp_path / "metrics.jsonl",
        resume=resume,
    )


_SPEC_LW = BaselineSpec("nbn", "cat", "mle", "lw")
_SPEC_VE = BaselineSpec("nbn", "cat", "mle", "ve")


@pytest.fixture()
def fake_run_cell(monkeypatch):
    """Replace Runner._run_cell with an instant fake that records the cells it
    ran and writes/yields one ok row per cell (no subprocess)."""
    calls: list[tuple] = []

    def fake(self, cfg, problem, spec, writer, **kwargs):
        from benchmarking.core.config import build_adapter
        name = build_adapter(spec).name
        calls.append(checkpoint.cell_key(problem, name))
        row = _row(problem, name)
        writer.write(row)
        yield row

    monkeypatch.setattr(Runner, "_run_cell", fake)
    return calls


# ---------------------------------------------------------------------------
# TestCellKey
# ---------------------------------------------------------------------------

class TestCellKey:
    def test_key_fields(self):
        p = _discrete_problem(n_nodes=3, seed=7)
        assert checkpoint.cell_key(p, "nbn-cat-lw") == (
            "discrete", "3", 7, "nbn-cat-lw", 100)

    def test_n_train_distinguishes_sweep_problems(self):
        # learning-curves sweep (#109 PR 6): problems differing ONLY in
        # training-set size share family/problem_id/seed — the key must
        # still tell them apart.
        p_small = _discrete_problem(n_train=64)
        p_large = _discrete_problem(n_train=256)
        assert (checkpoint.cell_key(p_small, "b")
                != checkpoint.cell_key(p_large, "b"))


# ---------------------------------------------------------------------------
# TestSidecar
# ---------------------------------------------------------------------------

class TestSidecar:
    def test_append_load_roundtrip(self, tmp_path):
        path = tmp_path / "completed_cells.jsonl"
        keys = [
            ("discrete", "3", 0, "nbn-cat-lw", 100),
            ("continuous_lg", "8", 1, "nbn-mdn-lw", None),
        ]
        for k in keys:
            checkpoint.append_completed(path, k)
        assert checkpoint.load_completed(path) == set(keys)

    def test_load_missing_is_empty(self, tmp_path):
        assert checkpoint.load_completed(tmp_path / "nope.jsonl") == set()

    def test_sidecar_path_is_sibling(self, tmp_path):
        assert (checkpoint.completed_cells_path(tmp_path / "metrics.jsonl")
                == tmp_path / "completed_cells.jsonl")


# ---------------------------------------------------------------------------
# TestCompaction
# ---------------------------------------------------------------------------

class TestCompaction:
    def test_partial_rows_dropped(self, tmp_path):
        import dataclasses
        p = _discrete_problem()
        jsonl = tmp_path / "metrics.jsonl"
        with open(jsonl, "w") as fh:
            for baseline in ("done-baseline", "partial-baseline"):
                d = dataclasses.asdict(_row(p, baseline))
                fh.write(json.dumps(d) + "\n")
        completed = {checkpoint.cell_key(p, "done-baseline")}
        kept, dropped = checkpoint.compact_jsonl(jsonl, completed)
        assert (kept, dropped) == (1, 1)
        rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
        assert [r["baseline"] for r in rows] == ["done-baseline"]

    def test_missing_jsonl_noop(self, tmp_path):
        assert checkpoint.compact_jsonl(tmp_path / "nope.jsonl", set()) == (0, 0)

    def test_rebuild_seed_skip_registry(self, tmp_path):
        import dataclasses
        p = _discrete_problem()
        jsonl = tmp_path / "metrics.jsonl"
        rows = [
            _row(p, "b1", status="oom", batch_size=8),
            _row(p, "b1", status="ok", batch_size=1),
            _row(p, "b2", status="timeout", batch_size=32),
        ]
        with open(jsonl, "w") as fh:
            for r in rows:
                fh.write(json.dumps(dataclasses.asdict(r)) + "\n")
        registry = checkpoint.rebuild_seed_skip_registry(jsonl)
        assert registry == {
            ("discrete", "3", "b1", 8): "oom",
            ("discrete", "3", "b2", 32): "timeout",
        }


# ---------------------------------------------------------------------------
# TestRunnerResume
# ---------------------------------------------------------------------------

class TestRunnerResume:
    def test_markers_written_per_cell(self, tmp_path, fake_run_cell):
        p = _discrete_problem()
        cfg = _runner_cfg(p, [_SPEC_LW, _SPEC_VE], tmp_path)
        rows = list(Runner().run(cfg))
        assert len(rows) == 2
        marked = checkpoint.load_completed(
            checkpoint.completed_cells_path(cfg.jsonl_path))
        assert marked == {
            checkpoint.cell_key(p, "nbn-cat-lw"),
            checkpoint.cell_key(p, "nbn-cat-ve"),
        }

    def test_resume_skips_completed_cells(self, tmp_path, fake_run_cell):
        p = _discrete_problem()
        cfg = _runner_cfg(p, [_SPEC_LW, _SPEC_VE], tmp_path)
        list(Runner().run(cfg))
        assert len(fake_run_cell) == 2

        cfg2 = _runner_cfg(p, [_SPEC_LW, _SPEC_VE], tmp_path, resume=True)
        rows2 = list(Runner().run(cfg2))
        assert rows2 == []                    # nothing left to run
        assert len(fake_run_cell) == 2        # no cell re-ran
        # JSONL untouched: still exactly one row per cell.
        lines = cfg.jsonl_path.read_text().splitlines()
        assert len(lines) == 2

    def test_resume_compacts_partial_and_reruns_only_missing(
            self, tmp_path, fake_run_cell):
        import dataclasses
        p = _discrete_problem()
        cfg = _runner_cfg(p, [_SPEC_LW], tmp_path)
        list(Runner().run(cfg))
        assert len(fake_run_cell) == 1

        # Simulate a job killed mid-cell: partial rows for the VE cell are in
        # the JSONL but the cell was never marked complete.
        with open(cfg.jsonl_path, "a") as fh:
            d = dataclasses.asdict(_row(p, "nbn-cat-ve"))
            fh.write(json.dumps(d) + "\n")

        cfg2 = _runner_cfg(p, [_SPEC_LW, _SPEC_VE], tmp_path, resume=True)
        list(Runner().run(cfg2))
        # Only the VE cell ran on resume …
        assert len(fake_run_cell) == 2
        assert fake_run_cell[-1] == checkpoint.cell_key(p, "nbn-cat-ve")
        # … and the partial row was compacted, so no cell is duplicated.
        rows = [json.loads(line)
                for line in cfg.jsonl_path.read_text().splitlines()]
        assert sorted(r["baseline"] for r in rows) == [
            "nbn-cat-lw", "nbn-cat-ve"]

    def test_preemption_stops_between_cells(self, tmp_path, monkeypatch):
        p = _discrete_problem()

        def fake(self, cfg, problem, spec, writer, **kwargs):
            from benchmarking.core.config import build_adapter
            name = build_adapter(spec).name
            row = _row(problem, name)
            writer.write(row)
            # Signal arrives while the first cell is running.
            checkpoint._handle_stop_signal(signal.SIGUSR1, None)
            yield row

        monkeypatch.setattr(Runner, "_run_cell", fake)
        try:
            cfg = _runner_cfg(p, [_SPEC_LW, _SPEC_VE], tmp_path)
            runner = Runner()
            rows = list(runner.run(cfg))
        finally:
            checkpoint.reset_stop()
        # First cell finished and was marked; the loop stopped before the second.
        assert runner.preempted is True
        assert len(rows) == 1
        marked = checkpoint.load_completed(
            checkpoint.completed_cells_path(cfg.jsonl_path))
        assert marked == {checkpoint.cell_key(p, "nbn-cat-lw")}

    def test_preempted_exit_code_is_124(self):
        assert checkpoint.PREEMPTED_EXIT_CODE == 124


# ---------------------------------------------------------------------------
# TestCliResultsDir
# ---------------------------------------------------------------------------

class TestCliResultsDir:
    def test_no_dir_passthrough(self):
        from benchmarking.cli import _resolve_results_dir
        args = SimpleNamespace(results_dir=None, resume=False)
        assert _resolve_results_dir(args) == (None, False)

    def test_resume_requires_dir(self):
        from benchmarking.cli import _resolve_results_dir
        args = SimpleNamespace(results_dir=None, resume=True)
        with pytest.raises(SystemExit):
            _resolve_results_dir(args)

    def test_fresh_dir_ok(self, tmp_path):
        from benchmarking.cli import _resolve_results_dir
        d = tmp_path / "run"
        args = SimpleNamespace(results_dir=str(d), resume=False)
        jsonl, resume = _resolve_results_dir(args)
        assert jsonl == d / "metrics.jsonl"
        assert resume is False
        assert d.is_dir()

    def test_existing_jsonl_requires_resume(self, tmp_path):
        from benchmarking.cli import _resolve_results_dir
        d = tmp_path / "run"
        d.mkdir()
        (d / "metrics.jsonl").write_text("")
        args = SimpleNamespace(results_dir=str(d), resume=False)
        with pytest.raises(SystemExit):
            _resolve_results_dir(args)
        args.resume = True
        assert _resolve_results_dir(args) == (d / "metrics.jsonl", True)
