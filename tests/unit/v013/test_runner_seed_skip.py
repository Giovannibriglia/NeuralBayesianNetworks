"""Tests for the speed-benchmark execution-level seed-skip (#148 PR 2/2).

Once a (family, problem_id, baseline, batch_size) config fails on one seed,
the runner records skip sentinels for its remaining seeds instead of re-running
them. Speed-benchmark-only (gated on cfg.batch_sizes); accuracy runs untouched.

Two layers:
  * the pure registry helpers (_register_failures, _skip_sentinel) — all the
    failure-shape logic, fast and subprocess-free;
  * Runner().run() integration with _run_cell monkeypatched to canned rows, so
    the loop-level orchestration (gating, skip emission, registry threading) is
    exercised without spawning real cell subprocesses.

Reference: docs/v0.14-batched-queries-design.md; PR #189 (reporting side).
"""
from __future__ import annotations

from typing import Iterator

import pytest
import torch

from benchmarking.core.config import BaselineSpec, RunnerConfig, build_adapter
from benchmarking.core.results import CellResult
from benchmarking.core.runner import (
    Runner,
    _register_failures,
    _skip_sentinel,
)
from benchmarking.domains.base import BenchmarkProblem, GroundTruth
from benchmarking.measurements import TimingOnly
from benchmarking.selectors.uniform import UniformRandomSelector


# --- fixtures ----------------------------------------------------------------

def _problem(seed: int, *, family: str = "discrete", n_nodes: int = 4) -> BenchmarkProblem:
    nodes = [f"X{i}" for i in range(n_nodes)]
    return BenchmarkProblem(
        name=f"{family}_n{n_nodes}_seed{seed}",
        dag=[(nodes[i], nodes[i + 1]) for i in range(n_nodes - 1)],
        variables=dict.fromkeys(nodes, ("discrete", 4)),
        train_data={n: torch.randint(0, 4, (64,)).float() for n in nodes},
        test_data={n: torch.randint(0, 4, (16,)).float() for n in nodes},
        queries=[],
        ground_truth=GroundTruth(),
        family=family,
        problem_id=str(n_nodes),
        seed=seed,
    )


def _row(*, baseline="nbn-cat-ve", batch_size=1, seed=0, status="ok",
         metric="query_time_s", value=0.01, family="discrete", problem_id="4"):
    return CellResult(
        benchmark="synthetic", family=family, problem_id=problem_id, seed=seed,
        baseline=baseline, query_role="hub", metric=metric,
        value=(float("nan") if status != "ok" else value), status=status,
        fit_time_s=0.1, query_time_s=value, metrics_time_s=0.0,
        batch_size=batch_size,
    )


# --- pure helper: _register_failures -----------------------------------------

class TestRegisterFailures:
    def _key(self, b, name="nbn-cat-ve"):
        return ("discrete", "4", name, b)

    def test_per_batch_size_only_failed_registered(self):
        reg = {}
        rows = [_row(batch_size=1, status="ok"),
                _row(batch_size=512, status="oom")]
        _register_failures(reg, _problem(0), "nbn-cat-ve", [1, 512], rows)
        assert reg == {self._key(512): "oom"}      # 1 ran ok -> not registered

    def test_fit_failure_registers_all(self):
        reg = {}
        rows = [_row(batch_size=1, status="error"),
                _row(batch_size=512, status="error")]
        _register_failures(reg, _problem(0), "nbn-cat-ve", [1, 512], rows)
        assert reg == {self._key(1): "error", self._key(512): "error"}

    def test_wholesale_death_registers_all_via_fallback(self):
        # Subprocess SIGKILL: one synthesized oom row at B=1, nothing for B=512.
        reg = {}
        rows = [_row(batch_size=1, status="oom", metric="status")]
        _register_failures(reg, _problem(0), "nbn-cat-ve", [1, 512], rows)
        assert reg == {self._key(1): "oom", self._key(512): "oom"}

    def test_partial_failure_at_batch_size_registers_it(self):
        # B=512 has both an ok group and a timed-out group this seed -> failed
        # (matches PR #189 "any failure row -> config failed"); B=1 ran clean.
        reg = {}
        rows = [_row(batch_size=1, status="ok"),
                _row(batch_size=512, status="ok"),
                _row(batch_size=512, status="timeout")]
        _register_failures(reg, _problem(0), "nbn-cat-ve", [1, 512], rows)
        assert reg == {self._key(512): "timeout"}

    def test_not_supported_registers_nothing(self):
        reg = {}
        rows = [_row(batch_size=1, status="not_supported", metric="status"),
                _row(batch_size=512, status="not_supported", metric="status")]
        _register_failures(reg, _problem(0), "nbn-cat-ve", [1, 512], rows)
        assert reg == {}

    def test_all_ok_registers_nothing(self):
        reg = {}
        rows = [_row(batch_size=1, status="ok"), _row(batch_size=512, status="ok")]
        _register_failures(reg, _problem(0), "nbn-cat-ve", [1, 512], rows)
        assert reg == {}


# --- pure helper: _skip_sentinel ---------------------------------------------

class TestSkipSentinel:
    def test_row_shape(self):
        r = _skip_sentinel(_problem(3), "nbn-cat-ve", "synthetic", 512, "oom",
                           device="cpu")
        assert r.status == "oom" and r.batch_size == 512 and r.seed == 3
        assert r.family == "discrete" and r.problem_id == "4"
        assert r.baseline == "nbn-cat-ve"
        assert "skipped" in r.error_msg


# --- run() integration (subprocess-free via _run_cell monkeypatch) -----------

class _MultiSeedSource:
    def __init__(self, problems):
        self._problems = problems

    def iter_problems(self, _cfg) -> Iterator[BenchmarkProblem]:
        yield from self._problems


def _speed_cfg(tmp_path, problems, *, batch_sizes):
    return RunnerConfig(
        benchmark="synthetic",
        config_name="batch_speed",
        problem_source=_MultiSeedSource(problems),
        source_config=None,
        selector=UniformRandomSelector(),
        measurement=TimingOnly(),
        baselines=[BaselineSpec(library="nbn", mechanism="cat",
                                param_method="mle", inference_method="ve")],
        n_queries_per_cell=2,
        per_cell_timeout_s=60.0,
        batch_sizes=batch_sizes,
        jsonl_path=tmp_path / "out.jsonl",
    )


def _stub_run_cell(calls, fail_plan):
    """Replacement for Runner._run_cell: records the batch_sizes it was asked to
    run per seed and yields canned rows (oom for fail_plan[seed], else ok)."""
    def _stub(self, cfg, problem, spec, writer, *, fit_budget_s,
             default_role, batch_sizes=None):
        calls.append((problem.seed, tuple(batch_sizes)))
        name = build_adapter(spec).name
        fails = fail_plan.get(problem.seed, set())
        for b in batch_sizes:
            st = "oom" if b in fails else "ok"
            yield _row(baseline=name, batch_size=b, seed=problem.seed,
                       status=st, family=problem.family,
                       problem_id=problem.problem_id)
    return _stub


class TestRunSeedSkip:
    def test_failed_config_skips_later_seed(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(Runner, "_run_cell",
                            _stub_run_cell(calls, {0: {512}}))
        cfg = _speed_cfg(tmp_path, [_problem(0), _problem(1)],
                         batch_sizes=[1, 512])
        rows = list(Runner().run(cfg))

        # seed 0 ran both; seed 1 ran only the live B=1 (512 skipped).
        assert (0, (1, 512)) in calls
        assert (1, (1,)) in calls
        # seed 1 emitted exactly one skip sentinel for B=512 with the code.
        skips = [r for r in rows if r.seed == 1 and r.batch_size == 512
                 and "skipped" in (r.error_msg or "")]
        assert len(skips) == 1 and skips[0].status == "oom"
        # seed 1's live B=1 still actually ran (ok row present).
        assert any(r.seed == 1 and r.batch_size == 1 and r.status == "ok"
                   for r in rows)

    def test_whole_cell_failure_skips_entire_later_seed(self, tmp_path, monkeypatch):
        calls = []
        # seed 0 fails at BOTH batch sizes (fit-failure shape).
        monkeypatch.setattr(Runner, "_run_cell",
                            _stub_run_cell(calls, {0: {1, 512}}))
        cfg = _speed_cfg(tmp_path, [_problem(0), _problem(1)],
                         batch_sizes=[1, 512])
        rows = list(Runner().run(cfg))

        seed1_calls = [c for c in calls if c[0] == 1]
        assert seed1_calls == []          # subprocess never invoked for seed 1
        skip_bs = {r.batch_size for r in rows if r.seed == 1
                   and "skipped" in (r.error_msg or "")}
        assert skip_bs == {1, 512}

    def test_accuracy_run_never_skips(self, tmp_path, monkeypatch):
        """cfg.batch_sizes=None (accuracy benchmark) -> gating off: seed 1 runs
        even though seed 0 failed, and no skip sentinels are emitted."""
        calls = []
        monkeypatch.setattr(Runner, "_run_cell",
                            _stub_run_cell(calls, {0: {1}}))
        cfg = _speed_cfg(tmp_path, [_problem(0), _problem(1)], batch_sizes=None)
        rows = list(Runner().run(cfg))

        # No sweep -> resolved batch_sizes is [1]; both seeds run, none skipped.
        assert (0, (1,)) in calls and (1, (1,)) in calls
        assert not any("skipped" in (r.error_msg or "") for r in rows)


@pytest.mark.slow
class TestRunSeedSkipRealSubprocess:
    """End-to-end through the real cell subprocess (no monkeypatch). A
    near-zero fit_timeout_s forces the fit safety net to fail every batch size
    on seed 0; seed 1 must then be skipped entirely."""

    def test_skip_fires_through_subprocess(self, tmp_path):
        from benchmarking.problems.synthetic import (
            SyntheticConfig, SyntheticProblemSource,
        )

        cfg = RunnerConfig(
            benchmark="synthetic",
            config_name="batch_speed",
            problem_source=SyntheticProblemSource(),
            source_config=SyntheticConfig(
                families=["discrete"], n_nodes_list=[6], seeds=[0, 1],
                n_train=128, n_test=32, n_reference=128, cardinality=2,
            ),
            selector=UniformRandomSelector(),
            measurement=TimingOnly(),
            baselines=[BaselineSpec(library="nbn", mechanism="cat",
                                    param_method="mle", inference_method="ve")],
            n_queries_per_cell=2,
            per_cell_timeout_s=60.0,
            fit_timeout_s=1e-6,          # any real fit exceeds this -> timeout
            batch_sizes=[1, 4],
            jsonl_path=tmp_path / "out.jsonl",
        )
        rows = list(Runner().run(cfg))

        # seed 0 actually ran and the cell failed (fit safety-net timeout).
        # (Fit-failure sentinels are stamped batch_size=1 — a pre-existing
        # quirk; the registry still covers B=4 via its wholesale fallback.)
        assert any(r.seed == 0 and r.status in ("timeout", "oom", "error")
                   for r in rows)
        assert not any(r.seed == 0 and "skipped" in (r.error_msg or "")
                       for r in rows)
        # seed 1 is all skip sentinels at BOTH batch sizes — never executed.
        seed1 = [r for r in rows if r.seed == 1]
        assert seed1 and all("skipped" in (r.error_msg or "") for r in seed1)
        assert {r.batch_size for r in seed1} == {1, 4}
