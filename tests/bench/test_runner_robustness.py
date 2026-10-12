"""Tests for v0.13 runner robustness (feat/v0.13-runner-robustness).

Covers the four fixes that let bnlearn_complete survive the failure mode that
silently killed the 2026-06-03 overnight run at munin1:

  TestBifUrlMapping              — munin1/2/3 resolve to the /munin4/ directory
  TestFailedProblem              — the sentinel dataclass
  TestIterProblemsResilience     — a failed load yields a sentinel and the
                                   generator stays alive for the next network
  TestRunnerFailedProblem        — runner emits error rows + continues
  TestUncaughtExceptionLogging   — a crash is logged to run.log before exit

Reference: investigation report 2026-06-04 (bnlearn_complete munin1 404).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import urllib.error
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch

from nbn.bench.core.config import BaselineSpec, RunnerConfig
from nbn.bench.core.runner import Runner
from nbn.bench.domains.base import BenchmarkProblem, FailedProblem
from nbn.bench.measurements import TimingOnly
from nbn.bench.problems.bnlearn import (
    _DOWNLOAD_TIMEOUT_S,
    BnlearnConfig,
    BnlearnProblemSource,
    _bif_url,
    _kind_to_family,
)
from nbn.bench.selectors.uniform import UniformRandomSelector


# --- 1. URL mapping ----------------------------------------------------------

class TestBifUrlMapping:
    """munin1/2/3 live under the /munin4/ directory on bnlearn.com (404 fix)."""

    @pytest.mark.parametrize("name, expected_dir", [
        ("asia", "asia"),
        ("link", "link"),
        ("munin", "munin"),     # the full network: per-name dir, unchanged
        ("munin4", "munin4"),   # coincidentally its own dir
        ("munin1", "munin4"),
        ("munin2", "munin4"),
        ("munin3", "munin4"),
    ])
    def test_directory_mapping(self, name, expected_dir):
        assert _bif_url(name) == (
            f"https://www.bnlearn.com/bnrepository/{expected_dir}/{name}.bif.gz"
        )

    def test_munin_partitions_use_munin4_dir(self):
        for n in ("munin1", "munin2", "munin3"):
            assert "/bnrepository/munin4/" in _bif_url(n)
            assert f"/{n}/" not in _bif_url(n)

    def test_download_timeout_plumbed(self):
        assert _DOWNLOAD_TIMEOUT_S == 30.0

    def test_kind_to_family(self):
        assert _kind_to_family("discrete") == "discrete"
        assert _kind_to_family("gaussian") == "continuous_gauss"
        assert _kind_to_family("clg") == "clg"


# --- 2. FailedProblem dataclass ---------------------------------------------

class TestFailedProblem:
    def test_basic_fields(self):
        fp = FailedProblem(problem_id="munin1", family="discrete",
                           error_msg="HTTPError: 404", benchmark="bnlearn")
        assert fp.problem_id == "munin1"
        assert fp.family == "discrete"
        assert fp.error_msg == "HTTPError: 404"
        assert fp.benchmark == "bnlearn"

    def test_frozen(self):
        fp = FailedProblem("a", "discrete", "e", "bnlearn")
        with pytest.raises(dataclasses.FrozenInstanceError):
            fp.problem_id = "b"  # type: ignore[misc]


# --- 3. iter_problems is resilient to per-network load failures --------------

class TestIterProblemsResilience:
    def test_yields_failed_problem_on_download_error(self, monkeypatch):
        def boom(name):
            raise urllib.error.HTTPError(_bif_url(name), 404, "Not Found", {}, None)
        monkeypatch.setattr(
            "nbn.bench.problems.bnlearn._load_discrete_model", boom)
        cfg = BnlearnConfig(networks=["asia"], seeds=[0],
                            n_train=10, n_test=5, n_reference=5)
        items = list(BnlearnProblemSource().iter_problems(cfg))
        assert len(items) == 1
        fp = items[0]
        assert isinstance(fp, FailedProblem)
        assert fp.problem_id == "asia"
        assert fp.family == "discrete"
        assert fp.benchmark == "bnlearn"
        assert "404" in fp.error_msg

    def test_generator_continues_after_failure(self, monkeypatch):
        """The critical property: one bad network does not stop the rest.

        Pre-fix this was impossible — the exception killed the generator and
        every later network was silently skipped.
        """
        def boom(name):
            raise RuntimeError(f"load failed for {name}")
        monkeypatch.setattr(
            "nbn.bench.problems.bnlearn._load_discrete_model", boom)
        cfg = BnlearnConfig(networks=["asia", "alarm", "child"], seeds=[0],
                            n_train=10, n_test=5, n_reference=5)
        items = list(BnlearnProblemSource().iter_problems(cfg))
        # All three failed, yet the generator reached every one, in order.
        assert [type(i) for i in items] == [FailedProblem, FailedProblem, FailedProblem]
        assert [i.problem_id for i in items] == ["asia", "alarm", "child"]
        assert all(i.error_msg.startswith("RuntimeError:") for i in items)

    def test_failure_then_success_continues(self, monkeypatch):
        """A failed network is followed by a *successfully loaded* one."""
        real = _minimal_discrete_problem(problem_id="good")

        def selective(name):
            if name == "asia":
                raise RuntimeError("boom on asia")
            raise AssertionError("should not load real model in this test")

        # asia fails at load; for "alarm" we bypass the loader entirely by
        # patching the per-network problem generator to yield a ready problem.
        monkeypatch.setattr(
            "nbn.bench.problems.bnlearn._load_discrete_model", selective)

        def fake_discrete(self, net_name, config):
            if net_name == "asia":
                # Exercise the real loader path -> raises -> caught upstream.
                from nbn.bench.problems.bnlearn import _load_discrete_model
                _load_discrete_model(net_name)
            yield real

        monkeypatch.setattr(BnlearnProblemSource, "_discrete_problems", fake_discrete)
        cfg = BnlearnConfig(networks=["asia", "alarm"], seeds=[0])
        items = list(BnlearnProblemSource().iter_problems(cfg))
        assert isinstance(items[0], FailedProblem)
        assert items[0].problem_id == "asia"
        assert isinstance(items[1], BenchmarkProblem)
        assert items[1].problem_id == "good"

    def test_unknown_network_still_fatal(self):
        """A typo'd network name stays a hard error — it's a config bug, not a
        transient load failure, so it should surface loudly."""
        cfg = BnlearnConfig(networks=["nope_not_real"], seeds=[0])
        with pytest.raises(ValueError, match="Unknown bnlearn network"):
            list(BnlearnProblemSource().iter_problems(cfg))


# --- 4. Runner turns FailedProblem into error rows and continues -------------

class _ScriptedSource:
    """Yields a pre-set list of problems / sentinels; ignores source_config."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def iter_problems(self, _config: Any) -> Iterator[Any]:
        yield from self._items


def _minimal_discrete_problem(problem_id: str = "4", seed: int = 0) -> BenchmarkProblem:
    nodes = ["X0", "X1", "X2", "X3"]
    return BenchmarkProblem(
        name=f"d_{problem_id}",
        dag=[("X0", "X1"), ("X1", "X2"), ("X2", "X3")],
        variables=dict.fromkeys(nodes, ("discrete", 2)),
        train_data={n: torch.randint(0, 2, (64,)) for n in nodes},
        test_data={n: torch.randint(0, 2, (16,)) for n in nodes},
        queries=[],
        family="discrete",
        problem_id=problem_id,
        seed=seed,
    )


def _minimal_continuous_problem(problem_id: str = "4", seed: int = 0) -> BenchmarkProblem:
    nodes = ["X0", "X1", "X2", "X3"]
    return BenchmarkProblem(
        name=f"clg_{problem_id}",
        dag=[("X0", "X1"), ("X1", "X2"), ("X2", "X3")],
        variables=dict.fromkeys(nodes, ("continuous", 1)),
        train_data={n: torch.randn(64) for n in nodes},
        test_data={n: torch.randn(16) for n in nodes},
        queries=[],
        family="continuous_lg",
        problem_id=problem_id,
        seed=seed,
    )


class TestRunnerFailedProblem:
    def _cfg(self, items, baselines, tmp_path) -> RunnerConfig:
        return RunnerConfig(
            benchmark="bnlearn",
            config_name="test",
            problem_source=_ScriptedSource(items),
            source_config=None,
            selector=UniformRandomSelector(),
            measurement=TimingOnly(),
            baselines=baselines,
            n_queries_per_cell=2,
            per_cell_timeout_s=60.0,
            fit_timeout_s=1000.0,
            jsonl_path=tmp_path / "out.jsonl",
        )

    def test_emits_one_error_row_per_baseline(self, tmp_path):
        fp = FailedProblem(problem_id="munin1", family="discrete",
                           error_msg="HTTPError: HTTP Error 404: Not Found",
                           benchmark="bnlearn")
        baselines = [
            BaselineSpec(library="pgmpy", mechanism="discrete",
                         param_method="mle", inference_method="ve"),
            BaselineSpec(library="nbn", mechanism="cat",
                         param_method="mle", inference_method="ve"),
        ]
        rows = list(Runner().run(self._cfg([fp], baselines, tmp_path)))
        assert len(rows) == 2
        for r in rows:
            assert r.status == "error"
            assert r.problem_id == "munin1"
            assert r.family == "discrete"
            assert r.benchmark == "bnlearn"
            assert r.seed == -1
            assert r.metric == "status"
            assert math.isnan(r.value)
            assert r.error_msg.startswith("problem load failed:")
            assert "404" in r.error_msg
        assert {r.baseline for r in rows} == {"pgmpy-mle-ve", "nbn-cat-ve"}

    def test_continues_to_real_problem_after_failure(self, tmp_path):
        fp = FailedProblem("munin1", "discrete", "RuntimeError: boom", "bnlearn")
        real = _minimal_continuous_problem(problem_id="realprob")
        # pgmpy-discrete is not applicable to continuous_lg, so the real cell
        # yields a fast not_supported row (no actual fit) — enough to prove the
        # runner advanced past the FailedProblem.
        spec = BaselineSpec(library="pgmpy", mechanism="discrete",
                            param_method="mle", inference_method="ve")
        rows = list(Runner().run(self._cfg([fp, real], [spec], tmp_path)))
        by_problem: dict[str, set[str]] = {}
        for r in rows:
            by_problem.setdefault(r.problem_id, set()).add(r.status)
        assert "error" in by_problem["munin1"]
        assert "realprob" in by_problem          # reached the real problem
        assert "not_supported" in by_problem["realprob"]

    def test_error_rows_written_to_jsonl(self, tmp_path):
        fp = FailedProblem("munin2", "discrete", "RuntimeError: boom", "bnlearn")
        spec = BaselineSpec(library="pgmpy", mechanism="discrete",
                            param_method="mle", inference_method="ve")
        cfg = self._cfg([fp], [spec], tmp_path)
        list(Runner().run(cfg))
        lines = cfg.jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["status"] == "error"
        assert row["problem_id"] == "munin2"
        assert row["seed"] == -1


# --- 5. Uncaught exceptions reach run.log before the handler detaches --------

class TestUncaughtExceptionLogging:
    _MINIMAL_CFG = """
version: "v0.13"
benchmark: synthetic
config_name: crashtest
metrics: timing
source:
  families: [discrete]
  n_nodes_list: [5]
  seeds: [0]
  n_train: 64
  n_test: 16
  n_reference: 64
  edge_density: 0.2
  max_in_degree: 2
  cardinality: 2
  fraction_continuous: 0.0
baselines:
  - {library: pgmpy, mechanism: discrete, param_method: mle, inference_method: ve}
n_queries_per_cell: 2
per_cell_timeout_s: 30.0
"""

    def test_unhandled_exception_routed_to_run_log(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "crash.yaml"
        cfg_path.write_text(self._MINIMAL_CFG)
        run_log = tmp_path / "run.log"

        # Redirect the run.log FileHandler into tmp (don't touch the repo
        # results dir), mirroring cli._attach_run_log's setup.
        def fake_attach(_results_dir):
            h = logging.FileHandler(run_log, mode="w", encoding="utf-8")
            h.setLevel(logging.INFO)
            h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            logging.getLogger().addHandler(h)
            return h
        monkeypatch.setattr("nbn.bench.cli._attach_run_log", fake_attach)

        # Force a crash inside the cell loop.
        def boom(self, cfg):
            if False:  # pragma: no cover — makes boom a generator (never yields)
                yield
            raise RuntimeError("synthetic explosion")
        monkeypatch.setattr("nbn.bench.core.runner.Runner.run", boom)

        from nbn.bench import cli
        with pytest.raises(RuntimeError, match="synthetic explosion"):
            cli.main(["inference", "--config", str(cfg_path)])

        assert run_log.exists()
        contents = run_log.read_text()
        assert "Unhandled exception during inference run" in contents
        # The traceback (exc_info) carries the original message.
        assert "synthetic explosion" in contents
        assert "Traceback" in contents
