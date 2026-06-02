"""Tests for the v0.13 runner orchestrator (Phase 1b-v).

Covers:
  TestBaselineSpec       — dataclass fields and defaults
  TestRunnerConfig       — dataclass construction and defaults
  TestBuildAdapter       — dispatch to the correct v0.13 adapter
  TestClassifyException  — exception → status mapping
  TestJsonlWriter        — write / read / context-manager
  TestNotSupportedPath   — is_applicable()=False → not_supported sentinel
  TestFitFailurePath     — fit exception → error/oom rows
  TestEndToEnd           — full matrix behavioral tests (@pytest.mark.slow)
    Discrete:          nbn-cat-lw, nbn-cat-ve, nbn-neuralcat-lw, nbn-neuralcat-ve,
                       pgmpy-mle-ve, pgmpy-bayes-ve, pomegranate-discrete-ve,
                       pyro-empirical-importance
    Continuous_lg:     nbn-lg-lw, nbn-mdn-lw, nbn-flow-lw, pgmpy-lg-predict,
                       pyro-empirical-importance
    Continuous_nongauss: nbn-mdn-lw, nbn-flow-lw
    Hybrid:            nbn-hybrid-router, pyro-empirical-importance

Reference: docs/v0.13-benchmark-redesign.md §3, §4.1, §6
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch

from benchmarking.adapters import NBNAdapter, PgmpyAdapter, PomegranateAdapter, PyroAdapter
from benchmarking.core.config import BaselineSpec, RunnerConfig, build_adapter
from benchmarking.core.output import JsonlWriter
from benchmarking.core.results import CellResult, VALID_STATUSES
from benchmarking.core.runner import Runner, _classify_exception, _is_structural_limit
from benchmarking.domains.base import BenchmarkProblem, GroundTruth
from benchmarking.measurements import AccuracyAndTiming, TimingOnly
from benchmarking.problems.synthetic import SyntheticConfig, SyntheticProblemSource
from benchmarking.selectors.uniform import UniformRandomSelector


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_N_QUERIES = 4
_FIT_EPOCHS = 5


class _FixedProblemSource:
    """Yields a single pre-built problem; source_config IS the problem."""

    def iter_problems(self, problem: BenchmarkProblem) -> Iterator[BenchmarkProblem]:
        yield problem


def _minimal_continuous_problem(n_nodes: int = 4, seed: int = 0) -> BenchmarkProblem:
    """Hand-crafted continuous_lg problem — no make_synthetic_bn needed."""
    nodes = [f"X{i}" for i in range(n_nodes)]
    return BenchmarkProblem(
        name=f"continuous_lg_n{n_nodes}_seed{seed}",
        dag=[(nodes[i], nodes[i + 1]) for i in range(n_nodes - 1)],
        variables=dict.fromkeys(nodes, ("continuous", 1)),
        train_data={n: torch.randn(100) for n in nodes},
        test_data={n: torch.randn(20) for n in nodes},
        queries=[],
        family="continuous_lg",
        problem_id=str(n_nodes),
        seed=seed,
    )


def _minimal_discrete_problem(n_nodes: int = 4, seed: int = 0) -> BenchmarkProblem:
    """Hand-crafted discrete problem — no make_synthetic_bn needed."""
    nodes = [f"X{i}" for i in range(n_nodes)]
    return BenchmarkProblem(
        name=f"discrete_n{n_nodes}_seed{seed}",
        dag=[(nodes[i], nodes[i + 1]) for i in range(n_nodes - 1)],
        variables=dict.fromkeys(nodes, ("discrete", 4)),
        train_data={n: torch.randint(0, 4, (100,)).float() for n in nodes},
        test_data={n: torch.randint(0, 4, (20,)).float() for n in nodes},
        queries=[],
        ground_truth=GroundTruth(),
        family="discrete",
        problem_id=str(n_nodes),
        seed=seed,
    )


def _make_source_cfg(
    family: str, n_nodes: int = 4, seed: int = 0,
) -> tuple[SyntheticProblemSource, SyntheticConfig]:
    src = SyntheticProblemSource()
    cfg = SyntheticConfig(
        families=[family],
        n_nodes_list=[n_nodes],
        seeds=[seed],
        n_train=200,
        n_test=50,
        n_reference=500,
    )
    return src, cfg


def _make_runner_cfg(
    problem: BenchmarkProblem,
    spec: BaselineSpec,
    tmp_path: Path,
    *,
    per_cell_timeout_s: float = 120.0,
    measurement: Any = None,
    fit_timeout_s_multiplier: float = 10.0,
    config_name: str = "test",
) -> RunnerConfig:
    if measurement is None:
        measurement = TimingOnly()
    return RunnerConfig(
        benchmark="synthetic",
        config_name=config_name,
        problem_source=_FixedProblemSource(),
        source_config=problem,
        selector=UniformRandomSelector(),
        measurement=measurement,
        baselines=[spec],
        n_queries_per_cell=_N_QUERIES,
        per_cell_timeout_s=per_cell_timeout_s,
        fit_timeout_s_multiplier=fit_timeout_s_multiplier,
        jsonl_path=tmp_path / "out.jsonl",
    )


def _build_test_ctx(
    problem: BenchmarkProblem,
    spec: BaselineSpec,
    *,
    selector: Any = None,
    measurement: Any = None,
    benchmark: str = "synthetic",
    n_queries_per_cell: int = _N_QUERIES,
    per_cell_timeout_s: float = 120.0,
    fit_budget_s: float = 1200.0,
    default_role: str = "random",
) -> dict:
    """Build the ctx dict that Runner._run_cell passes to the subprocess.

    Bug 4 Stage 2: the per-cell work now runs in cell_worker._run_cell.
    Tests that monkeypatch cell internals (adapter.fit, build_adapter, a
    spy measurement) call _run_cell directly in-process — where the
    patches/spies apply — instead of going through Runner().run(), which
    would execute the cell in a subprocess that never sees them.
    """
    return {
        "problem": problem,
        "baseline_spec": spec,
        "seed": problem.seed,
        "selector": selector or UniformRandomSelector(),
        "measurement": measurement or TimingOnly(),
        "benchmark": benchmark,
        "n_queries_per_cell": n_queries_per_cell,
        "per_cell_timeout_s": per_cell_timeout_s,
        "fit_budget_s": fit_budget_s,
        "default_role": default_role,
    }


# ---------------------------------------------------------------------------
# TestBaselineSpec
# ---------------------------------------------------------------------------

class TestBaselineSpec:
    def test_required_fields(self):
        spec = BaselineSpec(
            library="nbn", mechanism="cat", param_method="mle", inference_method="lw"
        )
        assert spec.library == "nbn"
        assert spec.mechanism == "cat"
        assert spec.param_method == "mle"
        assert spec.inference_method == "lw"

    def test_device_defaults_none(self):
        spec = BaselineSpec(library="pgmpy", mechanism="discrete", param_method="mle")
        assert spec.device is None

    def test_inference_method_defaults_none(self):
        spec = BaselineSpec(library="pomegranate", mechanism="discrete", param_method="mle")
        assert spec.inference_method is None

    def test_extra_kwargs_defaults_empty(self):
        spec = BaselineSpec(library="nbn", mechanism="cat", param_method="mle", inference_method="lw")
        assert spec.extra_kwargs == {}

    def test_extra_kwargs_custom(self):
        spec = BaselineSpec(
            library="nbn", mechanism="cat", param_method="mle",
            inference_method="lw", extra_kwargs={"n_samples": 512},
        )
        assert spec.extra_kwargs["n_samples"] == 512


# ---------------------------------------------------------------------------
# TestRunnerConfig
# ---------------------------------------------------------------------------

class TestRunnerConfig:
    def test_builds_correctly(self, tmp_path):
        src, src_cfg = _make_source_cfg("discrete")
        cfg = RunnerConfig(
            benchmark="synthetic",
            config_name="test",
            problem_source=src,
            source_config=src_cfg,
            selector=UniformRandomSelector(),
            measurement=TimingOnly(),
            baselines=[BaselineSpec("nbn", "cat", "mle", "lw")],
            n_queries_per_cell=4,
            per_cell_timeout_s=60.0,
            jsonl_path=tmp_path / "out.jsonl",
        )
        assert cfg.benchmark == "synthetic"
        assert cfg.n_queries_per_cell == 4
        assert cfg.fit_timeout_s_multiplier == 10.0

    def test_fit_timeout_s_multiplier_custom(self, tmp_path):
        src, src_cfg = _make_source_cfg("discrete")
        cfg = RunnerConfig(
            benchmark="synthetic",
            config_name="test",
            problem_source=src,
            source_config=src_cfg,
            selector=UniformRandomSelector(),
            measurement=TimingOnly(),
            baselines=[],
            n_queries_per_cell=4,
            per_cell_timeout_s=60.0,
            fit_timeout_s_multiplier=5.0,
            jsonl_path=tmp_path / "out.jsonl",
        )
        assert cfg.fit_timeout_s_multiplier == 5.0

    def test_source_config_held_separately(self, tmp_path):
        src, src_cfg = _make_source_cfg("discrete")
        cfg = RunnerConfig(
            benchmark="synthetic",
            config_name="test",
            problem_source=src,
            source_config=src_cfg,
            selector=UniformRandomSelector(),
            measurement=TimingOnly(),
            baselines=[],
            n_queries_per_cell=4,
            per_cell_timeout_s=60.0,
            jsonl_path=tmp_path / "out.jsonl",
        )
        assert cfg.problem_source is src
        assert cfg.source_config is src_cfg


# ---------------------------------------------------------------------------
# TestBuildAdapter
# ---------------------------------------------------------------------------

class TestBuildAdapter:
    def test_nbn_cat_lw(self):
        spec = BaselineSpec("nbn", "cat", "mle", "lw")
        adapter = build_adapter(spec)
        assert isinstance(adapter, NBNAdapter)
        assert adapter.name == "nbn-cat-lw"

    def test_nbn_mdn_lw(self):
        spec = BaselineSpec("nbn", "mdn", "mle", "lw")
        adapter = build_adapter(spec)
        assert isinstance(adapter, NBNAdapter)
        assert adapter.name == "nbn-mdn-lw"

    def test_nbn_hybrid_router(self):
        spec = BaselineSpec("nbn", "hybrid", "mle", "router")
        adapter = build_adapter(spec)
        assert isinstance(adapter, NBNAdapter)
        assert adapter.name == "nbn-hybrid-router"

    def test_pgmpy_mle_ve(self):
        spec = BaselineSpec("pgmpy", "discrete", "mle", "ve")
        adapter = build_adapter(spec)
        assert isinstance(adapter, PgmpyAdapter)
        assert adapter.name == "pgmpy-mle-ve"

    def test_pgmpy_lg_predict(self):
        spec = BaselineSpec("pgmpy", "lg", "lg", "predict")
        adapter = build_adapter(spec)
        assert isinstance(adapter, PgmpyAdapter)
        assert adapter.name == "pgmpy-lg-predict"

    def test_pomegranate(self):
        spec = BaselineSpec("pomegranate", "discrete", "mle")
        adapter = build_adapter(spec)
        assert isinstance(adapter, PomegranateAdapter)
        assert adapter.name == "pomegranate-discrete-ve"

    def test_pyro(self):
        spec = BaselineSpec("pyro", "empirical", "empirical", "importance")
        adapter = build_adapter(spec)
        assert isinstance(adapter, PyroAdapter)
        assert adapter.name == "pyro-empirical-importance"

    def test_nbn_device_propagated(self):
        spec = BaselineSpec("nbn", "cat", "mle", "lw", device="cpu")
        adapter = build_adapter(spec)
        assert str(adapter.device) == "cpu"

    def test_nbn_missing_inference_method_raises(self):
        spec = BaselineSpec("nbn", "cat", "mle", inference_method=None)
        with pytest.raises(ValueError, match="inference_method"):
            build_adapter(spec)

    def test_pgmpy_missing_inference_method_raises(self):
        spec = BaselineSpec("pgmpy", "discrete", "mle", inference_method=None)
        with pytest.raises(ValueError, match="inference_method"):
            build_adapter(spec)

    def test_unknown_library_raises(self):
        spec = BaselineSpec("unknown", "cat", "mle", "lw")
        with pytest.raises(ValueError, match="Unknown library"):
            build_adapter(spec)

    def test_extra_kwargs_forwarded_to_nbn(self):
        spec = BaselineSpec("nbn", "cat", "mle", "lw", extra_kwargs={"n_samples": 256})
        adapter = build_adapter(spec)
        assert adapter.n_samples == 256


# ---------------------------------------------------------------------------
# TestClassifyException
# ---------------------------------------------------------------------------

class TestClassifyException:
    def test_memory_error_is_oom(self):
        assert _classify_exception(MemoryError()) == "oom"

    def test_runtime_error_out_of_memory_is_oom(self):
        assert _classify_exception(RuntimeError("CUDA out of memory.")) == "oom"

    def test_runtime_error_alloc_cpu_is_oom(self):
        assert _classify_exception(RuntimeError("alloc_cpu failed: cannot alloc")) == "oom"

    def test_runtime_error_defaultcpuallocator_is_oom(self):
        assert _classify_exception(RuntimeError("DefaultCPUAllocator: not enough")) == "oom"

    def test_runtime_error_cannot_allocate_is_oom(self):
        assert _classify_exception(RuntimeError("cannot allocate memory")) == "oom"

    def test_runtime_error_cuda_oom_is_oom(self):
        assert _classify_exception(RuntimeError("cuda oom on device 0")) == "oom"

    def test_runtime_error_other_is_error(self):
        assert _classify_exception(RuntimeError("some other failure")) == "error"

    def test_import_error_is_not_supported(self):
        assert _classify_exception(ImportError("zuko not installed")) == "not_supported"

    def test_not_implemented_error_is_not_supported(self):
        assert _classify_exception(NotImplementedError("not implemented")) == "not_supported"

    # -- ValueError: marker-based classification (#117) -----------------------

    def test_value_error_with_structural_marker_is_not_supported(self):
        """ValueError containing a structural-limit marker → not_supported."""
        assert _classify_exception(ValueError("is discrete-only adapter")) == "not_supported"

    def test_value_error_with_only_supports_marker_is_not_supported(self):
        assert _classify_exception(ValueError("only supports discrete families")) == "not_supported"

    def test_value_error_with_cannot_condition_marker_is_not_supported(self):
        assert _classify_exception(ValueError("cannot condition on continuous evidence")) == "not_supported"

    def test_value_error_with_not_applicable_marker_is_not_supported(self):
        assert _classify_exception(ValueError("not applicable to continuous_nongauss")) == "not_supported"

    def test_value_error_without_marker_is_error(self):
        """ValueError without a structural-limit marker → error (real bug, #117)."""
        assert _classify_exception(ValueError("invalid tensor shape [3, 4]")) == "error"

    def test_value_error_index_out_of_range_is_error(self):
        """Generic IndexError / ValueError without marker → error."""
        assert _classify_exception(ValueError("index 5 out of bounds for axis 0")) == "error"

    # -- _is_structural_limit helper ------------------------------------------

    def test_is_structural_limit_true_for_not_implemented_error(self):
        # _is_structural_limit only handles ValueError; NotImplementedError
        # is caught separately — but the function must return False for it.
        assert not _is_structural_limit(NotImplementedError("n/a"))

    def test_is_structural_limit_true_for_marker_value_error(self):
        assert _is_structural_limit(ValueError("is discrete-only"))
        assert _is_structural_limit(ValueError("only supports discrete"))
        assert _is_structural_limit(ValueError("refused"))

    def test_is_structural_limit_false_for_generic_value_error(self):
        assert not _is_structural_limit(ValueError("tensor shape mismatch"))

    def test_generic_exception_is_error(self):
        assert _classify_exception(Exception("generic")) == "error"

    def test_torch_cuda_oom_if_available(self):
        try:
            import torch
            exc = torch.cuda.OutOfMemoryError("CUDA OOM")
            assert _classify_exception(exc) == "oom"
        except (ImportError, AttributeError):
            pytest.skip("torch.cuda.OutOfMemoryError not available in this environment")


# ---------------------------------------------------------------------------
# TestJsonlWriter
# ---------------------------------------------------------------------------

class TestJsonlWriter:
    def _dummy_row(self, idx: int = 0) -> CellResult:
        return CellResult(
            benchmark="synthetic",
            family="discrete",
            problem_id="4",
            seed=0,
            baseline="nbn-cat-lw",
            query_role="random",
            metric="tv_per_node",
            value=0.12 + idx * 0.01,
            status="ok",
            fit_time_s=0.5,
            query_time_s=0.01 + idx * 0.001,
            metrics_time_s=0.05,
        )

    def test_writes_one_line_per_row(self, tmp_path):
        path = tmp_path / "out.jsonl"
        with JsonlWriter(path) as w:
            w.write(self._dummy_row(0))
            w.write(self._dummy_row(1))
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_each_line_is_valid_json(self, tmp_path):
        path = tmp_path / "out.jsonl"
        with JsonlWriter(path) as w:
            w.write(self._dummy_row())
        d = json.loads(path.read_text().strip())
        assert d["benchmark"] == "synthetic"
        assert d["status"] == "ok"

    def test_nan_serialised_as_null(self, tmp_path):
        path = tmp_path / "out.jsonl"
        row = CellResult(
            benchmark="synthetic", family="discrete", problem_id="4", seed=0,
            baseline="nbn-cat-lw", query_role="random", metric="tv_per_node",
            value=float("nan"), status="timeout",
            fit_time_s=0.5, query_time_s=float("nan"), metrics_time_s=float("nan"),
        )
        with JsonlWriter(path) as w:
            w.write(row)
        raw = path.read_text()
        assert "null" in raw
        assert "NaN" not in raw

    def test_appends_on_existing_file(self, tmp_path):
        path = tmp_path / "out.jsonl"
        with JsonlWriter(path) as w:
            w.write(self._dummy_row(0))
        with JsonlWriter(path) as w:
            w.write(self._dummy_row(1))
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_parent_directory_created(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "out.jsonl"
        with JsonlWriter(path) as w:
            w.write(self._dummy_row())
        assert path.exists()

    def test_context_manager_closes_file(self, tmp_path):
        path = tmp_path / "out.jsonl"
        with JsonlWriter(path) as w:
            w.write(self._dummy_row())
        # File should be readable (closed properly)
        content = path.read_text()
        assert content.strip()


# ---------------------------------------------------------------------------
# TestNotSupportedPath (fast — uses hand-crafted problem, no synthesis)
# ---------------------------------------------------------------------------

class TestNotSupportedPath:
    def test_pgmpy_mle_ve_not_applicable_on_continuous_lg(self, tmp_path):
        """pgmpy-mle-ve is discrete-only → not_supported on continuous_lg."""
        problem = _minimal_continuous_problem()
        spec = BaselineSpec("pgmpy", "discrete", "mle", "ve")
        cfg = _make_runner_cfg(problem, spec, tmp_path)

        rows = list(Runner().run(cfg))

        assert len(rows) == 1
        assert rows[0].status == "not_supported"
        assert rows[0].baseline == "pgmpy-mle-ve"
        assert rows[0].fit_time_s == 0.0
        assert rows[0].query_time_s == 0.0

    def test_not_supported_written_to_jsonl(self, tmp_path):
        problem = _minimal_continuous_problem()
        spec = BaselineSpec("pgmpy", "discrete", "mle", "ve")
        cfg = _make_runner_cfg(problem, spec, tmp_path)

        list(Runner().run(cfg))

        assert cfg.jsonl_path.exists()
        lines = cfg.jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["status"] == "not_supported"

    def test_multiple_baselines_only_applicable_ones_run(self, tmp_path):
        """pgmpy-mle-ve not_supported; nbn-lg-lw applicable on continuous_lg."""
        problem = _minimal_continuous_problem()
        specs = [
            BaselineSpec("pgmpy", "discrete", "mle", "ve"),   # not applicable
            BaselineSpec("nbn", "lg", "mle", "lw"),            # applicable
        ]
        cfg = RunnerConfig(
            benchmark="synthetic",
            config_name="test",
            problem_source=_FixedProblemSource(),
            source_config=problem,
            selector=UniformRandomSelector(),
            measurement=TimingOnly(),
            baselines=specs,
            n_queries_per_cell=2,
            per_cell_timeout_s=120.0,
            jsonl_path=tmp_path / "out.jsonl",
        )
        rows = list(Runner().run(cfg))
        statuses = {r.status for r in rows}
        assert "not_supported" in statuses
        assert "ok" in statuses


# ---------------------------------------------------------------------------
# TestFitFailurePath (fast — monkeypatches adapter.fit)
# ---------------------------------------------------------------------------

class TestFitFailurePath:
    # Bug 4 Stage 2: these tests monkeypatch PgmpyAdapter.fit, which now
    # runs inside the cell subprocess.  An in-process patch can't reach a
    # subprocess, so they call cell_worker._run_cell(ctx) directly (the
    # same per-cell logic, in-process).  _run_cell returns a list of dicts
    # (dataclasses.asdict of each CellResult), so assertions use row[...]
    # rather than attribute access.  Behaviour/intent unchanged.

    def test_fit_runtime_error_emits_error_rows(self, monkeypatch):
        """RuntimeError in fit() → error rows for all selected queries."""
        from benchmarking.core.cell_worker import _run_cell

        problem = _minimal_discrete_problem()
        spec = BaselineSpec("pgmpy", "discrete", "mle", "ve")

        def bad_fit(self, problem, **kwargs):
            raise RuntimeError("simulated fit failure")

        monkeypatch.setattr(PgmpyAdapter, "fit", bad_fit)

        rows = _run_cell(_build_test_ctx(problem, spec))

        assert len(rows) == _N_QUERIES
        for r in rows:
            assert r["status"] == "error"
            assert math.isnan(r["fit_time_s"])
            assert math.isnan(r["query_time_s"])
            assert r["error_msg"] is not None
            assert "simulated fit failure" in r["error_msg"]

    def test_fit_memory_error_emits_oom_rows(self, monkeypatch):
        """MemoryError in fit() → oom rows."""
        from benchmarking.core.cell_worker import _run_cell

        problem = _minimal_discrete_problem()
        spec = BaselineSpec("pgmpy", "discrete", "mle", "ve")

        def oom_fit(self, problem, **kwargs):
            raise MemoryError("CPU out of memory")

        monkeypatch.setattr(PgmpyAdapter, "fit", oom_fit)

        rows = _run_cell(_build_test_ctx(problem, spec))

        assert len(rows) == _N_QUERIES
        for r in rows:
            assert r["status"] == "oom"

    def test_fit_failure_rows_written_to_jsonl(self, tmp_path, monkeypatch):
        problem = _minimal_discrete_problem()
        spec = BaselineSpec("pgmpy", "discrete", "mle", "ve")

        monkeypatch.setattr(PgmpyAdapter, "fit",
                            lambda self, p, **kw: (_ for _ in ()).throw(RuntimeError("x")))

        cfg = _make_runner_cfg(problem, spec, tmp_path)
        rows = list(Runner().run(cfg))

        lines = cfg.jsonl_path.read_text().strip().splitlines()
        assert len(lines) == len(rows)

    def test_fit_failure_fit_time_s_is_nan(self, monkeypatch):
        from benchmarking.core.cell_worker import _run_cell

        problem = _minimal_discrete_problem()
        spec = BaselineSpec("pgmpy", "discrete", "mle", "ve")
        monkeypatch.setattr(PgmpyAdapter, "fit",
                            lambda self, p, **kw: (_ for _ in ()).throw(RuntimeError("x")))
        rows = _run_cell(_build_test_ctx(problem, spec))
        for r in rows:
            assert math.isnan(r["fit_time_s"])


# ---------------------------------------------------------------------------
# TestEndToEnd — behavioral matrix (@pytest.mark.slow)
# ---------------------------------------------------------------------------

# Each entry: (adapter_id, family, BaselineSpec, requires_module_or_None)
_MATRIX = [
    # ---- Discrete ----
    ("nbn-cat-lw",              "discrete",           BaselineSpec("nbn", "cat",       "mle", "lw"),         None),
    ("nbn-cat-ve",              "discrete",           BaselineSpec("nbn", "cat",       "mle", "ve"),         None),
    ("nbn-neuralcat-lw",        "discrete",           BaselineSpec("nbn", "neuralcat", "mle", "lw"),         None),
    ("nbn-neuralcat-ve",        "discrete",           BaselineSpec("nbn", "neuralcat", "mle", "ve"),         None),
    ("pgmpy-mle-ve",            "discrete",           BaselineSpec("pgmpy", "discrete","mle", "ve"),         None),
    ("pgmpy-bayes-ve",          "discrete",           BaselineSpec("pgmpy", "discrete","bayes","ve"),        None),
    ("pomegranate-discrete-ve", "discrete",           BaselineSpec("pomegranate","discrete","mle"),          None),
    ("pyro-empirical-importance","discrete",          BaselineSpec("pyro","empirical","empirical","importance"), "pyro"),
    # ---- Continuous LG ----
    ("nbn-lg-lw",               "continuous_lg",      BaselineSpec("nbn", "lg",        "mle", "lw"),         None),
    ("nbn-mdn-lw",              "continuous_lg",      BaselineSpec("nbn", "mdn",       "mle", "lw"),         None),
    ("nbn-flow-lw",             "continuous_lg",      BaselineSpec("nbn", "flow",      "mle", "lw"),         "zuko"),
    ("pgmpy-lg-predict",        "continuous_lg",      BaselineSpec("pgmpy", "lg",      "lg",  "predict"),    None),
    ("pyro-empirical-importance","continuous_lg",     BaselineSpec("pyro","empirical","empirical","importance"), "pyro"),
    # ---- Continuous non-Gaussian ----
    ("nbn-mdn-lw",              "continuous_nongauss",BaselineSpec("nbn", "mdn",       "mle", "lw"),         None),
    ("nbn-flow-lw",             "continuous_nongauss",BaselineSpec("nbn", "flow",      "mle", "lw"),         "zuko"),
    # ---- Hybrid ----
    ("nbn-hybrid-router",       "hybrid",             BaselineSpec("nbn", "hybrid",    "mle", "router"),     None),
    ("pyro-empirical-importance","hybrid",            BaselineSpec("pyro","empirical","empirical","importance"), "pyro"),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "adapter_id,family,spec,requires",
    _MATRIX,
    ids=[f"{t[0]}@{t[1]}" for t in _MATRIX],
)
def test_end_to_end_cell(adapter_id, family, spec, requires, tmp_path):
    """Full runner pipeline: problem → adapter → measurement → JSONL."""
    if requires:
        pytest.importorskip(requires)

    src, src_cfg = _make_source_cfg(family, n_nodes=4, seed=0)
    measurement = AccuracyAndTiming(n_oracle_samples=100)
    cfg = RunnerConfig(
        benchmark="synthetic",
        config_name="e2e_test",
        problem_source=src,
        source_config=src_cfg,
        selector=UniformRandomSelector(),
        measurement=measurement,
        baselines=[spec],
        n_queries_per_cell=_N_QUERIES,
        per_cell_timeout_s=120.0,
        jsonl_path=tmp_path / f"{adapter_id}_{family}.jsonl",
    )

    rows = list(Runner().run(cfg))

    # JSONL written and line count matches
    assert cfg.jsonl_path.exists()
    lines = cfg.jsonl_path.read_text().strip().splitlines()
    assert len(lines) == len(rows)

    # Rows produced, all valid statuses
    assert len(rows) > 0
    for r in rows:
        assert r.status in VALID_STATUSES, f"Invalid status {r.status!r} in {r}"

    # No unexpected error/oom rows
    bad_rows = [r for r in rows if r.status in ("error", "oom")]
    assert len(bad_rows) == 0, (
        f"Got error/oom rows for {adapter_id}@{family}: "
        + str([(r.status, r.error_msg) for r in bad_rows[:3]])
    )

    # fit_time_s identical across all rows (per-cell cost)
    fit_times = {r.fit_time_s for r in rows}
    assert len(fit_times) == 1, f"fit_time_s not identical: {fit_times}"
    assert next(iter(fit_times)) > 0.0, "fit_time_s should be > 0"

    # query_time_s > 0 for ok rows
    ok_rows = [r for r in rows if r.status == "ok"]
    for r in ok_rows:
        assert r.query_time_s > 0.0, f"query_time_s should be > 0 for ok row: {r}"

    # Schema fields correctly populated
    for r in rows:
        assert r.benchmark == "synthetic"
        assert r.family == family
        assert r.problem_id == "4"
        assert r.seed == 0
        assert r.baseline == adapter_id


@pytest.mark.slow
def test_zero_budget_forces_timeout_rows(tmp_path):
    """budget=0.0: cumulative 0.0 >= 0.0 at query 0 → all queries timeout.

    Uses fit_timeout_s=1000.0 (explicit) to keep the fit safety net from
    firing, since fit_budget_s = multiplier * 0.0 = 0.0 would also trigger.
    """
    src, src_cfg = _make_source_cfg("discrete", n_nodes=4, seed=0)
    spec = BaselineSpec("pomegranate", "discrete", "mle")
    cfg = RunnerConfig(
        benchmark="synthetic",
        config_name="timeout_test",
        problem_source=src,
        source_config=src_cfg,
        selector=UniformRandomSelector(),
        measurement=AccuracyAndTiming(n_oracle_samples=50),
        baselines=[spec],
        n_queries_per_cell=_N_QUERIES,
        per_cell_timeout_s=0.0,
        fit_timeout_s=1000.0,  # explicit fit budget so safety net doesn't fire
        jsonl_path=tmp_path / "timeout.jsonl",
    )

    rows = list(Runner().run(cfg))

    # All rows are timeout (budget=0.0 fires before query 0 starts)
    timeout_rows = [r for r in rows if r.status == "timeout"]
    assert len(timeout_rows) == _N_QUERIES * 6  # 4 queries × 6 metrics (3 accuracy + 3 timing)

    for r in timeout_rows:
        assert math.isnan(r.query_time_s)
        assert math.isnan(r.metrics_time_s)
        assert not math.isnan(r.fit_time_s)  # fit completed before budget check
        assert r.fit_time_s >= 0.0
        assert r.error_msg == "query budget exceeded"


@pytest.mark.slow
def test_multiple_baselines_iterate_correctly(tmp_path):
    """Runner iterates all baselines for each problem."""
    src, src_cfg = _make_source_cfg("discrete", n_nodes=4, seed=0)
    specs = [
        BaselineSpec("nbn", "cat", "mle", "lw"),
        BaselineSpec("pgmpy", "discrete", "mle", "ve"),
    ]
    cfg = RunnerConfig(
        benchmark="synthetic",
        config_name="multi_test",
        problem_source=src,
        source_config=src_cfg,
        selector=UniformRandomSelector(),
        measurement=AccuracyAndTiming(n_oracle_samples=50),
        baselines=specs,
        n_queries_per_cell=2,
        per_cell_timeout_s=120.0,
        jsonl_path=tmp_path / "multi.jsonl",
    )

    rows = list(Runner().run(cfg))

    baselines_seen = {r.baseline for r in rows}
    assert "nbn-cat-lw" in baselines_seen
    assert "pgmpy-mle-ve" in baselines_seen

    # JSONL line count matches total rows
    lines = cfg.jsonl_path.read_text().strip().splitlines()
    assert len(lines) == len(rows)


@pytest.mark.slow
def test_fit_time_s_identical_across_rows_in_cell(tmp_path):
    """fit_time_s is a per-cell cost: identical across all metric rows for a cell."""
    src, src_cfg = _make_source_cfg("discrete", n_nodes=4, seed=0)
    spec = BaselineSpec("nbn", "cat", "mle", "lw")
    cfg = RunnerConfig(
        benchmark="synthetic",
        config_name="timing_test",
        problem_source=src,
        source_config=src_cfg,
        selector=UniformRandomSelector(),
        measurement=AccuracyAndTiming(n_oracle_samples=50),
        baselines=[spec],
        n_queries_per_cell=_N_QUERIES,
        per_cell_timeout_s=120.0,
        jsonl_path=tmp_path / "timing.jsonl",
    )

    rows = list(Runner().run(cfg))

    fit_times = {r.fit_time_s for r in rows}
    assert len(fit_times) == 1, f"fit_time_s varied: {fit_times}"
    assert next(iter(fit_times)) > 0.0


@pytest.mark.slow
def test_problem_seed_propagated_to_cell_result(tmp_path):
    """problem.seed is used as CellResult.seed for every row in the cell."""
    src, src_cfg = _make_source_cfg("discrete", n_nodes=4, seed=7)
    spec = BaselineSpec("pgmpy", "discrete", "mle", "ve")
    cfg = RunnerConfig(
        benchmark="synthetic",
        config_name="seed_test",
        problem_source=src,
        source_config=src_cfg,
        selector=UniformRandomSelector(),
        measurement=TimingOnly(),
        baselines=[spec],
        n_queries_per_cell=2,
        per_cell_timeout_s=120.0,
        jsonl_path=tmp_path / "seed.jsonl",
    )

    rows = list(Runner().run(cfg))

    seeds = {r.seed for r in rows}
    assert seeds == {7}
