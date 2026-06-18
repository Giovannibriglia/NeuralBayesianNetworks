"""Stage 4 wiring tests: per-query metadata propagation (Phase 2, PR #124).

Verifies that ``query_kind`` and ``evidence_strategy`` flow end-to-end:

  TopologicalAllocator  →  Query objects  →  Runner  →  Measurement  →  CellResult

and that pre-Phase-2 selectors (UniformRandomSelector) keep producing rows
with the backward-compat defaults ("prediction" / "random").

Two layers are exercised:
  * Runner extraction — the runner reads per-query metadata off the Query
    objects and passes parallel lists to ``measure()`` (uses a recording
    fake Measurement + a monkeypatched ``build_adapter`` so no real adapter
    or library is needed).
  * Measurement emission — ``AccuracyAndTiming`` / ``TimingOnly`` stamp each
    emitted ``CellResult`` with the per-query values.

Reference: PR #124, #109, #74; docs/v0.13-benchmark-redesign.md §3, §6.
"""
from __future__ import annotations

from typing import Any, Iterator

import torch

import benchmarking.core.config as config_mod
from benchmarking.core.cell_worker import _run_cell
from benchmarking.core.config import BaselineSpec
from benchmarking.core.results import CellResult
from benchmarking.domains.base import BenchmarkProblem, GroundTruth, Query
from benchmarking.measurements import AccuracyAndTiming, TimingOnly
from benchmarking.selectors.topological import TopologicalAllocator
from benchmarking.selectors.uniform import UniformRandomSelector


# --- fixtures / fakes ---

_ASIA_EDGES = [
    ("asia", "tub"),
    ("smoke", "lung"),
    ("smoke", "bronc"),
    ("tub", "either"),
    ("lung", "either"),
    ("either", "xray"),
    ("either", "dysp"),
    ("bronc", "dysp"),
]


def _asia_problem(n_rows: int = 50) -> BenchmarkProblem:
    """Discrete ASIA-shaped problem with random 0/1 test data."""
    nodes = sorted({n for e in _ASIA_EDGES for n in e})
    gen = torch.Generator().manual_seed(0)
    test_data = {
        n: torch.randint(0, 2, (n_rows,), generator=gen) for n in nodes
    }
    return BenchmarkProblem(
        name="asia",
        dag=list(_ASIA_EDGES),
        variables=dict.fromkeys(nodes, ("discrete", 2)),
        train_data={},
        test_data=test_data,
        queries=[],
        ground_truth=GroundTruth(),
        family="discrete",
        problem_id="asia_test",
        seed=0,
    )


class _FakeAdapter:
    """Applicable no-op adapter (query is never called by the fake measurement)."""

    name = "fake-adapter"

    def is_applicable(self, problem: Any) -> bool:
        return True

    def fit(self, problem: Any, **kwargs: Any) -> None:
        pass

    def query(self, q: Query) -> Any:  # pragma: no cover - not exercised
        raise AssertionError("query() should not be called by the fake measurement")


class _RecordingMeasurement:
    """Records the parallel metadata lists the runner passes to measure()."""

    def __init__(self) -> None:
        self.received: dict[str, Any] = {}

    def measure(
        self,
        problem: Any,
        adapter: Any,
        queries: list[Query],
        *,
        fit_time_s: float = 0.0,
        benchmark: str = "synthetic",
        seed: int = 0,
        query_roles: list[str] | None = None,
        query_kinds: list[str] | None = None,
        evidence_strategies: list[str] | None = None,
        evidence_modes: list[str] | None = None,
        query_budget_s: float = float("inf"),
        query_groups: list[list[Query]] | None = None,
    ) -> list[CellResult]:
        self.received = {
            "query_roles": query_roles,
            "query_kinds": query_kinds,
            "evidence_strategies": evidence_strategies,
            "evidence_modes": evidence_modes,
            "queries": queries,
            "query_groups": query_groups,
        }
        return []


class _RaisingAdapter:
    """Adapter whose query() raises — drives the measurement error path."""

    name = "raising-adapter"

    def is_applicable(self, problem: Any) -> bool:
        return True

    def fit(self, problem: Any, **kwargs: Any) -> None:
        pass

    def query(self, q: Query) -> Any:
        raise RuntimeError("boom")


_N_QUERIES = 20


def _wiring_ctx(problem, selector, measurement) -> dict:
    """ctx mirroring what Runner._run_cell passes to the cell worker."""
    return {
        "problem": problem,
        "baseline_spec": BaselineSpec(
            library="pgmpy", mechanism="discrete", param_method="mle",
        ),
        "seed": problem.seed,
        "selector": selector,
        "measurement": measurement,
        "benchmark": "synthetic",
        "n_queries_per_cell": _N_QUERIES,
        "per_cell_timeout_s": 120.0,
        "fit_budget_s": 1200.0,
        "default_role": "random",
    }


class TestRunnerPhase2Wiring:
    """Verify per-query metadata fields propagate through the cell + measurements.

    Bug 4 Stage 2: the cell runs in a subprocess, so these tests (which
    monkeypatch ``build_adapter`` and use a recording-spy measurement that
    must be observed afterwards) call ``cell_worker._run_cell`` directly
    in-process — where the patch and the spy's ``.received`` apply. The
    patch targets ``benchmarking.core.config.build_adapter`` (the name the
    worker imports), not the runner module. Intent unchanged.
    """

    def test_topological_metadata_in_results(self, monkeypatch):
        """TopologicalAllocator → cell → measure() preserves metadata."""
        monkeypatch.setattr(config_mod, "build_adapter", lambda spec, **_kw: _FakeAdapter())

        problem = _asia_problem()
        selector = TopologicalAllocator()
        measurement = _RecordingMeasurement()

        _run_cell(_wiring_ctx(problem, selector, measurement))

        # Recompute the exact queries the selector produced (deterministic).
        expected = selector.select(problem, _N_QUERIES, problem.seed)
        assert len(expected) > 0

        rec = measurement.received
        assert rec["query_kinds"] == [q.query_kind for q in expected]
        assert rec["evidence_strategies"] == [q.evidence_strategy for q in expected]
        assert rec["query_roles"] == [q.query_role for q in expected]

        # Metadata must come from the Query objects, not hardcoded defaults.
        assert all(k in {"prediction", "diagnosis"} for k in rec["query_kinds"])
        assert all(
            s in {"longest_path", "mb_neighbors", "random"}
            for s in rec["evidence_strategies"]
        )
        assert all(
            r in {"hub", "cut", "terminal", "random"} for r in rec["query_roles"]
        )

    def test_uniform_random_selector_unchanged(self, monkeypatch):
        """UniformRandomSelector queries carry default metadata; cell doesn't crash."""
        monkeypatch.setattr(config_mod, "build_adapter", lambda spec, **_kw: _FakeAdapter())

        problem = _asia_problem()
        selector = UniformRandomSelector()
        measurement = _RecordingMeasurement()

        _run_cell(_wiring_ctx(problem, selector, measurement))

        rec = measurement.received
        assert len(rec["queries"]) > 0
        # Backward-compat defaults from the Query dataclass.
        assert all(k == "prediction" for k in rec["query_kinds"])
        assert all(s == "random" for s in rec["evidence_strategies"])
        assert all(r == "random" for r in rec["query_roles"])

    def test_accuracy_timing_emits_metadata(self):
        """AccuracyAndTiming stamps each CellResult with per-query metadata."""
        problem = _asia_problem()
        queries = [
            Query(
                targets=("either",),
                evidence={"tub": torch.tensor(1.0)},
                query_role="hub",
                query_kind="diagnosis",
                evidence_strategy="mb_neighbors",
            ),
            Query(
                targets=("dysp",),
                evidence={"asia": torch.tensor(0.0)},
                query_role="terminal",
                query_kind="prediction",
                evidence_strategy="longest_path",
            ),
        ]
        rows = AccuracyAndTiming().measure(
            problem,
            _RaisingAdapter(),
            queries,
            query_roles=[q.query_role for q in queries],
            query_kinds=[q.query_kind for q in queries],
            evidence_strategies=[q.evidence_strategy for q in queries],
        )
        assert rows
        # Every row for query 0 → diagnosis/mb_neighbors; query 1 → prediction/longest_path.
        by_role = {"hub": set(), "terminal": set()}
        for r in rows:
            by_role.setdefault(r.query_role, set())
            by_role[r.query_role].add((r.query_kind, r.evidence_strategy))
        assert by_role["hub"] == {("diagnosis", "mb_neighbors")}
        assert by_role["terminal"] == {("prediction", "longest_path")}

    def test_timing_only_emits_metadata(self):
        """TimingOnly stamps each CellResult with per-query metadata."""
        problem = _asia_problem()
        queries = [
            Query(
                targets=("either",),
                evidence={"tub": torch.tensor(1.0)},
                query_role="cut",
                query_kind="diagnosis",
                evidence_strategy="random",
            ),
        ]
        rows = TimingOnly().measure(
            problem,
            _RaisingAdapter(),
            queries,
            query_roles=["cut"],
            query_kinds=["diagnosis"],
            evidence_strategies=["random"],
        )
        assert rows
        for r in rows:
            assert r.query_role == "cut"
            assert r.query_kind == "diagnosis"
            assert r.evidence_strategy == "random"

    def test_measure_defaults_when_not_passed(self):
        """Backward-compat: measure() without the new lists → default metadata."""
        problem = _asia_problem()
        queries = [
            Query(targets=("either",), evidence={"tub": torch.tensor(1.0)}),
        ]
        rows = TimingOnly().measure(problem, _RaisingAdapter(), queries)
        assert rows
        for r in rows:
            assert r.query_kind == "prediction"
            assert r.evidence_strategy == "random"
