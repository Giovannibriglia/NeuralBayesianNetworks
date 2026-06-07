"""Structural typing tests for v0.13 Protocol interfaces.

These tests verify that the Protocol definitions in
benchmarking/core/interfaces.py are well-formed and that minimal
concrete implementations satisfy them via runtime_checkable. They
do NOT test behavior — there are no real implementations yet.

Reference: docs/v0.13-benchmark-redesign.md §4.1
"""
from __future__ import annotations

from typing import Any, Iterator

import pytest
import torch

from benchmarking.core.interfaces import (
    BaselineAdapter,
    Measurement,
    ProblemSource,
    QuerySelector,
    default_query_batch,
)
from benchmarking.core.results import CellResult
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.domains.posterior import Posterior


# ---- Minimal stub implementations for structural conformance ----

class StubProblemSource:
    def iter_problems(self, cfg: Any) -> Iterator[BenchmarkProblem]:
        return iter([])


class StubQuerySelector:
    def select(self, problem: BenchmarkProblem, n_queries: int, seed: int) -> list[Query]:
        return []

    def select_groups(
        self, problem: BenchmarkProblem, n_queries: int, seed: int,
        *, batch_size: int = 1,
    ) -> list[list[Query]]:
        return [[q] for q in self.select(problem, n_queries, seed)]


class StubMeasurement:
    def measure(
        self,
        problem: BenchmarkProblem,
        adapter: BaselineAdapter,
        queries: list[Query],
    ) -> list[CellResult]:
        return []


class StubBaselineAdapter:
    name: str = "stub-adapter"

    def fit(self, problem: BenchmarkProblem, **kwargs: Any) -> None:
        pass

    def query(self, query: Query) -> Posterior:
        return Posterior(probs=torch.tensor([0.5, 0.5]))

    def query_batch(self, queries: list[Query]) -> list[Posterior]:
        return default_query_batch(self, queries)

    def is_applicable(self, problem: BenchmarkProblem) -> bool:
        return True


# ---- Tests ----

class TestProtocolConformance:
    """Each stub satisfies its protocol via runtime_checkable isinstance."""

    def test_problem_source_stub_conforms(self):
        assert isinstance(StubProblemSource(), ProblemSource)

    def test_query_selector_stub_conforms(self):
        assert isinstance(StubQuerySelector(), QuerySelector)

    def test_measurement_stub_conforms(self):
        assert isinstance(StubMeasurement(), Measurement)

    def test_baseline_adapter_stub_conforms(self):
        assert isinstance(StubBaselineAdapter(), BaselineAdapter)


class TestProtocolRejection:
    """Objects missing required methods do NOT satisfy the protocol."""

    def test_empty_object_rejected_as_problem_source(self):
        class Empty:
            pass
        assert not isinstance(Empty(), ProblemSource)

    def test_missing_query_method_rejected_as_baseline_adapter(self):
        class MissingQuery:
            name = "missing"
            def fit(self, problem, **kwargs): pass
            def is_applicable(self, problem): return True
        # No query() method
        assert not isinstance(MissingQuery(), BaselineAdapter)
