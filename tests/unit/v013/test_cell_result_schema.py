"""Schema tests for v3 CellResult dataclass.

Reference: docs/v0.13-benchmark-redesign.md §6
"""
from __future__ import annotations

import pytest

from nbn.bench.core.results import (
    CellResult,
    VALID_BENCHMARKS,
    VALID_QUERY_ROLES_ALL,
    VALID_QUERY_ROLES_SCALABILITY,
    VALID_QUERY_ROLES_SYNTHETIC,
    VALID_STATUSES,
)


def _make_valid_cell_result(**overrides) -> CellResult:
    """Helper: minimal valid CellResult, with overridable fields."""
    defaults = dict(
        benchmark="synthetic",
        family="discrete",
        problem_id="100",
        seed=0,
        baseline="nbn-cat-lw",
        query_role="random",
        metric="tv_per_node",
        value=0.1,
        status="ok",
        fit_time_s=0.5,
        query_time_s=0.05,
        metrics_time_s=0.02,
        error_msg=None,
    )
    defaults.update(overrides)
    return CellResult(**defaults)


class TestCellResultBasics:
    def test_valid_cell_result_constructs(self):
        row = _make_valid_cell_result()
        assert row.benchmark == "synthetic"
        assert row.fit_time_s == 0.5

    def test_cell_result_is_frozen(self):
        row = _make_valid_cell_result()
        with pytest.raises(AttributeError):  # FrozenInstanceError is a subclass of AttributeError
            row.value = 999.0

    def test_error_msg_defaults_to_none(self):
        row = _make_valid_cell_result()
        assert row.error_msg is None

    def test_error_msg_can_be_set(self):
        row = _make_valid_cell_result(status="error", error_msg="adapter crashed")
        assert row.error_msg == "adapter crashed"


class TestSchemaConstants:
    """Validate the constants used elsewhere for status/benchmark validation."""

    def test_valid_statuses_include_expected(self):
        assert "ok" in VALID_STATUSES
        assert "timeout" in VALID_STATUSES
        assert "oom" in VALID_STATUSES
        assert "error" in VALID_STATUSES
        assert "not_supported" in VALID_STATUSES

    def test_valid_benchmarks(self):
        assert {"synthetic", "scalability", "bnlearn"} == VALID_BENCHMARKS

    def test_synthetic_roles(self):
        assert {"hub", "cut", "terminal", "random"} == VALID_QUERY_ROLES_SYNTHETIC

    def test_scalability_roles(self):
        assert {
            "heaviest_hub", "heaviest_cut", "longest_propagation",
        } == VALID_QUERY_ROLES_SCALABILITY

    def test_all_roles_is_union(self):
        assert (VALID_QUERY_ROLES_SYNTHETIC | VALID_QUERY_ROLES_SCALABILITY) == VALID_QUERY_ROLES_ALL


class TestTimingColumns:
    """Verify timing columns are all present and accept float values."""

    def test_all_three_timing_columns_present(self):
        row = _make_valid_cell_result(
            fit_time_s=1.0, query_time_s=0.01, metrics_time_s=0.05,
        )
        assert row.fit_time_s == 1.0
        assert row.query_time_s == 0.01
        assert row.metrics_time_s == 0.05

    def test_zero_timings_acceptable(self):
        # Status=not_supported rows have zero timings — valid
        row = _make_valid_cell_result(
            status="not_supported",
            fit_time_s=0.0, query_time_s=0.0, metrics_time_s=0.0,
        )
        assert row.fit_time_s == 0.0
