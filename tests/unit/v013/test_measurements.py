"""Tests for AccuracyAndTiming and TimingOnly measurements.

Four test classes:
  - TestAccuracyAndTimingProtocol: isinstance + return-type invariants.
  - TestTimingOnlyProtocol: same for TimingOnly.
  - TestAccuracyAndTimingBehavioral: end-to-end fit + measure with real
    adapters on small synthetic problems (marked @pytest.mark.slow).
  - TestTimingOnlyBehavioral: timing-only end-to-end (marked slow).

For behavioral tests, we use SyntheticProblemSource + UniformRandomSelector
to build realistic problems and query lists, then check CellResult shape,
timing semantics, and field contracts.

Adapter coverage:
  - NBNAdapter (cat/lw on discrete; mdn/lw on continuous)
  - PgmpyAdapter (mle/ve on discrete)
  - PomegranateAdapter (discrete)

Pyro is excluded from the fast behavioral gate (requires GPU-optional
setup; its adapter-level tests cover it sufficiently).

Reference: docs/v0.13-benchmark-redesign.md §3, §4.1
"""
from __future__ import annotations

import time

import pytest
import torch

from benchmarking.adapters import NBNAdapter, PgmpyAdapter, PomegranateAdapter
from benchmarking.core.interfaces import Measurement
from benchmarking.core.results import CellResult, VALID_STATUSES
from benchmarking.domains.base import BenchmarkProblem, GroundTruth, Query
from benchmarking.domains.posterior import Posterior
from benchmarking.measurements import AccuracyAndTiming, TimingOnly
from benchmarking.problems.synthetic import SyntheticConfig, SyntheticProblemSource
from benchmarking.selectors.uniform import UniformRandomSelector


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_N_QUERIES = 4   # small for speed
_FIT_EPOCHS = 5  # fast fit


def _get_discrete_problem_and_queries():
    """Discrete 4-node BN, 8 queries."""
    src = SyntheticProblemSource()
    cfg = SyntheticConfig(
        families=["discrete"],
        n_nodes_list=[4],
        seeds=[0],
        n_train=300,
        n_test=50,
        n_reference=500,
    )
    problem = next(iter(src.iter_problems(cfg)))
    sel = UniformRandomSelector()
    queries = sel.select(problem, n_queries=_N_QUERIES, seed=0)
    return problem, queries


def _get_continuous_problem_and_queries():
    """Continuous_lg 4-node BN, 4 queries."""
    src = SyntheticProblemSource()
    cfg = SyntheticConfig(
        families=["continuous_lg"],
        n_nodes_list=[4],
        seeds=[0],
        n_train=300,
        n_test=50,
        n_reference=500,
    )
    problem = next(iter(src.iter_problems(cfg)))
    sel = UniformRandomSelector()
    queries = sel.select(problem, n_queries=_N_QUERIES, seed=0)
    return problem, queries


# ---------------------------------------------------------------------------
# Stub adapter for structural tests (no fitting)
# ---------------------------------------------------------------------------

class _StubAdapter:
    """Minimal stub satisfying BaselineAdapter protocol for structural tests."""
    name = "stub-adapter"

    def fit(self, problem, **kwargs): pass

    def query(self, q: Query) -> Posterior:
        # Return a valid Posterior based on target kind from the query's evidence
        # (we don't have the problem here, but we can return probs for discrete).
        # Structural tests don't care about correctness, just contract shape.
        return Posterior(probs=torch.tensor([0.5, 0.5]))

    def is_applicable(self, problem): return True


class _StubContinuousAdapter:
    """Stub adapter returning continuous posterior samples."""
    name = "stub-continuous"

    def fit(self, problem, **kwargs): pass

    def query(self, q: Query) -> Posterior:
        return Posterior(samples=torch.randn(64))

    def is_applicable(self, problem): return True


class _StubErrorAdapter:
    """Stub adapter that always raises on query()."""
    name = "stub-error"

    def fit(self, problem, **kwargs): pass

    def query(self, q: Query) -> Posterior:
        raise RuntimeError("simulated query failure")

    def is_applicable(self, problem): return True


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestAccuracyAndTimingProtocol:
    """AccuracyAndTiming satisfies the Measurement protocol."""

    def test_satisfies_measurement_protocol(self):
        m = AccuracyAndTiming()
        assert isinstance(m, Measurement)

    def test_returns_list(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = AccuracyAndTiming()
        result = m.measure(problem, _StubAdapter(), queries)
        assert isinstance(result, list)

    def test_each_element_is_cell_result(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = AccuracyAndTiming()
        result = m.measure(problem, _StubAdapter(), queries)
        for row in result:
            assert isinstance(row, CellResult)

    def test_empty_queries_returns_empty(self):
        problem, _ = _get_discrete_problem_and_queries()
        m = AccuracyAndTiming()
        result = m.measure(problem, _StubAdapter(), [])
        assert result == []


class TestTimingOnlyProtocol:
    """TimingOnly satisfies the Measurement protocol."""

    def test_satisfies_measurement_protocol(self):
        m = TimingOnly()
        assert isinstance(m, Measurement)

    def test_returns_list(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), queries)
        assert isinstance(result, list)

    def test_each_element_is_cell_result(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), queries)
        for row in result:
            assert isinstance(row, CellResult)

    def test_empty_queries_returns_empty(self):
        problem, _ = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), [])
        assert result == []

    def test_one_row_per_query(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), queries)
        assert len(result) == len(queries)

    def test_metric_is_query_time_s(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), queries)
        for row in result:
            assert row.metric == "query_time_s"

    def test_value_equals_query_time_s(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), queries)
        for row in result:
            assert row.value == pytest.approx(row.query_time_s, abs=1e-9)

    def test_metrics_time_s_is_zero(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), queries)
        for row in result:
            assert row.metrics_time_s == 0.0


class TestCellResultSchema:
    """CellResult field contracts for both measurement types."""

    def test_fit_time_s_propagated(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), queries, fit_time_s=1.234)
        for row in result:
            assert row.fit_time_s == pytest.approx(1.234)

    def test_fit_time_s_identical_across_rows(self):
        """fit_time_s is a per-cell cost, repeated for every query row."""
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), queries, fit_time_s=0.77)
        fit_times = {row.fit_time_s for row in result}
        assert len(fit_times) == 1

    def test_metrics_time_s_identical_across_rows_accuracy(self):
        """metrics_time_s is a per-cell cost, repeated for every query row.

        Note: query_time_s varies per row (one timer per adapter.query call).
        fit_time_s and metrics_time_s are recorded identically across all N
        rows for a given (cell, baseline) — they are per-cell costs, repeated
        for query-row joinability in the parquet. This matches the v0.13
        schema documented in docs/v0.13-benchmark-redesign.md §3.1.
        """
        problem, queries = _get_discrete_problem_and_queries()
        m = AccuracyAndTiming()
        result = m.measure(problem, _StubAdapter(), queries)
        metrics_times = {row.metrics_time_s for row in result}
        assert len(metrics_times) == 1

    def test_status_values_are_valid(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = AccuracyAndTiming()
        result = m.measure(problem, _StubAdapter(), queries)
        for row in result:
            assert row.status in VALID_STATUSES

    def test_baseline_from_adapter_name(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), queries)
        for row in result:
            assert row.baseline == "stub-adapter"

    def test_family_from_problem(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), queries)
        for row in result:
            assert row.family == "discrete"

    def test_problem_id_from_problem(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), queries)
        for row in result:
            assert row.problem_id == "4"

    def test_query_role_default_random(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), queries)
        for row in result:
            assert row.query_role == "random"

    def test_custom_query_roles_propagated(self):
        problem, queries = _get_discrete_problem_and_queries()
        roles = ["hub"] * len(queries)
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), queries, query_roles=roles)
        for row in result:
            assert row.query_role == "hub"

    def test_query_roles_length_mismatch_raises(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        with pytest.raises(ValueError, match="query_roles length"):
            m.measure(problem, _StubAdapter(), queries,
                      query_roles=["random"] * (len(queries) + 1))

    def test_error_adapter_gives_error_status(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubErrorAdapter(), queries)
        for row in result:
            assert row.status == "error"
            assert row.error_msg is not None

    def test_query_time_s_is_nonnegative(self):
        problem, queries = _get_discrete_problem_and_queries()
        m = TimingOnly()
        result = m.measure(problem, _StubAdapter(), queries)
        for row in result:
            assert row.query_time_s >= 0.0


# ---------------------------------------------------------------------------
# Behavioral tests — real adapters, real fitting
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestAccuracyAndTimingBehavioral:
    """End-to-end fit + measure with real adapters.

    Validates:
      - Timing fields: query_time_s > 0, fit_time_s correctly threaded.
      - Accuracy fields: ok/no_result/not_supported for each metric.
      - Row count: N * 3 (tv + jsd + w1) for N queries.
      - metrics_time_s identical across all rows in a cell.
    """

    @pytest.fixture(scope="class")
    def discrete_data(self):
        return _get_discrete_problem_and_queries()

    @pytest.fixture(scope="class")
    def continuous_data(self):
        return _get_continuous_problem_and_queries()

    def _fit_and_measure(self, problem, queries, adapter, *, n_oracle_samples=200):
        m = AccuracyAndTiming(n_oracle_samples=n_oracle_samples)
        t0 = time.perf_counter()
        adapter.fit(problem, epochs=_FIT_EPOCHS)
        fit_time_s = time.perf_counter() - t0
        return m.measure(problem, adapter, queries, fit_time_s=fit_time_s)

    def _validate_rows(self, result: list, n_queries: int) -> None:
        """Assert basic structural contracts on a measure() result."""
        # Three metric rows per query (tv, jsd, w1).
        assert len(result) == n_queries * 3
        for row in result:
            assert isinstance(row, CellResult)
            assert row.status in VALID_STATUSES
            assert row.metric in {"tv_per_node", "jsd_per_node", "w1_per_node"}
            assert row.fit_time_s >= 0.0
            assert row.query_time_s >= 0.0
            assert row.metrics_time_s >= 0.0

    # ---- NBN discrete ----

    def test_nbn_cat_lw_on_discrete(self, discrete_data):
        problem, queries = discrete_data
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        result = self._fit_and_measure(problem, queries, adapter)
        self._validate_rows(result, len(queries))

    def test_nbn_cat_ve_on_discrete(self, discrete_data):
        problem, queries = discrete_data
        adapter = NBNAdapter(mechanism="cat", engine="ve")
        result = self._fit_and_measure(problem, queries, adapter)
        self._validate_rows(result, len(queries))

    # ---- NBN continuous ----

    def test_nbn_mdn_lw_on_continuous(self, continuous_data):
        problem, queries = continuous_data
        adapter = NBNAdapter(mechanism="mdn", engine="lw")
        result = self._fit_and_measure(problem, queries, adapter, n_oracle_samples=100)
        self._validate_rows(result, len(queries))

    def test_nbn_lg_lw_on_continuous(self, continuous_data):
        problem, queries = continuous_data
        adapter = NBNAdapter(mechanism="lg", engine="lw")
        result = self._fit_and_measure(problem, queries, adapter, n_oracle_samples=100)
        self._validate_rows(result, len(queries))

    # ---- pgmpy discrete ----

    def test_pgmpy_mle_ve_on_discrete(self, discrete_data):
        problem, queries = discrete_data
        adapter = PgmpyAdapter(param_method="mle", inference_method="ve")
        result = self._fit_and_measure(problem, queries, adapter)
        self._validate_rows(result, len(queries))

    # ---- pomegranate discrete ----

    def test_pomegranate_on_discrete(self, discrete_data):
        problem, queries = discrete_data
        adapter = PomegranateAdapter()
        if not adapter.is_applicable(problem):
            pytest.skip("pomegranate not applicable to this family")
        result = self._fit_and_measure(problem, queries, adapter)
        self._validate_rows(result, len(queries))

    # ---- Timing semantics (the #107 fix) ----

    def test_fit_time_s_recorded_in_every_row(self, discrete_data):
        problem, queries = discrete_data
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        t0 = time.perf_counter()
        adapter.fit(problem, epochs=_FIT_EPOCHS)
        fit_time_s = time.perf_counter() - t0
        m = AccuracyAndTiming()
        result = m.measure(problem, adapter, queries, fit_time_s=fit_time_s)
        for row in result:
            assert row.fit_time_s == pytest.approx(fit_time_s)

    def test_fit_time_s_identical_across_all_rows(self, discrete_data):
        problem, queries = discrete_data
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        adapter.fit(problem, epochs=_FIT_EPOCHS)
        m = AccuracyAndTiming()
        result = m.measure(problem, adapter, queries, fit_time_s=0.99)
        fit_times = {row.fit_time_s for row in result}
        assert len(fit_times) == 1

    def test_metrics_time_s_identical_across_all_rows(self, discrete_data):
        """metrics_time_s is a per-cell cost repeated for query-row joinability.

        Note: query_time_s varies per row (one timer per adapter.query call).
        fit_time_s and metrics_time_s are recorded identically across all N
        rows for a given (cell, baseline) — they are per-cell costs, repeated
        for query-row joinability in the parquet. This matches the v0.13
        schema documented in docs/v0.13-benchmark-redesign.md §3.1.
        """
        problem, queries = discrete_data
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        adapter.fit(problem, epochs=_FIT_EPOCHS)
        m = AccuracyAndTiming()
        result = m.measure(problem, adapter, queries)
        ok_rows = [r for r in result if r.status in ("ok", "no_result", "not_supported")]
        if ok_rows:
            metrics_times = {row.metrics_time_s for row in ok_rows}
            assert len(metrics_times) == 1

    def test_query_time_s_positive_on_ok_rows(self, discrete_data):
        problem, queries = discrete_data
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        adapter.fit(problem, epochs=_FIT_EPOCHS)
        m = AccuracyAndTiming()
        result = m.measure(problem, adapter, queries)
        ok_rows = [r for r in result if r.status == "ok"]
        for row in ok_rows:
            assert row.query_time_s > 0.0

    # ---- Row count ----

    def test_row_count_is_n_queries_times_3(self, discrete_data):
        """AccuracyAndTiming emits 3 metric rows per query (tv + jsd + w1)."""
        problem, queries = discrete_data
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        adapter.fit(problem, epochs=_FIT_EPOCHS)
        m = AccuracyAndTiming()
        result = m.measure(problem, adapter, queries)
        assert len(result) == len(queries) * 3

    def test_metrics_covered_are_tv_jsd_w1(self, discrete_data):
        problem, queries = discrete_data
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        adapter.fit(problem, epochs=_FIT_EPOCHS)
        m = AccuracyAndTiming()
        result = m.measure(problem, adapter, queries)
        metrics = {row.metric for row in result}
        assert metrics == {"tv_per_node", "jsd_per_node", "w1_per_node"}

    def test_w1_not_supported_for_discrete(self, discrete_data):
        """W1 is not meaningful on unordered discrete categories."""
        problem, queries = discrete_data
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        adapter.fit(problem, epochs=_FIT_EPOCHS)
        m = AccuracyAndTiming()
        result = m.measure(problem, adapter, queries)
        w1_rows = [r for r in result if r.metric == "w1_per_node"]
        assert all(r.status == "not_supported" for r in w1_rows)


@pytest.mark.slow
class TestTimingOnlyBehavioral:
    """TimingOnly end-to-end with a real adapter."""

    def test_nbn_cat_lw_discrete(self):
        problem, queries = _get_discrete_problem_and_queries()
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        t0 = time.perf_counter()
        adapter.fit(problem, epochs=_FIT_EPOCHS)
        fit_time_s = time.perf_counter() - t0

        m = TimingOnly()
        result = m.measure(problem, adapter, queries, fit_time_s=fit_time_s)

        assert len(result) == len(queries)
        for row in result:
            assert row.metric == "query_time_s"
            assert row.query_time_s > 0.0
            assert row.metrics_time_s == 0.0
            assert row.fit_time_s == pytest.approx(fit_time_s)
            assert row.status == "ok"

    def test_nbn_mdn_lw_continuous(self):
        problem, queries = _get_continuous_problem_and_queries()
        adapter = NBNAdapter(mechanism="mdn", engine="lw")
        t0 = time.perf_counter()
        adapter.fit(problem, epochs=_FIT_EPOCHS)
        fit_time_s = time.perf_counter() - t0

        m = TimingOnly()
        result = m.measure(problem, adapter, queries, fit_time_s=fit_time_s)

        assert len(result) == len(queries)
        for row in result:
            assert row.status == "ok"
            assert row.query_time_s > 0.0
