"""Fit-once query-many worker contract (#174, design doc §5).

These tests drive ``cell_worker._run_cell`` in-process with a fake
adapter (``build_adapter`` monkeypatched) and the real ``TimingOnly``
measurement, so the fit-count spy is observable and the per-batch
sweep loop is exercised end-to-end without the subprocess boundary.

Asserts the design-doc contract:
  1. fit runs exactly ONCE per cell regardless of how many batch_sizes
     follow (the optimization);
  2. each batch_size produces rows stamped with its value;
  3. a fit failure emits sentinel rows for EVERY batch_size (decision #2);
  4. a query failure at one batch_size does not break the sweep (#3);
  5. a missing / length-1 batch_sizes is identity behavior;
  6. fit_timeout_s covers the single fit, not fit × N;
  7. per_cell_timeout_s is applied fresh per batch_size pass (#1).

Reference: docs/v0.14-fit-once-query-many-design.md §4, §5, §6.
"""
from __future__ import annotations

import math
import time
import types

import pytest

import benchmarking.core.config as config_mod
from benchmarking.core.cell_worker import _run_cell
from benchmarking.core.config import BaselineSpec
from benchmarking.domains.base import Query
from benchmarking.measurements.timing_only import TimingOnly


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeAdapter:
    """Minimal adapter implementing just what the worker + TimingOnly need.

    Records every ``fit`` call (the fit-once spy) and every dispatched
    batch size. ``fit_should`` / ``query_should`` hooks let individual
    tests inject failures or delays.
    """

    name = "fake-adapter"
    device = "cpu"
    supports_batched_queries = True

    def __init__(self, *, fit_should=None, query_should=None,
                 fit_sleep_s=0.0, query_sleep_s=0.0, applicable=True):
        self.fit_calls = 0
        self.query_batch_sizes: list[int] = []
        self._fit_should = fit_should
        self._query_should = query_should
        self._fit_sleep_s = fit_sleep_s
        self._query_sleep_s = query_sleep_s
        self._applicable = applicable

    def is_applicable(self, problem) -> bool:
        return self._applicable

    def fit(self, problem, **kwargs) -> None:
        self.fit_calls += 1
        if self._fit_sleep_s:
            time.sleep(self._fit_sleep_s)
        if self._fit_should is not None:
            self._fit_should()

    def _maybe_fail_or_delay(self, b: int) -> None:
        if self._query_sleep_s:
            time.sleep(self._query_sleep_s)
        if self._query_should is not None:
            self._query_should(b)

    def query(self, q):
        self.query_batch_sizes.append(1)
        self._maybe_fail_or_delay(1)
        return {"posterior": 0.0}

    def query_batch(self, group):
        b = len(group)
        self.query_batch_sizes.append(b)
        self._maybe_fail_or_delay(b)
        return [{"posterior": 0.0} for _ in group]


class _FakeSelector:
    """Deterministic selector: emits ``n`` queries chunked by batch_size."""

    query_role = "random"

    def select_groups(self, problem, n, seed, *, batch_size=1):
        queries = [
            Query(targets=("x",), evidence={"y": 0}, query_role="random")
            for _ in range(n)
        ]
        return [
            queries[i:i + batch_size]
            for i in range(0, len(queries), batch_size)
        ]


def _fake_problem():
    return types.SimpleNamespace(
        family="synthetic", problem_id="p0", seed=0, name="p0",
    )


def _ctx(*, batch_sizes, adapter, n_queries=16, per_cell_timeout_s=1e9,
         fit_budget_s=1e9, include_batch_sizes=True):
    """Build a worker ctx dict wired to the fake adapter/selector."""
    ctx = {
        "problem": _fake_problem(),
        "baseline_spec": BaselineSpec(
            library="nbn", mechanism="cat", param_method="mle",
            inference_method="ve",
        ),
        "seed": 0,
        "selector": _FakeSelector(),
        "measurement": TimingOnly(),
        "benchmark": "synthetic",
        "n_queries_per_cell": n_queries,
        "per_cell_timeout_s": per_cell_timeout_s,
        "fit_budget_s": fit_budget_s,
        "default_role": "random",
    }
    if include_batch_sizes:
        ctx["batch_sizes"] = batch_sizes
    return ctx


@pytest.fixture
def patch_build_adapter(monkeypatch):
    """Return a helper that pins build_adapter to a given fake instance."""
    def _install(adapter):
        # Accept the keyword-only require_engine the cell worker now passes (#109).
        monkeypatch.setattr(
            config_mod, "build_adapter", lambda spec, **_kw: adapter,
        )
        return adapter
    return _install


def _query_rows(rows):
    return [r for r in rows if r["metric"] == "query_time_s"]


# ---------------------------------------------------------------------------
# 1. Fit count
# ---------------------------------------------------------------------------

def test_fit_called_exactly_once_across_sweep(patch_build_adapter):
    adapter = patch_build_adapter(_FakeAdapter())
    rows = _run_cell(_ctx(batch_sizes=[1, 4, 16], adapter=adapter))
    assert adapter.fit_calls == 1, (
        f"fit must run once per cell, ran {adapter.fit_calls} times"
    )
    # All three sweep values were dispatched.
    assert {1, 4, 16}.issubset(set(adapter.query_batch_sizes))


# ---------------------------------------------------------------------------
# 2. Sweep rows present, batch_size stamped
# ---------------------------------------------------------------------------

def test_sweep_rows_present_and_stamped(patch_build_adapter):
    adapter = patch_build_adapter(_FakeAdapter())
    n = 16
    batch_sizes = [1, 4, 16]
    rows = _run_cell(_ctx(batch_sizes=batch_sizes, adapter=adapter, n_queries=n))
    q = _query_rows(rows)
    # Every sweep value is present in the stamped batch_size column.
    assert sorted({r["batch_size"] for r in q}) == [1, 4, 16]
    # One query row per planned query, per batch_size pass.
    for b in batch_sizes:
        b_rows = [r for r in q if r["batch_size"] == b]
        assert len(b_rows) == n, (
            f"batch_size={b}: expected {n} query rows, got {len(b_rows)}"
        )
        assert all(r["status"] == "ok" for r in b_rows)


# ---------------------------------------------------------------------------
# 3. Fit failure → sentinel rows for ALL batch_sizes
# ---------------------------------------------------------------------------

def test_fit_failure_emits_sentinels_for_all_batch_sizes(patch_build_adapter):
    def _boom():
        raise ValueError("fit blew up")

    adapter = patch_build_adapter(_FakeAdapter(fit_should=_boom))
    n = 16
    batch_sizes = [1, 4, 16]
    rows = _run_cell(_ctx(batch_sizes=batch_sizes, adapter=adapter, n_queries=n))

    assert adapter.fit_calls == 1
    # No queries ran.
    assert adapter.query_batch_sizes == []
    # Sentinels are one-per-group per batch_size (§4.3); total reflects
    # every batch_size having been planned (decision #2).
    expected = sum(math.ceil(n / b) for b in batch_sizes)
    assert len(rows) == expected, (
        f"expected {expected} fit-failure sentinels "
        f"(sum of groups across {batch_sizes}), got {len(rows)}"
    )
    assert all(r["status"] == "error" for r in rows)
    assert all(r["metric"] == "status" for r in rows)


# ---------------------------------------------------------------------------
# 4. Query failure at one batch_size does not block the next
# ---------------------------------------------------------------------------

def test_query_failure_midsweep_continues(patch_build_adapter):
    def _fail_at_4(b):
        if b == 4:
            raise RuntimeError("out of memory")

    adapter = patch_build_adapter(_FakeAdapter(query_should=_fail_at_4))
    n = 16
    rows = _run_cell(_ctx(batch_sizes=[1, 4, 16], adapter=adapter, n_queries=n))
    q = _query_rows(rows)

    def _statuses(b):
        return {r["status"] for r in q if r["batch_size"] == b}

    assert _statuses(1) == {"ok"}, "B=1 should succeed"
    assert _statuses(16) == {"ok"}, "B=16 must still be attempted after B=4 failed"
    # B=4 classified as a failure (oom here), not silently dropped.
    assert _statuses(4) <= {"oom", "error"} and _statuses(4), (
        f"B=4 should be a failure status, got {_statuses(4)}"
    )
    # Fit still ran exactly once despite the mid-sweep query failure.
    assert adapter.fit_calls == 1


# ---------------------------------------------------------------------------
# 5. Non-sweep identity (length-1 / missing batch_sizes)
# ---------------------------------------------------------------------------

def test_missing_batch_sizes_falls_back_to_single_pass(patch_build_adapter):
    adapter = patch_build_adapter(_FakeAdapter())
    # ctx WITHOUT a "batch_sizes" key → worker uses [spec.batch_size] (1).
    rows = _run_cell(_ctx(batch_sizes=None, adapter=adapter, n_queries=8,
                          include_batch_sizes=False))
    q = _query_rows(rows)
    assert adapter.fit_calls == 1
    assert {r["batch_size"] for r in q} == {1}
    assert len(q) == 8
    assert all(r["status"] == "ok" for r in q)


def test_length_one_batch_sizes_matches_missing_key(patch_build_adapter):
    # Explicit [1] and a missing key must produce identical rows (both the
    # single-pass identity path).
    a_explicit = patch_build_adapter(_FakeAdapter())
    rows_explicit = _run_cell(_ctx(batch_sizes=[1], adapter=a_explicit,
                                   n_queries=8))

    a_missing = patch_build_adapter(_FakeAdapter())
    rows_missing = _run_cell(_ctx(batch_sizes=None, adapter=a_missing,
                                  n_queries=8, include_batch_sizes=False))

    # Compare on the structural columns (timing values differ as noise).
    cols = ("metric", "status", "batch_size", "query_role")
    proj_e = [tuple(r[c] for c in cols) for r in rows_explicit]
    proj_m = [tuple(r[c] for c in cols) for r in rows_missing]
    assert proj_e == proj_m


# ---------------------------------------------------------------------------
# 6. fit_timeout_s covers the single fit, not fit × N
# ---------------------------------------------------------------------------

def test_fit_timeout_covers_single_fit(patch_build_adapter):
    # fit sleeps just over the budget → the safety net trips; the point is
    # fit runs ONCE (so the wall-clock is one fit, not N fits).
    adapter = patch_build_adapter(_FakeAdapter(fit_sleep_s=0.15))
    batch_sizes = [1, 4, 16]
    t0 = time.perf_counter()
    rows = _run_cell(_ctx(batch_sizes=batch_sizes, adapter=adapter,
                          n_queries=8, fit_budget_s=0.05))
    elapsed = time.perf_counter() - t0

    assert adapter.fit_calls == 1
    assert adapter.query_batch_sizes == []  # over-budget fit → no queries
    assert all(r["status"] == "timeout" for r in rows)
    # One fit (~0.15s), not three (~0.45s). Generous bound for CI noise.
    assert elapsed < 0.15 * len(batch_sizes), (
        f"elapsed {elapsed:.3f}s suggests fit ran more than once"
    )


# ---------------------------------------------------------------------------
# 7. per_cell_timeout_s applied fresh per batch_size pass
# ---------------------------------------------------------------------------

def test_per_cell_timeout_applies_per_batch_size(patch_build_adapter):
    # Each query group sleeps SLEEP_S; the budget is below one group, so
    # exactly the first group of EACH pass runs (the budget check is before
    # the group) and the rest time out. If the budget were shared across
    # passes, later passes would be entirely timed out — which this asserts
    # against by requiring every batch_size to have at least one ok row.
    SLEEP_S = 0.05
    adapter = patch_build_adapter(_FakeAdapter(query_sleep_s=SLEEP_S))
    batch_sizes = [1, 2]
    n = 4  # B=1 → 4 groups, B=2 → 2 groups; both have ≥2 groups
    rows = _run_cell(_ctx(batch_sizes=batch_sizes, adapter=adapter,
                          n_queries=n, per_cell_timeout_s=SLEEP_S * 0.5))
    q = _query_rows(rows)

    for b in batch_sizes:
        statuses = [r["status"] for r in q if r["batch_size"] == b]
        assert "ok" in statuses, (
            f"batch_size={b} pass got no fresh budget (no ok rows) — "
            f"per-pass timeout not applied independently"
        )
        assert "timeout" in statuses, (
            f"batch_size={b} pass should have timed-out groups too"
        )
    assert adapter.fit_calls == 1
