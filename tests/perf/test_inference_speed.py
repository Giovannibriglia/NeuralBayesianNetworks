"""Speed contracts of the amortised inference engines (wall-clock).

Moved from tests/unit/test_amortized_{is,vi}_engine.py: they compare
timings, so they belong on a quiet runner (``-m perf``), not in the
correctness suites that run under ``pytest -n auto``.
"""
from __future__ import annotations

import time

import pytest

from tests.unit.test_amortized_is_engine import (
    _discrete_evidence_query,
    _make_small_discrete_problem as _ais_problem,
)
from tests.unit.test_amortized_is_engine import _fit_adapter as _fit_ais_adapter
from tests.unit.test_amortized_vi_engine import (
    _disc_q,
    _make_small_discrete_problem as _avi_problem,
)
from tests.unit.test_amortized_vi_engine import _fit_adapter as _fit_avi_adapter

pytestmark = [pytest.mark.perf, pytest.mark.slow]


@pytest.fixture(scope="module")
def ais_problem():
    return _ais_problem()


@pytest.fixture(scope="module")
def avi_problem():
    return _avi_problem()


# ---- 3. Batching speedup: per-query time falls as B grows ---------------------

def test_ais_batching_speedup(ais_problem):
    # n_samples=64 keeps the per-call overhead -- what batching amortises --
    # dominant. At 1024 the work is compute-bound and B=64 beat B=1 by only
    # ~1.2x on one core, a margin CI-runner noise flipped (master after #292,
    # macOS lane); at 64 it is ~6x.
    ais = _fit_ais_adapter("cat", "ais", ais_problem, n_samples=64, epochs=5)

    def per_query_time(b: int) -> float:
        queries = [_discrete_evidence_query(v % 2) for v in range(b)]
        ais.query_batch(queries)  # warmup
        # Best of 5, not the mean: on a shared CI runner (xdist workers on the
        # same cores) a single preempted call can swamp the mean and flip the
        # comparison; the minimum is what the engine can do.
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            ais.query_batch(queries)
            times.append(time.perf_counter() - t0)
        return min(times) / b

    pq1 = per_query_time(1)
    pq_big = per_query_time(64)
    assert pq_big < pq1, (
        f"batching contract violated: per-query at B=64 ({pq_big:.6f}s) "
        f"not < per-query at B=1 ({pq1:.6f}s)"
    )


# ---- 5. Single-forward-pass speed: faster than Engine A -----------------------

def test_single_forward_pass_faster_than_ais(avi_problem):
    avi = _fit_avi_adapter("cat", "avi", avi_problem, n_samples=1024, epochs=10)
    ais = _fit_avi_adapter("cat", "ais", avi_problem, n_samples=1024, epochs=5)

    def batch_time(adapter, b: int = 64) -> float:
        queries = [_disc_q(v % 2) for v in range(b)]
        adapter.query_batch(queries)  # warmup
        t0 = time.perf_counter()
        for _ in range(5):
            adapter.query_batch(queries)
        return (time.perf_counter() - t0) / 5

    t_avi = batch_time(avi)
    t_ais = batch_time(ais)
    assert t_avi < 0.5 * t_ais, (
        f"AVI single-forward-pass ({t_avi:.6f}s) not < 50% of AIS "
        f"({t_ais:.6f}s) at B=64"
    )
