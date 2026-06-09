"""Tests for AmortizedISEngine — Engine A (#181).

Stage (a) coverage: B=1 correctness vs VE, convergence to VE as the
particle count grows, the batching-speedup contract (per-query time
falls as B grows), per-mechanism training (cat / neuralcat / mdn), the
under-trained-proposal diagnostic, and the fit-once contract (the
proposal trains exactly once and is reused across query_batch calls).

Stage (b) appends lg / flow / hybrid coverage.

Engine A is self-normalized importance sampling with a learned proposal:
asymptotically correct like LW, so posteriors match VE within Monte
Carlo tolerance and tighten as N → ∞.  See
docs/v0.14-batched-inference-engines-research.md (Engine A section).
"""
from __future__ import annotations

import logging
import statistics
import time

import pytest
import torch

from benchmarking.adapters import NBNAdapter
from benchmarking.domains.base import BenchmarkProblem, Query
from nbn.inference.amortized_is import AmortizedISEngine


# ---- Fixtures ----------------------------------------------------------------

def _make_small_discrete_problem(n_samples: int = 500, seed: int = 0) -> BenchmarkProblem:
    """4-node binary BN: X0 → X1 → X2, X0 → X3 (correlated, non-degenerate)."""
    g = torch.Generator().manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2"), ("X0", "X3")]
    x0 = torch.randint(0, 2, (n_samples,), generator=g)
    flip = lambda x, p: torch.where(  # noqa: E731
        torch.rand(n_samples, generator=g) < p, 1 - x, x
    )
    train_data = {
        "X0": x0,
        "X1": flip(x0, 0.3),
        "X2": flip(flip(x0, 0.3), 0.3),
        "X3": flip(x0, 0.2),
    }
    variables = dict.fromkeys(train_data, ("discrete", 2))
    return BenchmarkProblem(
        name="ais_discrete", dag=dag, variables=variables,
        train_data=train_data, test_data=train_data, queries=[],
    )


def _make_small_continuous_problem(n_samples: int = 400, seed: int = 0) -> BenchmarkProblem:
    """3-node linear-Gaussian chain: X0 → X1 → X2 (mean X2|X0=v is 0.25·v)."""
    torch.manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2")]
    x0 = torch.randn(n_samples)
    x1 = 0.5 * x0 + 0.5 * torch.randn(n_samples)
    x2 = 0.5 * x1 + 0.5 * torch.randn(n_samples)
    train_data = {"X0": x0, "X1": x1, "X2": x2}
    variables = dict.fromkeys(train_data, ("continuous", None))
    return BenchmarkProblem(
        name="ais_continuous", dag=dag, variables=variables,
        train_data=train_data, test_data=train_data, queries=[],
    )


def _discrete_evidence_query(v: int = 0) -> Query:
    return Query(targets=("X2",), evidence={"X0": torch.tensor(v)}, kind="marginal")


@pytest.fixture(scope="module")
def discrete_problem() -> BenchmarkProblem:
    return _make_small_discrete_problem()


@pytest.fixture(scope="module")
def continuous_problem() -> BenchmarkProblem:
    return _make_small_continuous_problem()


def _fit_adapter(mechanism: str, engine: str, problem, n_samples: int = 8192,
                 epochs: int = 10) -> NBNAdapter:
    torch.manual_seed(0)
    adapter = NBNAdapter(
        mechanism=mechanism, engine=engine, device="cpu", n_samples=n_samples
    )
    adapter.fit(problem, epochs=epochs)
    return adapter


# ---- 1 & 4. B=1 correctness vs VE (cat) --------------------------------------

@pytest.mark.slow
class TestB1Correctness:
    def test_cat_matches_ve(self, discrete_problem):
        ve = _fit_adapter("cat", "ve", discrete_problem, epochs=5)
        ais = _fit_adapter("cat", "ais", discrete_problem, n_samples=8192, epochs=5)
        torch.manual_seed(1)
        for v in (0, 1):
            q = _discrete_evidence_query(v)
            p_ve = ve.query(q).probs
            p_ais = ais.query(q).probs
            assert p_ais is not None and p_ais.shape == p_ve.shape
            maxdiff = (p_ve - p_ais).abs().max()
            assert maxdiff < 0.05, f"X0={v}: AIS vs VE maxdiff {maxdiff:.4f} > 0.05"

    # ---- 5. neuralcat vs VE --------------------------------------------------
    def test_neuralcat_matches_ve(self, discrete_problem):
        ve = _fit_adapter("neuralcat", "ve", discrete_problem, epochs=20)
        ais = _fit_adapter("neuralcat", "ais", discrete_problem, n_samples=8192, epochs=20)
        torch.manual_seed(1)
        q = _discrete_evidence_query(0)
        maxdiff = (ve.query(q).probs - ais.query(q).probs).abs().max()
        assert maxdiff < 0.07, f"neuralcat AIS vs VE maxdiff {maxdiff:.4f} > 0.07"


# ---- 2. Convergence to VE as N grows -----------------------------------------

@pytest.mark.slow
def test_convergence_to_ve(discrete_problem):
    """Mean |AIS − VE| over seeds must shrink as the particle count grows."""
    ve = _fit_adapter("cat", "ve", discrete_problem, epochs=5)
    target = ve.query(_discrete_evidence_query(0)).probs
    ais = _fit_adapter("cat", "ais", discrete_problem, n_samples=64, epochs=5)

    def mean_err(n_samples: int) -> float:
        ais._engine_obj.n_samples = n_samples
        errs = []
        for sd in range(15):
            torch.manual_seed(sd)
            p = ais.query(_discrete_evidence_query(0)).probs
            errs.append(float((target - p).abs().max()))
        return statistics.mean(errs)

    err_low = mean_err(64)
    err_high = mean_err(4096)
    assert err_high < err_low, (
        f"convergence violated: err(N=4096)={err_high:.4f} "
        f"not < err(N=64)={err_low:.4f}"
    )
    # Roughly the 1/√N Monte-Carlo rate: 64× more particles → clearly tighter.
    assert err_high < 0.6 * err_low


# ---- 3. Batching speedup: per-query time falls as B grows ---------------------

@pytest.mark.slow
def test_batching_speedup(discrete_problem):
    ais = _fit_adapter("cat", "ais", discrete_problem, n_samples=1024, epochs=5)

    def per_query_time(b: int) -> float:
        queries = [_discrete_evidence_query(v % 2) for v in range(b)]
        ais.query_batch(queries)  # warmup
        t0 = time.perf_counter()
        for _ in range(3):
            ais.query_batch(queries)
        return (time.perf_counter() - t0) / 3 / b

    pq1 = per_query_time(1)
    pq_big = per_query_time(64)
    assert pq_big < pq1, (
        f"batching contract violated: per-query at B=64 ({pq_big:.6f}s) "
        f"not < per-query at B=1 ({pq1:.6f}s)"
    )


# ---- 6. mdn continuous: AIS agrees with LW (same fitted model) ---------------

@pytest.mark.slow
def test_mdn_continuous_matches_lw(continuous_problem):
    lw = _fit_adapter("mdn", "lw", continuous_problem, n_samples=8192, epochs=30)
    ais = _fit_adapter("mdn", "ais", continuous_problem, n_samples=8192, epochs=30)
    q = Query(targets=("X2",), evidence={"X0": torch.tensor(1.0)}, kind="marginal")
    torch.manual_seed(1)
    lw_samp = lw.query(q).samples
    ais_samp = ais.query(q).samples
    assert ais_samp is not None and torch.isfinite(ais_samp).all()
    assert ais_samp.shape == lw_samp.shape
    # Both are MC estimates of the same model posterior; means agree loosely
    # (MDN on tiny data is diffuse — this is a sanity bound, not a tight one).
    assert abs(float(lw_samp.mean()) - float(ais_samp.mean())) < 0.25


# ---- 7. Under-trained proposal → warning, no crash ---------------------------

@pytest.mark.slow
def test_undertrained_proposal_warns(discrete_problem, caplog):
    """A low held-out ESS triggers the diagnostic warning but not a failure."""
    fitted = _fit_adapter("cat", "ve", discrete_problem, epochs=5)
    model = fitted.model

    eng = AmortizedISEngine(n_samples=256)
    # Force the ESS proxy below threshold so the branch is deterministic.
    eng._estimate_ess_fraction = lambda *a, **k: 0.01  # type: ignore[assignment]
    with caplog.at_level(logging.WARNING):
        metrics = eng.train_proposal(
            model, n_training_samples=500, n_epochs=2, device="cpu"
        )
    assert eng.recognition_net is not None  # still produced a usable proposal
    assert any(
        "under-trained" in rec.getMessage() for rec in caplog.records
    ), "expected under-trained ESS warning"
    # And the engine still answers a query without crashing.
    torch.manual_seed(0)
    p = eng.query(model, ["X2"], {"X0": torch.tensor([0])})
    assert torch.isfinite(p).all()


# ---- 8. Fit-once: proposal trains once, reused across batch sizes -------------

@pytest.mark.slow
def test_proposal_trains_once_and_is_reused(discrete_problem, monkeypatch):
    calls = {"n": 0}
    orig = AmortizedISEngine.train_proposal

    def _spy(self, *args, **kwargs):
        calls["n"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(AmortizedISEngine, "train_proposal", _spy)

    adapter = _fit_adapter("cat", "ais", discrete_problem, n_samples=512, epochs=5)
    assert calls["n"] == 1, f"proposal trained {calls['n']}× during fit, expected 1"

    net_before = adapter._engine_obj.recognition_net
    # Sweep several batch sizes (mimics fit-once-query-many) — no retraining.
    for b in (1, 4, 16):
        queries = [_discrete_evidence_query(v % 2) for v in range(b)]
        adapter.query_batch(queries)
    assert calls["n"] == 1, "proposal must not retrain across query_batch calls"
    assert adapter._engine_obj.recognition_net is net_before
