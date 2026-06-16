"""Tests for AmortizedISEngine — Engine A (#181).

Stage (a) coverage: B=1 correctness vs VE, convergence to VE as the
particle count grows, the batching-speedup contract (per-query time
falls as B grows), per-mechanism training (cat / neuralcat / mdn), the
low-ESS proposal-rejection fallback to LW, and the fit-once contract (the
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


# ---- 7. Low fit-time ESS → reject proposal, fall back to LW ------------------

@pytest.mark.slow
def test_low_ess_triggers_fallback(discrete_problem, caplog):
    """A low held-out ESS rejects the learned proposal → engine falls back to LW.

    Was ``test_undertrained_proposal_warns``: pre-fallback the engine kept the
    proposal and only warned.  This PR makes the warning actionable — the same
    forced-low-ESS fixture now drives the fallback path (recognition_net unset),
    addressing the discrete-network accuracy catastrophe from the June 12 run.
    """
    fitted = _fit_adapter("cat", "ve", discrete_problem, epochs=5)
    model = fitted.model

    eng = AmortizedISEngine(n_samples=256)
    # Force the ESS proxy below threshold so the branch is deterministic.
    eng._estimate_ess_fraction = lambda *a, **k: 0.01  # type: ignore[assignment]
    with caplog.at_level(logging.WARNING):
        metrics = eng.train_proposal(
            model, n_training_samples=500, steps_per_node=1, device="cpu"
        )
    # Proposal rejected → recognition_net unset → _run takes the inherited LW
    # path; the engine still answers a query without crashing.
    assert eng.recognition_net is None
    assert metrics["proposal_used"] == "lw_fallback"
    torch.manual_seed(0)
    p = eng.query(model, ["X2"], {"X0": torch.tensor([0])})
    assert torch.isfinite(p).all()


@pytest.mark.slow
def test_good_ess_keeps_learned_proposal(discrete_problem):
    """A proposal that clears the ESS gate is retained (no fallback)."""
    fitted = _fit_adapter("cat", "ve", discrete_problem, epochs=5)
    model = fitted.model

    eng = AmortizedISEngine(n_samples=256)
    # Real training on the trivial 4-node fixture clears the 10% ESS gate.
    metrics = eng.train_proposal(model, device="cpu")
    assert metrics["ess_fraction"] is None or metrics["ess_fraction"] >= 0.1
    assert eng.recognition_net is not None
    assert metrics["proposal_used"] == "learned"


@pytest.mark.slow
def test_fallback_logged(discrete_problem, caplog):
    """The fallback action logs a message distinct from the low-ESS diagnostic."""
    fitted = _fit_adapter("cat", "ve", discrete_problem, epochs=5)
    model = fitted.model

    eng = AmortizedISEngine(n_samples=256)
    eng._estimate_ess_fraction = lambda *a, **k: 0.01  # type: ignore[assignment]
    with caplog.at_level(logging.WARNING):
        eng.train_proposal(model, n_training_samples=500, steps_per_node=1, device="cpu")
    msgs = [rec.getMessage() for rec in caplog.records]
    # Diagnostic ("why") and action ("what we did") are distinct log records.
    assert any("under-trained" in m for m in msgs), "expected ESS diagnostic"
    assert any("falling back to LW" in m for m in msgs), "expected fallback action"
    assert not any(
        "under-trained" in m and "falling back to LW" in m for m in msgs
    ), "diagnostic and action should be distinct records"


@pytest.mark.slow
def test_proposal_used_column(discrete_problem):
    """train_proposal reports proposal_used in both the learned and fallback branches."""
    fitted = _fit_adapter("cat", "ve", discrete_problem, epochs=5)
    model = fitted.model

    good = AmortizedISEngine(n_samples=256)
    assert good.train_proposal(model, device="cpu")["proposal_used"] == "learned"

    bad = AmortizedISEngine(n_samples=256)
    bad._estimate_ess_fraction = lambda *a, **k: 0.01  # type: ignore[assignment]
    m_bad = bad.train_proposal(
        model, n_training_samples=500, steps_per_node=1, device="cpu")
    assert m_bad["proposal_used"] == "lw_fallback"


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


# =============================================================================
# Stage (b): flow + lg output heads, plus mixed (hybrid) per-node dispatch
# =============================================================================

def _make_nongauss_problem(n_samples: int = 400, seed: int = 0) -> BenchmarkProblem:
    """3-node non-Gaussian chain (exercises the flow head)."""
    torch.manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2")]
    x0 = torch.randn(n_samples)
    x1 = torch.sin(2 * x0) + 0.3 * torch.randn(n_samples)
    x2 = 0.5 * x1 ** 2 + 0.3 * torch.randn(n_samples)
    train_data = {"X0": x0, "X1": x1, "X2": x2}
    variables = dict.fromkeys(train_data, ("continuous", None))
    return BenchmarkProblem(
        name="ais_nongauss", dag=dag, variables=variables,
        train_data=train_data, test_data=train_data, queries=[],
    )


def _make_hybrid_problem(n_samples: int = 500, seed: int = 0) -> BenchmarkProblem:
    """Mixed discrete+continuous net: D0 → C1 → C2, D0 → D3."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    dag = [("D0", "C1"), ("C1", "C2"), ("D0", "D3")]
    d0 = torch.randint(0, 2, (n_samples,), generator=g)
    c1 = d0.float() + 0.5 * torch.randn(n_samples)
    c2 = 0.5 * c1 + 0.5 * torch.randn(n_samples)
    d3 = torch.where(torch.rand(n_samples, generator=g) < 0.2, 1 - d0, d0)
    train_data = {"D0": d0, "C1": c1, "C2": c2, "D3": d3}
    variables = {
        "D0": ("discrete", 2), "C1": ("continuous", None),
        "C2": ("continuous", None), "D3": ("discrete", 2),
    }
    return BenchmarkProblem(
        name="ais_hybrid", dag=dag, variables=variables,
        train_data=train_data, test_data=train_data, queries=[],
    )


_CONT_Q = Query(targets=("X2",), evidence={"X0": torch.tensor(1.0)}, kind="marginal")


# ---- 9. lg continuous: AIS agrees with LW (closed-form Gaussian) --------------

@pytest.mark.slow
def test_lg_continuous_matches_lw(continuous_problem):
    lw = _fit_adapter("lg", "lw", continuous_problem, n_samples=8192, epochs=10)
    ais = _fit_adapter("lg", "ais", continuous_problem, n_samples=8192, epochs=10)
    torch.manual_seed(1)
    lw_samp = lw.query(_CONT_Q).samples
    ais_samp = ais.query(_CONT_Q).samples
    assert ais_samp is not None and torch.isfinite(ais_samp).all()
    assert ais_samp.shape == lw_samp.shape
    # LG posterior is exact Gaussian → tight agreement expected.
    assert abs(float(lw_samp.mean()) - float(ais_samp.mean())) < 0.1


# ---- 10. flow continuous: trains, produces reasonable finite output ----------

@pytest.mark.slow
def test_flow_continuous_runs(continuous_problem):
    """Flow head trains and produces finite, reasonably-located samples.

    The proposal is a Gaussian surrogate; the weight uses the *true* flow
    density, so the estimate stays asymptotically correct.  Flow on tiny
    data is high-variance, so this is a sanity bound (per the issue's
    'training succeeds and outputs are reasonable' bar), not a tight match.
    """
    prob = _make_nongauss_problem()
    lw = _fit_adapter("flow", "lw", prob, n_samples=4096, epochs=20)
    ais = _fit_adapter("flow", "ais", prob, n_samples=4096, epochs=20)
    torch.manual_seed(1)
    lw_samp = lw.query(_CONT_Q).samples
    ais_samp = ais.query(_CONT_Q).samples
    assert ais_samp is not None and torch.isfinite(ais_samp).all()
    assert ais_samp.shape == lw_samp.shape
    assert abs(float(lw_samp.mean()) - float(ais_samp.mean())) < 0.3


# ---- 11. hybrid: recognition net dispatches the right head per node ----------

@pytest.mark.slow
def test_hybrid_per_node_heads():
    """Mixed discrete+continuous net → cat heads for D*, mdn heads for C*."""
    prob = _make_hybrid_problem()
    ais = _fit_adapter("hybrid", "ais", prob, n_samples=4096, epochs=10)
    net = ais._engine_obj.recognition_net
    kinds = {n: net.heads[n].kind for n in net.node_order}
    assert kinds["D0"] == "discrete" and kinds["D3"] == "discrete"
    assert kinds["C1"] == "mdn" and kinds["C2"] == "mdn"
    # A discrete-target query and a continuous-target query both work.
    torch.manual_seed(1)
    p_disc = ais.query(
        Query(targets=("D3",), evidence={"D0": torch.tensor(0)}, kind="marginal")
    ).probs
    assert p_disc is not None and torch.isfinite(p_disc).all()
    assert abs(float(p_disc.sum()) - 1.0) < 1e-4
    s_cont = ais.query(
        Query(targets=("C2",), evidence={"D0": torch.tensor(1)}, kind="marginal")
    ).samples
    assert s_cont is not None and torch.isfinite(s_cont).all()
