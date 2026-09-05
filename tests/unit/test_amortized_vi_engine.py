"""Tests for AmortizedVIEngine — Engine B (#182).

Stage (a) coverage: ELBO is a verified lower bound on log p(evidence),
discrete marginals close to VE (within the wider KL-gap tolerance),
single-forward-pass speed (faster than Engine A's particle loop),
per-mechanism training (cat / neuralcat / mdn), discrete gradient flow,
the variational-gap diagnostic, and the fit-once contract.

Stage (b) appends lg / flow / hybrid coverage.

Engine B is amortized variational inference: a *bounded* approximation
(ELBO ≤ log p(evidence)), not asymptotically exact — so marginals match
VE within a wider tolerance than Engine A.  See
docs/v0.14-batched-inference-engines-research.md (Engine B section).
"""
from __future__ import annotations

import logging
import time

import pytest
import torch

from nbn.bench.adapters import NBNAdapter
from nbn.bench.domains.base import BenchmarkProblem, Query
from nbn.inference.amortized_vi import AmortizedVIEngine
from nbn.sampling.ancestral import ancestral_sample


# ---- Fixtures ----------------------------------------------------------------

def _make_small_discrete_problem(n_samples: int = 500, seed: int = 0) -> BenchmarkProblem:
    """4-node binary BN: X0 → X1 → X2, X0 → X3."""
    g = torch.Generator().manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2"), ("X0", "X3")]
    x0 = torch.randint(0, 2, (n_samples,), generator=g)
    flip = lambda x, p: torch.where(  # noqa: E731
        torch.rand(n_samples, generator=g) < p, 1 - x, x
    )
    train_data = {
        "X0": x0, "X1": flip(x0, 0.3),
        "X2": flip(flip(x0, 0.3), 0.3), "X3": flip(x0, 0.2),
    }
    variables = dict.fromkeys(train_data, ("discrete", 2))
    return BenchmarkProblem(
        name="avi_discrete", dag=dag, variables=variables,
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
        name="avi_continuous", dag=dag, variables=variables,
        train_data=train_data, test_data=train_data, queries=[],
    )


def _disc_q(v: int = 0) -> Query:
    return Query(targets=("X2",), evidence={"X0": torch.tensor(v)}, kind="marginal")


@pytest.fixture(scope="module")
def discrete_problem() -> BenchmarkProblem:
    return _make_small_discrete_problem()


@pytest.fixture(scope="module")
def continuous_problem() -> BenchmarkProblem:
    return _make_small_continuous_problem()


def _fit_adapter(mechanism, engine, problem, n_samples=1024, epochs=20):
    torch.manual_seed(0)
    adapter = NBNAdapter(
        mechanism=mechanism, engine=engine, device="cpu", n_samples=n_samples
    )
    adapter.fit(problem, epochs=epochs)
    return adapter


# ---- 1. ELBO is a lower bound on log p(evidence) -----------------------------

@pytest.mark.slow
def test_elbo_is_lower_bound(discrete_problem):
    avi = _fit_adapter("cat", "avi", discrete_problem, epochs=20)
    eng = avi._engine_obj
    for v in (0, 1):
        ev = {"X0": torch.tensor([float(v)])}
        elbo = eng.elbo(avi.model, ev, n_mc=2048)
        log_pe = float(eng._log_marginal_discrete(
            avi.model, eng.recognition_net,
            {"X0": torch.tensor([[float(v)]])}, torch.device("cpu"),
        ).item())
        assert elbo <= log_pe + 0.1, (
            f"ELBO {elbo:.4f} exceeds log p(e) {log_pe:.4f} (not a lower bound)"
        )
        assert elbo > log_pe - 20.0, f"ELBO {elbo:.4f} implausibly low vs {log_pe:.4f}"


# ---- 2 & 3. Discrete marginals close to VE (wider KL-gap tolerance) ----------

@pytest.mark.slow
def test_cat_marginals_close_to_ve(discrete_problem):
    ve = _fit_adapter("cat", "ve", discrete_problem, epochs=5)
    avi = _fit_adapter("cat", "avi", discrete_problem, epochs=20)
    for v in (0, 1):
        q = _disc_q(v)
        maxdiff = (ve.query(q).probs - avi.query(q).probs).abs().max()
        assert maxdiff < 0.15, f"X0={v}: cat AVI vs VE maxdiff {maxdiff:.4f} > 0.15"


@pytest.mark.slow
def test_neuralcat_marginals_close_to_ve(discrete_problem):
    ve = _fit_adapter("neuralcat", "ve", discrete_problem, epochs=20)
    avi = _fit_adapter("neuralcat", "avi", discrete_problem, epochs=20)
    q = _disc_q(0)
    maxdiff = (ve.query(q).probs - avi.query(q).probs).abs().max()
    assert maxdiff < 0.15, f"neuralcat AVI vs VE maxdiff {maxdiff:.4f} > 0.15"


# ---- 4. MDN continuous: finite, well-shaped output ---------------------------

@pytest.mark.slow
def test_mdn_continuous_finite(continuous_problem):
    lw = _fit_adapter("mdn", "lw", continuous_problem, n_samples=4096, epochs=30)
    avi = _fit_adapter("mdn", "avi", continuous_problem, n_samples=4096, epochs=20)
    q = Query(targets=("X2",), evidence={"X0": torch.tensor(1.0)}, kind="marginal")
    torch.manual_seed(1)
    lw_samp = lw.query(q).samples
    avi_samp = avi.query(q).samples
    assert avi_samp is not None and torch.isfinite(avi_samp).all()
    assert avi_samp.shape == lw_samp.shape
    # Variational posterior sits in a plausible range vs the LW estimate.
    assert abs(float(lw_samp.mean()) - float(avi_samp.mean())) < 0.5


# ---- 5. Single-forward-pass speed: faster than Engine A -----------------------

@pytest.mark.slow
def test_single_forward_pass_faster_than_ais(discrete_problem):
    avi = _fit_adapter("cat", "avi", discrete_problem, n_samples=1024, epochs=10)
    ais = _fit_adapter("cat", "ais", discrete_problem, n_samples=1024, epochs=5)

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


# ---- 6. Discrete gradient flow (Rao-Blackwellized ELBO) ----------------------

@pytest.mark.slow
def test_discrete_gradient_flow(discrete_problem):
    ve = _fit_adapter("cat", "ve", discrete_problem, epochs=5)
    model = ve.model
    eng = AmortizedVIEngine(n_samples=256)
    eng.train_proposal(model, n_training_samples=512, n_epochs=1, device="cpu")
    net = eng.recognition_net

    samples = ancestral_sample(model, n=256, device="cpu")
    x = eng._stack_values(samples, net.node_order, torch.device("cpu"))
    mask = (torch.rand_like(x) < 0.5).float()
    net.zero_grad()
    loss = eng._elbo_loss(model, net, x, mask)
    loss.backward()
    grad_norms = [float(p.grad.norm()) for p in net.parameters() if p.grad is not None]
    assert grad_norms, "no recognition-net gradients produced"
    assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms)
    assert sum(grad_norms) > 0.0, "discrete ELBO produced all-zero gradients"


# ---- 7. Variational-gap warning ----------------------------------------------

@pytest.mark.slow
def test_variational_gap_warns(discrete_problem, caplog):
    ve = _fit_adapter("cat", "ve", discrete_problem, epochs=5)
    model = ve.model
    eng = AmortizedVIEngine(n_samples=256)
    eng._estimate_elbo_gap = lambda *a, **k: 50.0  # type: ignore[assignment]
    with caplog.at_level(logging.WARNING):
        eng.train_proposal(model, n_training_samples=500, n_epochs=2, device="cpu")
    assert eng.recognition_net is not None
    assert any("variational gap" in rec.getMessage() for rec in caplog.records), (
        "expected variational-gap warning"
    )
    torch.manual_seed(0)
    p = eng.query(model, ["X2"], {"X0": torch.tensor([0])})
    assert torch.isfinite(p).all()


# ---- 8. Fit-once: trains once, reused across query_batch ---------------------

@pytest.mark.slow
def test_proposal_trains_once_and_is_reused(discrete_problem, monkeypatch):
    calls = {"n": 0}
    orig = AmortizedVIEngine.train_proposal

    def _spy(self, *args, **kwargs):
        calls["n"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(AmortizedVIEngine, "train_proposal", _spy)
    adapter = _fit_adapter("cat", "avi", discrete_problem, n_samples=512, epochs=5)
    assert calls["n"] == 1, f"variational net trained {calls['n']}× during fit, expected 1"

    net_before = adapter._engine_obj.recognition_net
    for b in (1, 4, 16):
        adapter.query_batch([_disc_q(v % 2) for v in range(b)])
    assert calls["n"] == 1, "must not retrain across query_batch calls"
    assert adapter._engine_obj.recognition_net is net_before


# =============================================================================
# Stage (b): lg + flow + mixed (hybrid) per-node dispatch
# =============================================================================

def _make_nongauss_problem(n_samples: int = 400, seed: int = 0) -> BenchmarkProblem:
    torch.manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2")]
    x0 = torch.randn(n_samples)
    x1 = torch.sin(2 * x0) + 0.3 * torch.randn(n_samples)
    x2 = 0.5 * x1 ** 2 + 0.3 * torch.randn(n_samples)
    train_data = {"X0": x0, "X1": x1, "X2": x2}
    variables = dict.fromkeys(train_data, ("continuous", None))
    return BenchmarkProblem(
        name="avi_nongauss", dag=dag, variables=variables,
        train_data=train_data, test_data=train_data, queries=[],
    )


def _make_hybrid_problem(n_samples: int = 500, seed: int = 0) -> BenchmarkProblem:
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
        name="avi_hybrid", dag=dag, variables=variables,
        train_data=train_data, test_data=train_data, queries=[],
    )


_CONT_Q = Query(targets=("X2",), evidence={"X0": torch.tensor(1.0)}, kind="marginal")


# ---- 9. lg continuous: AVI agrees with LW (analytic Gaussian posterior) -------

@pytest.mark.slow
def test_lg_continuous_matches_lw(continuous_problem):
    lw = _fit_adapter("lg", "lw", continuous_problem, n_samples=4096, epochs=10)
    avi = _fit_adapter("lg", "avi", continuous_problem, n_samples=4096, epochs=20)
    torch.manual_seed(1)
    lw_samp = lw.query(_CONT_Q).samples
    avi_samp = avi.query(_CONT_Q).samples
    assert avi_samp is not None and torch.isfinite(avi_samp).all()
    assert avi_samp.shape == lw_samp.shape
    # LG posterior is exact Gaussian → mean-field Normal q matches well.
    assert abs(float(lw_samp.mean()) - float(avi_samp.mean())) < 0.15


# ---- 10. flow continuous: runs, produces finite reasonable output ------------

@pytest.mark.slow
def test_flow_continuous_runs():
    prob = _make_nongauss_problem()
    lw = _fit_adapter("flow", "lw", prob, n_samples=2048, epochs=20)
    avi = _fit_adapter("flow", "avi", prob, n_samples=2048, epochs=20)
    torch.manual_seed(1)
    lw_samp = lw.query(_CONT_Q).samples
    avi_samp = avi.query(_CONT_Q).samples
    assert avi_samp is not None and torch.isfinite(avi_samp).all()
    assert avi_samp.shape == lw_samp.shape
    assert abs(float(lw_samp.mean()) - float(avi_samp.mean())) < 0.4


# ---- 11. hybrid: recognition net dispatches the right head per node ----------

@pytest.mark.slow
def test_hybrid_per_node_heads():
    prob = _make_hybrid_problem()
    avi = _fit_adapter("hybrid", "avi", prob, n_samples=2048, epochs=10)
    net = avi._engine_obj.recognition_net
    kinds = {n: net.heads[n].kind for n in net.node_order}
    assert kinds["D0"] == "discrete" and kinds["D3"] == "discrete"
    assert kinds["C1"] == "mdn" and kinds["C2"] == "mdn"
    torch.manual_seed(1)
    p_disc = avi.query(
        Query(targets=("D3",), evidence={"D0": torch.tensor(0)}, kind="marginal")
    ).probs
    assert p_disc is not None and torch.isfinite(p_disc).all()
    assert abs(float(p_disc.sum()) - 1.0) < 1e-4
    s_cont = avi.query(
        Query(targets=("C2",), evidence={"D0": torch.tensor(1)}, kind="marginal")
    ).samples
    assert s_cont is not None and torch.isfinite(s_cont).all()
