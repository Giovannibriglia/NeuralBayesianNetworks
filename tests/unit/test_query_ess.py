"""Query-time ESS-fraction reporting (X1, PR 2 of the AIS+LW arc).

Covers the four layers the per-query ESS fraction flows through:
- engine: ``LikelihoodWeightingEngine.query(..., return_ess=True)`` returns
  ``(payload, ess_frac[B])``; AIS inherits it unchanged;
- ``Posterior.ess`` field is optional/additive;
- ``NBNAdapter`` sets ``Posterior.ess`` for lw/ais and leaves it None for ve/avi;
- ``AccuracyAndTiming.measure`` stamps the fraction onto every row of a query.
"""
from __future__ import annotations

import pytest
import torch

from benchmarking.adapters import NBNAdapter
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.domains.posterior import Posterior
from benchmarking.measurements.accuracy_timing import AccuracyAndTiming
from nbn.inference.amortized_is import AmortizedISEngine
from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine


# ---- Fixtures ----------------------------------------------------------------

def _discrete_problem(n: int = 400, seed: int = 0) -> BenchmarkProblem:
    g = torch.Generator().manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2"), ("X0", "X3")]
    x0 = torch.randint(0, 2, (n,), generator=g)
    flip = lambda x, p: torch.where(  # noqa: E731
        torch.rand(n, generator=g) < p, 1 - x, x
    )
    data = {"X0": x0, "X1": flip(x0, 0.3),
            "X2": flip(flip(x0, 0.3), 0.3), "X3": flip(x0, 0.2)}
    return BenchmarkProblem(
        name="ess_d", dag=dag, variables=dict.fromkeys(data, ("discrete", 2)),
        train_data=data, test_data=data, queries=[],
        family="discrete", problem_id="ess_d", seed=seed,
    )


def _continuous_problem(n: int = 400, seed: int = 0) -> BenchmarkProblem:
    torch.manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2")]
    x0 = torch.randn(n)
    x1 = 0.5 * x0 + 0.5 * torch.randn(n)
    x2 = 0.5 * x1 + 0.5 * torch.randn(n)
    data = {"X0": x0, "X1": x1, "X2": x2}
    return BenchmarkProblem(
        name="ess_c", dag=dag, variables=dict.fromkeys(data, ("continuous", None)),
        train_data=data, test_data=data, queries=[],
        family="continuous_gauss", problem_id="ess_c", seed=seed,
    )


def _fit(mech: str, engine: str, problem: BenchmarkProblem,
         n_samples: int = 512, epochs: int = 5) -> NBNAdapter:
    torch.manual_seed(0)
    a = NBNAdapter(mechanism=mech, engine=engine, device="cpu", n_samples=n_samples)
    a.fit(problem, epochs=epochs)
    return a


def _disc_query(v: int = 0) -> Query:
    return Query(targets=("X2",), evidence={"X0": torch.tensor(v)}, kind="marginal")


# ---- 1-2. Engine return contract ---------------------------------------------

@pytest.mark.slow
def test_lw_returns_ess_when_requested():
    model = _fit("cat", "ve", _discrete_problem()).model
    eng = LikelihoodWeightingEngine(n_samples=256)
    out = eng.query(model, ["X2"], {"X0": torch.tensor([0])}, return_ess=True)
    assert isinstance(out, tuple) and len(out) == 2
    _, ess = out
    assert ess.shape == (1,)                          # [B], B=1
    assert torch.isfinite(ess).all()
    assert float(ess.min()) > 0.0 and float(ess.max()) <= 1.0 + 1e-6


@pytest.mark.slow
def test_lw_default_behavior_unchanged():
    model = _fit("cat", "ve", _discrete_problem()).model
    eng = LikelihoodWeightingEngine(n_samples=256)
    out = eng.query(model, ["X2"], {"X0": torch.tensor([0])})
    # Discrete single-target default path is a bare probs tensor, never a tuple.
    assert isinstance(out, torch.Tensor)


# ---- 3-4. ESS-fraction formula edge cases ------------------------------------

@pytest.mark.slow
def test_ess_fraction_uniform_weights_is_one():
    model = _fit("lg", "lw", _continuous_problem()).model
    S = 64
    eng = LikelihoodWeightingEngine(n_samples=S)
    real_run = eng._run

    def _uniform(*a, **k):
        log_w, buf = real_run(*a, **k)
        return torch.zeros_like(log_w), buf       # uniform weights, real buffer

    eng._run = _uniform
    _, ess = eng.query(model, ["X2"], {"X0": torch.tensor([0.0])}, return_ess=True)
    assert torch.allclose(ess, torch.ones_like(ess), atol=1e-5)


@pytest.mark.slow
def test_ess_fraction_degenerate_weights_is_near_zero():
    model = _fit("lg", "lw", _continuous_problem()).model
    S = 64
    eng = LikelihoodWeightingEngine(n_samples=S)
    real_run = eng._run

    def _degenerate(*a, **k):
        log_w, buf = real_run(*a, **k)
        z = torch.full_like(log_w, -50.0)
        z[..., 0] = 0.0                               # all mass on one particle
        return z, buf

    eng._run = _degenerate
    _, ess = eng.query(model, ["X2"], {"X0": torch.tensor([0.0])}, return_ess=True)
    assert torch.allclose(ess, torch.full_like(ess, 1.0 / S), atol=1e-3)


# ---- 5. AIS inherits the ESS-capable query -----------------------------------

@pytest.mark.slow
def test_ais_inherits_ess_path():
    model = _fit("cat", "ve", _discrete_problem()).model
    # Untrained proposal → _run falls back to LW; query() is inherited unchanged.
    eng = AmortizedISEngine(n_samples=256)
    out = eng.query(model, ["X2"], {"X0": torch.tensor([0])}, return_ess=True)
    assert isinstance(out, tuple) and len(out) == 2
    _, ess = out
    assert float(ess.min()) > 0.0 and float(ess.max()) <= 1.0 + 1e-6


# ---- 6. Posterior.ess is optional --------------------------------------------

def test_posterior_ess_field_optional():
    p = Posterior(probs=torch.tensor([0.5, 0.5]))
    assert p.ess is None                              # default, no validation error
    p2 = Posterior(samples=torch.randn(8), ess=0.42)
    assert p2.ess == 0.42


# ---- 7-8. Adapter wiring -----------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("engine", ["lw", "ais"])
def test_adapter_sets_ess_for_lw_and_ais(engine):
    a = _fit("cat", engine, _discrete_problem())
    post = a.query(_disc_query(0))
    assert post.ess is not None
    assert 0.0 < post.ess <= 1.0 + 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("engine", ["ve", "avi"])
def test_adapter_sets_ess_none_for_ve_avi(engine):
    a = _fit("cat", engine, _discrete_problem())
    post = a.query(_disc_query(0))
    assert post.ess is None


# ---- 9. Measurement threads ess onto every row of a query --------------------

class _EssAdapter:
    """Adapter double whose posteriors carry a fixed ess."""

    name = "mock-lw"
    device = "cpu"

    def __init__(self, ess):
        self._ess = ess

    def query(self, q: Query) -> Posterior:
        return Posterior(probs=torch.tensor([0.5, 0.5]), ess=self._ess)

    def query_batch(self, queries: list[Query]) -> list[Posterior]:
        return [self.query(q) for q in queries]


def test_cellresult_ess_threaded_through_measurement():
    problem = _discrete_problem()
    q = _disc_query(0)

    rows = AccuracyAndTiming().measure(
        problem, _EssAdapter(ess=0.73), [q], query_groups=[[q]],
    )
    assert rows
    assert all(r.ess == 0.73 for r in rows), \
        "every metric+timing row of the query should carry its ESS fraction"

    rows_none = AccuracyAndTiming().measure(
        problem, _EssAdapter(ess=None), [q], query_groups=[[q]],
    )
    assert rows_none
    assert all(r.ess is None for r in rows_none)
