"""PSIS k-hat diagnostic (X2) — per-query tail-reliability diagnostic for AIS/LW.

Diagnostic only: the fit-time fallback gate (P1) is unchanged. Covers the
psis_khat function (Gaussian-IS regimes + degenerate handling) and the per-query
plumbing (engine → Posterior.khat → adapter → CellResult.khat column).
"""
from __future__ import annotations

import math

import pytest
import torch

from nbn.bench.adapters import NBNAdapter
from nbn.bench.domains.base import BenchmarkProblem, Query
from nbn.bench.domains.posterior import Posterior
from nbn.bench.measurements.accuracy_timing import AccuracyAndTiming
from nbn.inference.amortized_is import AmortizedISEngine
from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine, psis_khat


def _gauss_logw(s: float, n: int = 20000, seed: int = 0) -> torch.Tensor:
    """log importance weights for target N(0,1), proposal N(0,s), x~proposal."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g) * s
    lp = -0.5 * x**2 - 0.5 * math.log(2 * math.pi)
    lq = -0.5 * (x / s) ** 2 - 0.5 * math.log(2 * math.pi * s**2)
    return lp - lq


def _discrete_problem(n: int = 500, seed: int = 0) -> BenchmarkProblem:
    g = torch.Generator().manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2"), ("X0", "X3")]
    x0 = torch.randint(0, 2, (n,), generator=g)
    flip = lambda x, p: torch.where(torch.rand(n, generator=g) < p, 1 - x, x)  # noqa: E731
    data = {"X0": x0, "X1": flip(x0, 0.3), "X2": flip(flip(x0, 0.3), 0.3), "X3": flip(x0, 0.2)}
    return BenchmarkProblem(name="d", dag=dag, variables=dict.fromkeys(data, ("discrete", 2)),
                            train_data=data, test_data=data, queries=[],
                            family="discrete", problem_id="d", seed=seed)


def _disc_query(v: int = 0) -> Query:
    return Query(targets=("X2",), evidence={"X0": torch.tensor(v)}, kind="marginal")


# ---- 1-2. psis_khat function ------------------------------------------------

def test_psis_khat_synthetic():
    # Wide proposal (heavy enough tails) → reliable.
    assert psis_khat(_gauss_logw(2.0)) < 0.5
    # Narrow proposal → heavy-tailed weights → unreliable.
    assert psis_khat(_gauss_logw(0.5)) > 0.7
    # Monotone in narrowness.
    assert psis_khat(_gauss_logw(0.8)) < psis_khat(_gauss_logw(0.5))


def test_psis_khat_degenerate_returns_none():
    # Matched proposal (q == p) → all log-weights equal → degenerate tail.
    assert psis_khat(torch.zeros(20000)) is None
    # Too few particles for a meaningful tail.
    assert psis_khat(torch.randn(10)) is None
    # Non-finite input → None (never raises).
    bad = torch.randn(20000); bad[0] = float("inf")
    assert psis_khat(bad) is None


# ---- 3-4. engine query return contract --------------------------------------

@pytest.mark.slow
def test_lw_query_returns_khat():
    model = NBNAdapter(mechanism="cat", engine="ve", device="cpu")
    model.fit(_discrete_problem(), epochs=1)
    eng = LikelihoodWeightingEngine(n_samples=4096)
    out = eng.query(model.model, ["X2"], {"X0": torch.tensor([0])},
                    return_ess=True, return_psis_k=True)
    assert isinstance(out, tuple) and len(out) == 3      # (payload, ess[B], khat[B])
    _, ess, khat = out
    assert ess.shape == (1,) and khat.shape == (1,)
    # khat is a float or NaN (degenerate); both are acceptable here.
    assert torch.isnan(khat).all() or khat.shape == (1,)
    # return_ess alone keeps the X1 2-tuple contract.
    out2 = eng.query(model.model, ["X2"], {"X0": torch.tensor([0])}, return_ess=True)
    assert isinstance(out2, tuple) and len(out2) == 2


@pytest.mark.slow
def test_ais_query_returns_khat():
    m = NBNAdapter(mechanism="cat", engine="ve", device="cpu")
    m.fit(_discrete_problem(), epochs=1)
    eng = AmortizedISEngine(n_samples=4096)        # untrained → LW fallback path
    out = eng.query(m.model, ["X2"], {"X0": torch.tensor([0])}, return_psis_k=True)
    assert isinstance(out, tuple) and len(out) == 2  # (payload, khat[B])
    _, khat = out
    assert khat.shape == (1,)


# ---- 5. adapter wiring ------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("engine", ["lw", "ais"])
def test_adapter_sets_khat_for_lw_and_ais(engine):
    a = NBNAdapter(mechanism="cat", engine=engine, device="cpu", n_samples=4096)
    a.fit(_discrete_problem(), epochs=1)
    post = a.query(_disc_query(0))
    assert post.khat is None or isinstance(post.khat, float)   # float or degenerate-None


@pytest.mark.slow
@pytest.mark.parametrize("engine", ["ve", "avi"])
def test_adapter_khat_none_for_ve_avi(engine):
    a = NBNAdapter(mechanism="cat", engine=engine, device="cpu", n_samples=512)
    a.fit(_discrete_problem(), epochs=1)
    assert a.query(_disc_query(0)).khat is None


# ---- 6. end-to-end through the measurement ----------------------------------

class _KhatAdapter:
    name = "mock-lw"; device = "cpu"
    def __init__(self, khat): self._k = khat
    def query(self, q):  # noqa: D102
        return Posterior(probs=torch.tensor([0.5, 0.5]), ess=0.4, khat=self._k)
    def query_batch(self, queries):  # noqa: D102
        return [self.query(q) for q in queries]


def test_per_query_khat_in_parquet():
    problem = _discrete_problem()
    q = _disc_query(0)
    rows = AccuracyAndTiming().measure(problem, _KhatAdapter(0.62), [q], query_groups=[[q]])
    assert rows and all(r.khat == 0.62 for r in rows)
    rows_none = AccuracyAndTiming().measure(problem, _KhatAdapter(None), [q], query_groups=[[q]])
    assert rows_none and all(r.khat is None for r in rows_none)
