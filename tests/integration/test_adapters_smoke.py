"""Smoke-test the baseline adapters: each one fits and answers one query.

Adapters that need optional libraries are skipped when those libs aren't
installed. This keeps the test suite green on minimal environments.
"""
from __future__ import annotations

import importlib

import pytest
import torch

from benchmarking.baselines import get_adapter
from benchmarking.domains import get_domain
from benchmarking.domains.base import Query


def _has(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


def _hybrid_problem(n=10):
    return get_domain("synthetic_hybrid").load_problem(
        f"hybrid_{n}", n_train=200, n_test=50, seed=0, device=torch.device("cpu"),
    )


def _bnlearn_problem(name="asia"):
    return get_domain("bnlearn").load_problem(
        name, n_train=200, n_test=50, seed=0, device=torch.device("cpu"),
    )


def _smoke_query(adapter, problem):
    """Fit + answer one marginal query."""
    adapter.fit(problem)
    marg = next(q for q in problem.queries if q.kind == "marginal")
    res = adapter.query(marg)
    assert isinstance(res, torch.Tensor)
    adapter.teardown()


def test_nbn_adapter_smoke_hybrid():
    _smoke_query(get_adapter("nbn", device="cpu"), _hybrid_problem())


@pytest.mark.skipif(not _has("pgmpy"), reason="needs pgmpy")
def test_pgmpy_adapter_smoke_discrete():
    _smoke_query(get_adapter("pgmpy"), _bnlearn_problem("asia"))


@pytest.mark.skipif(not _has("pomegranate"), reason="needs pomegranate")
def test_pomegranate_adapter_smoke_discrete():
    _smoke_query(get_adapter("pomegranate"), _bnlearn_problem("asia"))


@pytest.mark.skipif(not _has("pyro"), reason="needs pyro-ppl")
def test_pyro_adapter_smoke_discrete():
    _smoke_query(get_adapter("pyro"), _bnlearn_problem("asia"))


@pytest.mark.skipif(not _has("gpytorch"), reason="needs gpytorch")
def test_gpytorch_adapter_skips_discrete_evidence():
    """GPyTorch is continuous-only — discrete evidence must raise NotImplementedError."""
    p = _hybrid_problem()
    a = get_adapter("gpytorch", device="cpu")
    a.fit(p)
    # Find a query whose evidence has at least one discrete var
    discrete_set = {n for n, (kind, _) in p.variables.items() if kind == "discrete"}
    discrete_q = next(
        (q for q in p.queries
         if q.kind == "conditional" and any(k in discrete_set for k in q.evidence)),
        None,
    )
    if discrete_q is None:
        pytest.skip("no discrete-evidence query in this problem instance")
    with pytest.raises((NotImplementedError, Exception)):
        a.query(discrete_q)
    a.teardown()
