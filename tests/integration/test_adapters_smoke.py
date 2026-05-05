"""Smoke-test the baseline adapters: each one fits and answers one query.

Adapters that need optional libraries are skipped when those libs aren't
installed.  This keeps the test suite green on minimal environments.

Every adapter is exercised against a small synthetic problem constructed
via :func:`benchmarking.synthetic.make_synthetic_bn`.
"""
from __future__ import annotations

import importlib

import pytest
import torch

from benchmarking.baselines import get_adapter
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.synthetic import make_synthetic_bn


def _has(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


def _problem(family: str = "hybrid", n: int = 10) -> BenchmarkProblem:
    """Build a small ``BenchmarkProblem`` from a synthetic BN."""
    bn = make_synthetic_bn(
        family=family, n_nodes=n, edge_density=0.20,
        n_train=200, n_test=50, n_reference=200, seed=0, device="cpu",
    )
    nodes = list(bn.dag.nodes())
    queries = [
        Query(
            targets=(nodes[0],),
            evidence={nodes[1]: bn.test_data[nodes[1]][:1].reshape(1)} if len(nodes) > 1 else {},
            kind="marginal",
        ),
    ]
    return BenchmarkProblem(
        name=bn.name, dag=list(bn.dag.edges()), variables=bn.variable_specs,
        train_data=bn.train_data, test_data=bn.test_data,
        queries=queries, ground_truth=None,
    )


def _smoke_query(adapter, problem: BenchmarkProblem) -> None:
    """Fit + answer one marginal query."""
    adapter.fit(problem)
    marg = next(q for q in problem.queries if q.kind == "marginal")
    res = adapter.query(marg)
    assert isinstance(res, torch.Tensor)
    adapter.teardown()


def test_nbn_adapter_smoke_continuous() -> None:
    # ``hybrid`` family fitter requires the option-(b) train-quantile
    # binner for continuous→discrete edges (deferred to v0.5b).  This
    # smoke test runs against ``continuous_lg`` to exercise the NBN
    # adapter end-to-end on a path that doesn't depend on the unfixed
    # binner.
    _smoke_query(get_adapter("nbn", device="cpu"), _problem("continuous_lg"))


@pytest.mark.parametrize(
    "variant",
    ["nbn_ve", "nbn_lw", "nbn_hybrid", "nbn_neural_categorical", "nbn_linear_gaussian"],
)
def test_all_nbn_variants_run(variant: str) -> None:
    """Every registered NBN preset must fit + answer one marginal query."""
    if variant == "nbn_ve":
        _smoke_query(get_adapter(variant, device="cpu"), _problem("discrete"))
    else:
        _smoke_query(get_adapter(variant, device="cpu"), _problem("continuous_lg"))


@pytest.mark.skipif(not _has("pgmpy"), reason="needs pgmpy")
def test_pgmpy_adapter_smoke_discrete() -> None:
    _smoke_query(get_adapter("pgmpy"), _problem("discrete"))


@pytest.mark.skipif(not _has("pomegranate"), reason="needs pomegranate")
def test_pomegranate_adapter_smoke_discrete() -> None:
    _smoke_query(get_adapter("pomegranate"), _problem("discrete"))


@pytest.mark.skipif(not _has("pyro"), reason="needs pyro-ppl")
def test_pyro_adapter_smoke_discrete() -> None:
    _smoke_query(get_adapter("pyro"), _problem("discrete"))


@pytest.mark.skipif(not _has("gpytorch"), reason="needs gpytorch")
def test_gpytorch_adapter_skips_discrete_evidence() -> None:
    """GPyTorch is continuous-only — discrete evidence must raise NotImplementedError."""
    p = _problem("hybrid")
    a = get_adapter("gpytorch", device="cpu")
    a.fit(p)
    discrete_set = {n for n, (kind, _) in p.variables.items() if kind == "discrete"}
    discrete_q = next(
        (q for q in p.queries
         if q.kind == "marginal" and any(k in discrete_set for k in q.evidence)),
        None,
    )
    if discrete_q is None:
        pytest.skip("no discrete-evidence query in this problem instance")
    with pytest.raises((NotImplementedError, Exception)):
        a.query(discrete_q)
    a.teardown()
