"""The LG posterior must be built from the network's actual edges.

``_lg_posterior_moments`` assembles a structural equation model by reading
each CPD's regression coefficients and placing them at the parents' indices.
It used to source the parent list as
``list(cpd.evidence) if hasattr(cpd, "evidence") else []`` — and that ``else``
is a silent-wrong-number trap: an empty list leaves the coefficient row all
zeros, which drops the node's parents from the SEM entirely.  Every posterior
below would then be computed from a *different, edge-free* network, and
nothing would raise.

The guard never fired (pgmpy 1.1.2 still exposes ``.evidence``), but pgmpy has
removed sibling attributes before — ``from_bif`` died on exactly that — so
this pins both halves: the attribute is there, and its absence would be loud.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pgmpy")


def test_linear_gaussian_cpd_still_exposes_evidence():
    """If this fails, _lg_posterior_moments must be ported, not defaulted."""
    from pgmpy.factors.continuous import LinearGaussianCPD

    cpd = LinearGaussianCPD(variable="Y", beta=[0.5, 2.0], std=1.0, evidence=["X"])
    assert hasattr(cpd, "evidence")
    assert list(cpd.evidence) == ["X"]


def test_missing_evidence_attribute_raises_instead_of_dropping_edges():
    """The failure mode must be an exception, not an edge-free SEM."""
    from nbn.bench.adapters.pgmpy_adapter import PgmpyAdapter

    class _NoEvidenceCPD:
        variable = "Y"
        beta = [0.5, 2.0]
        std = 1.0

    class _Model:
        @staticmethod
        def get_cpds():
            return [_NoEvidenceCPD()]

    adapter = PgmpyAdapter.__new__(PgmpyAdapter)
    adapter._model = _Model()
    adapter._lg_topo = ["Y"]

    class _Q:
        targets = ["Y"]
        evidence: dict = {}

    with pytest.raises(AttributeError, match="no longer exposes"):
        adapter._lg_posterior_moments(_Q())


def test_lg_joint_is_built_once_per_fit_not_per_query(monkeypatch):
    """D9: the joint Gaussian (an O(n^3) inverse) was rebuilt on every query."""
    import torch

    from nbn.bench.adapters.pgmpy_adapter import PgmpyAdapter
    from nbn.bench.domains.base import Query
    from nbn.bench.problems import SyntheticConfig, SyntheticProblemSource

    cfg = SyntheticConfig(families=["continuous_lg"], n_nodes_list=[6], seeds=[0],
                          n_train=300, n_test=10, n_reference=50, edge_density=0.5)
    problem = next(iter(SyntheticProblemSource().iter_problems(cfg)))
    nodes = list(problem.variables)
    queries = [Query(targets=(nodes[-1 - i],), evidence={nodes[i]: torch.tensor(0.1 * i)},
                     kind="marginal") for i in range(5)]

    fresh = []
    for q in queries:                     # reference: a new adapter per query
        a = PgmpyAdapter("lg", "predict")
        a.fit(problem)
        fresh.append(a._lg_posterior_moments(q))

    adapter = PgmpyAdapter("lg", "predict")
    adapter.fit(problem)
    calls = []
    real_inv = torch.linalg.inv
    monkeypatch.setattr(torch.linalg, "inv", lambda m: calls.append(1) or real_inv(m))
    cached = [adapter._lg_posterior_moments(q) for q in queries]
    assert len(calls) == 1
    for (m1, s1), (m2, s2) in zip(fresh, cached):
        assert torch.allclose(m1, m2) and torch.allclose(s1, s2)
    adapter.fit(problem)                  # a refit invalidates the cache
    adapter._lg_posterior_moments(queries[0])
    assert len(calls) == 2
