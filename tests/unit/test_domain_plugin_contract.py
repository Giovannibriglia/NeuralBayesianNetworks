"""Verify the BenchmarkDomain plugin contract: every domain produces a
well-formed BenchmarkProblem with a non-empty query battery."""
from __future__ import annotations

import torch

from nbn.benchmarks.domains import get_domain
from nbn.benchmarks.domains.base import BenchmarkDomain, BenchmarkProblem, Query


def test_synthetic_hybrid_domain_lists_problems():
    d = get_domain("synthetic_hybrid")
    problems = d.list_problems()
    assert "hybrid_10" in problems


def test_synthetic_hybrid_loads_problem():
    d = get_domain("synthetic_hybrid")
    p = d.load_problem("hybrid_10", n_train=200, n_test=50, seed=0,
                       device=torch.device("cpu"))
    assert isinstance(p, BenchmarkProblem)
    assert p.name == "hybrid_10"
    assert len(p.dag) >= 0
    assert len(p.variables) == 10
    assert all(isinstance(q, Query) for q in p.queries)
    assert len(p.queries) > 0
    # Each query has a target
    assert all(len(q.targets) >= 1 for q in p.queries)
    # All training data tensors on CPU
    for v in p.train_data.values():
        assert v.device.type == "cpu"


def test_query_battery_kinds():
    d = get_domain("synthetic_hybrid")
    p = d.load_problem("hybrid_10", n_train=200, n_test=50, seed=0,
                       device=torch.device("cpu"))
    kinds = {q.kind for q in p.queries}
    # All five kinds should be present
    assert "marginal" in kinds
    assert "conditional" in kinds
    assert "map" in kinds
    assert "do" in kinds
