"""Tests for the v0.3 network-size scaling-grid builder."""
from __future__ import annotations

import torch

from benchmarking.domains import get_domain
from benchmarking.domains.synthetic_hybrid import make_scaling_grid


def test_make_scaling_grid_n_nodes_axis():
    grid = make_scaling_grid(
        n_nodes=[10, 25], n_replicates=2, n_train=100, n_test=20,
        device=torch.device("cpu"),
    )
    assert len(grid) == 4   # 2 sizes x 2 reps
    names = [p.name for p in grid]
    assert any("hybrid_n10_" in n for n in names)
    assert any("hybrid_n25_" in n for n in names)


def test_scaling_grid_domain_round_trip():
    d = get_domain("synthetic_hybrid_scaling",
                   n_nodes=[10], cardinality=4, n_replicates=1)
    names = d.list_problems()
    assert names, "expected at least one problem"
    p = d.load_problem(names[0], n_train=100, n_test=20, seed=0,
                       device=torch.device("cpu"))
    assert p.name == names[0]
    assert len(p.variables) == 10
