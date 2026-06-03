"""Unit tests for n_parameters computation (#133).

Tests the pure function on known structures, the problem-reading wrapper on a
tiny fake problem, and an end-to-end check on a real bnlearn network.
"""
from __future__ import annotations

import types

import pytest

from benchmarking.domains._n_parameters import (
    n_parameters_from_cardinalities,
    n_parameters_from_problem,
)


class TestNParametersFromCardinalities:
    """Pure-function tests on known structures."""

    def test_single_root_node(self):
        assert n_parameters_from_cardinalities({"A": 4}, {"A": []}) == 4

    def test_chain(self):
        # A -> B -> C with K=2,3,4: 2 + 2*3 + 3*4 = 20
        result = n_parameters_from_cardinalities(
            {"A": 2, "B": 3, "C": 4},
            {"A": [], "B": ["A"], "C": ["B"]},
        )
        assert result == 2 + 2 * 3 + 3 * 4

    def test_v_structure(self):
        # A -> C <- B with K=3,3,2: 3 + 3 + 2*3*3 = 24
        result = n_parameters_from_cardinalities(
            {"A": 3, "B": 3, "C": 2},
            {"A": [], "B": [], "C": ["A", "B"]},
        )
        assert result == 3 + 3 + 2 * 3 * 3

    def test_continuous_node_contributes_zero(self):
        # B is continuous (K=0): A=3, B=0*3=0 -> 3
        result = n_parameters_from_cardinalities(
            {"A": 3, "B": 0}, {"A": [], "B": ["A"]},
        )
        assert result == 3

    def test_discrete_child_of_continuous_parent_contributes_zero(self):
        # A continuous (0), B discrete (3) with parent A: B = 3*0 = 0 -> 0
        result = n_parameters_from_cardinalities(
            {"A": 0, "B": 3}, {"A": [], "B": ["A"]},
        )
        assert result == 0

    def test_empty_network(self):
        assert n_parameters_from_cardinalities({}, {}) == 0


class TestNParametersFromProblem:
    """The problem-reading wrapper."""

    def test_reads_variables_and_dag(self):
        problem = types.SimpleNamespace(
            variables={"A": ("discrete", 2), "B": ("discrete", 3)},
            dag=[("A", "B")],
        )
        # A: 2, B: 3*2 = 6 -> 8
        assert n_parameters_from_problem(problem) == 8

    def test_continuous_variables_zeroed(self):
        problem = types.SimpleNamespace(
            variables={"A": ("discrete", 2), "B": ("continuous", 1)},
            dag=[("A", "B")],
        )
        # A: 2, B continuous -> 0 ; total 2
        assert n_parameters_from_problem(problem) == 2

    def test_missing_structure_returns_none(self):
        assert n_parameters_from_problem(types.SimpleNamespace()) is None
        assert n_parameters_from_problem(
            types.SimpleNamespace(variables={}, dag=[])
        ) is None


class TestNParametersIntegration:
    """End-to-end on a real bnlearn network."""

    @pytest.mark.slow
    def test_asia_n_parameters(self):
        """asia: 8 binary nodes. Total discrete CPT cells:
        asia(2) + tub|asia(4) + smoke(2) + lung|smoke(4) + bronc|smoke(4)
        + either|lung,tub(8) + xray|either(4) + dysp|bronc,either(8) = 36."""
        from benchmarking.problems.bnlearn import BnlearnConfig, BnlearnProblemSource

        cfg = BnlearnConfig(networks=["asia"], seeds=[0],
                            n_train=100, n_test=20, n_reference=20)
        problem = next(BnlearnProblemSource().iter_problems(cfg))
        assert n_parameters_from_problem(problem) == 36
