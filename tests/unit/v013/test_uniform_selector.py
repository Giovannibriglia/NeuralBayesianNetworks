"""Tests for UniformRandomSelector.

Three test classes:
  - TestProtocolConformance: isinstance check + query_role label.
  - TestSelectionInvariants: target ≠ evidence, length, scalar evidence.
  - TestCollapsePolicy: cap-at-n_test behaviour, determinism, small DAGs.

No adapter fitting — all tests are fast.
"""
from __future__ import annotations

import pytest
import torch

from benchmarking.selectors.uniform import UniformRandomSelector
from benchmarking.core.interfaces import QuerySelector
from benchmarking.domains.base import BenchmarkProblem, Query


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_problem(
    n_nodes: int = 10,
    n_test: int = 50,
    *,
    family: str = "discrete",
    cardinality: int = 3,
) -> BenchmarkProblem:
    """Build a simple chain DAG problem for testing selectors."""
    nodes = [f"X{i}" for i in range(n_nodes)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(n_nodes - 1)]
    variables = dict.fromkeys(nodes, ("discrete", cardinality))
    test_data = {n: torch.randint(0, cardinality, (n_test,)) for n in nodes}
    train_data = {n: torch.randint(0, cardinality, (n_test,)) for n in nodes}
    return BenchmarkProblem(
        name=f"test_{family}_n{n_nodes}",
        dag=edges,
        variables=variables,
        train_data=train_data,
        test_data=test_data,
        queries=[],
        family=family,
        problem_id=str(n_nodes),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProtocolConformance:
    """UniformRandomSelector satisfies the QuerySelector protocol."""

    def test_satisfies_query_selector_protocol(self):
        sel = UniformRandomSelector()
        assert isinstance(sel, QuerySelector)

    def test_query_role_is_random(self):
        sel = UniformRandomSelector()
        assert sel.query_role == "random"

    def test_select_returns_list(self):
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=5, n_test=20)
        result = sel.select(problem, n_queries=5, seed=0)
        assert isinstance(result, list)

    def test_each_element_is_query(self):
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=5, n_test=20)
        result = sel.select(problem, n_queries=5, seed=0)
        for q in result:
            assert isinstance(q, Query)


class TestSelectionInvariants:
    """Core selection invariants."""

    def test_target_not_in_evidence(self):
        """Target node must never appear in evidence dict."""
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=8, n_test=30)
        queries = sel.select(problem, n_queries=20, seed=0)
        for q in queries:
            target = q.targets[0]
            assert target not in q.evidence, (
                f"Target {target!r} found in evidence keys {list(q.evidence)}"
            )

    def test_evidence_values_are_scalars(self):
        """Each evidence value must be a 0-dim or 1-element tensor."""
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=8, n_test=30)
        queries = sel.select(problem, n_queries=10, seed=0)
        for q in queries:
            for node, val in q.evidence.items():
                assert isinstance(val, torch.Tensor)
                assert val.numel() == 1, (
                    f"Evidence for {node!r}: expected scalar, got shape {val.shape}"
                )

    def test_all_targets_from_problem_variables(self):
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=8, n_test=30)
        queries = sel.select(problem, n_queries=20, seed=0)
        for q in queries:
            assert q.targets[0] in problem.variables

    def test_all_evidence_nodes_from_problem_variables(self):
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=8, n_test=30)
        queries = sel.select(problem, n_queries=20, seed=0)
        for q in queries:
            for node in q.evidence:
                assert node in problem.variables

    def test_query_kind_is_marginal(self):
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=5, n_test=20)
        queries = sel.select(problem, n_queries=10, seed=0)
        for q in queries:
            assert q.kind == "marginal"

    def test_same_target_across_all_queries(self):
        """All queries in one select() call share the same target."""
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=8, n_test=30)
        queries = sel.select(problem, n_queries=15, seed=0)
        targets = {q.targets[0] for q in queries}
        assert len(targets) == 1


class TestCollapsePolicy:
    """Cap-at-n_test collapse policy and edge cases."""

    def test_n_queries_equals_n_test(self):
        n_test = 15
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=6, n_test=n_test)
        queries = sel.select(problem, n_queries=n_test, seed=0)
        assert len(queries) == n_test

    def test_n_queries_exceeds_n_test_caps_at_n_test(self):
        """When n_queries > n_test, result is capped at n_test (collapse policy)."""
        n_test = 10
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=6, n_test=n_test)
        queries = sel.select(problem, n_queries=n_test + 100, seed=0)
        assert len(queries) == n_test

    def test_n_queries_below_n_test_gives_exact_count(self):
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=6, n_test=50)
        queries = sel.select(problem, n_queries=12, seed=0)
        assert len(queries) == 12

    def test_zero_n_queries_returns_empty(self):
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=6, n_test=50)
        queries = sel.select(problem, n_queries=0, seed=0)
        assert queries == []

    def test_deterministic_given_same_seed(self):
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=8, n_test=30, cardinality=4)
        q1 = sel.select(problem, n_queries=10, seed=42)
        q2 = sel.select(problem, n_queries=10, seed=42)
        assert len(q1) == len(q2)
        for a, b in zip(q1, q2):
            assert a.targets == b.targets
            for node in a.evidence:
                assert torch.allclose(a.evidence[node], b.evidence[node])

    def test_different_seeds_may_give_different_targets(self):
        """Different seeds typically give different targets (probabilistic)."""
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=10, n_test=30)
        targets_0 = {sel.select(problem, n_queries=1, seed=s)[0].targets[0]
                     for s in range(20)}
        # With 10 nodes, at least some seeds should pick different targets.
        assert len(targets_0) > 1

    def test_small_dag_2_nodes_no_crash(self):
        """2-node DAG: target + at most 1 evidence node."""
        sel = UniformRandomSelector()
        problem = _make_problem(n_nodes=2, n_test=20)
        queries = sel.select(problem, n_queries=5, seed=0)
        assert len(queries) == 5
        for q in queries:
            assert q.targets[0] not in q.evidence

    def test_small_dag_1_node_returns_empty_or_queries(self):
        """1-node DAG: can still select a target, just no evidence."""
        nodes = ["X0"]
        variables = {"X0": ("discrete", 3)}
        test_data = {"X0": torch.randint(0, 3, (10,))}
        problem = BenchmarkProblem(
            name="singleton",
            dag=[],
            variables=variables,
            train_data=test_data,
            test_data=test_data,
            queries=[],
        )
        sel = UniformRandomSelector()
        queries = sel.select(problem, n_queries=5, seed=0)
        # Either 5 queries with empty evidence, or empty list — both acceptable.
        # Invariant: no query has target in evidence.
        for q in queries:
            assert q.targets[0] not in q.evidence

    def test_evidence_count_at_most_n_nodes_minus_1(self):
        """Can never have more evidence nodes than non-target nodes."""
        sel = UniformRandomSelector(n_evidence=10)  # request more than possible
        problem = _make_problem(n_nodes=4, n_test=30)
        queries = sel.select(problem, n_queries=10, seed=0)
        for q in queries:
            assert len(q.evidence) <= len(problem.variables) - 1

    def test_large_n_evidence_capped_gracefully(self):
        """Requesting more evidence nodes than available doesn't crash."""
        sel = UniformRandomSelector(n_evidence=100)
        problem = _make_problem(n_nodes=6, n_test=20)
        queries = sel.select(problem, n_queries=5, seed=0)
        assert len(queries) == 5
        for q in queries:
            assert q.targets[0] not in q.evidence
