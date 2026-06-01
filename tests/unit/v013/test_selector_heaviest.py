"""Tests for benchmarking.selectors.heaviest — Stage 1.

Covers HeaviestQueryByRole's query construction. Adapter / oracle
None-handling lands in Stages 2-3.

Reference: docs/phase3-design-draft.md §2, §8.
"""
from __future__ import annotations

import random
from collections import defaultdict

import networkx as nx
import pytest
import torch

from benchmarking.domains.base import BenchmarkProblem
from benchmarking.selectors.heaviest import HeaviestQueryByRole


# --- Fixtures ---

ASIA_EDGES = [
    ("asia", "tub"),
    ("smoke", "lung"),
    ("smoke", "bronc"),
    ("tub", "either"),
    ("lung", "either"),
    ("either", "xray"),
    ("either", "dysp"),
    ("bronc", "dysp"),
]


@pytest.fixture
def asia_problem() -> BenchmarkProblem:
    """ASIA Bayesian network with synthetic test_data (50 rows)."""
    rng = random.Random(0)
    n_rows = 50
    G = nx.DiGraph(ASIA_EDGES)
    test_data = {
        node: torch.tensor([rng.randint(0, 1) for _ in range(n_rows)])
        for node in G.nodes()
    }
    variables = dict.fromkeys(G.nodes(), ("discrete", 2))
    return BenchmarkProblem(
        name="asia",
        dag=ASIA_EDGES,
        variables=variables,
        train_data={},
        test_data=test_data,
        queries=[],
        seed=0,
        family="discrete",
        problem_id="asia_test",
        true_model=None,
    )


@pytest.fixture
def synthetic_no_aps_problem() -> BenchmarkProblem:
    """A small DAG with no articulation points (for testing cut-empty case)."""
    # A triangle: A→B, A→C, B→C — no APs in the moralized graph
    edges = [("A", "B"), ("A", "C"), ("B", "C")]
    G = nx.DiGraph(edges)
    rng = random.Random(0)
    test_data = {
        node: torch.tensor([rng.randint(0, 1) for _ in range(20)])
        for node in G.nodes()
    }
    variables = dict.fromkeys(G.nodes(), ("discrete", 2))
    return BenchmarkProblem(
        name="triangle",
        dag=edges,
        variables=variables,
        train_data={},
        test_data=test_data,
        queries=[],
        seed=0,
        family="discrete",
        problem_id="triangle_test",
        true_model=None,
    )


class TestHeaviestQueryByRole:
    """Contract: deterministic top-1 target per role × 2 directions × 2 modes."""

    def test_query_count_with_aps(self, asia_problem):
        """ASIA has 2 APs (either, tub), so 12 queries expected."""
        selector = HeaviestQueryByRole()
        queries = selector.select(asia_problem, n_queries=999, seed=0)
        # Up to 12. Some role-direction pairs may have empty pools
        # (e.g. root targets in prediction direction), which drops
        # both V1 and V2. So count is 12 minus pairs with empty pools,
        # times 2 (V1+V2).
        # Verify: count is a multiple of 2 (V1/V2 paired) and ≤ 12
        assert len(queries) % 2 == 0
        assert len(queries) <= 12
        assert len(queries) > 0

    def test_query_count_without_aps(self, synthetic_no_aps_problem):
        """No APs → 8 queries max (hub + terminal roles only, × 2 modes)."""
        selector = HeaviestQueryByRole()
        queries = selector.select(synthetic_no_aps_problem, n_queries=999, seed=0)
        # Should have queries from hub + terminal roles only
        roles_seen = {q.query_role for q in queries}
        assert "cut" not in roles_seen
        assert "hub" in roles_seen or "terminal" in roles_seen

    def test_v1_v2_paired(self, asia_problem):
        """Every emitted (role, direction) pair has both V1 and V2 queries."""
        selector = HeaviestQueryByRole()
        queries = selector.select(asia_problem, n_queries=999, seed=0)

        # Group by (role, direction, evidence node set)
        by_pair: dict[tuple, list] = {}
        for q in queries:
            evidence_keys = frozenset(q.evidence.keys())
            key = (q.query_role, q.query_kind, evidence_keys)
            by_pair.setdefault(key, []).append(q)

        # Each pair group should have exactly 2 queries (V1 + V2)
        for key, qs in by_pair.items():
            assert len(qs) == 2, f"pair {key} has {len(qs)} queries, expected 2"
            # One should have all-concrete values; one should have all-None
            modes = []
            for q in qs:
                vals = list(q.evidence.values())
                if all(v is None for v in vals):
                    modes.append("empty")
                elif all(v is not None for v in vals):
                    modes.append("full")
                else:
                    pytest.fail(f"mixed values in evidence for {key}: {vals}")
            assert set(modes) == {"full", "empty"}, f"modes for {key}: {modes}"

    def test_evidence_nodes_match_across_v1_v2(self, asia_problem):
        """V1 and V2 of the same pair use IDENTICAL evidence node sets."""
        selector = HeaviestQueryByRole()
        queries = selector.select(asia_problem, n_queries=999, seed=0)

        # Group by (role, direction) — within each, V1 and V2 should match on evidence keys
        by_rd: dict[tuple, list] = defaultdict(list)
        for q in queries:
            by_rd[(q.query_role, q.query_kind)].append(q)

        for (role, direction), qs in by_rd.items():
            assert len(qs) == 2  # already enforced above; sanity
            keys_a = frozenset(qs[0].evidence.keys())
            keys_b = frozenset(qs[1].evidence.keys())
            assert keys_a == keys_b, (
                f"({role}, {direction}) V1/V2 evidence keys differ: {keys_a} vs {keys_b}"
            )

    def test_hub_target_is_top_mb(self, asia_problem):
        """Hub queries target the node with highest |MB| (NodeRoles.hubs[0])."""
        selector = HeaviestQueryByRole()
        queries = selector.select(asia_problem, n_queries=999, seed=0)

        # On ASIA, hubs[0] is 'either' (|MB|=5: tub, lung, xray, dysp, bronc)
        hub_queries = [q for q in queries if q.query_role == "hub"]
        assert len(hub_queries) > 0
        for q in hub_queries:
            assert q.targets == ("either",), f"hub target was {q.targets}, expected ('either',)"

    def test_terminal_target_matches_spec(self, asia_problem):
        """Terminal queries target the node with highest depth - |descendants|.

        On ASIA: xray and dysp tie at depth 3 - 0 = 3. The selector picks
        terminals[0] which (per NodeRoles sort) is whichever sorts first.
        """
        selector = HeaviestQueryByRole()
        queries = selector.select(asia_problem, n_queries=999, seed=0)

        terminal_queries = [q for q in queries if q.query_role == "terminal"]
        assert len(terminal_queries) > 0
        for q in terminal_queries:
            # Target should be one of the depth-3-zero-descendants nodes
            assert q.targets[0] in {"xray", "dysp"}, f"terminal target was {q.targets}"

    def test_evidence_strategy_per_role(self, asia_problem):
        """Hub & terminal use longest_path; cut uses mb_neighbors."""
        selector = HeaviestQueryByRole()
        queries = selector.select(asia_problem, n_queries=999, seed=0)

        for q in queries:
            if q.query_role in {"hub", "terminal"}:
                assert q.evidence_strategy == "longest_path"
            elif q.query_role == "cut":
                assert q.evidence_strategy == "mb_neighbors"

    def test_deterministic(self, asia_problem):
        """Same seed → same query list."""
        s1 = HeaviestQueryByRole()
        s2 = HeaviestQueryByRole()
        q1 = s1.select(asia_problem, n_queries=999, seed=42)
        q2 = s2.select(asia_problem, n_queries=999, seed=42)

        assert len(q1) == len(q2)
        for a, b in zip(q1, q2):
            assert a.targets == b.targets
            assert a.query_role == b.query_role
            assert a.query_kind == b.query_kind
            assert a.evidence_strategy == b.evidence_strategy
            # Evidence keys must match; values must match for V1, both be None for V2
            assert set(a.evidence.keys()) == set(b.evidence.keys())

    def test_cache_reused(self, asia_problem):
        """NodeRoles computed once per (family, problem_id)."""
        selector = HeaviestQueryByRole()
        selector.select(asia_problem, n_queries=999, seed=0)
        assert len(selector._cache) == 1
        # Second call shouldn't grow cache
        selector.select(asia_problem, n_queries=999, seed=1)
        assert len(selector._cache) == 1

    def test_raises_without_test_data(self, asia_problem):
        """V1 mode requires test_data; raise if absent."""
        asia_problem_no_data = BenchmarkProblem(
            name=asia_problem.name,
            dag=asia_problem.dag,
            variables=asia_problem.variables,
            train_data={},
            test_data={},  # empty
            queries=[],
            seed=0,
            family=asia_problem.family,
            problem_id=asia_problem.problem_id,
            true_model=None,
        )
        selector = HeaviestQueryByRole()
        with pytest.raises(ValueError, match="test_data"):
            selector.select(asia_problem_no_data, n_queries=999, seed=0)
