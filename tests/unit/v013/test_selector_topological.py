"""Tests for benchmarking.selectors.topological — Stage 1.

Covers NodeRoles + compute_node_roles + _TargetAllocator. Steps 2-4
tested in later stages.

Reference: docs/phase2-design-draft.md §4.4 test surface.
"""
from __future__ import annotations

import random
import time

import networkx as nx
import pytest

from benchmarking.selectors.topological import (
    _TargetAllocator,
    compute_node_roles,
)


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
def asia_dag() -> nx.DiGraph:
    """ASIA Bayesian network (n=8). Hand-typed for Stage 1;
    Phase 4 will replace with BnlearnProblemSource."""
    return nx.DiGraph(ASIA_EDGES)


@pytest.fixture
def synthetic_n1000() -> nx.DiGraph:
    """Synthetic n=1000 continuous_lg DAG."""
    from benchmarking.synthetic import make_synthetic_bn
    bn = make_synthetic_bn(n_nodes=1000, family="continuous_lg", seed=0)
    return nx.DiGraph(bn.dag)


# --- NodeRoles tests ---

class TestNodeRolesAsia:
    """Verify NodeRoles correctness on ASIA fixture."""

    def test_articulation_points(self, asia_dag):
        roles = compute_node_roles(asia_dag)
        # ASIA has 2 APs: either, tub (verified in Phase 0)
        assert set(roles.cuts) == {"either", "tub"}

    def test_hub_ranking(self, asia_dag):
        roles = compute_node_roles(asia_dag)
        # 'either' has the largest MB (5: tub, lung, xray, dysp, bronc)
        assert roles.hubs[0] == "either"
        assert roles.mb_size["either"] == 5

    def test_terminal_ranking(self, asia_dag):
        roles = compute_node_roles(asia_dag)
        # dysp and xray are tied at depth=3, desc=0 (no descendants)
        # but their score is 3*0 = 0. Top score by depth*descendants:
        # bronc: 1*1 = 1; either: 2*2 = 4
        assert roles.terminals[0] == "either"

    def test_direct_neighbors(self, asia_dag):
        roles = compute_node_roles(asia_dag)
        # either has parents = {tub, lung}, children = {xray, dysp}
        assert roles.parents["either"] == frozenset({"tub", "lung"})
        assert roles.children["either"] == frozenset({"xray", "dysp"})

    def test_co_parents(self, asia_dag):
        roles = compute_node_roles(asia_dag)
        # tub and lung share 'either' as child → they are co-parents
        assert "lung" in roles.co_parents["tub"]
        assert "tub" in roles.co_parents["lung"]

    def test_ancestors_descendants(self, asia_dag):
        roles = compute_node_roles(asia_dag)
        assert "asia" in roles.ancestors["either"]
        assert "dysp" in roles.descendants["either"]


class TestNodeRolesSynthetic:
    """Verify NodeRoles handles large DAG correctly."""

    def test_compute_time_under_one_second(self, synthetic_n1000):
        t0 = time.perf_counter()
        compute_node_roles(synthetic_n1000)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"NodeRoles took {elapsed:.2f}s (budget 1s)"

    def test_disconnected_or_no_aps_handled(self, synthetic_n1000):
        """Synthetic n=1000 may have 0 APs (verified in Phase 0).
        Must not crash; cuts list may be empty."""
        roles = compute_node_roles(synthetic_n1000)
        # No assertion on count — just verify no exception
        assert isinstance(roles.cuts, list)

    def test_all_nodes_have_entries(self, synthetic_n1000):
        roles = compute_node_roles(synthetic_n1000)
        n = synthetic_n1000.number_of_nodes()
        assert len(roles.mb_size) == n
        assert len(roles.depth) == n
        assert len(roles.parents) == n


# --- _TargetAllocator tests ---

class TestTargetAllocator:
    """Spec §3.2 shortfall rule + budget compliance."""

    def test_asia_shortfall_to_random(self, asia_dag):
        """ASIA has only 2 APs. A cut quota of 4 → 2 cut + surplus to random."""
        roles = compute_node_roles(asia_dag)
        allocator = _TargetAllocator()

        # Budget 16; allocation {hub: 0.25, cut: 0.25, terminal: 0.25, random: 0.25}
        # Quotas: 4 each (hub, cut, terminal, random)
        # Cut has only 2 APs → 2 surplus → random gets 4+2 = 6
        result = allocator.allocate(
            roles=roles,
            budget=16,
            allocation={"hub": 0.25, "cut": 0.25, "terminal": 0.25, "random": 0.25},
            rng=random.Random(0),
            all_nodes=list(asia_dag.nodes()),
        )

        by_bucket = dict.fromkeys(("hub", "cut", "terminal", "random"), 0)
        for _, bucket in result:
            by_bucket[bucket] += 1

        assert by_bucket["cut"] == 2  # only 2 APs available
        # The other 3 buckets should hit their quota of 4 each (8 nodes available for each)
        # Total result must be <= 16
        assert len(result) <= 16

    def test_returns_at_most_budget(self, synthetic_n1000):
        roles = compute_node_roles(synthetic_n1000)
        allocator = _TargetAllocator()
        all_nodes = list(synthetic_n1000.nodes())
        result = allocator.allocate(
            roles=roles,
            budget=100,
            allocation={"hub": 0.3, "cut": 0.25, "terminal": 0.25, "random": 0.2},
            rng=random.Random(0),
            all_nodes=all_nodes,
        )
        assert len(result) <= 100

    def test_deterministic(self, asia_dag):
        """Same seed → same allocation."""
        roles = compute_node_roles(asia_dag)
        allocator = _TargetAllocator()
        all_nodes = list(asia_dag.nodes())

        result1 = allocator.allocate(
            roles=roles, budget=8,
            allocation={"hub": 0.5, "random": 0.5},
            rng=random.Random(42), all_nodes=all_nodes,
        )
        result2 = allocator.allocate(
            roles=roles, budget=8,
            allocation={"hub": 0.5, "random": 0.5},
            rng=random.Random(42), all_nodes=all_nodes,
        )
        assert result1 == result2
