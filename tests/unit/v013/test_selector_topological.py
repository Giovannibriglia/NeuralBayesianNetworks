"""Tests for nbn.bench.selectors.topological — Stage 1.

Covers NodeRoles + compute_node_roles + _TargetAllocator. Steps 2-4
tested in later stages.

Reference: docs/phase2-design-draft.md §4.4 test surface.
"""
from __future__ import annotations

import random
import time

import networkx as nx
import pytest

import torch

from nbn.bench.domains.base import BenchmarkProblem
from nbn.bench.selectors.topological import (
    TopologicalAllocator,
    _KindEvidenceAllocator,
    _TargetAllocator,
    _ValueAllocator,
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
    from nbn.bench.synthetic import make_synthetic_bn
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
        # Spec: depth − |descendants|. xray and dysp tie at 3 - 0 = 3.
        # The next best is either at 2 - 2 = 0.
        assert roles.terminals[0] in {"xray", "dysp"}
        # Specifically, both xray and dysp should be in the top 2
        assert set(roles.terminals[:2]) == {"xray", "dysp"}

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

        # Cut had 2 surplus (quota 4 - 2 actual APs). Surplus goes to random.
        # Random's base quota is 4; +2 surplus = 6 expected.
        assert by_bucket["cut"] == 2
        assert by_bucket["random"] >= 4  # at least its base quota
        # Total should match the requested budget (largest-remainder apportionment).
        assert len(result) == 16

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

    def test_quota_distribution(self, synthetic_n1000):
        """Per-bucket counts approximately match requested fractions."""
        roles = compute_node_roles(synthetic_n1000)
        allocator = _TargetAllocator()
        budget = 1000
        result = allocator.allocate(
            roles=roles,
            budget=budget,
            allocation={"hub": 0.30, "cut": 0.25, "terminal": 0.25, "random": 0.20},
            rng=random.Random(0),
            all_nodes=list(synthetic_n1000.nodes()),
        )

        by_bucket = dict.fromkeys(("hub", "cut", "terminal", "random"), 0)
        for _, b in result:
            by_bucket[b] += 1

        # Total must exactly equal budget (largest-remainder guarantees this).
        assert sum(by_bucket.values()) == budget

        # Cut may be 0 (synthetic n=1000 has 0 APs); surplus goes to random.
        # Verify total budget is exact, and that hub/terminal got their share.
        assert by_bucket["hub"] >= 300 - 10  # 30% ± rounding
        assert by_bucket["terminal"] >= 250 - 10  # 25% ± rounding
        # Random gets its 20% + any cut surplus (250):
        assert by_bucket["random"] >= 200


# --- Edge-case tests (PR #124 review probes) ---

class TestEdgeCases:
    """Edge-case fixtures verified in PR #124 review probes."""

    def test_empty_dag(self):
        G = nx.DiGraph()
        roles = compute_node_roles(G)
        assert roles.hubs == []
        assert roles.cuts == []
        assert roles.terminals == []

    def test_single_node_dag(self):
        G = nx.DiGraph()
        G.add_node("X")
        roles = compute_node_roles(G)
        assert roles.hubs == ["X"]
        assert roles.cuts == []
        assert roles.mb_size["X"] == 0

    def test_disconnected_dag(self):
        G = nx.DiGraph()
        G.add_nodes_from(["A", "B", "C"])
        roles = compute_node_roles(G)
        # All isolated; no APs; no MB
        assert roles.cuts == []
        assert all(roles.mb_size[v] == 0 for v in ["A", "B", "C"])

    def test_chain_dag_articulation(self):
        G = nx.DiGraph([("A", "B"), ("B", "C")])
        roles = compute_node_roles(G)
        # B is the articulation point in the chain
        assert roles.cuts == ["B"]


# --- Stage 2 tests ---

class TestKindEvidenceAllocator:
    """Spec §3.3, §3.4: kind assignment + evidence-set selection."""

    DEFAULT_KIND_ALLOC = {
        "hub":      {"prediction": 0.5, "diagnosis": 0.5},
        "cut":      {"prediction": 0.5, "diagnosis": 0.5},
        "terminal": {"prediction": 1.0, "diagnosis": 0.0},
        "random":   {"prediction": 0.5, "diagnosis": 0.5},
    }
    DEFAULT_EV_ALLOC = {
        "prediction": {"longest_path": 0.4, "mb_neighbors": 0.4, "random": 0.2},
        "diagnosis":  {"longest_path": 0.4, "mb_neighbors": 0.4, "random": 0.2},
    }
    N_EVIDENCE = {"prediction": 3, "diagnosis": 3}

    def test_terminal_bucket_always_prediction(self, asia_dag):
        """K1: terminal bucket has kind_alloc[terminal] = {prediction: 1.0}.
        Every terminal query must have query_kind='prediction'."""
        roles = compute_node_roles(asia_dag)
        allocator = _KindEvidenceAllocator()

        # Sample 50 terminal-bucket queries
        kinds = []
        for i in range(50):
            result = allocator.assign(
                target="xray",  # a known terminal
                bucket="terminal",
                roles=roles,
                kind_alloc=self.DEFAULT_KIND_ALLOC,
                evidence_alloc=self.DEFAULT_EV_ALLOC,
                n_evidence=self.N_EVIDENCE,
                rng=random.Random(i),
            )
            if result is not None:
                kinds.append(result[0])

        # All should be prediction
        assert all(k == "prediction" for k in kinds), f"unexpected kinds: {set(kinds)}"

    def test_hub_bucket_balanced_50_50(self, asia_dag):
        """K1: hub bucket has 50/50 prediction/diagnosis. Verify statistically."""
        roles = compute_node_roles(asia_dag)
        allocator = _KindEvidenceAllocator()

        # Sample 200 hub-bucket queries with different seeds
        kinds = []
        for i in range(200):
            result = allocator.assign(
                target="either",  # a known hub
                bucket="hub",
                roles=roles,
                kind_alloc=self.DEFAULT_KIND_ALLOC,
                evidence_alloc=self.DEFAULT_EV_ALLOC,
                n_evidence=self.N_EVIDENCE,
                rng=random.Random(i),
            )
            if result is not None:
                kinds.append(result[0])

        # Should be roughly 50/50 with reasonable tolerance
        prediction_count = sum(1 for k in kinds if k == "prediction")
        fraction = prediction_count / len(kinds)
        assert 0.35 < fraction < 0.65, f"hub kind imbalance: {fraction:.2f}"

    def test_evidence_pool_prediction_longest_path(self, asia_dag):
        """Prediction + longest_path: pool is ancestors, sorted by depth distance desc."""
        roles = compute_node_roles(asia_dag)
        allocator = _KindEvidenceAllocator()

        # Force prediction + longest_path with overwhelming allocation
        ev_alloc = {"prediction": {"longest_path": 1.0}}

        # Target 'either' has ancestors {asia, tub, smoke, lung}
        # depth(either)=2; depth(asia)=0, depth(tub)=1, depth(smoke)=0, depth(lung)=1
        # Distance: asia=2, tub=1, smoke=2, lung=1
        # Top by distance descending: asia and smoke tied at 2
        result = allocator.assign(
            target="either",
            bucket="hub",
            roles=roles,
            kind_alloc={"hub": {"prediction": 1.0}},  # force prediction
            evidence_alloc=ev_alloc,
            n_evidence={"prediction": 2},
            rng=random.Random(0),
        )
        assert result is not None
        query_kind, ev_strategy, ev_nodes = result
        assert query_kind == "prediction"
        assert ev_strategy == "longest_path"
        # Top 2 by depth distance should include the deepest ancestors
        assert set(ev_nodes).issubset({"asia", "tub", "smoke", "lung"})
        assert len(ev_nodes) == 2

    def test_evidence_pool_prediction_mb_neighbors(self, asia_dag):
        """Prediction + mb_neighbors: pool is parents ∪ co_parents (spec §3.4)."""
        roles = compute_node_roles(asia_dag)
        allocator = _KindEvidenceAllocator()

        # 'either' has parents={tub, lung}. co_parents(either)={bronc}: bronc
        # shares child 'dysp' with either (dysp's parents are {either, bronc}).
        # Pool = parents ∪ co_parents = {tub, lung, bronc}.
        result = allocator.assign(
            target="either",
            bucket="hub",
            roles=roles,
            kind_alloc={"hub": {"prediction": 1.0}},
            evidence_alloc={"prediction": {"mb_neighbors": 1.0}},
            n_evidence={"prediction": 5},  # ask for more than pool has
            rng=random.Random(0),
        )
        assert result is not None
        _, _, ev_nodes = result
        # Should get the full pool (3 nodes), not crash on shortfall
        assert set(ev_nodes) == {"tub", "lung", "bronc"}

    def test_evidence_pool_diagnosis_children(self, asia_dag):
        """Diagnosis + mb_neighbors: pool is children + co_parents (semantic co_children)."""
        roles = compute_node_roles(asia_dag)
        allocator = _KindEvidenceAllocator()

        # 'either' has children={xray, dysp}. co_parents(either) = nodes sharing
        # a child with either: dysp's other parent is bronc → co_parents={bronc}.
        result = allocator.assign(
            target="either",
            bucket="hub",
            roles=roles,
            kind_alloc={"hub": {"diagnosis": 1.0}},  # force diagnosis
            evidence_alloc={"diagnosis": {"mb_neighbors": 1.0}},
            n_evidence={"diagnosis": 5},
            rng=random.Random(0),
        )
        assert result is not None
        query_kind, _, ev_nodes = result
        assert query_kind == "diagnosis"
        # Pool = children(either) | co_parents(either) = {xray, dysp} | {bronc}
        assert set(ev_nodes).issubset({"xray", "dysp", "bronc"})

    def test_empty_pool_returns_none(self, asia_dag):
        """Root node with prediction kind has no ancestors → None."""
        roles = compute_node_roles(asia_dag)
        allocator = _KindEvidenceAllocator()

        # 'asia' is a root (no ancestors). prediction + longest_path → empty pool.
        result = allocator.assign(
            target="asia",
            bucket="random",
            roles=roles,
            kind_alloc={"random": {"prediction": 1.0}},
            evidence_alloc={"prediction": {"longest_path": 1.0}},
            n_evidence={"prediction": 3},
            rng=random.Random(0),
        )
        assert result is None

    def test_deterministic_with_seed(self, asia_dag):
        """Same rng state → same output."""
        roles = compute_node_roles(asia_dag)
        allocator = _KindEvidenceAllocator()

        result1 = allocator.assign(
            target="either",
            bucket="hub",
            roles=roles,
            kind_alloc=self.DEFAULT_KIND_ALLOC,
            evidence_alloc=self.DEFAULT_EV_ALLOC,
            n_evidence=self.N_EVIDENCE,
            rng=random.Random(42),
        )
        result2 = allocator.assign(
            target="either",
            bucket="hub",
            roles=roles,
            kind_alloc=self.DEFAULT_KIND_ALLOC,
            evidence_alloc=self.DEFAULT_EV_ALLOC,
            n_evidence=self.N_EVIDENCE,
            rng=random.Random(42),
        )
        assert result1 == result2


# --- Stage 3 tests ---

class TestValueAllocator:
    """Step 4 contract: dedup, retry, parametrized determinism."""

    def _make_test_data(self, nodes: list[str], n_rows: int) -> dict:
        """Build a small test_data dict with integer tensors."""
        return {n: torch.arange(n_rows) for n in nodes}

    def test_returns_evidence_dict(self):
        """Successful sample returns dict keyed by evidence_nodes."""
        allocator = _ValueAllocator()
        test_data = self._make_test_data(["A", "B", "C"], n_rows=20)
        seen = set()
        result = allocator.assign_values(
            target="T",
            evidence_nodes=["A", "B"],
            test_data=test_data,
            seen=seen,
            problem_seed=0,
            query_idx=0,
        )
        assert result is not None
        assert set(result.keys()) == {"A", "B"}

    def test_dedup_by_triple(self):
        """Calling twice with same (target, evidence_nodes, query_idx)
        uses retries; eventually picks different row_idx."""
        allocator = _ValueAllocator()
        test_data = self._make_test_data(["A"], n_rows=20)
        seen = set()

        r1 = allocator.assign_values(
            target="T", evidence_nodes=["A"], test_data=test_data,
            seen=seen, problem_seed=0, query_idx=0,
        )
        assert r1 is not None
        assert len(seen) == 1

        # Same query_idx but seen is now populated — should still succeed
        # by trying next retry (different row_idx)
        r2 = allocator.assign_values(
            target="T", evidence_nodes=["A"], test_data=test_data,
            seen=seen, problem_seed=0, query_idx=0,
        )
        assert r2 is not None
        assert len(seen) == 2

    def test_dedup_exhaustion_returns_none(self):
        """When all retries hit duplicates, return None."""
        allocator = _ValueAllocator(max_retry=2)
        test_data = self._make_test_data(["A"], n_rows=2)
        # Pre-populate seen with all possible triples
        seen = {
            ("T", frozenset({"A"}), 0),
            ("T", frozenset({"A"}), 1),
        }
        result = allocator.assign_values(
            target="T", evidence_nodes=["A"], test_data=test_data,
            seen=seen, problem_seed=0, query_idx=0,
            max_retry=2,
        )
        # All row indices already in seen → None
        assert result is None

    def test_deterministic(self):
        """Same inputs and seen state → same output."""
        allocator = _ValueAllocator()
        test_data = self._make_test_data(["A", "B"], n_rows=20)

        r1 = allocator.assign_values(
            target="T", evidence_nodes=["A", "B"], test_data=test_data,
            seen=set(), problem_seed=42, query_idx=5,
        )
        r2 = allocator.assign_values(
            target="T", evidence_nodes=["A", "B"], test_data=test_data,
            seen=set(), problem_seed=42, query_idx=5,
        )
        assert r1 is not None and r2 is not None
        # Same row → same tensor values
        assert torch.equal(r1["A"], r2["A"])
        assert torch.equal(r1["B"], r2["B"])


class TestTopologicalAllocator:
    """Public class contract: end-to-end query construction."""

    def _make_benchmark_problem(self, asia_dag):
        """Build a minimal BenchmarkProblem from the ASIA fixture."""
        # ASIA-shaped synthetic test_data: 50 rows per node, integer values 0-1
        rng = random.Random(0)
        n_rows = 50
        test_data = {
            node: torch.tensor([rng.randint(0, 1) for _ in range(n_rows)])
            for node in asia_dag.nodes()
        }
        dag_edges = list(asia_dag.edges())
        variables = dict.fromkeys(asia_dag.nodes(), ("discrete", 2))
        return BenchmarkProblem(
            name="asia",
            dag=dag_edges,
            variables=variables,
            train_data={},  # not used by selector
            test_data=test_data,
            queries=[],
            seed=0,
            family="discrete",
            problem_id="asia_test",
            true_model=None,
        )

    def test_returns_queries_with_metadata(self, asia_dag):
        """All returned Query objects have populated metadata fields."""
        problem = self._make_benchmark_problem(asia_dag)
        selector = TopologicalAllocator()
        queries = selector.select(problem, n_queries=20, seed=0)

        assert len(queries) > 0
        for q in queries:
            assert q.query_role in {"hub", "cut", "terminal", "random"}
            assert q.query_kind in {"prediction", "diagnosis"}
            assert q.evidence_strategy in {"longest_path", "mb_neighbors", "random"}
            # Query targets/evidence must be populated
            assert len(q.targets) == 1
            assert isinstance(q.evidence, dict)
            assert len(q.evidence) > 0  # evidence_nodes was non-empty

    def test_returns_at_most_budget(self, asia_dag):
        """Query count ≤ budget per spec §3.7."""
        problem = self._make_benchmark_problem(asia_dag)
        selector = TopologicalAllocator()
        queries = selector.select(problem, n_queries=50, seed=0)
        assert len(queries) <= 50

    def test_deterministic(self, asia_dag):
        """Same seed → same query list."""
        problem = self._make_benchmark_problem(asia_dag)
        selector1 = TopologicalAllocator()
        selector2 = TopologicalAllocator()

        q1 = selector1.select(problem, n_queries=30, seed=0)
        q2 = selector2.select(problem, n_queries=30, seed=0)

        assert len(q1) == len(q2)
        for a, b in zip(q1, q2):
            assert a.targets == b.targets
            assert a.query_role == b.query_role
            assert a.query_kind == b.query_kind
            assert a.evidence_strategy == b.evidence_strategy

    def test_cache_reused_across_calls(self, asia_dag):
        """NodeRoles is computed once per (family, problem_id, seed)."""
        import dataclasses

        problem = self._make_benchmark_problem(asia_dag)
        selector = TopologicalAllocator()

        # First call populates cache for this (family, problem_id, seed)
        selector.select(problem, n_queries=10, seed=0)
        assert len(selector._cache) == 1
        cache_key = next(iter(selector._cache))
        assert cache_key == ("discrete", "asia_test", 0)

        # Second call with the SAME problem hits the cache (size unchanged).
        # The select(seed=...) arg drives query-value RNG, not the cache key,
        # which is derived from problem.seed.
        selector.select(problem, n_queries=10, seed=1)
        assert len(selector._cache) == 1

        # A problem sharing problem_id but with a different seed (hence a
        # different DAG in real runs) gets its own cache entry — no stale reuse.
        problem_seed1 = dataclasses.replace(problem, seed=1)
        selector.select(problem_seed1, n_queries=10, seed=0)
        assert len(selector._cache) == 2
        assert ("discrete", "asia_test", 1) in selector._cache

    def test_backward_compat_random_only(self, asia_dag):
        """allocation={'random': 1.0} → all queries have query_role='random'
        (acceptance criterion 5 from #74)."""
        problem = self._make_benchmark_problem(asia_dag)
        selector = TopologicalAllocator(target_allocation={"random": 1.0})
        queries = selector.select(problem, n_queries=20, seed=0)

        assert all(q.query_role == "random" for q in queries)

    def test_no_evidence_no_query(self, asia_dag):
        """Targets with empty evidence pools produce no queries (spec §3.7)."""
        # Force prediction kind everywhere; root nodes (asia, smoke) have no ancestors
        problem = self._make_benchmark_problem(asia_dag)
        selector = TopologicalAllocator(
            target_allocation={"random": 1.0},
            kind_allocation={"random": {"prediction": 1.0}},
            evidence_allocation={"prediction": {"longest_path": 1.0}},
        )
        queries = selector.select(problem, n_queries=10, seed=0)

        # No query should target a root node (asia or smoke)
        for q in queries:
            assert q.targets[0] not in {"asia", "smoke"}
            # Evidence must be non-empty (we never construct queries with empty evidence)
            assert len(q.evidence) > 0
