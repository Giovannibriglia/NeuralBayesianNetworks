"""Heaviest-query-per-role selector (Phase 3 of v0.13, spec §2).

Picks the structurally hardest query of each role × direction × mode
combination, emitting up to 12 queries per DAG (8 if no articulation
points). Reuses Phase 2's NodeRoles + _ValueAllocator infrastructure.
"""
from __future__ import annotations

import logging

import networkx as nx

from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.selectors.topological import (
    NodeRoles,
    _KindEvidenceAllocator,
    _ValueAllocator,
    compute_node_roles,
)

logger = logging.getLogger(__name__)

# Role → evidence strategy (spec §2.1)
_ROLE_STRATEGY = {
    "hub": "longest_path",
    "cut": "mb_neighbors",
    "terminal": "longest_path",
}

_DIRECTIONS = ("prediction", "diagnosis")
_MODES = ("full", "empty")


class HeaviestQueryByRole:
    """Deterministic selector for the scalability benchmark (spec §2).

    Produces up to 12 queries per DAG: top-1 target per role × 2
    directions × 2 evidence modes. Cut queries dropped if no APs.
    """

    # Class-level fallback for runner.py's getattr(cfg.selector, "query_role", ...)
    query_role: str = "random"

    _DEFAULT_N_EVIDENCE = {"prediction": 3, "diagnosis": 3}
    _DEFAULT_MAX_RETRY = 10

    def __init__(
        self,
        n_evidence: dict | None = None,
        max_retry: int | None = None,
    ) -> None:
        self.n_evidence = dict(n_evidence or self._DEFAULT_N_EVIDENCE)
        self.max_retry = (
            max_retry if max_retry is not None else self._DEFAULT_MAX_RETRY
        )
        self._kind_evidence_allocator = _KindEvidenceAllocator()
        self._value_allocator = _ValueAllocator(max_retry=self.max_retry)
        self._cache: dict[tuple[str, str, int], NodeRoles] = {}

    def _get_roles(self, problem: BenchmarkProblem, G: nx.DiGraph) -> NodeRoles:
        # Key includes seed: the synthetic DAG varies by seed, so two problems
        # sharing a problem_id (e.g. n_nodes="10") but differing in seed have
        # distinct topologies and must not share a NodeRoles cache entry.
        key = (problem.family, problem.problem_id, problem.seed)
        roles = self._cache.get(key)
        if roles is None:
            roles = compute_node_roles(G)
            self._cache[key] = roles
        return roles

    def select(
        self,
        problem: BenchmarkProblem,
        n_queries: int,
        seed: int,
    ) -> list[Query]:
        """Emit up to 12 queries (8 if no APs).

        ``n_queries`` is ignored — the selector produces a fixed count by
        spec §2.1. A debug message is logged if it is non-zero, since the
        caller's budget has no effect here.
        """
        if not problem.test_data:
            raise ValueError(
                "HeaviestQueryByRole requires problem.test_data for V1 mode."
            )

        if n_queries:
            logger.debug(
                "HeaviestQueryByRole ignores n_queries=%d; it emits a fixed "
                "count (up to 12) per DAG.",
                n_queries,
            )

        G = nx.DiGraph(problem.dag)
        G.add_nodes_from(problem.variables.keys())
        roles = self._get_roles(problem, G)

        # Pick the top-1 target per role (or None if bucket empty)
        targets_by_role = {
            "hub":      roles.hubs[0]      if roles.hubs      else None,
            "cut":      roles.cuts[0]      if roles.cuts      else None,
            "terminal": roles.terminals[0] if roles.terminals else None,
        }

        seen: set[tuple] = set()  # for _ValueAllocator dedup across V1 queries
        queries: list[Query] = []
        query_idx = 0

        for role in ("hub", "cut", "terminal"):
            target = targets_by_role[role]
            if target is None:
                continue  # spec §2.1: drop cut queries if no APs

            strategy = _ROLE_STRATEGY[role]

            for direction in _DIRECTIONS:
                # Build the evidence pool once per (role, direction) pair.
                # Reuse _KindEvidenceAllocator's pool logic; we don't use
                # its kind-sampling. Call _evidence_pool directly with the
                # fixed strategy.
                pool = self._kind_evidence_allocator._evidence_pool(
                    target=target,
                    query_kind=direction,
                    strategy=strategy,
                    roles=roles,
                )
                if not pool:
                    # Empty pool (e.g. root node with prediction direction).
                    # Both V1 and V2 of this (role, direction) are dropped.
                    continue

                n = self.n_evidence.get(direction, 3)
                evidence_nodes = pool[:n]  # top-n by hardness ordering

                # V1: full mode with concrete values from test_data
                v1_values = self._value_allocator.assign_values(
                    target=target,
                    evidence_nodes=evidence_nodes,
                    test_data=problem.test_data,
                    seen=seen,
                    problem_seed=seed,
                    query_idx=query_idx,
                )
                query_idx += 1

                if v1_values is None:
                    # Dedup exhausted for V1; skip both (V1 and V2 are paired)
                    continue

                v1_query = Query(
                    targets=(target,),
                    evidence=v1_values,
                    kind="marginal",
                    query_role=role,
                    query_kind=direction,
                    evidence_strategy=strategy,
                )
                queries.append(v1_query)

                # V2: empty mode — same evidence nodes, all values None
                v2_evidence = dict.fromkeys(evidence_nodes, None)
                v2_query = Query(
                    targets=(target,),
                    evidence=v2_evidence,
                    kind="marginal",
                    query_role=role,
                    query_kind=direction,
                    evidence_strategy=strategy,
                )
                queries.append(v2_query)

        return queries
