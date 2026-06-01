"""Topology-aware query selector — Stage 1 (NodeRoles + _TargetAllocator).

Phase 2 of v0.13 (issue #74). Implements four-step query construction:
  Step 1 — Target selection (THIS STAGE)
  Step 2 — Kind assignment (Stage 2)
  Step 3 — Evidence-set selection (Stage 2)
  Step 4 — Value assignment (Stage 3)

See docs/phase2-design-draft.md for the full design.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping

import networkx as nx


@dataclass(frozen=True)
class NodeRoles:
    """Topological roles for all nodes in a DAG. Spec §4.2."""

    # Ranked lists (Step 1: target selection)
    hubs:      list[str]          # sorted desc by |MB(v)|
    cuts:      list[str]          # sorted desc by degree centrality
    terminals: list[str]          # sorted desc by depth(v) − |desc(v)|

    # Per-node scalar maps
    mb_size:           dict[str, int]
    depth:             dict[str, int]
    descendant_count:  dict[str, int]

    # Per-node set maps (for Step 3 evidence pools)
    parents:           dict[str, frozenset[str]]
    children:          dict[str, frozenset[str]]
    co_parents:        dict[str, frozenset[str]]
    co_children:       dict[str, frozenset[str]]
    ancestors:         dict[str, frozenset[str]]
    descendants:       dict[str, frozenset[str]]


def compute_node_roles(G: nx.DiGraph) -> NodeRoles:
    """Compute NodeRoles for a DAG. Spec §3.2, §3.8.

    Runtime: ~16 ms at n=1000 (measured in Phase 0).
    """
    # 1) Direct neighbor sets
    parents = {v: frozenset(G.predecessors(v)) for v in G.nodes()}
    children = {v: frozenset(G.successors(v)) for v in G.nodes()}

    # 2) Co-parents: nodes sharing ≥1 child with v (excluding v).
    co_parents: dict[str, frozenset[str]] = {}
    for v in G.nodes():
        cp: set[str] = set()
        for child in children[v]:
            cp.update(parents[child])
        cp.discard(v)
        co_parents[v] = frozenset(cp)

    # 3) Co-children. Per spec §5 Q3, the diagnosis evidence pool is
    #    children(target) ∪ co_children(target), where
    #        co_children = {n for child in children[v]
    #                       for n in parents[child] if n != v}
    #    i.e. nodes that share at least one child with v. That set is
    #    identical to co_parents (same formula); the distinct name marks
    #    its diagnostic-direction semantic role, not a different set.
    co_children = co_parents

    # 4) MB size: |parents ∪ children ∪ co_parents|
    mb_size = {
        v: len(parents[v] | children[v] | co_parents[v])
        for v in G.nodes()
    }

    # 5) Depth (longest path from a root, via topological order)
    depth: dict[str, int] = {}
    for v in nx.topological_sort(G):
        if not parents[v]:
            depth[v] = 0
        else:
            depth[v] = max(depth[p] for p in parents[v]) + 1

    # 6) Ancestors + descendants
    ancestors = {v: frozenset(nx.ancestors(G, v)) for v in G.nodes()}
    descendants = {v: frozenset(nx.descendants(G, v)) for v in G.nodes()}
    descendant_count = {v: len(descendants[v]) for v in G.nodes()}

    # 7) Hub ranking
    hubs = sorted(G.nodes(), key=lambda v: -mb_size[v])

    # 8) Cut ranking — articulation points of the moralized undirected
    #    graph, sub-ranked by degree centrality (spec §5 Q1 recommendation).
    G_moral_und = nx.moral_graph(G)  # already undirected per nx convention
    try:
        ap_list = list(nx.articulation_points(G_moral_und))
    except nx.NetworkXNotImplemented:
        # Defensive: articulation_points is defined for undirected graphs.
        ap_list = []

    if ap_list:
        degree_cent = nx.degree_centrality(G_moral_und)
        cuts = sorted(ap_list, key=lambda v: -degree_cent.get(v, 0.0))
    else:
        cuts = []

    # 9) Terminal ranking by depth(v) − |descendants(v)| (spec §1, §3.2, §3.8):
    #    deep nodes with few descendants rank highest (leaf-like, long paths).
    terminals = sorted(
        G.nodes(),
        key=lambda v: -(depth[v] - descendant_count[v]),
    )

    return NodeRoles(
        hubs=hubs,
        cuts=cuts,
        terminals=terminals,
        mb_size=mb_size,
        depth=depth,
        descendant_count=descendant_count,
        parents=parents,
        children=children,
        co_parents=co_parents,
        co_children=co_children,
        ancestors=ancestors,
        descendants=descendants,
    )


class _TargetAllocator:
    """Step 1: target selection across hub/cut/terminal/random buckets.

    Spec §3.2. Hardest-first within each bucket; shortfall to random.
    """

    def allocate(
        self,
        roles: NodeRoles,
        budget: int,
        allocation: Mapping[str, float],
        rng: random.Random,
        all_nodes: list[str],
    ) -> list[tuple[str, str]]:
        """Return list of (target_node, bucket_name), length ≤ budget.

        Spec §3.2: shortfall rule = if a bucket has fewer nodes than its
        quota, cap at min(quota, available) and redistribute the surplus
        to the random bucket.
        """
        # Largest-remainder method (Hamilton): quotas sum exactly to budget.
        # Each bucket gets floor(budget × frac); the remaining seats go to the
        # buckets with the largest fractional remainder. Avoids the ceil-per-
        # bucket overshoot that previously biased truncation against random.
        raw = {bucket: budget * frac for bucket, frac in allocation.items()}
        floors = {bucket: int(r) for bucket, r in raw.items()}
        remainders = {bucket: raw[bucket] - floors[bucket] for bucket in raw}
        seats_remaining = budget - sum(floors.values())
        ordered_buckets = sorted(
            remainders.items(),
            key=lambda x: (-x[1], x[0]),  # secondary sort by name for determinism
        )
        quotas = dict(floors)
        for bucket, _ in ordered_buckets[:seats_remaining]:
            quotas[bucket] += 1

        available = {
            "hub": roles.hubs,
            "cut": roles.cuts,
            "terminal": roles.terminals,
            "random": all_nodes,
        }

        result: list[tuple[str, str]] = []
        surplus = 0

        # Process structural buckets first; accumulate shortfall as surplus.
        for bucket in ("hub", "cut", "terminal"):
            quota = quotas.get(bucket, 0)
            pool = available[bucket]
            picks = pool[: min(quota, len(pool))]
            surplus += quota - len(picks)
            for node in picks:
                result.append((node, bucket))

        # Random bucket: original quota + accumulated surplus.
        # Spec §3.2: cross-bucket repetition is allowed; dedup is enforced at
        # Step 4 by the full (target, evidence_nodes, evidence_values) triple.
        # The random bucket may include nodes already in structural buckets.
        random_quota = quotas.get("random", 0) + surplus
        random_candidates = list(all_nodes)
        rng.shuffle(random_candidates)
        random_picks = random_candidates[: min(random_quota, len(random_candidates))]
        for node in random_picks:
            result.append((node, "random"))

        # Defensive: quotas sum to budget, but a structural shortfall whose
        # surplus exceeds the remaining node pool can still leave us short
        # (never over). Slice is a no-op in the common case.
        return result[:budget]
