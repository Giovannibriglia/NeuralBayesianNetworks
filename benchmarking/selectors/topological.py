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

    # Note: the diagnosis evidence pool (spec §3.4/§5 Q3) is
    # children(target) ∪ co_children(target). co_children — nodes sharing
    # ≥1 child with v — is structurally identical to co_parents (same
    # formula), so _KindEvidenceAllocator reuses co_parents directly rather
    # than storing a duplicate set.

    # 3) MB size: |parents ∪ children ∪ co_parents|
    mb_size = {
        v: len(parents[v] | children[v] | co_parents[v])
        for v in G.nodes()
    }

    # 4) Depth (longest path from a root, via topological order)
    depth: dict[str, int] = {}
    for v in nx.topological_sort(G):
        if not parents[v]:
            depth[v] = 0
        else:
            depth[v] = max(depth[p] for p in parents[v]) + 1

    # 5) Ancestors + descendants
    ancestors = {v: frozenset(nx.ancestors(G, v)) for v in G.nodes()}
    descendants = {v: frozenset(nx.descendants(G, v)) for v in G.nodes()}
    descendant_count = {v: len(descendants[v]) for v in G.nodes()}

    # 6) Hub ranking
    hubs = sorted(G.nodes(), key=lambda v: -mb_size[v])

    # 7) Cut ranking — articulation points of the moralized undirected
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

    # 8) Terminal ranking by depth(v) − |descendants(v)| (spec §1, §3.2, §3.8):
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


class _KindEvidenceAllocator:
    """Steps 2 + 3 of Phase 2 pipeline: kind assignment + evidence-set selection.

    Spec §3.3, §3.4, §4.2.
    """

    def assign(
        self,
        target: str,
        bucket: str,
        roles: NodeRoles,
        kind_alloc: Mapping[str, Mapping[str, float]],
        evidence_alloc: Mapping[str, Mapping[str, float]],
        n_evidence: Mapping[str, int],
        rng: random.Random,
    ) -> tuple[str, str, list[str]] | None:
        """Return ``(query_kind, evidence_strategy, evidence_nodes)`` or None
        if the evidence pool has zero candidates.

        - query_kind: prediction | diagnosis
        - evidence_strategy: longest_path | mb_neighbors | random
        - evidence_nodes: list of node names, length <= n_evidence[query_kind]
        """
        # Step 2: kind assignment (per spec §3.3)
        bucket_kind_probs = kind_alloc.get(
            bucket, {"prediction": 0.5, "diagnosis": 0.5}
        )
        query_kind = self._weighted_choice(bucket_kind_probs, rng)

        # Step 3: evidence-set selection (per spec §3.4)
        # 3a. Sub-bucket assignment
        ev_strategy_probs = evidence_alloc.get(query_kind, {})
        if not ev_strategy_probs:
            # Defensive: no allocation defined for this kind
            return None
        evidence_strategy = self._weighted_choice(ev_strategy_probs, rng)

        # 3b. Build candidate pool based on (kind, strategy)
        pool = self._evidence_pool(target, query_kind, evidence_strategy, roles)
        if not pool:
            # Empty pool (e.g. root node with prediction kind has no ancestors)
            return None

        # 3c. Pick n_evidence nodes from pool with hardness ordering
        n = n_evidence.get(query_kind, 3)
        chosen = self._pick_evidence(pool, evidence_strategy, n, target, roles, rng)

        return (query_kind, evidence_strategy, chosen)

    def _weighted_choice(
        self,
        probs: Mapping[str, float],
        rng: random.Random,
    ) -> str:
        """Draw a key from a {key: probability} dict using rng."""
        items = sorted(probs.items())  # deterministic order
        total = sum(p for _, p in items)
        if total <= 0:
            # All zero probabilities — degenerate; pick first key
            return items[0][0]
        r = rng.random() * total
        cumulative = 0.0
        for key, p in items:
            cumulative += p
            if r < cumulative:
                return key
        return items[-1][0]  # numerical floor case

    def _evidence_pool(
        self,
        target: str,
        query_kind: str,
        strategy: str,
        roles: NodeRoles,
    ) -> list[str]:
        """Build the candidate pool for evidence nodes.

        Spec §3.4 evidence pools table:
        - prediction + longest_path: ancestors of target, sorted by depth
          distance desc
        - prediction + mb_neighbors: parents(target) ∪ co_parents(target)
        - prediction + random: ancestors of target (uniform)
        - diagnosis + longest_path: descendants of target, sorted by depth
          distance desc
        - diagnosis + mb_neighbors: children(target) ∪ co_parents(target)
          (co_parents = co_children semantically; same structural set per
          spec §5 Q3)
        - diagnosis + random: all non-target, non-descendant nodes
        """
        if query_kind == "prediction":
            if strategy == "longest_path":
                ancestors = roles.ancestors.get(target, frozenset())
                # Deepest ancestors are farthest in topological depth from target.
                return sorted(
                    ancestors,
                    key=lambda v: -(roles.depth.get(target, 0) - roles.depth.get(v, 0)),
                )
            elif strategy == "mb_neighbors":
                parents = roles.parents.get(target, frozenset())
                co_parents = roles.co_parents.get(target, frozenset())
                pool = list(parents | co_parents)
                return sorted(pool, key=lambda v: -roles.mb_size.get(v, 0))
            elif strategy == "random":
                return list(roles.ancestors.get(target, frozenset()))
            else:
                return []
        elif query_kind == "diagnosis":
            if strategy == "longest_path":
                descendants = roles.descendants.get(target, frozenset())
                return sorted(
                    descendants,
                    key=lambda v: -(roles.depth.get(v, 0) - roles.depth.get(target, 0)),
                )
            elif strategy == "mb_neighbors":
                children = roles.children.get(target, frozenset())
                # Per spec §5 Q3: co_children = co_parents (same structural set)
                co_children = roles.co_parents.get(target, frozenset())
                pool = list(children | co_children)
                return sorted(pool, key=lambda v: -roles.mb_size.get(v, 0))
            elif strategy == "random":
                descendants = roles.descendants.get(target, frozenset())
                all_nodes = set(roles.mb_size.keys())
                return list(all_nodes - {target} - descendants)
            else:
                return []
        else:
            return []

    def _pick_evidence(
        self,
        pool: list[str],
        strategy: str,
        n: int,
        target: str,
        roles: NodeRoles,
        rng: random.Random,
    ) -> list[str]:
        """Pick up to n evidence nodes from the pool.

        For structured strategies (longest_path, mb_neighbors): take top-n
        by hardness ordering (descending). Pool is already sorted.
        For random strategy: uniform shuffle, then take first n.
        """
        if strategy == "random":
            pool_copy = list(pool)
            rng.shuffle(pool_copy)
            return pool_copy[:n]
        else:
            # longest_path or mb_neighbors: pool already sorted by hardness
            return pool[:n]
