"""Min-fill elimination ordering for variable elimination.

Given the moralised undirected graph induced by the BN's DAG, return an
order in which non-target, non-evidence variables can be eliminated such
that the maximum induced clique (and hence peak intermediate factor
size) is approximately minimised.

This is the standard greedy heuristic from Koller & Friedman §9.4.3.  It
does not produce optimal orderings (NP-hard) but is provably within a
constant factor and runs in ``O(n^3)`` for our scale.

Round-1 diagnostic on PR #22 verified: on the v0.5b §A.5 OOM case
(``n_nodes=20, K=4, max_in_degree=4, B=16``) min-fill drops the peak
algebraic intermediate from 16 384 MiB to 256 MiB — a **64× reduction**.

Public API
----------
``min_fill_order(dag, *, targets, evidence)``
    Return a min-fill elimination order.

``get_order(name, dag, *, targets, evidence)``
    Dispatch by string name (``'min_fill'`` or ``'topological'``).

``ORDER_FUNCTIONS``
    Module-level dict of strategy name → callable.

The lower-level helpers ``moralise``, ``fill_count``, ``eliminate`` are
exposed primarily for tests and for the diagnostic in
``nbn/bench/diagnostics/ve_profile_n20.py``.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Mapping

import networkx as nx


def moralise(dag: nx.DiGraph) -> nx.Graph:
    """Return the moralised undirected graph of ``dag``.

    Each parent of every node is connected pairwise (the "marriage"
    edges); then direction is dropped.
    """
    g: nx.Graph = nx.Graph()
    g.add_nodes_from(dag.nodes())
    for node in dag.nodes():
        parents = list(dag.predecessors(node))
        for p in parents:
            g.add_edge(p, node)
        for i, p in enumerate(parents):
            for q in parents[i + 1:]:
                g.add_edge(p, q)
    return g


def fill_count(graph: nx.Graph, node: str) -> int:
    """Number of edges that would be added to make the neighbourhood of
    ``node`` into a clique (i.e., the "fill cost" of eliminating it)."""
    nbrs = list(graph.neighbors(node))
    n = len(nbrs)
    if n < 2:
        return 0
    existing = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if graph.has_edge(nbrs[i], nbrs[j])
    )
    return n * (n - 1) // 2 - existing


def eliminate(graph: nx.Graph, node: str) -> nx.Graph:
    """Return a copy of ``graph`` with ``node`` eliminated: its
    neighbours are fully connected, then the node is removed."""
    g = graph.copy()
    nbrs = list(g.neighbors(node))
    for i, u in enumerate(nbrs):
        for v in nbrs[i + 1:]:
            g.add_edge(u, v)
    g.remove_node(node)
    return g


def min_fill_order(
    dag: nx.DiGraph,
    *,
    targets: Iterable[str],
    evidence: Iterable[str],
    cards: Mapping[str, int] | None = None,
) -> List[str]:
    """Compute a min-fill elimination order over non-target, non-evidence
    variables.

    Parameters
    ----------
    dag:
        The BN's directed graph (an ``nx.DiGraph``).
    targets:
        Nodes to keep (never eliminated).
    evidence:
        Observed nodes (pruned from the moralised graph since they are
        conditioned on; their neighbours are connected pairwise so the
        residual still contains every induced clique).

    Returns
    -------
    A list of variable names in elimination order.  Length equals
    ``|nodes(dag)| - |targets| - |evidence|``.

    Tie-breaking
    ------------
    When multiple variables share the minimum fill cost, the variable
    with the smallest neighbourhood is chosen; alphabetical name breaks
    the remaining tie for determinism.  Weighted-min-fill (cardinality-
    aware) is a v0.7 enhancement.
    """
    target_set = set(targets)
    evidence_set = set(evidence)
    reduced = _moral_adjacency(dag)
    legacy = {v: set(s) for v, s in reduced.items()}

    # Evidence (and do-) nodes are conditioned on: every factor is reduced to
    # the observed value before elimination, so the variable simply leaves
    # the graph -- no fill edges [Koller & Friedman 2009, §9.3.2, pp. 303-304;
    # the reduced network, p. 307].  The marriage edges among the parents of
    # an observed child stay, because the child's reduced CPD still couples
    # them.  Up to v0.25 the neighbours of each evidence node were joined as
    # if it were eliminated ("legacy" graph).  That never changes an answer,
    # only the order; it is usually worse (by up to 4 in width on 500-node
    # random networks) but, min-fill being a heuristic, not always.  Both
    # orders are computed and the cheaper one is kept (ordering is cheap next
    # to inference; K&F Box 9.C, p. 315, recommends trying several), so the
    # result is never costlier than the legacy order.  Ties keep the legacy
    # order.
    for ev in evidence_set:
        if ev in reduced:
            for u in reduced.pop(ev):
                reduced[u].discard(ev)
        if ev in legacy:
            _eliminate_inplace(legacy, ev)
    o_red = _greedy_min_fill({v: set(x) for v, x in reduced.items()}, target_set)
    o_leg = _greedy_min_fill(legacy, target_set)
    if o_red == o_leg:
        return o_leg
    c_red = elimination_cost(reduced, o_red, cards)
    c_leg = elimination_cost(reduced, o_leg, cards)
    return o_red if c_red < c_leg else o_leg


def min_fill_legacy_order(
    dag: nx.DiGraph,
    *,
    targets: Iterable[str],
    evidence: Iterable[str],
    **_: object,
) -> List[str]:
    """The order NBN used up to v0.25: min-fill after *joining* the
    neighbours of each evidence node (as if eliminated).  Kept for
    reproducing historical measurements; :func:`min_fill_order` is never
    costlier by :func:`elimination_cost`."""
    adj = _moral_adjacency(dag)
    for ev in set(evidence):
        if ev in adj:
            _eliminate_inplace(adj, ev)
    return _greedy_min_fill(adj, set(targets))


def elimination_cost(adj: dict, order: List[str],
                     cards: Mapping[str, int] | None = None) -> tuple[int, int]:
    """``(largest, total)`` factor size of eliminating ``order`` on ``adj``.

    The sizes of the product factors ``psi_i`` that dominate the cost of
    variable elimination [K&F, §9.4.1, p. 306]; each is the product of the
    cardinalities over the eliminated node and its current neighbours.
    ``cards`` defaults to 2 for every variable.  ``adj`` is not mutated.
    """
    g = {v: set(x) for v, x in adj.items()}
    card = (lambda v: 2) if cards is None else (lambda v: int(cards.get(v, 2)))
    big = tot = 0
    for v in order:
        if v not in g:
            continue
        size = card(v)
        for u in g[v]:
            size *= card(u)
        big = max(big, size)
        tot += size
        _eliminate_inplace(g, v)
    return big, tot


def _moral_adjacency(dag: nx.DiGraph) -> dict:
    """Moral graph as a dict of neighbour sets (cheap to mutate in place)."""
    adj: dict = {v: set() for v in dag.nodes()}
    for node in dag.nodes():
        parents = list(dag.predecessors(node))
        for p in parents:
            adj[p].add(node)
            adj[node].add(p)
        for i, p in enumerate(parents):
            for q in parents[i + 1:]:
                adj[p].add(q)
                adj[q].add(p)
    return adj


def _eliminate_inplace(adj: dict, node: str) -> set:
    """Connect ``node``'s neighbours pairwise, remove it; return its neighbours."""
    nbrs = adj.pop(node)
    for u in nbrs:
        s = adj[u]
        s.discard(node)
        s.update(nbrs)
        s.discard(u)
    return nbrs


def _fill(adj: dict, node: str) -> int:
    """Fill-in count of ``node`` in ``adj`` (same value as :func:`fill_count`)."""
    nbrs = adj[node]
    n = len(nbrs)
    if n < 2:
        return 0
    # every edge inside the neighbourhood is counted from both endpoints
    inside = sum(len(adj[u] & nbrs) for u in nbrs) // 2
    return n * (n - 1) // 2 - inside


def _greedy_min_fill(adj: dict, keep: set, rank: dict | None = None) -> List[str]:
    """Greedy min-fill on ``adj`` (mutated), never eliminating ``keep``.

    Exact incremental form of the textbook loop that eliminates
    ``argmin_v (fill(v), degree(v), rank(v))`` at every step (``rank`` defaults
    to the node name) [Koller & Friedman
    2009, Alg. 9.4, p. 314]: it returns the *same* order as rescoring every
    node at every step.  Eliminating ``v`` changes only the neighbourhoods of
    ``N(v)``, and only a node adjacent to a pair inside ``N(v)`` can gain an
    edge inside its own neighbourhood, so only the scores of
    ``N(v) ∪ N(N(v))`` can move.  Stale heap entries are skipped lazily.

    The historical loop rescored all ``n`` nodes and copied the whole graph at
    every step (``O(n^2)`` graph work per query); on a 2000-node network this
    was ~97% of the exact-query time.
    """
    import heapq

    tb = (lambda v: v) if rank is None else rank.__getitem__
    key: dict = {}
    heap: list = []
    for v in adj:
        if v not in keep:
            k = (_fill(adj, v), len(adj[v]), tb(v), v)
            key[v] = k
            heap.append(k)
    heapq.heapify(heap)
    order: List[str] = []
    while heap:
        k = heapq.heappop(heap)
        v = k[-1]
        if key.get(v) != k:
            continue  # stale entry
        del key[v]
        order.append(v)
        nbrs = _eliminate_inplace(adj, v)
        touched = set(nbrs)
        for u in nbrs:
            touched |= adj[u]
        for u in touched:
            if u in key:
                nk = (_fill(adj, u), len(adj[u]), tb(u), u)
                if nk != key[u]:
                    key[u] = nk
                    heapq.heappush(heap, nk)
    return order


def _topological_order(
    dag: nx.DiGraph,
    *,
    targets: Iterable[str],
    evidence: Iterable[str],
    **_: object,
) -> List[str]:
    """Legacy topological order (pre-v0.6b behaviour).  Kept for
    comparability and as a fallback strategy."""
    target_set = set(targets)
    evidence_set = set(evidence)
    return [
        n
        for n in nx.topological_sort(dag)
        if n not in target_set and n not in evidence_set
    ]


# Public API: factory for elimination orders by name.
OrderFn = Callable[..., List[str]]
ORDER_FUNCTIONS: dict[str, OrderFn] = {
    "min_fill": min_fill_order,
    "min_fill_legacy": min_fill_legacy_order,
    "topological": _topological_order,
}


def get_order(
    name: str,
    dag: nx.DiGraph,
    *,
    targets: Iterable[str],
    evidence: Iterable[str],
    cards: Mapping[str, int] | None = None,
) -> List[str]:
    """Dispatch by string name.

    Raises
    ------
    ValueError
        If ``name`` is not a registered strategy.
    """
    fn = ORDER_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown elimination order {name!r}; "
            f"valid: {sorted(ORDER_FUNCTIONS)}"
        )
    if cards is None:
        return fn(dag, targets=targets, evidence=evidence)
    return fn(dag, targets=targets, evidence=evidence, cards=cards)


# --------------------------------------------------------------------------- #
# Width bounds: how large must exact elimination be?
# --------------------------------------------------------------------------- #


def induced_width(adj: dict, order: List[str]) -> int:
    """Width of ``order`` on the undirected graph ``adj`` (not mutated).

    The largest neighbourhood met when eliminating in this order, i.e. the
    largest factor scope minus one [Koller & Friedman 2009, Def. 9.6, p. 310].
    """
    g = {v: set(s) for v, s in adj.items()}
    w = 0
    for v in order:
        if v in g:
            w = max(w, len(g[v]))
            _eliminate_inplace(g, v)
    return w


def minor_min_width(adj: dict) -> int:
    """Minor-min-width lower bound on the tree-width of ``adj`` (not mutated).

    Repeatedly take a vertex of minimum degree, record that degree, and
    contract it into its minimum-degree neighbour.  Tree-width does not grow
    under taking minors and is at least the minimum degree, so the largest
    recorded degree is a lower bound (Gogate & Dechter, "A Complete Anytime
    Algorithm for Treewidth", UAI 2004, pp. 201-208,
    https://arxiv.org/abs/1207.4109).  Every
    elimination order therefore creates a factor over at least
    ``bound + 1`` variables [K&F, Thm 9.6, p. 309].
    """
    g = {v: set(s) for v, s in adj.items()}
    lb = 0
    while len(g) > 1:
        v = min(g, key=lambda x: (len(g[x]), x))
        d = len(g[v])
        lb = max(lb, d)
        if d == 0:
            del g[v]
            continue
        u = min(g[v], key=lambda x: (len(g[x]), x))
        # contract v into u
        for w in g.pop(v):
            g[w].discard(v)
            if w != u:
                g[w].add(u)
                g[u].add(w)
        g[u].discard(u)
    return lb


def query_width_bounds(
    dag: nx.DiGraph, *, targets: Iterable[str], evidence: Iterable[str],
) -> tuple[int, int]:
    """``(lower, upper)`` bounds on the width of exact elimination for a query.

    Computed on the moral graph of ``dag`` (pass the query's pruned factor
    scope, as ``TensorVariableElimination`` does) after evidence reduction.
    ``upper`` is the width of the greedy min-fill order NBN uses;
    ``lower`` is the minor-min-width bound, valid for every order.  The
    targets are kept, so the bound covers the final joint over them too.
    """
    target_set = set(targets)
    adj = _moral_adjacency(dag)
    for ev in set(evidence):
        if ev in adj:
            for u in adj.pop(ev):
                adj[u].discard(ev)
    order = _greedy_min_fill({v: set(s) for v, s in adj.items()}, target_set)
    # Width of eliminating everything but the targets, then the targets'
    # joint (a clique on them) — the target scope is part of the last factor.
    upper = max(induced_width(adj, order), len(target_set) - 1)
    lower = minor_min_width(adj)
    return lower, max(upper, lower)
