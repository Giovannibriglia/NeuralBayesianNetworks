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

from typing import Callable, Iterable, List

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
    adj = _moral_adjacency(dag)

    # Prune evidence nodes (we condition on them; they don't participate
    # in elimination).  Connect their neighbours pairwise first so the
    # residual contains every induced clique.  The residual graph does not
    # depend on the order in which evidence nodes are removed.
    for ev in evidence_set:
        if ev in adj:
            _eliminate_inplace(adj, ev)

    return _greedy_min_fill(adj, target_set)


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


def _greedy_min_fill(adj: dict, keep: set) -> List[str]:
    """Greedy min-fill on ``adj`` (mutated), never eliminating ``keep``.

    Exact incremental form of the textbook loop that eliminates
    ``argmin_v (fill(v), degree(v), name)`` at every step [Koller & Friedman
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

    key: dict = {}
    heap: list = []
    for v in adj:
        if v not in keep:
            k = (_fill(adj, v), len(adj[v]), v)
            key[v] = k
            heap.append(k)
    heapq.heapify(heap)
    order: List[str] = []
    while heap:
        k = heapq.heappop(heap)
        v = k[2]
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
                nk = (_fill(adj, u), len(adj[u]), u)
                if nk != key[u]:
                    key[u] = nk
                    heapq.heappush(heap, nk)
    return order


def _topological_order(
    dag: nx.DiGraph,
    *,
    targets: Iterable[str],
    evidence: Iterable[str],
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
    "topological": _topological_order,
}


def get_order(
    name: str,
    dag: nx.DiGraph,
    *,
    targets: Iterable[str],
    evidence: Iterable[str],
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
    return fn(dag, targets=targets, evidence=evidence)
