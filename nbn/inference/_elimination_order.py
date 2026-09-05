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
    g = moralise(dag)

    # Prune evidence nodes (we condition on them; they don't participate
    # in elimination).  Connect their neighbours pairwise first so the
    # residual contains every induced clique.
    for ev in evidence_set:
        if ev in g:
            nbrs = list(g.neighbors(ev))
            for i, u in enumerate(nbrs):
                for v in nbrs[i + 1:]:
                    g.add_edge(u, v)
            g.remove_node(ev)

    # Iteratively eliminate the non-target node with smallest fill cost.
    order: List[str] = []
    eliminable = set(g.nodes()) - target_set
    while eliminable:
        best = min(
            eliminable,
            key=lambda v: (fill_count(g, v), g.degree(v), v),
        )
        order.append(best)
        g = eliminate(g, best)
        eliminable = set(g.nodes()) - target_set
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
