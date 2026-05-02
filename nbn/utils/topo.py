from __future__ import annotations

from typing import Dict, List

import networkx as nx


def elimination_order(graph: nx.Graph, heuristic: str = "min_fill") -> List[str]:
    """Compute variable elimination order using greedy heuristic.

    Parameters
    ----------
    graph:
        Undirected (moral) graph.
    heuristic:
        ``"min_fill"`` (default) or ``"min_degree"``.

    Returns
    -------
    List[str]
        Ordered list of variable names for elimination.
    """
    g = graph.copy()
    order = []
    nodes = set(g.nodes)
    while nodes:
        if heuristic == "min_fill":
            best = min(
                nodes,
                key=lambda n: _fill_edges(g, n),
            )
        else:
            best = min(nodes, key=lambda n: g.degree(n))
        order.append(best)
        nbrs = list(g.neighbors(best))
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                g.add_edge(nbrs[i], nbrs[j])
        g.remove_node(best)
        nodes.discard(best)
    return order


def _fill_edges(g: nx.Graph, node: str) -> int:
    """Number of fill edges needed to make node's neighbours a clique."""
    nbrs = list(g.neighbors(node))
    existing = sum(1 for i in range(len(nbrs)) for j in range(i + 1, len(nbrs)) if g.has_edge(nbrs[i], nbrs[j]))
    total = len(nbrs) * (len(nbrs) - 1) // 2
    return total - existing
