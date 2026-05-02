from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import networkx as nx


class DAG:
    """Immutable DAG wrapper around ``networkx.DiGraph``.

    Validates acyclicity on construction and caches topological order and
    per-node parent lists so hot-path inference code never calls networkx
    again after construction.

    Parameters
    ----------
    edges:
        Either a ``networkx.DiGraph`` or a sequence of ``(parent, child)``
        string tuples.  Isolated nodes (no edges) must be passed as a
        ``DiGraph`` with explicit ``add_node`` calls, or added via
        ``extra_nodes``.
    extra_nodes:
        Additional node names to register even if they have no edges.
    """

    def __init__(
        self,
        edges: nx.DiGraph | Sequence[Tuple[str, str]],
        extra_nodes: Sequence[str] | None = None,
    ) -> None:
        if isinstance(edges, nx.DiGraph):
            g = edges
        else:
            g = nx.DiGraph()
            g.add_edges_from(edges)
        if extra_nodes:
            g.add_nodes_from(extra_nodes)
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("Graph must be a DAG (no directed cycles).")
        self._g = g
        self._topo: List[str] = list(nx.topological_sort(g))
        self._parents: Dict[str, List[str]] = {
            n: list(g.predecessors(n)) for n in g.nodes
        }
        self._children: Dict[str, List[str]] = {
            n: list(g.successors(n)) for n in g.nodes
        }

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def nodes(self) -> List[str]:
        return list(self._g.nodes)

    def edges(self) -> List[Tuple[str, str]]:
        return list(self._g.edges)

    def parents(self, node: str) -> List[str]:
        return self._parents.get(node, [])

    def children(self, node: str) -> List[str]:
        return self._children.get(node, [])

    def topological_order(self) -> List[str]:
        return list(self._topo)

    def ancestors(self, node: str) -> List[str]:
        return list(nx.ancestors(self._g, node))

    def descendants(self, node: str) -> List[str]:
        return list(nx.descendants(self._g, node))

    def induced_width(self) -> int:
        """Approximate induced width via greedy min-fill triangulation."""
        moral = nx.moral_graph(self._g.to_undirected())
        order, _ = nx.minimum_fill_in(moral)
        width = 0
        g = moral.copy()
        for node in order:
            nbrs = list(g.neighbors(node))
            width = max(width, len(nbrs))
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    g.add_edge(nbrs[i], nbrs[j])
            g.remove_node(node)
        return width

    @property
    def networkx_graph(self) -> nx.DiGraph:
        return self._g

    def __len__(self) -> int:
        return len(self._g)

    def __contains__(self, node: str) -> bool:
        return node in self._g
