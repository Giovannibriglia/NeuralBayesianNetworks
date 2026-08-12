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
        # Memoized induced_width() result. The DAG is immutable (class
        # contract above), so the greedy min-fill pass — O(n * deg^2) pure
        # Python — need only ever run once. HybridRouter calls
        # induced_width() on every query/query_batch dispatch; without this
        # cache the graph work can dominate the actual inference.
        self._induced_width: int | None = None

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
        """Approximate induced width via greedy min-fill triangulation.

        Moralises the DAG, then repeatedly eliminates the node whose
        removal introduces the fewest fill-in edges among its neighbours
        (ties broken by smallest neighbourhood), tracking the largest
        neighbourhood seen — an upper bound on the treewidth that
        ``HybridRouter`` uses to choose exact VE versus likelihood
        weighting.  ``nx.moral_graph`` requires the *directed* graph
        (it is ``@not_implemented_for("undirected")``); moralising the
        DiGraph directly also adds the parent-marrying edges that an
        undirected projection would omit.

        The result is memoized on first call (the DAG is immutable).
        """
        if self._induced_width is not None:
            return self._induced_width
        g = nx.moral_graph(self._g)
        width = 0
        while g.number_of_nodes():
            def _fill_cost(node: str) -> tuple[int, int]:
                nbrs = list(g.neighbors(node))
                fill = sum(
                    not g.has_edge(nbrs[i], nbrs[j])
                    for i in range(len(nbrs))
                    for j in range(i + 1, len(nbrs))
                )
                return (fill, len(nbrs))

            node = min(g.nodes, key=_fill_cost)
            nbrs = list(g.neighbors(node))
            width = max(width, len(nbrs))
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    g.add_edge(nbrs[i], nbrs[j])
            g.remove_node(node)
        self._induced_width = width
        return width

    @property
    def networkx_graph(self) -> nx.DiGraph:
        return self._g

    def __len__(self) -> int:
        return len(self._g)

    def __contains__(self, node: str) -> bool:
        return node in self._g
