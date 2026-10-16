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
            # Copy: extra_nodes below (and any later caller edit) must not
            # leak between this immutable DAG and the caller's graph.
            g = nx.DiGraph(edges)
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

    def ordered_edges(self) -> List[Tuple[str, str]]:
        """Edges grouped by child, each child's parents in ``parents()`` order.

        Use this — not ``edges()`` — whenever a DAG is going to be
        *reconstructed*, e.g. persisted to a checkpoint or mutilated by the
        do-operator.

        ``DAG(g.edges())`` is not a faithful round-trip.  networkx iterates
        edges in source-node order (all of A's out-edges, then all of B's),
        so rebuilding from that list can permute a child's predecessor list:
        ``A→B, B→C, A→C`` has ``parents(C) == ['B', 'A']`` but its
        ``edges()`` is ``[(A,B), (A,C), (B,C)]``, which rebuilds to
        ``parents(C) == ['A', 'B']``.  That matters because every mechanism's
        tabulation is laid out in its *fit-time* parent order: permuting the
        parent list transposes the CPT's axes and inference returns
        confidently wrong probabilities with no error anywhere.
        """
        return [(p, n) for n in self.nodes() for p in self.parents(n)]

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
        # Incremental greedy min-fill (same key as the historical loop: fill,
        # then neighbourhood size, then the node's position in the graph,
        # which is what ``min`` over ``g.nodes`` tie-broke on), so the width
        # is unchanged; only the cost drops from O(n) rescoring plus graph
        # edits per step to rescoring the 2-neighbourhood of the eliminated
        # node.
        from nbn.inference._elimination_order import (
            _greedy_min_fill,
            _moral_adjacency,
            induced_width,
        )
        adj = _moral_adjacency(self._g)
        rank = {v: i for i, v in enumerate(self._g.nodes)}
        order = _greedy_min_fill({v: set(x) for v, x in adj.items()}, set(), rank)
        width = induced_width(adj, order)
        self._induced_width = width
        return width

    @property
    def networkx_graph(self) -> nx.DiGraph:
        return self._g

    def __len__(self) -> int:
        return len(self._g)

    def __contains__(self, node: str) -> bool:
        return node in self._g
