"""Unit tests for relevant_subnetwork (Bug 2 of #127).

Bayes-ball algorithm verification: each test specifies a known DAG +
target + evidence, asserts the returned relevant set matches the
m-connected (requisite probability node) set computed by hand.

Reference: docs/v0.13-bug2-subnetwork-pruning.md §6.1
"""
from __future__ import annotations

import networkx as nx
import pytest

from nbn.inference._pruning import relevant_subnetwork


def _dag(edges: list[tuple[str, str]]) -> nx.DiGraph:
    """Build a DAG from an edge list, including isolated nodes if they
    appear only on one side."""
    g = nx.DiGraph()
    for src, dst in edges:
        g.add_edge(src, dst)
    return g


class TestRelevantSubnetwork:
    """Cases from design doc §6.1."""

    def test_target_only(self):
        """Query with no evidence: ancestors are relevant; barren
        siblings are not."""
        # A -> B -> C ; A -> D (D is barren wrt target=C, no evidence)
        dag = _dag([("A", "B"), ("B", "C"), ("A", "D")])
        result = relevant_subnetwork(dag, target="C", evidence=None)
        # Relevant: C (target), B (parent of C), A (parent of B)
        # NOT relevant: D (barren -- has no descendant in {C})
        assert result == {"A", "B", "C"}

    def test_barren_nodes_excluded(self):
        """Barren descendants of siblings should be excluded."""
        # A -> B (target) ; A -> C -> D ; A -> E
        # C, D, E are all barren wrt target=B with no evidence
        dag = _dag([("A", "B"), ("A", "C"), ("C", "D"), ("A", "E")])
        result = relevant_subnetwork(dag, target="B", evidence=None)
        assert result == {"A", "B"}

    def test_evidence_separates(self):
        """Evidence on the middle of a chain blocks the far end."""
        # A -> B -> C -> D ; target=A, evidence={C}
        # Conditioning on C makes A independent of D
        dag = _dag([("A", "B"), ("B", "C"), ("C", "D")])
        result = relevant_subnetwork(dag, target="A", evidence={"C": 0})
        # A and C are in the set; B is between them (m-connected)
        # D is m-separated from A given C
        assert result == {"A", "B", "C"}

    def test_v_structure_unblocked(self):
        """Collider with evidence on the collider: parents become
        m-connected."""
        # A -> C <- B ; target=A, evidence={C}
        # Conditioning on C activates the v-structure: A and B become
        # m-connected
        dag = _dag([("A", "C"), ("B", "C")])
        result = relevant_subnetwork(dag, target="A", evidence={"C": 0})
        assert result == {"A", "B", "C"}

    def test_v_structure_blocked(self):
        """Collider with no evidence on it: parents stay m-separated."""
        # A -> C <- B ; target=A, evidence=None
        # Without conditioning, A and B are marginally independent
        dag = _dag([("A", "C"), ("B", "C")])
        result = relevant_subnetwork(dag, target="A", evidence=None)
        # Only A is relevant -- C and B are barren
        assert result == {"A"}

    def test_v_structure_descendant_evidence(self):
        """Collider with evidence on a DESCENDANT of the collider also
        activates the v-structure."""
        # A -> C <- B ; C -> D ; target=A, evidence={D}
        # Conditioning on a descendant of the collider also activates
        dag = _dag([("A", "C"), ("B", "C"), ("C", "D")])
        result = relevant_subnetwork(dag, target="A", evidence={"D": 0})
        assert result == {"A", "B", "C", "D"}

    def test_hub_given_mb(self):
        """Hub query given full Markov blanket: only the MB + hub are
        relevant.

        This is the case from the investigation that motivated Bug 2 --
        pre-fix nbn-cat-ve eliminated 38 irrelevant variables and
        allocated 612 GB. Post-fix it should restrict to the hub + its
        MB."""
        # Hub H has parents P1, P2; children C1, C2.
        # Each child has co-parent CP1, CP2.
        # MB(H) = {P1, P2, C1, C2, CP1, CP2}
        # Outside MB: ancestors of parents (G1), and barren
        # descendants (X1)
        # Target=H, evidence=MB -> only MB u {H} relevant
        dag = _dag([
            ("G1", "P1"),     # grandparent above the MB
            ("P1", "H"),
            ("P2", "H"),
            ("H", "C1"),
            ("H", "C2"),
            ("CP1", "C1"),
            ("CP2", "C2"),
            ("C1", "X1"),     # barren descendant
        ])
        mb = {"P1": 0, "P2": 0, "C1": 0, "C2": 0, "CP1": 0, "CP2": 0}
        result = relevant_subnetwork(dag, target="H", evidence=mb)
        # Should be exactly H + the MB. G1 and X1 are pruned.
        expected = {"H", "P1", "P2", "C1", "C2", "CP1", "CP2"}
        assert result == expected, (
            f"hub-given-MB pruning failed. "
            f"expected={expected}, got={result}, "
            f"extra={result - expected}, missing={expected - result}"
        )

    def test_no_evidence_explicit_empty(self):
        """Explicit empty evidence dict equivalent to None."""
        dag = _dag([("A", "B"), ("B", "C")])
        result_none = relevant_subnetwork(dag, target="C", evidence=None)
        result_empty = relevant_subnetwork(dag, target="C", evidence={})
        assert result_none == result_empty

    def test_chain_evidence_blocks_both_ways(self):
        """Evidence in the middle of a chain blocks both directions."""
        # A -> B -> C -> D -> E ; target=A, evidence={C}
        # D, E are m-separated from A given C
        dag = _dag([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
        result = relevant_subnetwork(dag, target="A", evidence={"C": 0})
        assert result == {"A", "B", "C"}

        # And from the other side: target=E, evidence={C}
        result2 = relevant_subnetwork(dag, target="E", evidence={"C": 0})
        # E is target, D is parent of E, C is evidence. A and B m-sep.
        assert result2 == {"C", "D", "E"}


class TestRelevantSubnetworkAPI:
    """Defensive tests for the function's input/output contract."""

    def test_target_as_string(self):
        dag = _dag([("A", "B")])
        assert relevant_subnetwork(dag, "B") == {"A", "B"}

    def test_target_as_iterable(self):
        dag = _dag([("A", "B"), ("C", "D")])
        # Multiple targets -- both ancestors should be relevant
        result = relevant_subnetwork(dag, target=["B", "D"])
        assert result == {"A", "B", "C", "D"}

    def test_evidence_as_set(self):
        dag = _dag([("A", "B"), ("B", "C")])
        # Pass evidence as a set (not a dict) -- values are ignored
        result = relevant_subnetwork(dag, target="A", evidence={"C"})
        # Equivalent to evidence={"C": 0}
        assert result == {"A", "B", "C"}

    def test_unknown_target_raises(self):
        dag = _dag([("A", "B")])
        with pytest.raises(KeyError, match="ZZZ"):
            relevant_subnetwork(dag, target="ZZZ")

    def test_unknown_evidence_raises(self):
        dag = _dag([("A", "B")])
        with pytest.raises(KeyError, match="ZZZ"):
            relevant_subnetwork(dag, target="A", evidence={"ZZZ": 0})
