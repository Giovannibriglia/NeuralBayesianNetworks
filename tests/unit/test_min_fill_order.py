"""Tests for ``nbn.inference._elimination_order``.

The min-fill order on a small chain, V-structure, and naive-Bayes is
hand-checkable.  This file pins those answers as regression tests, plus
the n=20 v0.5b §A.5 OOM case as the integration regression.
"""
from __future__ import annotations

import networkx as nx
import pytest

from nbn.inference._elimination_order import (
    eliminate,
    fill_count,
    get_order,
    min_fill_order,
    moralise,
)


# ---------------------------------------------------------------------- #
# moralise / fill_count / eliminate primitives
# ---------------------------------------------------------------------- #


def test_moralise_chain_is_chain() -> None:
    """A simple chain ``X0 → X1 → X2`` has moralised graph ``X0—X1—X2``."""
    dag = nx.DiGraph([("X0", "X1"), ("X1", "X2")])
    g = moralise(dag)
    assert set(map(frozenset, g.edges())) == {
        frozenset({"X0", "X1"}),
        frozenset({"X1", "X2"}),
    }


def test_moralise_v_structure_adds_marriage_edge() -> None:
    """V-structure ``X0 → X2 ← X1`` moralises with ``X0—X1`` added."""
    dag = nx.DiGraph([("X0", "X2"), ("X1", "X2")])
    g = moralise(dag)
    assert g.has_edge("X0", "X1")
    assert g.has_edge("X0", "X2")
    assert g.has_edge("X1", "X2")


def test_fill_count_on_clique_is_zero() -> None:
    g = nx.Graph([("A", "B"), ("B", "C"), ("A", "C")])
    assert fill_count(g, "A") == 0
    assert fill_count(g, "B") == 0
    assert fill_count(g, "C") == 0


def test_fill_count_on_path_is_combinatorial() -> None:
    """Path ``A—B—C``: eliminating ``B`` forces ``A—C`` (one fill edge)."""
    g = nx.Graph([("A", "B"), ("B", "C")])
    assert fill_count(g, "B") == 1
    assert fill_count(g, "A") == 0
    assert fill_count(g, "C") == 0


def test_eliminate_connects_neighbours_and_removes_node() -> None:
    g = nx.Graph([("A", "B"), ("B", "C")])
    g2 = eliminate(g, "B")
    assert "B" not in g2.nodes()
    assert g2.has_edge("A", "C")


# ---------------------------------------------------------------------- #
# min_fill_order on hand-checked cases
# ---------------------------------------------------------------------- #


def test_min_fill_chain_excludes_target() -> None:
    """Chain ``X0→X1→X2→X3→X4`` with ``target={X2}``: order must be a
    permutation of {X0, X1, X3, X4} (target excluded)."""
    dag = nx.DiGraph()
    dag.add_edges_from([
        ("X0", "X1"), ("X1", "X2"), ("X2", "X3"), ("X3", "X4"),
    ])
    order = min_fill_order(dag, targets={"X2"}, evidence=set())
    assert set(order) == {"X0", "X1", "X3", "X4"}
    assert "X2" not in order


def test_min_fill_excludes_evidence_and_targets() -> None:
    """V-structure ``X0 → X2 ← X1`` with ``evidence={X0}, target={X2}``.
    Only ``X1`` remains to eliminate."""
    dag = nx.DiGraph([("X0", "X2"), ("X1", "X2")])
    order = min_fill_order(dag, targets={"X2"}, evidence={"X0"})
    assert order == ["X1"]


def test_min_fill_naive_bayes_eliminates_leaves() -> None:
    """Naive Bayes: ``C → {X0..X4}`` with ``target=C, evidence={X0, X1}``.
    Order should permute ``{X2, X3, X4}``."""
    dag = nx.DiGraph()
    dag.add_edges_from([
        ("C", "X0"), ("C", "X1"), ("C", "X2"), ("C", "X3"), ("C", "X4"),
    ])
    order = min_fill_order(dag, targets={"C"}, evidence={"X0", "X1"})
    assert set(order) == {"X2", "X3", "X4"}
    assert "C" not in order
    assert "X0" not in order and "X1" not in order


# ---------------------------------------------------------------------- #
# get_order dispatch
# ---------------------------------------------------------------------- #


def test_get_order_rejects_unknown() -> None:
    dag = nx.DiGraph([("X0", "X1")])
    with pytest.raises(ValueError, match="Unknown elimination order"):
        get_order("not_a_real_strategy", dag,
                  targets={"X1"}, evidence=set())


def test_get_order_topological_fallback_works() -> None:
    """Legacy 'topological' option preserved for backward-compat."""
    dag = nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X2", "X3")])
    order = get_order("topological", dag,
                      targets={"X3"}, evidence={"X0"})
    assert order == ["X1", "X2"]


def test_get_order_min_fill_dispatches() -> None:
    dag = nx.DiGraph([("X0", "X2"), ("X1", "X2")])
    order = get_order("min_fill", dag,
                      targets={"X2"}, evidence={"X0"})
    assert order == ["X1"]


# ---------------------------------------------------------------------- #
# Integration: n=20 v0.5b §A.5 OOM case
# ---------------------------------------------------------------------- #


def test_min_fill_n20_case_pins_order_and_invariants() -> None:
    """Regression test for the v0.5b §A.5 OOM case.

    On the synthetic discrete BN with ``n_nodes=20, K=4, max_in_degree=4,
    edge_density=0.20, seed=0`` and query ``target=X2, evidence={X0, X10}``,
    min-fill produces a 17-element order.  The exact order pinned below
    is the algorithm's measured output with the tie-breaking documented
    in ``_elimination_order.py`` (fill cost, then graph-degree in the
    residual, then alphabetical name).

    The first 8 eliminated variables are the "low-fill leaves" (each with
    fill cost 0 in the residual graph at their elimination step), which
    is the source of the 64× peak reduction over topological order
    measured in PR #22's round-1 follow-up.
    """
    from nbn.bench.synthetic import make_synthetic_bn

    bn = make_synthetic_bn(
        family="discrete", n_nodes=20,
        cardinality=4, max_in_degree=4,
        edge_density=0.20, seed=0, device="cpu",
        n_train=200, n_test=50, n_reference=500,
    )
    order = min_fill_order(bn.dag, targets={"X2"}, evidence={"X0", "X10"})

    # Invariants
    expected_set = (set(bn.dag.nodes()) - {"X2"} - {"X0", "X10"})
    assert set(order) == expected_set
    assert len(order) == len(set(order)) == 17, "no duplicates; full coverage"
    assert "X2" not in order
    assert "X0" not in order and "X10" not in order

    # Pin the exact order so future tie-breaking changes are caught.
    assert order == [
        "X16", "X9", "X15", "X11", "X14", "X19", "X5", "X13",
        "X1", "X12", "X17", "X18", "X3", "X4", "X6", "X7", "X8",
    ], f"min-fill order changed; got {order}"
