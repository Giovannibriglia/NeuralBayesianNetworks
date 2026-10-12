"""Relevant-subnetwork pruning for variable elimination (Bug 2 of #127).

Implements Bayes-ball (Shachter 1998) to compute the set of nodes
m-connected to a target given evidence -- i.e., the variables that
actually contribute to the answer of a probabilistic query.

Used by ``TensorVariableElimination`` to restrict elimination to relevant
variables only, avoiding the eliminate-everything pathology that crashed
barley with 612 GB allocations (#127, Bug 2).

A node ``n`` is in the relevant set iff it is m-connected to the target
given the evidence: barren nodes (no descendant in target/evidence) and
nodes m-separated from the target by the evidence are excluded.  Target
and evidence nodes are always included.

Public API
----------
``relevant_subnetwork(dag, target, evidence=None)``
    Return the set of nodes m-connected to ``target`` given ``evidence``.

Reference: Shachter (1998) "Bayes-Ball: The Rational Pastime";
Geiger-Verma-Pearl (1990).
Tracker: #127 (Bug 2).
"""
from __future__ import annotations

from typing import Iterable, Mapping

import networkx as nx

# The two ball-arrival directions (Shachter 1998 §3).
_FROM_CHILD = 0   # ball arrived at a node from one of its children
_FROM_PARENT = 1  # ball arrived at a node from one of its parents


def relevant_subnetwork(
    dag: nx.DiGraph,
    target: str | Iterable[str],
    evidence: Mapping[str, object] | Iterable[str] | None = None,
) -> set[str]:
    """Compute the nodes m-connected to ``target`` given ``evidence``.

    Implements the Bayes-ball traversal: the relevant set is exactly the
    nodes reachable from the target via active paths under the evidence
    set.  This is the standard relevant-subnetwork pruning step that
    standard variable elimination performs and nbn's VE was missing.

    The function is a pure graph algorithm on the DAG structure -- it
    knows nothing about factors, CPTs, or mechanisms.  Evidence *values*
    are irrelevant to m-separation, so only the evidence *names* are used.

    Args:
        dag: the Bayesian network structure (a directed acyclic graph).
        target: query variable name, or an iterable of names.
        evidence: observed variables -- either a mapping whose keys are
            the observed names (values ignored) or an iterable of names.
            ``None`` means no evidence.

    Returns:
        The set of node names m-connected to ``target`` given
        ``evidence``.  Always includes ``target`` and ``evidence``
        themselves, even when they are isolated.

    Raises:
        KeyError: if any target or evidence name is not a node in ``dag``.
    """
    target_set = {target} if isinstance(target, str) else set(target)
    evidence_set = _as_name_set(evidence)

    _validate_names(dag, target_set | evidence_set)
    top = _top_marked(dag, target_set, evidence_set)

    # Top-marked nodes are the requisite ones; target and evidence are
    # relevant by definition, even when isolated.
    return top | target_set | evidence_set


def requisite_nodes(
    dag: nx.DiGraph,
    target: str | Iterable[str],
    evidence: Mapping[str, object] | Iterable[str] | None = None,
) -> set[str]:
    """Nodes whose CPT is needed for ``P(target | evidence)`` (#132).

    The top-marked set of the Bayes-ball traversal — Shachter (1998)'s
    requisite probability nodes — plus the target.  It differs from
    :func:`relevant_subnetwork` only by the *non-requisite evidence* nodes:
    observed nodes the ball never enters from a parent, e.g. an observed
    parent or ancestor of the target.  Their CPT, evaluated at the observed
    value, is a function of their own parents only and cancels in the
    normalised posterior, so dropping it is exact — and it keeps those
    parents (possibly high-cardinality) out of the elimination entirely.
    The evidence *values* of such nodes still matter: they are clamped
    wherever the node appears in a kept CPT's scope.
    """
    target_set = {target} if isinstance(target, str) else set(target)
    evidence_set = _as_name_set(evidence)
    _validate_names(dag, target_set | evidence_set)
    return _top_marked(dag, target_set, evidence_set) | target_set


def _as_name_set(evidence) -> set[str]:
    if evidence is None:
        return set()
    if isinstance(evidence, Mapping):
        return set(evidence.keys())
    return set(evidence)


def _validate_names(dag: nx.DiGraph, names: set[str]) -> None:
    # Validate every supplied name against the DAG up front -- a typo in
    # a query should fail loudly, not silently return an empty set.
    dag_nodes = set(dag.nodes())
    for name in names:
        if name not in dag_nodes:
            raise KeyError(
                f"node {name!r} not in DAG "
                f"(have {sorted(dag_nodes)[:5]}...)"
            )


def _top_marked(dag: nx.DiGraph, target_set: set[str], evidence_set: set[str]) -> set[str]:
    """Top-marked nodes of the Bayes-ball traversal from ``target_set``."""
    # Bayes-ball search for requisite probability nodes (Shachter 1998).
    #
    # The ball visits each node having arrived from either a child
    # (travelling up) or a parent (travelling down).  Each node carries
    # up to two marks:
    #   - "top"    : reached from a child -> the node's CPT is requisite
    #   - "bottom" : reached from a parent
    # The marks also serve as the termination guard (a node is expanded
    # at most once per direction).  The relevant set is exactly the
    # top-marked nodes -- nodes reached only from a parent are barren
    # (their factor integrates to 1) and contribute nothing.
    top: set[str] = set()
    bottom: set[str] = set()

    # Inquiring about a target is modelled as a ball arriving from a
    # (hypothetical) child, so it propagates up to parents and down to
    # children just like an ordinary FROM_CHILD visit.
    queue: list[tuple[str, int]] = [(t, _FROM_CHILD) for t in target_set]

    while queue:
        node, direction = queue.pop()

        if direction == _FROM_CHILD:
            if node in evidence_set:
                # Observed node hit from below: the ball is blocked -- a
                # chain/fork through an observed node is inactive.
                continue
            if node not in top:
                top.add(node)
                for parent in dag.predecessors(node):
                    queue.append((parent, _FROM_CHILD))
            if node not in bottom:
                bottom.add(node)
                for child in dag.successors(node):
                    queue.append((child, _FROM_PARENT))
        else:  # _FROM_PARENT
            if node in evidence_set:
                # Observed collider: the ball bounces back up to parents
                # (the v-structure activation) but cannot continue down
                # to children.  This also fires when an *unobserved* node
                # below has carried the ball down to an observed
                # descendant -- handling the descendant-evidence case.
                if node not in top:
                    top.add(node)
                    for parent in dag.predecessors(node):
                        queue.append((parent, _FROM_CHILD))
            elif node not in bottom:
                bottom.add(node)
                for child in dag.successors(node):
                    queue.append((child, _FROM_PARENT))

    return top
