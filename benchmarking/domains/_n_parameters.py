"""Compute n_parameters (total discrete CPT cell count) for a network (#133).

``n_parameters = sum over nodes of cardinality(node) x prod(cardinality(p)
                 for p in parents(node))``

This is the total number of probability values in the network's *discrete*
CPTs -- a tighter proxy for inference cost than ``n_nodes`` alone. Two networks
with the same node count but different cardinalities (barley vs alarm) have very
different parameter counts.

**Continuous nodes contribute 0** (strict interpretation): a continuous node is
parameterized by mean+variance, not an enumerated table, so it has no CPT cells.
A discrete table that is conditioned on a continuous parent likewise contributes
0 (its parent product includes a 0-cardinality factor) -- only fully-discrete
tables are counted. Networks with no discrete CPTs therefore have
``n_parameters == 0``; the paper figure script (docs/v0.13-paper-figures.md
§5.4b) log-skips scaling plots for such uninformative axes.

Used by the paper figures for the cardinality-aware scaling axis. Issue: #133.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def n_parameters_from_cardinalities(
    cardinalities: Mapping[str, int],
    parents: Mapping[str, Iterable[str]],
) -> int:
    """Compute n_parameters from explicit cardinality + parent mappings.

    Args:
        cardinalities: node name -> cardinality (number of discrete states;
            use 0 for continuous nodes, which contribute nothing).
        parents: node name -> iterable of parent node names.

    Returns:
        Sum over nodes of ``K_node * prod(K_parent)``. A table touching any
        0-cardinality (continuous) variable contributes 0.
    """
    total = 0
    for node, k in cardinalities.items():
        parent_product = 1
        for p in parents.get(node, []):
            parent_product *= cardinalities.get(p, 0)
        total += k * parent_product
    return total


def n_parameters_from_problem(problem: Any) -> int | None:
    """Compute n_parameters for a ``BenchmarkProblem``.

    Reads cardinalities from ``problem.variables`` (``{node: (kind, K)}``;
    discrete nodes use ``K``, continuous nodes contribute 0) and parents from
    ``problem.dag`` (an edge list ``[(parent, child), ...]``).

    Returns ``None`` if the problem lacks the structure to compute it (e.g. a
    bare unit-test fixture), so callers can treat it as "absent" rather than
    erroring.
    """
    variables = getattr(problem, "variables", None)
    dag = getattr(problem, "dag", None)
    if not variables or dag is None:
        return None

    cardinalities: dict[str, int] = {}
    for node, spec in variables.items():
        kind = spec[0] if isinstance(spec, (tuple, list)) and spec else None
        k = spec[1] if isinstance(spec, (tuple, list)) and len(spec) > 1 else 0
        cardinalities[node] = int(k) if kind == "discrete" else 0

    parents: dict[str, list[str]] = {node: [] for node in cardinalities}
    for edge in dag:
        parent, child = edge[0], edge[1]
        if child in parents:
            parents[child].append(parent)

    return n_parameters_from_cardinalities(cardinalities, parents)


def n_nodes_from_problem(problem: Any) -> int | None:
    """Number of variables (nodes) in a problem; per-problem constant.

    Reads ``len(problem.variables)`` -- the same structure
    :func:`n_parameters_from_problem` consumes. Returns ``None`` when the
    problem lacks variables (e.g. a bare unit-test fixture), mirroring
    ``n_parameters_from_problem``'s None-on-missing contract so the runner's
    inject sites stay parallel.
    """
    variables = getattr(problem, "variables", None)
    if not variables:
        return None
    return len(variables)
