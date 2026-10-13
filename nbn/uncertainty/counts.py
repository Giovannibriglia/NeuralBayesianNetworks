"""Count tables of a discrete node, straight from data (no fitted mechanism needed)."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from nbn.learning.weighting import validate_weights
from nbn.utils.device import accum_dtype, to_device_dtype


def parent_row_index(parents: torch.Tensor, parent_cards: list[int]) -> torch.Tensor:
    """Flat row index of each parent configuration, C order (last parent fastest).

    This is the layout of ``CategoricalTableMechanism`` and
    ``Mechanism.tabulate``: ``table.reshape(*parent_cards, K)`` recovers the
    per-parent axes.  Raises on values outside the declared cardinalities.
    """
    index = torch.zeros(parents.shape[:-1], dtype=torch.long, device=parents.device)
    for d, card in enumerate(parent_cards):
        values = parents[..., d].long()
        if values.numel() and (int(values.min()) < 0 or int(values.max()) >= card):
            raise ValueError(f"Parent {d} has values outside its {card} declared states.")
        index = index * card + values
    return index


def parent_grid(parent_cards: Sequence[int], device: Any = "cpu") -> torch.Tensor:
    """Every parent configuration ``[prod(cards), len(cards)]`` in C order (last fastest).

    Row ``r`` of the result is the configuration whose :func:`parent_row_index`
    is ``r``.
    """
    cards = [int(c) for c in parent_cards]
    if not cards:
        return torch.zeros(1, 0, dtype=torch.long, device=device)
    mesh = torch.meshgrid(*[torch.arange(c, device=device) for c in cards], indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=-1)


def scatter_counts(
    rows: torch.Tensor,
    values: torch.Tensor,
    n_rows: int,
    n_classes: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """``[n_rows, n_classes]`` (weighted) counts of ``values`` per row index.

    Accumulates in the widest float the device supports.  Raises on values
    outside ``0 .. n_classes - 1`` instead of spilling them into the next row.
    """
    values = values.reshape(-1).long()
    if values.numel() and (int(values.min()) < 0 or int(values.max()) >= n_classes):
        raise ValueError(f"Child values outside the {n_classes} declared states.")
    dtype = accum_dtype(values.device)
    contrib = (
        torch.ones(values.shape[0], dtype=dtype, device=values.device)
        if weights is None else to_device_dtype(weights, values.device, dtype)
    )
    counts = torch.zeros(n_rows * n_classes, dtype=dtype, device=values.device)
    counts.scatter_add_(0, rows.reshape(-1) * n_classes + values, contrib)
    return counts.reshape(n_rows, n_classes)


def count_table(
    model,
    data: Mapping[str, torch.Tensor],
    node: str,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Counts ``[R, K]`` of ``node`` per parent configuration.

    Rows follow the DAG's parent order and the declared cardinalities of the
    model's variables, so the table lines up with the node's mechanism.
    Same as :func:`count_table_for` with ``parents=model.dag.parents(node)``.

    Returns
    -------
    counts:
        ``counts[r, k]`` = (weighted) number of rows with parent configuration
        ``r`` and child value ``k``.
    parent_cards:
        Cardinality of each parent, in DAG order.
    """
    return count_table_for(model.variables, data, node, model.dag.parents(node), weights)


def count_table_for(
    variables: Mapping[str, Any],
    data: Mapping[str, torch.Tensor],
    node: str,
    parents: Sequence[str],
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Counts ``[R, K]`` of ``node`` against any tuple of discrete ``parents``.

    Unlike :func:`count_table`, the parents need not be the node's parents in
    a model: any tuple of other discrete variables works, in the order given.
    This is what a cache of counts over several candidate parent sets needs.
    Rows are in C order (last parent fastest), the layout of
    ``CategoricalTableMechanism``, so ``counts.reshape(*parent_cards, K)``
    recovers one axis per parent.

    Parameters
    ----------
    variables:
        Variable specs by name (e.g. ``model.variables``); each needs
        ``is_discrete`` and ``cardinality``.
    data:
        Columns by name; ``data[v]`` holds the integer values of ``v``.
    node, parents:
        The child and the parent tuple to count against.
    weights:
        Optional ``[N]`` per-row weights (replication semantics), validated
        like ``fit``'s.  Weighted tables suit fitting and Bayesian sets, not
        the PAC builders (``interval_set_from_counts``,
        ``l1_ball_set_from_counts``): a weight is not a number of independent
        draws.

    Returns
    -------
    counts, parent_cards:
        As in :func:`count_table`, with ``parent_cards`` in the order of
        ``parents``.
    """
    if isinstance(parents, str):
        raise TypeError(f"parents must be a sequence of names, not the string {parents!r}; "
                        f"write ({parents!r},) for one parent.")
    parents = tuple(parents)
    if node not in variables:
        raise KeyError(f"Unknown variable '{node}'.")
    if node in parents:
        raise ValueError(f"'{node}' cannot be its own parent.")
    if len(set(parents)) != len(parents):
        raise ValueError(f"Repeated parents in {parents}.")
    variable = variables[node]
    if not variable.is_discrete:
        raise ValueError(f"count_table_for needs a discrete node; '{node}' is continuous.")
    parent_cards = []
    for p in parents:
        if p not in variables:
            raise KeyError(f"Unknown parent '{p}'.")
        pv = variables[p]
        if not pv.is_discrete:
            raise ValueError(f"count_table_for needs discrete parents; '{p}' is continuous.")
        parent_cards.append(int(pv.cardinality))
    absent = [v for v in (node, *parents) if v not in data]
    if absent:
        raise KeyError(f"No data column for {absent}.")

    x = data[node].reshape(-1)
    weights = validate_weights(weights, x.shape[0], where="count_table_for", device=x.device)
    if parents:
        pa = torch.stack([data[p].reshape(-1) for p in parents], dim=-1)
    else:
        pa = torch.zeros(x.shape[0], 0, device=x.device)
    rows = parent_row_index(pa, parent_cards)
    n_rows = math.prod(parent_cards)
    return scatter_counts(rows, x, n_rows, int(variable.cardinality), weights), parent_cards
