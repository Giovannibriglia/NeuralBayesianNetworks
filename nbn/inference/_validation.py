"""Validation of discrete evidence / do values shared by the engines.

A discrete node's states are the indices ``0 .. K-1``.  Out-of-range or
fractional values used to be accepted silently, and every engine read them
differently: torch indexing wraps ``-1`` to the *last* state (VE, and LW's
histogram/CPT lookups), and VE truncated ``1.5`` to state 1.  The same query
thus returned plausible but different answers per engine.  The engines now
reject such values up front, with the core's message
(:func:`nbn.core.variables.validate_discrete_value`).
"""
from __future__ import annotations

from typing import Iterable, Mapping

import torch

from nbn.core.variables import validate_discrete_value


def _cardinality(model, node: str) -> int | None:
    # The declared cardinality is always known; a mechanism's n_classes is 0
    # until it is fitted (reading it would mask the clearer "has not been
    # fitted" error raised later).
    var = model.variables.get(node) if hasattr(model, "variables") else None
    k = getattr(var, "cardinality", None)
    if not k:
        k = getattr(model.mechanisms[node], "n_classes", None)
    return int(k) if k else None


def _is_discrete(model, node: str) -> bool:
    var = model.variables.get(node) if hasattr(model, "variables") else None
    if var is not None and getattr(var, "kind", None) is not None:
        return var.kind == "discrete"
    return bool(getattr(model.mechanisms[node], "is_discrete", False))


def check_discrete_values(
    model,
    values: Mapping[str, object],
    do_keys: Iterable[str] = (),
) -> None:
    """Raise ``ValueError`` if a discrete node's value is not a valid state.

    ``values`` maps node → scalar / tensor of any shape (all entries are
    checked); continuous nodes are skipped; an unknown node raises
    ``KeyError``.  The per-node min / max / non-integral reductions are
    stacked so a batched query pays one device→host sync, not one per node.
    """
    do_keys = set(do_keys)
    names, rows = [], []
    for node, v in values.items():
        if node not in model.mechanisms:
            kind = "do" if node in do_keys else "evidence"
            raise KeyError(f"{kind} node {node!r} is not a node of the model.")
        if v is None or not _is_discrete(model, node):
            continue
        t = (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)).detach()
        if t.numel() == 0:
            continue
        # float32 everywhere: MPS has no float64, and state indices are small.
        tf = t.to(torch.float32)
        frac = (tf != tf.round()).any().to(torch.float32)
        rows.append(torch.stack([tf.min(), tf.max(), frac]))
        names.append(node)
    if not names:
        return
    stats = torch.stack([r.to(rows[0].device) for r in rows]).tolist()
    for node, (lo, hi, frac) in zip(names, stats):
        k = _cardinality(model, node)
        if frac or lo < 0 or (k is not None and hi >= k):
            what = "Intervention value" if node in do_keys else "Evidence value"
            validate_discrete_value(node, values[node], k, what=what)
            # validate_discrete_value raises on exactly these conditions; this
            # is a belt-and-braces guard should its contract ever loosen.
            raise ValueError(
                f"{what} for discrete node {node!r} is not a valid state "
                f"(min {lo}, max {hi}, cardinality {k})."
            )
