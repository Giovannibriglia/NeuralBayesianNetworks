"""Drive incremental updates node-by-node, mirroring ``learning.fit``.

This is the ``update`` counterpart to :func:`nbn.learning.fit.fit` (method
``"local"``): it walks the DAG in topological order, packs parents the same
way, threads declared cardinalities the same way, and dispatches to each
mechanism's ``update_local`` instead of ``fit_local``.  Mechanisms that do
not support incremental update are skipped (recorded in ``UpdateHistory``).
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict

import torch

from nbn.update.base import UpdateHistory
from nbn.utils.batching import pack_parents

if TYPE_CHECKING:
    from nbn.core.network import NeuralBayesianNetwork

logger = logging.getLogger(__name__)


def update(
    model: NeuralBayesianNetwork,
    data: Dict[str, torch.Tensor],
    *,
    forgetting: float = 1.0,
    device: str | None = None,
    **kwargs: Any,
) -> UpdateHistory:
    """Fold ``data`` into an already-fitted ``model`` without rehearsal.

    Parameters
    ----------
    model:
        A ``NeuralBayesianNetwork`` whose mechanisms have already been fitted.
    data:
        Dict mapping node name → new-data tensor of shape ``[N, D]`` or ``[N]``.
    forgetting:
        Exponential-forgetting factor in ``(0, 1]`` applied to each
        mechanism's persisted prior state (``1.0`` = pure recursive Bayes).

    Returns
    -------
    UpdateHistory
    """
    if not (0.0 < float(forgetting) <= 1.0):
        raise ValueError(f"forgetting factor must be in (0, 1], got {forgetting!r}")

    if device is None:
        device = str(model.device)
    dev = torch.device(device)
    model.to(dev)

    data_dev: Dict[str, torch.Tensor] = {k: v.to(dev) for k, v in data.items()}

    history = UpdateHistory(device=device)
    t0 = time.perf_counter()

    for node in model.dag.topological_order():
        if node not in model.mechanisms:
            raise RuntimeError(f"No mechanism registered for node '{node}'.")
        mech = model.mechanisms[node]
        parents = model.dag.parents(node)
        pa_tensor = pack_parents(data_dev, parents)
        x = data_dev[node]

        if not getattr(mech, "supports_update", False):
            history.skipped.append(node)
            continue

        # Mirror fit.py's declared-cardinality threading (Bug 1a, #127) so the
        # update spans the same fixed table the fit produced.
        mech_kwargs: Dict[str, Any] = dict(kwargs)
        node_var = model.variables.get(node)
        if node_var is not None and node_var.cardinality is not None:
            mech_kwargs.setdefault("n_classes", int(node_var.cardinality))
        parent_vars = [model.variables.get(p) for p in parents]
        if parents and all(
            pv is not None and pv.cardinality is not None for pv in parent_vars
        ):
            mech_kwargs.setdefault(
                "parent_cards", [int(pv.cardinality) for pv in parent_vars]
            )

        metrics = mech.update_local(x, pa_tensor, forgetting=forgetting, **mech_kwargs)
        history.node_methods[node] = (metrics or {}).get("method", "unknown")

        with torch.no_grad():
            history.node_log_likelihoods[node] = float(
                mech.log_prob(x, pa_tensor).mean()
            )

    history.wall_clock_s = time.perf_counter() - t0
    return history
