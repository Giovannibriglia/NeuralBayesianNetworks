"""Drive incremental updates node-by-node, mirroring ``learning.fit``.

This is the ``update`` counterpart to :func:`nbn.learning.fit.fit` (method
``"local"``): it walks the DAG in topological order, packs parents the same
way, threads declared cardinalities the same way, and dispatches to each
mechanism's ``update_local`` instead of ``fit_local``.  Mechanisms that do
not support incremental update are skipped (recorded in ``UpdateHistory``).
"""
from __future__ import annotations

import copy
import logging
import time
from typing import TYPE_CHECKING, Any, Dict

import torch

from nbn.update.base import UpdateHistory
from nbn.utils.batching import pack_parents
from nbn.utils.device import to_device

if TYPE_CHECKING:
    from nbn.core.network import NeuralBayesianNetwork

logger = logging.getLogger(__name__)


def update(
    model: NeuralBayesianNetwork,
    data: Dict[str, torch.Tensor],
    *,
    forgetting: float = 1.0,
    device: str | None = None,
    atomic: bool = True,
    **kwargs: Any,
) -> UpdateHistory:
    """Fold ``data`` into an already-fitted ``model`` without rehearsal.

    Mechanisms whose ``supports_update`` is False (e.g. the
    ``BinningCategoricalTable`` that ``auto_mechanisms`` assigns to a discrete
    node with a continuous parent) keep their fitted CPD unchanged; they are
    listed in ``UpdateHistory.skipped`` and in one INFO log line per call.

    Parameters
    ----------
    model:
        A ``NeuralBayesianNetwork`` whose mechanisms have already been fitted.
    data:
        Dict mapping node name → new-data tensor of shape ``[N, D]`` or ``[N]``.
    forgetting:
        Exponential-forgetting factor in ``(0, 1]`` applied to each
        mechanism's persisted prior state (``1.0`` = pure recursive Bayes).
    atomic:
        If True (default) the update is all-or-nothing: every updatable
        mechanism is snapshotted before it is touched and restored in place
        if any later node fails, so a failure never leaves the network half
        old, half new.  The restore re-installs the *original* parameter and
        buffer objects with their saved values, so optimisers and references
        to ``model.parameters()`` stay valid.  Costs one transient copy of each
        updated mechanism — pass False for very large non-parametric stores
        (KDE / kNN keep their whole sample) when memory is tight.

    Returns
    -------
    UpdateHistory
    """
    if not (0.0 < float(forgetting) <= 1.0):
        raise ValueError(f"forgetting factor must be in (0, 1], got {forgetting!r}")

    nodes = model.dag.topological_order()
    _preflight(model, data, nodes)

    if device is None:
        device = str(model.device)
    dev = torch.device(device)
    model.to(dev)

    data_dev: Dict[str, torch.Tensor] = to_device(dict(data), dev)
    _check_finite(data_dev, nodes)

    history = UpdateHistory(device=device)
    t0 = time.perf_counter()
    backups: Dict[str, Any] = {}
    try:
        _update_nodes(model, data_dev, nodes, forgetting, kwargs, history,
                      backups if atomic else None)
    except BaseException:
        # Restore *in place* (same module objects), so references the caller
        # holds to a mechanism see the pre-update state too.
        for saved in backups.values():
            _restore(saved)
        raise

    if history.skipped:
        logger.info(
            "update(): %d node(s) not updated (mechanism has no incremental "
            "update; refit to fold new data in): %s",
            len(history.skipped),
            ", ".join(f"{n} ({type(model.mechanisms[n]).__name__})" for n in history.skipped),
        )
    history.wall_clock_s = time.perf_counter() - t0
    return history


_CONTAINERS = ("_parameters", "_buffers", "_modules")


def _snapshot(module: torch.nn.Module):
    """Record ``module`` so ``_restore`` can undo an ``update_local`` in place.

    ``update_local`` may rebind attributes (LG's ``_set_params`` installs new
    ``nn.Parameter`` objects, EWC reassigns ``_ewc_mu``) and mutate tensors in
    place (Adam steps).  So keep: the original attribute *bindings* (shallow
    copies of ``__dict__`` and its parameter/buffer/module dicts), a clone of
    every tensor's data, and a deepcopy of plain mutable attributes (lists of
    parent cards, ...).  Restoring re-binds the original objects and writes
    their saved data back — the identities callers hold stay valid.
    """
    attrs = {}
    tensors = []
    for k, v in module.__dict__.items():
        if k in _CONTAINERS:
            attrs[k] = dict(v)
            if k != "_modules":
                tensors += [(t, t.detach().clone()) for t in v.values() if t is not None]
        elif isinstance(v, torch.Tensor):
            attrs[k] = v
            tensors.append((v, v.detach().clone()))
        elif isinstance(v, (list, dict, set)) and not k.startswith("_forward") \
                and not k.startswith("_backward") and not k.endswith("hooks"):
            attrs[k] = copy.deepcopy(v)
        else:
            attrs[k] = v
    children = [_snapshot(m) for m in module._modules.values() if m is not None]
    return module, attrs, tensors, children


def _restore(saved) -> None:
    module, attrs, tensors, children = saved
    for k in [k for k in module.__dict__ if k not in attrs]:
        del module.__dict__[k]
    for k, v in attrs.items():
        if k in _CONTAINERS:
            getattr(module, k).clear()
            getattr(module, k).update(v)
        else:
            module.__dict__[k] = v
    with torch.no_grad():
        for t, data in tensors:
            t.data = data       # .data (not copy_): tolerates a shape change
    for child in children:
        _restore(child)


def _preflight(model, data, nodes) -> None:
    """Check everything that can be checked before any mechanism is touched."""
    missing = [n for n in nodes if n not in data]
    if missing:
        raise ValueError(
            f"update() needs a column for every node; missing {sorted(missing)}."
        )

    unfitted = [n for n in nodes if n not in model.mechanisms]
    if unfitted:
        raise RuntimeError(f"No mechanism registered for node(s) {sorted(unfitted)}.")
    # EWC-updated mechanisms register an ``_ewc_mu`` buffer that stays None
    # until a fit with consolidate=True; ewc_update would raise mid-network.
    no_ewc = [
        n for n in nodes
        if getattr(model.mechanisms[n], "supports_update", False)
        and "_ewc_mu" in model.mechanisms[n]._buffers
        and model.mechanisms[n]._buffers["_ewc_mu"] is None
    ]
    if no_ewc:
        raise RuntimeError(
            f"EWC state missing for node(s) {sorted(no_ewc)}: they were fitted "
            f"with consolidate=False (or never fitted), so there is no "
            f"theta*/Fisher snapshot to anchor the update.  Refit with "
            f"consolidate=True before calling update()."
        )


def _update_nodes(model, data_dev, nodes, forgetting, kwargs, history, backups) -> None:
    for node in nodes:
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

        if backups is not None:
            backups[node] = _snapshot(mech)
        metrics = mech.update_local(x, pa_tensor, forgetting=forgetting, **mech_kwargs)
        history.node_methods[node] = (metrics or {}).get("method", "unknown")

        with torch.no_grad():
            history.node_log_likelihoods[node] = float(
                mech.log_prob(x, pa_tensor).mean()
            )


def _check_finite(data_dev, nodes) -> None:
    """Refuse NaN / inf before any mechanism is touched.

    A single NaN row would be folded into the persisted sufficient
    statistics (LG normal equations, CPT counts) and poison every later
    update for good — there is no way to subtract it back out.  One device
    sync for all columns (they share a device by now), not one per column.
    """
    cols = [n for n in nodes if torch.is_tensor(data_dev[n]) and data_dev[n].is_floating_point()]
    if not cols:
        return
    ok = torch.stack([torch.isfinite(data_dev[n]).all() for n in cols])
    if bool(ok.all()):
        return
    bad = [n for n, good in zip(cols, ok.tolist()) if not good]
    raise ValueError(
        f"update() data contains NaN or infinite values in column(s) "
        f"{sorted(bad)}; drop or impute those rows first."
    )
