"""Choose each node's mechanism from a portfolio, within a training budget.

Why
---
Fitting every candidate (say a table, an MDN and a flow) to completion costs
the sum of their budgets.  Successive halving spends little on the losers:
train every candidate briefly, score it on held-out rows, keep the best
``1 / eta``, train the survivors ``eta`` times longer, and repeat [the
SuccessiveHalving subroutine of Hyperband: Li, Jamieson, DeSalvo,
Rostamizadeh & Talwalkar, JMLR 18, 2018, arXiv:1603.06560].  With ``n``
candidates and a final budget ``R`` epochs, the cost is about
``n r_0 + (n / eta) r_0 eta + ... ~ r_0 n log_eta(n) + R`` epochs instead of
``n R``.

Selection is per node.  With complete data and disjoint per-node parameters
the likelihood of a network is a product of one local likelihood per family,
for any parameterisation of the CPDs [Koller & Friedman 2009, §17.2.2,
Prop. 17.1, pp. 725–726]; on a validation set too, so the held-out score is a
sum of per-node terms and the best mechanism of each node can be chosen
separately (the per-node argmax is our step; it needs the same split for
every node, which this function uses).  Holdout over training likelihood:
the training likelihood never prefers the simpler model (K&F §18.3.1.3,
pp. 793–794).  Nodes are independent, so ``workers`` threads can handle
several at once.

Caveats
-------
* Successive halving can drop a candidate that learns slowly but ends best:
  the early scores rank candidates, not final quality.  ``min_epochs``
  controls how early that is.
* The chosen mechanism's validation score is optimistic (it won on those
  rows; K&F Box 16.A, pp. 705–708): report performance on a separate test
  set.  Calibration or PAC tools that need held-out data must use rows that
  were *not* used here (see ``nbn.structure.continuous`` and
  ``nbn.scm.calibration``); ``PortfolioReport.train_index`` / ``val_index``
  are the ids to pass as ``fit_ids`` to the calibration builders.
* Scores compare densities (continuous) or probabilities (discrete) of one
  node, so all candidates of a node must model the same variable on the same
  scale: do not mix a discretised (binned) table with a density without the
  bin widths, and a candidate that transforms the variable must report the
  log-density of the original variable (with its log-Jacobian).
"""
from __future__ import annotations

import inspect
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence

import torch

from nbn.utils.batching import pack_parents
from nbn.utils.device import to_device

__all__ = ["select_mechanisms", "PortfolioReport"]


@dataclass
class PortfolioReport:
    """Per node: the chosen candidate and every rung ``(candidate, epochs, val_ll)``."""

    chosen: Dict[str, str] = field(default_factory=dict)
    rungs: Dict[str, List[tuple]] = field(default_factory=dict)
    epochs_spent: Dict[str, int] = field(default_factory=dict)
    epochs_full: Dict[str, int] = field(default_factory=dict)
    seconds: Dict[str, float] = field(default_factory=dict)
    # Row indices *into ``data``* used for training and for selection (None
    # when ``val_data`` was given: then every row of ``data`` and ``val_data``
    # is a fit row).  Both count as "fit" rows for calibration: map them to
    # global row ids (the same space as the calibration rows' ids) and pass
    # them as ``fit_ids``.
    train_index: torch.Tensor | None = None
    val_index: torch.Tensor | None = None

    def summary(self) -> str:
        lines = []
        for n, c in self.chosen.items():
            lines.append(f"{n}: {c} ({self.epochs_spent[n]} candidate-epochs vs "
                         f"{self.epochs_full[n]} to train all to the end; "
                         f"{self.seconds[n]:.2f}s)")
        return "\n".join(lines)


def _iterative(mech) -> bool:
    try:
        return "epochs" in inspect.signature(mech.fit_local).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


def _make(factory, node: str, n_classes: int | None):
    """A fresh candidate; ``n_classes`` goes to factories that take it (a
    mechanism such as ``NeuralCategoricalMechanism`` fixes its class count at
    construction, so a bare ``factory()`` would get the default)."""
    takes = False
    if n_classes is not None:
        try:
            takes = "n_classes" in inspect.signature(factory).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins
            takes = False
    mech = factory(n_classes=n_classes) if takes else factory()
    k = getattr(mech, "n_classes", None)
    if n_classes is not None and isinstance(k, int) and k > 0 and k != n_classes:
        raise ValueError(f"select_mechanisms: candidate {type(mech).__name__} for '{node}' has "
                         f"{k} classes, the variable {n_classes}; pass a factory that sets them.")
    return mech


def select_mechanisms(
    model,
    data: Mapping[str, torch.Tensor],
    candidates: Mapping[str, Any],
    *,
    nodes: Sequence[str] | None = None,
    val_data: Mapping[str, torch.Tensor] | None = None,
    val_fraction: float = 0.2,
    min_epochs: int = 10,
    max_epochs: int = 200,
    eta: int = 3,
    refit: bool = False,
    workers: int = 1,
    seed: int = 0,
    **fit_kwargs: Any,
) -> PortfolioReport:
    """Pick and fit one mechanism per node by successive halving on held-out
    log-likelihood; the winner is installed in ``model``.

    ``candidates``: ``{name: factory}`` (callables returning a fresh
    mechanism; a factory with an ``n_classes`` parameter, e.g. a mechanism
    class, gets the node's cardinality) used for every node in ``nodes``, or
    ``{node: {name: factory}}`` per node.  ``val_data``: held-out rows
    (default: a random ``val_fraction`` of ``data``, fixed by ``seed``).
    Rungs follow Hyperband's schedule: with ``n`` trainable candidates there
    are ``s = ceil(log_eta n)`` halvings, the first rung trains every
    candidate ``max(min_epochs, max_epochs / eta^s)`` epochs and the last one
    trains the survivor to ``max_epochs`` (``warm_start=True`` continues
    training); closed-form mechanisms (no ``epochs`` argument) are fitted once
    and ranked with the others.  ``refit``: refit the winner on all rows
    (``data`` plus ``val_data``) for ``max_epochs`` (default off: the winner
    keeps the parameters trained on ``data`` and the held-out score stays
    available).
    ``fit_kwargs`` are passed to every ``fit_local`` (e.g. ``lr``,
    ``batch_size``).
    """
    if eta < 2:
        raise ValueError("eta must be >= 2.")
    dev = model.device
    data = to_device(dict(data), dev)
    if val_data is None:
        n = next(iter(data.values())).shape[0]
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).to(dev)
        n_val = max(1, int(round(val_fraction * n)))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        train = {k: v[tr_idx] for k, v in data.items()}
        val = {k: v[val_idx] for k, v in data.items()}
    else:
        train, val = data, to_device(dict(val_data), dev)
    full = {k: torch.cat([train[k], val[k]]) for k in train}
    nodes = list(nodes) if nodes is not None else list(model.dag.topological_order())
    report = PortfolioReport()
    if val_data is None:
        report.train_index, report.val_index = tr_idx.cpu(), val_idx.cpu()

    def node_kwargs(node):
        kw = dict(fit_kwargs)
        var = model.variables.get(node)
        if var is not None and var.cardinality is not None:
            kw.setdefault("n_classes", int(var.cardinality))
        pvars = [model.variables.get(p) for p in model.dag.parents(node)]
        if pvars and all(p is not None and p.cardinality is not None for p in pvars):
            kw.setdefault("parent_cards", [int(p.cardinality) for p in pvars])
        return kw

    def run(node):
        t0 = time.perf_counter()
        cands = candidates.get(node) if isinstance(candidates.get(node), Mapping) else candidates
        if not cands:
            raise ValueError(f"select_mechanisms: no candidates for '{node}'.")
        parents = model.dag.parents(node)
        x_tr, pa_tr = train[node], pack_parents(train, parents)
        x_va, pa_va = val[node], pack_parents(val, parents)
        kw = node_kwargs(node)
        k_node = kw.get("n_classes")
        alive = {name: _make(f, node, k_node) for name, f in cands.items()}
        n_iter = sum(1 for m in alive.values() if _iterative(m))
        # Hyperband's schedule: s rungs so that the last one is max_epochs
        s_rungs = math.ceil(math.log(n_iter) / math.log(eta) - 1e-9) if n_iter > 1 else 0
        budget = max(int(min_epochs), int(max_epochs) // eta ** s_rungs)
        trained = {name: 0 for name in alive}
        rungs, spent = [], 0
        while True:
            scores = {}
            for name, mech in alive.items():
                if _iterative(mech):
                    more = budget - trained[name]
                    if more > 0:
                        mech.fit_local(x_tr, pa_tr, epochs=more,
                                       warm_start=trained[name] > 0, **kw)
                        spent += more
                        trained[name] = budget
                elif trained[name] == 0:
                    mech.fit_local(x_tr, pa_tr, **kw)
                    trained[name] = -1                     # closed form: fitted once
                with torch.no_grad():
                    lp = mech.log_prob(x_va, pa_va)
                    sc = float(lp.mean()) if torch.isfinite(lp).all() else float("-inf")
                scores[name] = sc
                rungs.append((name, max(trained[name], 0), sc))
            if budget >= max_epochs or not any(_iterative(m) for m in alive.values()):
                break
            keep = max(1, math.ceil(len(alive) / eta))
            best = sorted(alive, key=lambda n: scores[n], reverse=True)[:keep]
            alive = {n: alive[n] for n in best}
            budget = min(int(max_epochs), budget * eta)
        winner = max(alive, key=lambda n: scores[n])
        mech = alive[winner]
        if refit:
            mech = _make(cands[winner], node, k_node)
            if _iterative(mech):
                mech.fit_local(full[node], pack_parents(full, parents),
                               epochs=max(trained[winner], 1), **kw)
            else:
                mech.fit_local(full[node], pack_parents(full, parents), **kw)
        return node, winner, mech, rungs, spent, n_iter * int(max_epochs), \
            time.perf_counter() - t0

    if workers > 1 and len(nodes) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(run, nodes))
    else:
        results = [run(n) for n in nodes]
    for node, winner, mech, rungs, spent, full_cost, secs in results:
        model.set_mechanism(node, mech)
        report.chosen[node] = winner
        report.rungs[node] = rungs
        report.epochs_spent[node] = spent
        report.epochs_full[node] = full_cost
        report.seconds[node] = secs
    return report
