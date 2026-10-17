"""Stacked local fitting for any per-node network, via ``torch.func``.

Generalises :mod:`nbn.learning.stacked` (hand-written for MDNs) to the other
gradient-trained mechanisms: ``NeuralCategoricalMechanism`` (cross-entropy),
``FlexCodeMechanism`` (least squares on basis coefficients) and
``NormalizingFlowMechanism`` (zuko neural spline flow, with per-node early
stopping).  The argument is the same as for MDNs:

* the local objectives are independent (one parameter set per node), so the
  gradient of the summed loss with respect to node ``j``'s parameters is node
  ``j``'s own gradient [Koller & Friedman 2009, Prop. 17.1, p. 725];
* Adam is element-wise, so one Adam over the stacked tensors makes the same
  update per element as one Adam per node;
* gradient clipping, where the mechanism clips, is done per node;
* initialisation is each mechanism's own (its ``fit_local`` is called with
  ``epochs=0``, which builds the network and sets every standardisation
  buffer exactly as a normal fit does), and the trained parameters are written
  back into that same network.

Only the minibatch order is shared across the nodes of a group (``local``
draws one permutation per node), so each node's training run has the same law
as before.  Parameters are stacked with :func:`torch.func.stack_module_state`
and evaluated with ``vmap(functional_call)``; nodes whose parent count differs
are padded to the largest one with zero inputs and zero first-layer weights,
which receive zero gradient and never move.

Not stacked (sequential ``fit_local`` as before): roots, ``warm_start``,
neural-categorical nodes with parent embeddings, and any mechanism type not
listed here.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state, vmap

from nbn.utils.batching import ensure_2d


@dataclass
class _Task:
    node: str
    mech: Any
    module: nn.Module          # the mechanism's own network (true input size)
    inp: torch.Tensor          # [N, d_in] network input
    tgt: torch.Tensor          # [N, ...] training target
    d_in: int
    extra: Dict[str, Any] = field(default_factory=dict)


class _Family:
    """How to stack one mechanism type.  Subclasses fill the hooks."""

    clip: float | None = None
    early_stopping = False

    def eligible(self, mech, parents) -> bool:
        raise NotImplementedError

    def group_key(self, mech, kw) -> tuple:
        raise NotImplementedError

    def prepare(self, node, mech, x, parents, kw, weights) -> _Task:
        raise NotImplementedError

    def make_padded(self, task: _Task, d_in: int) -> nn.Module:
        raise NotImplementedError

    def loss_rows(self, out, tgt) -> torch.Tensor:  # per-row loss [B]
        raise NotImplementedError

    def schedule(self, mech, kw) -> tuple[int, float, int]:
        raise NotImplementedError

    def finish(self, task: _Task, kw, weights) -> None:
        task.mech.eval()


# --------------------------------------------------------------------- #
# Families
# --------------------------------------------------------------------- #

class _NeuralCategorical(_Family):
    clip = None  # NeuralCategoricalMechanism.fit_local does not clip

    def eligible(self, mech, parents):
        from nbn.mechanisms.parametric.neural_categorical import NeuralCategoricalMechanism
        return (type(mech) is NeuralCategoricalMechanism and mech.embedding_dim is None
                and parents is not None and parents.shape[-1] > 0)

    def schedule(self, mech, kw):
        return (int(kw.get("epochs", 100)), float(kw.get("lr", 1e-3)),
                int(kw.get("batch_size", 512)))

    def group_key(self, mech, kw):
        return ("ncat", mech.n_classes, tuple(mech.hidden), mech.activation, self.schedule(mech, kw))

    def prepare(self, node, mech, x, parents, kw, weights):
        pa = ensure_2d(parents)
        mech.fit_local(x, pa, **{**kw, "epochs": 0, "consolidate": False,
                                 "warm_start": False, "weights": weights})
        return _Task(node, mech, mech.net, pa.float(), x.long().reshape(-1), pa.shape[1])

    def make_padded(self, task, d_in):
        from nbn.mechanisms.parametric.mdn import _build_mlp
        m = task.mech
        return _build_mlp(d_in, m.hidden, m.n_classes, m.activation)

    def loss_rows(self, out, tgt):
        return nn.functional.cross_entropy(out, tgt, reduction="none")

    def finish(self, task, kw, weights):
        task.mech.eval()
        if kw.get("consolidate", True):
            from nbn.update import online_laplace
            online_laplace.consolidate(task.mech, task.tgt, task.extra["parents"], weights=weights)


class _FlexCode(_Family):
    clip = 5.0

    def eligible(self, mech, parents):
        from nbn.mechanisms.non_parametric.flexcode import FlexCodeMechanism
        return (type(mech) is FlexCodeMechanism and parents is not None
                and parents.shape[-1] > 0)

    def schedule(self, mech, kw):
        return (int(kw.get("epochs", mech.epochs)), float(kw.get("lr", mech.lr)),
                int(kw.get("batch_size", mech.batch_size)))

    def group_key(self, mech, kw):
        return ("flexcode", mech.n_basis, tuple(mech.hidden), mech.activation,
                self.schedule(mech, kw), mech.sharpen == "auto", mech.grid_size)

    def prepare(self, node, mech, x, parents, kw, weights):
        pa = ensure_2d(parents).float()
        y = ensure_2d(x).float()
        mech._fit_core(y, pa, epochs=0, weights=weights,
                       **{k: v for k, v in kw.items() if k in ("lr", "batch_size")})
        inp = (pa.to(mech._pa_mean.device) - mech._pa_mean) / mech._pa_std
        tgt = mech._basis(mech._scale_to_z(y).squeeze(-1))
        return _Task(node, mech, mech.net, inp, tgt, pa.shape[1])

    def make_padded(self, task, d_in):
        from nbn.mechanisms.parametric.mdn import _build_mlp
        m = task.mech
        return _build_mlp(d_in, m.hidden, m.n_basis, m.activation)

    def loss_rows(self, out, tgt):
        return (out - tgt).pow(2).mean(dim=-1)


class _Flow(_Family):
    clip = 5.0
    early_stopping = True

    def eligible(self, mech, parents):
        from nbn.mechanisms.parametric.normalizing_flow import NormalizingFlowMechanism
        return (type(mech) is NormalizingFlowMechanism and parents is not None
                and parents.shape[-1] > 0)

    def schedule(self, mech, kw):
        return (int(kw.get("epochs", 300)), float(kw.get("lr", 5e-4)),
                int(kw.get("batch_size", 512)))

    def group_key(self, mech, kw):
        es = (bool(kw.get("early_stopping", True)), int(kw.get("patience", 5)),
              float(kw.get("min_delta", 1e-3)), float(kw.get("val_fraction", 0.1)))
        return ("flow", mech.d_x, mech.num_transforms, tuple(mech.hidden), mech.bins,
                self.schedule(mech, kw), es)

    def prepare(self, node, mech, x, parents, kw, weights):
        pa = ensure_2d(parents)
        xx = ensure_2d(x)
        mech.fit_local(xx, pa, epochs=0, weights=weights, warm_start=False)
        inp = mech._std_ctx(pa.to(xx.device))
        tgt = mech._std_x(xx.to(torch.get_default_dtype()))
        task = _Task(node, mech, mech._flow, inp, tgt, pa.shape[1])
        task.extra["log_det"] = float(mech._log_det()) if mech._x_std is not None else 0.0
        return task

    def make_padded(self, task, d_in):
        import zuko
        m = task.mech
        return zuko.flows.NSF(features=m.d_x, context=d_in, transforms=m.num_transforms,
                              hidden_features=list(m.hidden), bins=m.bins)

    def loss_rows(self, out, tgt):
        return -out.log_prob(tgt)


FAMILIES: List[_Family] = [_NeuralCategorical(), _FlexCode(), _Flow()]


def family_of(mech, parents) -> _Family | None:
    for f in FAMILIES:
        if f.eligible(mech, parents):
            return f
    return None


# --------------------------------------------------------------------- #
# The trainer
# --------------------------------------------------------------------- #

def _pad_into(padded: nn.Module, own: nn.Module) -> None:
    """Copy ``own``'s tensors into ``padded``; extra input columns stay zero."""
    own_sd = own.state_dict()
    with torch.no_grad():
        for name, t in padded.state_dict(keep_vars=True).items():
            src = own_sd[name]
            if t.shape == src.shape:
                t.copy_(src)
            else:
                t.zero_()
                t[tuple(slice(0, s) for s in src.shape)].copy_(src)


def _write_back(own: nn.Module, stacked: Dict[str, torch.Tensor], j: int) -> None:
    with torch.no_grad():
        for name, t in own.state_dict(keep_vars=True).items():
            if name in stacked:
                src = stacked[name][j]
                t.copy_(src[tuple(slice(0, s) for s in t.shape)])


def fit_group(family: _Family, tasks: List[_Task], kw: Dict[str, Any],
              weights: torch.Tensor | None) -> Dict[str, Any]:
    """Train one group of same-architecture tasks together; write back; report."""
    device = tasks[0].inp.device
    epochs, lr, batch_size = family.schedule(tasks[0].mech, kw)
    J = len(tasks)
    d_max = max(t.d_in for t in tasks)
    n = tasks[0].inp.shape[0]

    padded = []
    for t in tasks:
        m = family.make_padded(t, d_max).to(device)
        _pad_into(m, t.module)
        padded.append(m)
    params, buffers = stack_module_state(padded)
    base = copy.deepcopy(padded[0]).to("meta")
    plist = list(params.values())
    opt = torch.optim.Adam(plist, lr=lr)

    X = torch.zeros(n, J, d_max, device=device, dtype=tasks[0].inp.dtype)
    for j, t in enumerate(tasks):
        X[:, j, :t.d_in] = t.inp
    T = torch.stack([t.tgt for t in tasks], dim=1)            # [N, J, ...]

    def rows_loss(p, b, inp, tgt):
        return family.loss_rows(functional_call(base, (p, b), (inp,)), tgt)

    batched = vmap(rows_loss, in_dims=(0, 0, 1, 1))           # -> [J, B]

    def per_node(idx, w):
        lr_ = batched(params, buffers, X[idx], T[idx])         # [J, B]
        if w is None:
            return lr_.mean(1)
        wb = w[idx].to(lr_.dtype)
        return (lr_ * wb).sum(1) / wb.sum().clamp_min(1e-30)

    # ---- per-node early stopping split (flows) ------------------------- #
    es = family.early_stopping and bool(kw.get("early_stopping", True)) and epochs > 0
    val_idx = None
    if es:
        n_val = int(n * float(kw.get("val_fraction", 0.1)))
        min_rows = getattr(tasks[0].mech, "_MIN_VAL_ROWS", 32)
        if n_val >= min_rows and n - n_val >= 1:
            split = torch.randperm(n, device=device)
            val_idx, tr_idx = split[:n_val], split[n_val:]
    tr_idx = tr_idx if val_idx is not None else torch.arange(n, device=device)
    n_tr = tr_idx.shape[0]
    patience = int(kw.get("patience", 5))
    min_delta = float(kw.get("min_delta", 1e-3))
    best_val = torch.full((J,), float("inf"))
    bad = torch.zeros(J, dtype=torch.long)
    stopped = torch.zeros(J, dtype=torch.bool)
    best_state = None
    epochs_run = torch.zeros(J, dtype=torch.long)
    log_det = torch.tensor([t.extra.get("log_det", 0.0) for t in tasks])

    for _ in range(epochs):
        perm = tr_idx[torch.randperm(n_tr, device=device)]
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i + batch_size]
            loss = per_node(idx, weights)
            opt.zero_grad(set_to_none=False)
            loss.sum().backward()
            if family.clip is not None:
                with torch.no_grad():
                    sq = torch.zeros(J, device=device)
                    for t in plist:
                        sq += t.grad.reshape(J, -1).pow(2).sum(-1)
                    coef = (family.clip / (sq.sqrt() + 1e-6)).clamp(max=1.0)
                    for t in plist:
                        t.grad.mul_(coef.view(J, *([1] * (t.dim() - 1))))
            opt.step()
        epochs_run += (~stopped).long()
        if val_idx is None:
            continue
        with torch.no_grad():
            v = (per_node(val_idx, weights).cpu() + log_det)
            v = torch.where(torch.isfinite(v), v, torch.full_like(v, float("inf")))
        improved = (v < best_val - min_delta) & ~stopped
        if best_state is None:
            best_state = {k: t.detach().clone() for k, t in params.items()}
        if bool(improved.any()):
            sel = improved.to(device)
            for k, t in params.items():
                best_state[k][sel] = t.detach()[sel]
            best_val = torch.where(improved, v, best_val)
        bad = torch.where(improved, torch.zeros_like(bad), bad + (~stopped).long())
        stopped = stopped | (bad >= patience)
        if bool(stopped.all()):
            break
    final = {k: t.detach() for k, t in params.items()}
    if best_state is not None:
        # Restore each node's best epoch, as the sequential loop does; a node
        # whose validation loss was never finite keeps its final parameters
        # (the sequential loop never stored a best state for it).
        keep_best = torch.isfinite(best_val).to(device)
        for k in final:
            final[k] = torch.where(keep_best.view(J, *([1] * (final[k].dim() - 1))),
                                   best_state[k], final[k])
    for j, t in enumerate(tasks):
        _write_back(t.module, final, j)
        family.finish(t, kw, weights)
    return {"nodes": [t.node for t in tasks], "epochs_run": epochs_run.tolist(),
            "early_stopped": stopped.tolist()}


def fit_stacked_modules(model, data_dev, order, *, kw: Dict[str, Any],
                        weights: torch.Tensor | None, pack_parents: Callable,
                        node_chunk: int | None = None) -> set:
    """Fit every eligible node of ``model`` by family; return the fitted names.

    ``kw`` holds the caller's overrides (epochs / lr / batch_size / consolidate,
    and the flow early-stopping options); nodes not returned must be fitted by
    the caller's sequential path.
    """
    groups: Dict[tuple, List[tuple]] = {}
    for node in order:
        mech = model.mechanisms[node]
        parents = pack_parents(data_dev, model.dag.parents(node))
        fam = family_of(mech, parents)
        if fam is None:
            continue
        groups.setdefault(fam.group_key(mech, kw), []).append((fam, node, mech, parents))
    done: set = set()
    for key, members in groups.items():
        fam = members[0][0]
        if isinstance(fam, _FlexCode) and weights is not None and bool((weights == 0).any()):
            continue  # FlexCode drops zero-weight rows first: keep the sequential path
        if isinstance(fam, _FlexCode) and members[0][2].sharpen == "auto":
            _flexcode_auto(fam, members, data_dev, kw, weights, node_chunk)
        else:
            _fit_members(fam, members, data_dev, kw, weights, node_chunk, rows=None)
        done.update(node for _, node, _, _ in members)
    return done


def _fit_members(fam, members, data_dev, kw, weights, node_chunk, rows):
    tasks = []
    for _, node, mech, parents in members:
        x = data_dev[node] if rows is None else ensure_2d(data_dev[node])[rows]
        pa = parents if rows is None else parents[rows]
        w = weights if (rows is None or weights is None) else weights[rows]
        t = fam.prepare(node, mech, x, pa, kw, w)
        t.extra["parents"] = pa
        tasks.append(t)
    w = weights if (rows is None or weights is None) else weights[rows]
    width = max(1, node_chunk or _auto_chunk(tasks))
    for s in range(0, len(tasks), width):
        fit_group(fam, tasks[s:s + width], kw, w)
    return tasks


def _flexcode_auto(fam, members, data_dev, kw, weights, node_chunk,
                   candidates=(1.0, 1.25, 1.5, 1.75, 2.0, 2.5), val_frac=0.25):
    """``sharpen='auto'`` for a group, as ``FlexCodeMechanism._select_sharpen``:
    the same seed-0 split, a sub-fit on 75% of the rows, the exponent picked
    per node by (weighted) held-out NLL, then the full fit."""
    from nbn.learning.weighting import weighted_mean
    n = ensure_2d(data_dev[members[0][1]]).shape[0]
    gen = torch.Generator(device="cpu").manual_seed(0)
    perm = torch.randperm(n, generator=gen)
    n_val = max(8, int(round(val_frac * n)))
    best = {node: 1.0 for _, node, _, _ in members}
    if n_val < n:
        dev = data_dev[members[0][1]].device
        val_idx, fit_idx = perm[:n_val].to(dev), perm[n_val:].to(dev)
        for _, _, mech, _ in members:
            mech.sharpen = 1.0
        _fit_members(fam, members, data_dev, kw, weights, node_chunk, rows=fit_idx)
        w_val = None if weights is None else weights[val_idx]
        for _, node, mech, parents in members:
            y = ensure_2d(data_dev[node]).float()
            best_s, best_nll = 1.0, float("inf")
            for s in candidates:
                mech.sharpen = float(s)
                with torch.no_grad():
                    nll = float(weighted_mean(-mech.log_prob(y[val_idx], parents[val_idx]), w_val))
                if nll == nll and nll < best_nll:
                    best_nll, best_s = nll, float(s)
            best[node] = best_s
            mech.sharpen = "auto"
    _fit_members(fam, members, data_dev, kw, weights, node_chunk, rows=None)
    for _, node, mech, _ in members:
        mech.sharpen = best[node]


def _auto_chunk(tasks: List[_Task], budget: float = 3e7) -> int:
    """Nodes per pass so the stacked activations stay moderate."""
    n_par = sum(p.numel() for p in tasks[0].module.parameters())
    return max(1, int(budget // max(1, 512 * max(64, int(math.sqrt(n_par)) * 8))))
