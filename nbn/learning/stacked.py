"""Stacked local fitting: train many per-node MDNs as one batched network.

Why this is the same estimator
------------------------------
With complete data the log-likelihood decomposes into one term per family,
and the local maximisers jointly maximise it [Koller & Friedman 2009,
Prop. 17.1, p. 725].  ``fit(method="local")`` exploits this by training each
node's mechanism on its own, *sequentially*: a Python loop over nodes, each
with its own optimiser loop.  For small networks per node (an MDN with a few
inputs) that loop is dominated by interpreter and kernel-launch overhead, and
the wall-clock grows linearly in the number of nodes with a large constant.

``fit(method="stacked")`` trains every eligible MDN *at the same time*: the
parameters of all nodes in a group are stored as tensors with a leading node
axis ``J`` (first-layer inputs zero-padded to the largest parent set), and one
step evaluates all ``J`` local losses with batched matrix products.  Per node
it keeps, exactly, what ``MDNMechanism.fit_local`` does:

* the model: same MLP architecture, same parent / child standardisation
  (weighted moments, the same ``unbiased`` conventions and clamps), same
  ``_pa_clamp`` / ``_log_scale_clamp`` / ``min_scale``;
* the objective: the (weighted) mean negative log-likelihood of that node;
  the stacked loss is the *sum* of the per-node losses, whose gradient with
  respect to node ``j``'s parameters is node ``j``'s own gradient;
* the optimiser: Adam acts element-wise, so one Adam over the stacked tensors
  makes the same update per element as one Adam per node (same ``lr``,
  betas, eps, step count); padded first-layer weights get zero gradient and
  never move;
* gradient clipping: per node (``clip_grad_norm_(node params, 5.0)``), not
  over the stack.

What differs: the minibatch order.  ``local`` draws an independent
permutation per node; ``stacked`` shares one permutation per epoch across the
nodes of a group.  Each node's training run therefore has the same law, only
the runs of different nodes are no longer independent -- which no
per-node quantity depends on.  Initial weights are drawn with the standard
``nn.Linear`` initialiser per node.  The fitted result is written back into
ordinary ``MDNMechanism`` objects, so everything downstream (queries,
``log_prob``, ``update``, save/load) is unchanged.

Eligible nodes: mechanisms of exact type ``MDNMechanism`` with at least one
parent.  Root MDNs are closed form and are fitted by ``fit_local``; any other
mechanism, and ``warm_start=True``, fall back to the sequential path.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from nbn.learning.weighting import weighted_moments
from nbn.mechanisms.parametric.mdn import MDNMechanism, _build_mlp
from nbn.utils.batching import ensure_2d

_ACTS = {"relu": torch.relu, "tanh": torch.tanh,
         "gelu": nn.functional.gelu, "elu": nn.functional.elu}


def stack_key(mech) -> tuple | None:
    """Group key for stacking, or ``None`` if the mechanism cannot be stacked."""
    if type(mech) is not MDNMechanism:
        return None
    return (mech.num_components, tuple(mech.hidden), mech.activation,
            float(mech.min_scale), float(mech._pa_clamp), float(mech._log_scale_clamp))


def _node_chunk(batch: int, widths: Sequence[int], budget_elems: float = 2.5e7) -> int:
    """Nodes per chunk so one layer's activations stay under ``budget_elems``."""
    return max(1, int(budget_elems // (batch * max(widths))))


def fit_stacked(
    mechs: Dict[str, MDNMechanism],
    xs: Dict[str, torch.Tensor],
    parents: Dict[str, torch.Tensor],
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 512,
    weights: torch.Tensor | None = None,
    node_chunk: int | None = None,
) -> None:
    """Fit a group of non-root MDNs with identical architecture, in place.

    ``xs[v]`` is ``[N, D_x]``, ``parents[v]`` is ``[N, D_pa_v]`` (raw units).
    All mechanisms must share ``stack_key``.
    """
    nodes = list(mechs)
    if not nodes:
        return
    m0 = mechs[nodes[0]]
    k = m0.num_components
    hidden = tuple(m0.hidden)
    act = _ACTS[m0.activation]
    d_x = ensure_2d(xs[nodes[0]]).shape[1]
    out_dim = k + 2 * k * d_x
    n_rows = ensure_2d(xs[nodes[0]]).shape[0]
    chunk = node_chunk or _node_chunk(batch_size, (*hidden, out_dim))
    for start in range(0, len(nodes), chunk):
        _fit_chunk(nodes[start:start + chunk], mechs, xs, parents, k=k, hidden=hidden,
                   act=act, d_x=d_x, out_dim=out_dim, n_rows=n_rows, epochs=epochs,
                   lr=lr, batch_size=batch_size, weights=weights)


def _fit_chunk(nodes, mechs, xs, parents, *, k, hidden, act, d_x, out_dim, n_rows,
               epochs, lr, batch_size, weights):
    device = ensure_2d(xs[nodes[0]]).device
    dtype = torch.get_default_dtype()
    J = len(nodes)
    d_pa = [ensure_2d(parents[v]).shape[1] for v in nodes]
    P = max(d_pa)
    m_ref = mechs[nodes[0]]

    # ---- data, standardisation (exactly as MDNMechanism.fit_local) ------ #
    X = torch.empty(n_rows, J, d_x, device=device, dtype=dtype)
    PA = torch.zeros(n_rows, J, P, device=device, dtype=dtype)
    mask = torch.zeros(J, P, device=device, dtype=dtype)
    pa_mean = torch.zeros(J, P, device=device, dtype=dtype)
    pa_std = torch.ones(J, P, device=device, dtype=dtype)
    x_mean = torch.zeros(J, d_x, device=device, dtype=dtype)
    x_std = torch.ones(J, d_x, device=device, dtype=dtype)
    for j, v in enumerate(nodes):
        x = ensure_2d(xs[v]).to(dtype)
        pa = ensure_2d(parents[v]).to(device=device, dtype=dtype)
        X[:, j] = x
        PA[:, j, :d_pa[j]] = pa
        mask[j, :d_pa[j]] = 1.0
        xm, xsd = weighted_moments(x, weights)
        xsd = xsd.nan_to_num(0.0)
        x_mean[j] = xm
        x_std[j] = torch.where(xsd > 0, xsd, torch.ones_like(xsd))
        pm, ps = weighted_moments(pa, weights, unbiased=True)
        pa_mean[j, :d_pa[j]] = pm
        pa_std[j, :d_pa[j]] = ps.nan_to_num(0.0).clamp_min(1e-3)

    # ---- parameters: per-node nn.Linear init, stacked ------------------ #
    widths = [P, *hidden, out_dim]
    Ws = [torch.zeros(J, widths[i], widths[i + 1], device=device, dtype=dtype)
          for i in range(len(widths) - 1)]
    Bs = [torch.zeros(J, widths[i + 1], device=device, dtype=dtype)
          for i in range(len(widths) - 1)]
    for j in range(J):
        net = _build_mlp(d_pa[j], hidden, out_dim, m_ref.activation)
        lins = [m for m in net if isinstance(m, nn.Linear)]
        for li, lin in enumerate(lins):
            w = lin.weight.detach().to(device=device, dtype=dtype).T  # [in, out]
            Ws[li][j, :w.shape[0]] = w
            Bs[li][j] = lin.bias.detach().to(device=device, dtype=dtype)
    params = [nn.Parameter(t) for pair in zip(Ws, Bs) for t in pair]
    Ws = params[0::2]
    Bs = params[1::2]
    opt = torch.optim.Adam(params, lr=lr)

    pa_clamp = float(m_ref._pa_clamp)
    ls_clamp = float(m_ref._log_scale_clamp)
    min_scale = float(m_ref.min_scale)
    log2pi = math.log(2.0 * math.pi)

    def nll(idx):
        xb = X[idx]                                     # [B, J, D_x]
        z = (PA[idx] - pa_mean) / pa_std               # [B, J, P]
        z = z.clamp(-pa_clamp, pa_clamp) * mask
        h = z
        for li in range(len(Ws)):
            h = torch.einsum("bji,jio->bjo", h, Ws[li]) + Bs[li]
            if li < len(Ws) - 1:
                h = act(h)
        logits = h[..., :k]
        rest = h[..., k:].reshape(*h.shape[:-1], k, 2 * d_x)
        loc = rest[..., :d_x]
        log_scale = rest[..., d_x:].clamp(max=ls_clamp)
        scale = torch.exp(log_scale)
        loc = x_mean[:, None, :] + x_std[:, None, :] * loc          # [B, J, K, D]
        scale = (x_std[:, None, :] * scale).clamp_min(min_scale)
        diff = (xb.unsqueeze(-2) - loc) / scale
        log_comp = (-0.5 * diff.pow(2) - torch.log(scale) - 0.5 * log2pi).sum(-1)
        lp = torch.logsumexp(torch.log_softmax(logits, -1) + log_comp, -1)  # [B, J]
        return -lp

    max_norm = 5.0
    for _ in range(epochs):
        perm = torch.randperm(n_rows, device=device)
        for i in range(0, n_rows, batch_size):
            idx = perm[i:i + batch_size]
            per = nll(idx)                               # [B, J]
            if weights is None:
                loss_j = per.mean(0)
            else:
                w = weights[idx].to(per.dtype)
                loss_j = (per * w[:, None]).sum(0) / w.sum().clamp_min(1e-30)
            opt.zero_grad(set_to_none=False)
            loss_j.sum().backward()
            # per-node gradient clipping == clip_grad_norm_(node params, 5.0)
            with torch.no_grad():
                sq = torch.zeros(J, device=device, dtype=dtype)
                for t in params:
                    sq += t.grad.reshape(J, -1).pow(2).sum(-1)
                coef = (max_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
                for t in params:
                    t.grad.mul_(coef.view(J, *([1] * (t.dim() - 1))))
            opt.step()

    # ---- write back into ordinary MDNMechanism objects ----------------- #
    with torch.no_grad():
        for j, v in enumerate(nodes):
            mech = mechs[v]
            mech._d_x = d_x
            mech.output_dim = d_x
            mech._d_pa = d_pa[j]
            mech._x_mean = x_mean[j].clone()
            mech._x_std = x_std[j].clone()
            mech._pa_mean = pa_mean[j, :d_pa[j]].unsqueeze(0).clone()
            mech._pa_std = pa_std[j, :d_pa[j]].unsqueeze(0).clone()
            net = _build_mlp(d_pa[j], hidden, out_dim, mech.activation).to(device)
            lins = [m for m in net if isinstance(m, nn.Linear)]
            for li, lin in enumerate(lins):
                lin.weight.copy_(Ws[li][j, :lin.in_features].T)
                lin.bias.copy_(Bs[li][j])
            mech.net = net
            mech._root_logits = None
            mech._root_loc = None
            mech._root_log_scale = None
            mech.eval()


def group_stackable(model, nodes: List[str]) -> Dict[tuple, List[str]]:
    """Group the stackable non-root nodes of ``model`` by architecture."""
    groups: Dict[tuple, List[str]] = {}
    for v in nodes:
        if not model.dag.parents(v):
            continue
        key = stack_key(model.mechanisms[v])
        if key is not None:
            groups.setdefault(key, []).append(v)
    return groups
