"""Problem generator for the NBN scaling study.

Two topologies, both in a fixed topological order X0 < X1 < ... :

* ``er``    -- parents drawn uniformly from all earlier nodes.  The moral graph
               of such a DAG has tree-width that grows roughly linearly in n,
               so exact inference becomes hopeless quickly.  This is the
               "graph complexity" stress case.
* ``local`` -- parents drawn uniformly from the ``window`` previous nodes.
               The moral graph has bandwidth <= window, hence tree-width <=
               window, whatever n is.  This isolates "graph size" from
               "graph complexity".

Each node's in-degree is ``min(j, cap, Poisson(deg))`` (``cap = window`` for
``local``), so the number of edges is close to ``deg * n``.

Families
--------
* ``discrete``  -- categorical CPTs with i.i.d. N(0,1) logits (as in
                   ``nbn.bench.synthetic``), cardinality ``k``.
* ``nongauss``  -- 3-component Gaussian mixture whose means are linear in
                   the parents, with skew-normal biases (``nbn.bench.synthetic``'s
                   ``continuous_nongauss`` mechanism).

A problem is written once to ``<out>/<name>.pt`` and read by every method, so
data generation is never charged to a method's budget.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from nbn.bench.synthetic import (
    _build_discrete_mechanism,
    _build_mdn_mechanism,
)
from nbn.core.network import NeuralBayesianNetwork


def make_dag(n: int, deg: float, topology: str, window: int, cap: int,
             rng: np.random.Generator) -> list[tuple[int, int]]:
    edges = []
    for j in range(1, n):
        if topology == "er":
            pool = j
            lim = cap
        elif topology == "local":
            pool = min(j, window)
            lim = min(cap, window)
        else:
            raise ValueError(topology)
        d = int(min(pool, lim, rng.poisson(deg)))
        if d == 0:
            continue
        lo = j - pool
        pa = rng.choice(pool, size=d, replace=False) + lo
        edges += [(int(i), j) for i in sorted(pa)]
    return edges


def _strengthen(mech, parents, pilot, strength=1.5, gen=None):
    """Strong-dependence, correctly laid-out MDN truth (family ``ngs``).

    ``MDNMechanism._params_from_parents`` reads the outputs after the ``K``
    logits as *interleaved* pairs ``(loc_c, log_scale_c)`` per component.
    ``nbn.bench.synthetic._build_mdn_mechanism`` writes them as blocks
    ``[locs | log_scales]``, so part of its mean weights land on log-scales
    and the scale becomes ``exp(linear in parents)`` (heavy tails; found in
    this study, to be filed).  Here the layout is the one the mechanism
    reads: mixture means linear in the *standardised* parents with weights of
    scale ``strength / sqrt(p)``, log-scales constant in the parents, mixing
    weights constant.  Parents are standardised with pilot-sample moments
    (``_pa_mean`` / ``_pa_std``), so chains stay bounded.
    """
    P = torch.stack([pilot[a] for a in parents], -1)
    mech._pa_mean = P.mean(0, keepdim=True)
    mech._pa_std = P.std(0, keepdim=True).clamp_min(1e-3)
    lin = mech.net[0]
    k = mech.num_components
    p = len(parents)
    old_b = lin.bias.detach().clone()
    W = torch.zeros_like(lin.weight)
    Bv = torch.zeros_like(lin.bias)
    Bv[:k] = old_b[:k]                                   # mixing logits
    for c in range(k):
        loc_row, ls_row = k + 2 * c, k + 2 * c + 1
        W[loc_row] = torch.randn(p, generator=gen) * strength / p ** 0.5
        Bv[loc_row] = 0.3 * torch.randn((), generator=gen) + 0.6 * (c - 1)
        Bv[ls_row] = torch.log(torch.empty(()).uniform_(0.3, 0.7, generator=gen))
    with torch.no_grad():
        lin.weight.copy_(W)
        lin.bias.copy_(Bv)


def problem_name(family, topology, n, deg, k, seed, window):
    w = f"_w{window}" if topology == "local" else ""
    return f"{family}_{topology}{w}_n{n}_deg{deg:g}_k{k}_s{seed}"


def build_additive(n: int, indeg: int, *, k: int = 3, n_train: int = 10_000,
                   n_test: int = 2_000, seed: int = 0) -> dict:
    """In-degree stress case: node j has exactly ``min(j, indeg)`` parents.

    The true CPD is additive in logit space,
    ``logit P(X_j = c | pa) = b_j[c] + sum_i W_ji[pa_i, c]``, so the truth is
    compact although a table of it has ``k^(indeg+1)`` entries; data are
    sampled directly (no dense table is ever built).
    """
    rng = np.random.default_rng(seed)
    names = [f"X{i}" for i in range(n)]
    parents, W, B = [], [], []
    for j in range(n):
        d = min(j, indeg)
        pa = sorted(rng.choice(j, size=d, replace=False).tolist()) if d else []
        parents.append(pa)
        W.append(rng.normal(0, 1.5 / max(1, d) ** 0.5, size=(d, k, k)))
        B.append(rng.normal(0, 1, size=k))
    def sample(m):
        X = np.zeros((m, n), dtype=np.int64)
        for j in range(n):
            lg = np.tile(B[j], (m, 1))
            for t, i in enumerate(parents[j]):
                lg += W[j][t][X[:, i]]
            p = np.exp(lg - lg.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            u = rng.random((m, 1))
            X[:, j] = np.minimum((u > p.cumsum(1)).sum(1), k - 1)
        return X
    Xtr, Xte = sample(n_train), sample(n_test)
    edges = [(names[i], names[j]) for j in range(n) for i in parents[j]]
    return dict(
        name=f"discadd_fixed_n{n}_deg{indeg}_k{k}_s{seed}", family="discadd",
        topology="fixed", n=n, deg=indeg, k=k, window=0, seed=seed, nodes=names,
        edges=edges, n_edges=len(edges), max_in=max(len(p) for p in parents),
        train={v: torch.from_numpy(Xtr[:, i]) for i, v in enumerate(names)},
        test={v: torch.from_numpy(Xte[:, i]) for i, v in enumerate(names)},
        true_model=None, add_parents=parents, add_W=W, add_B=B,
    )


def build(family: str, topology: str, n: int, deg: float, *, k: int = 3,
          window: int = 5, cap: int = 8, n_train: int = 10_000,
          n_test: int = 2_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    edges_i = make_dag(n, deg, topology, window, cap, rng)
    names = [f"X{i}" for i in range(n)]
    edges = [(names[a], names[b]) for a, b in edges_i]
    if family == "discrete":
        spec = dict.fromkeys(names, ("discrete", k))
    else:
        spec = dict.fromkeys(names, ("continuous", 1))
    strong = family == "ngs"
    model = NeuralBayesianNetwork(edges, variables=spec, device="cpu")
    kinds = dict.fromkeys(names, "discrete" if family == "discrete" else "continuous")
    pilot: dict = {}
    for v in names:  # names are already a topological order
        pa = list(model.dag.parents(v))
        if family == "discrete":
            mech = _build_discrete_mechanism(v, pa, kinds, k, gen,
                                             torch.device("cpu"), family="discrete")
        else:
            mech = _build_mdn_mechanism(pa, gen, torch.device("cpu"))
            if strong and pa:
                _strengthen(mech, pa, pilot, strength=1.5, gen=gen)
        model.set_mechanism(v, mech)
        if strong:
            with torch.no_grad():
                P = torch.stack([pilot[a] for a in pa], -1) if pa else None
                pilot[v] = (mech.sample(P, 1) if pa else mech.sample(None, 4000)).reshape(-1)
    torch.manual_seed(seed)
    with torch.no_grad():
        train = model.sample(n=n_train)
        test = model.sample(n=n_test)

    def _clean(d):
        out = {}
        for key, v in d.items():
            v = v.reshape(v.shape[0], -1)
            out[key] = v[:, 0].long() if family == "discrete" else v[:, 0].float()
        return out

    train, test = _clean(train), _clean(test)
    g = nx.DiGraph()
    g.add_nodes_from(names)
    g.add_edges_from(edges)
    return dict(
        name=problem_name(family, topology, n, deg, k, seed, window),
        family=family, topology=topology, n=n, deg=deg, k=k, window=window,
        seed=seed, nodes=names, edges=edges, n_edges=len(edges),
        max_in=max((g.in_degree(v) for v in names), default=0),
        train=train, test=test, true_model=model,
    )


def save(problem: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{problem['name']}.pt"
    torch.save(problem, p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="discrete")
    ap.add_argument("--topology", default="er")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--deg", type=float, default=2.0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmarks/scaling/problems")
    a = ap.parse_args()
    t = time.perf_counter()
    pr = build(a.family, a.topology, a.n, a.deg, k=a.k, window=a.window, seed=a.seed)
    p = save(pr, Path(a.out))
    print(p, pr["n_edges"], f"{time.perf_counter()-t:.2f}s")


if __name__ == "__main__":
    main()
