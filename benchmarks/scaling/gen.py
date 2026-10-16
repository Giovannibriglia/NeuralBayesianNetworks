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
import math
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


def problem_name(family, topology, n, deg, k, seed, window):
    w = f"_w{window}" if topology == "local" else ""
    return f"{family}_{topology}{w}_n{n}_deg{deg:g}_k{k}_s{seed}"


def build(family: str, topology: str, n: int, deg: float, *, k: int = 3,
          window: int = 5, cap: int = 8, n_train: int = 10_000,
          n_test: int = 2_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    edges_i = make_dag(n, deg, topology, window, cap, rng)
    names = [f"X{i}" for i in range(n)]
    edges = [(names[a], names[b]) for a, b in edges_i]
    if family == "discrete":
        spec = {v: ("discrete", k) for v in names}
    else:
        spec = {v: ("continuous", 1) for v in names}
    model = NeuralBayesianNetwork(edges, variables=spec, device="cpu")
    kinds = {v: ("discrete" if family == "discrete" else "continuous") for v in names}
    for v in names:  # names are already a topological order
        pa = list(model.dag.parents(v))
        if family == "discrete":
            mech = _build_discrete_mechanism(v, pa, kinds, k, gen,
                                             torch.device("cpu"), family="discrete")
        else:
            mech = _build_mdn_mechanism(pa, gen, torch.device("cpu"))
        model.set_mechanism(v, mech)
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
