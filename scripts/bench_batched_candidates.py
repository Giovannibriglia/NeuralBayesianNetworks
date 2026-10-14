"""Batched candidate fit vs one fit per candidate (#318): wall clock for C = 1, 4, 16.

Continuous target with an MLP location (two hidden layers of 64, SiLU).  The
baseline fits ``AdditiveNoiseMechanism(location="mlp")`` once per candidate;
the batched path trains all candidates of the node in one loop
(``HypothesisSet.fit``).  Same epochs, learning rate and batch size.  With
full-batch updates on many rows the work is compute-bound and both paths
cost about the same; with small mini-batches the per-step overhead dominates
and one stacked step for all candidates is faster.

Usage::

    python scripts/bench_batched_candidates.py [--device cpu|cuda|mps] [--n 20000] [--epochs 100] [--batch-size 256]
"""
from __future__ import annotations

import argparse
import itertools
import time

import torch

from nbn.scm.mechanisms import AdditiveNoiseMechanism
from nbn.structure import HypothesisSet


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()
    d = 6
    g = torch.Generator().manual_seed(0)
    inputs = [f"u{i}" for i in range(d)]
    data = {v: torch.randn(args.n, generator=g) for v in inputs}
    data["x'"] = torch.sin(data["u0"]) + data["u1"] * data["u2"] + 0.3 * torch.randn(args.n, generator=g)
    data = {k: v.to(args.device) for k, v in data.items()}
    variables = dict.fromkeys([*inputs, "x'"], ("continuous", 1))
    pool = list(itertools.combinations(inputs, 3))
    print(f"device={args.device} n={args.n} epochs={args.epochs} batch_size={args.batch_size}")
    print(f"{'C':>3} | {'batched s':>9} | {'separate s':>10} | speed-up")
    for c in (1, 4, 16):
        hs = HypothesisSet(variables, {"x'": pool[:c]})
        sync(args.device)
        t0 = time.perf_counter()
        hs.fit(data, location="mlp", epochs=args.epochs, batch_size=args.batch_size)
        sync(args.device)
        t_batched = time.perf_counter() - t0
        t0 = time.perf_counter()
        for parents in pool[:c]:
            mech = AdditiveNoiseMechanism(location="mlp", epochs=args.epochs,
                                          batch_size=args.batch_size).to(args.device)
            pa = torch.stack([data[p] for p in parents], -1)
            mech.fit_local(data["x'"].reshape(-1, 1), pa)
        sync(args.device)
        t_sep = time.perf_counter() - t0
        print(f"{c:>3} | {t_batched:>9.2f} | {t_sep:>10.2f} | {t_sep / t_batched:>6.1f}x")


if __name__ == "__main__":
    main()
