"""Factor cache vs one model per hypothesis (#312): time and stored table cells.

A 2-slice model with ``d`` binary-or-ternary state variables, one action and
``c`` candidate parent sets per next-state variable spans ``K = c ** d``
hypotheses and ``d * c`` factors.  The baseline builds one model per
hypothesis and counts / builds the interval set of every node of it; the
factor cache counts once per node (union table, then marginalise) and builds
one set per factor.

Usage::

    python scripts/bench_hypothesis_set.py [--device cpu|cuda|mps] [--n 200000]
"""
from __future__ import annotations

import argparse
import itertools
import time

import torch

from nbn import NeuralBayesianNetwork
from nbn.structure import HypothesisSet
from nbn.uncertainty import count_table, interval_set_from_counts


def spec(d: int, c: int):
    cards = {f"s{i}": 3 for i in range(d)}
    cards["a"] = 3
    inputs = list(cards)
    cand = {}
    for i in range(d):
        pool = [p for r in (2, 3) for p in itertools.combinations(inputs, r) if f"s{i}" in p]
        cand[f"s{i}'"] = pool[:c]
        cards[f"s{i}'"] = 3
    return {v: ("discrete", k) for v, k in cards.items()}, cand, cards


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=200_000)
    args = ap.parse_args()
    print(f"device={args.device} n={args.n}")
    print(f"{'K':>5} {'factors':>7} | {'cache s':>8} {'cells':>9} | {'per-hyp s':>9} {'cells':>9} | speed-up")
    for d, c in [(1, 1), (4, 2), (4, 4)]:
        variables, cand, cards = spec(d, c)
        gen = torch.Generator().manual_seed(0)
        data = {v: torch.randint(0, k, (args.n,), generator=gen).to(args.device) for v, k in cards.items()}
        hs = HypothesisSet(variables, cand)

        sync(args.device)
        t0 = time.perf_counter()
        fs = hs.factor_sets(data, delta=0.05)
        for h in hs.hypotheses():
            fs.sets_for(h)
        sync(args.device)
        t_cache = time.perf_counter() - t0
        cells_cache = sum(t.numel() for t in fs.counts.tables.values())

        t0 = time.perf_counter()
        cells_base = 0
        for h in hs.hypotheses():
            model = NeuralBayesianNetwork(hs.edges(h), dict(hs.variables), device=args.device)
            for node in hs.targets:
                counts, _ = count_table(model, data, node)
                interval_set_from_counts(counts, delta=0.05 / hs.n_factors)
                cells_base += counts.numel()
        sync(args.device)
        t_base = time.perf_counter() - t0
        print(f"{hs.n_hypotheses:>5} {hs.n_factors:>7} | {t_cache:>8.3f} {cells_cache:>9} | "
              f"{t_base:>9.3f} {cells_base:>9} | {t_base / t_cache:>6.1f}x")


if __name__ == "__main__":
    main()
