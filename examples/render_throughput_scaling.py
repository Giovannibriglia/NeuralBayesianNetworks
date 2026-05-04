"""Render the v0.3 page-1 *headline* scaling figure: batched throughput.

Per the F2 clarification on PR #6: the single-query latency figure is
honest but tells the wrong story (pgmpy's ≤1 ms Python-VE wins on small
nets and NBN-VE's amortised speedup needs CUDA-graph capture to land —
see #7). Batched-query throughput (queries / second when the workload
is ``query_batch(B=1024)``) is the metric NBN already wins on the data
we have today, because ``LikelihoodWeightingEngine.query_batch`` runs a
single vectorised einsum across the whole batch while pgmpy is forced
into a Python loop with full per-call overhead.

Sweeps the discrete bnlearn ladder ``cancer(5) → asia(8) → child(20) →
alarm(37)`` and writes
``examples/figures/scaling_nodes_batched_throughput.{pdf,svg,png}``.

The legacy single-query figure
``examples/render_scaling_figure.py`` stays as appendix material.

Usage:
    python examples/render_throughput_scaling.py [--smoke]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch  # noqa: E402

from benchmarking.baselines import get_adapter  # noqa: E402
from benchmarking.domains import get_domain  # noqa: E402
from benchmarking.domains.base import Query  # noqa: E402
from benchmarking.style import NBN_PALETTE, apply_style, savefig_multi  # noqa: E402
from nbn import seed_all  # noqa: E402

LADDER_FULL = [("cancer", 5), ("asia", 8), ("child", 20), ("alarm", 37)]
LADDER_SMOKE = [("cancer", 5), ("asia", 8)]
BASELINES = ["nbn_lw", "nbn_hybrid", "nbn_ve", "pgmpy"]
B = 1024
B_SMOKE = 64
N_WARMUP = 5
N_REPEAT = 3   # take median


def _bench_throughput(adapter, problem, target, batch: int) -> float:
    """Time `B` conditional queries; return queries-per-second."""
    g = torch.Generator().manual_seed(0)
    nodes = [n for n in problem.variables if n != target]
    if len(nodes) < 2:
        return float("nan")
    e_nodes = nodes[:2]
    cards = {n: int(problem.variables[n][1]) for n in e_nodes}
    evidence = {n: torch.randint(0, cards[n], (batch,), generator=g)
                for n in e_nodes}
    q_batch = Query(targets=(target,), evidence=evidence, kind="conditional")

    is_nbn_batched = adapter.name.startswith("nbn") and hasattr(adapter, "query_batch")
    # Warm up
    for _ in range(N_WARMUP):
        try:
            if is_nbn_batched:
                adapter.query_batch(q_batch)
            else:
                ev1 = {k: v[0:1] for k, v in evidence.items()}
                adapter.query(Query(targets=(target,), evidence=ev1, kind="conditional"))
        except Exception:
            return float("nan")
    # Measure
    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        try:
            if is_nbn_batched:
                adapter.query_batch(q_batch)
            else:
                for i in range(batch):
                    ev_i = {k: v[i:i + 1] for k, v in evidence.items()}
                    adapter.query(Query(targets=(target,), evidence=ev_i,
                                         kind="conditional"))
            times.append(time.perf_counter() - t0)
        except Exception:
            return float("nan")
    return batch / max(statistics.median(times), 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seed_all(0)

    ladder = LADDER_SMOKE if args.smoke else LADDER_FULL
    batch = B_SMOKE if args.smoke else B
    print(f"Sweeping bnlearn ladder ∈ {[n for n, _ in ladder]}, B={batch}…")

    rows = []
    for net_name, n_nodes in ladder:
        problem = get_domain("bnlearn").load_problem(
            net_name, n_train=1000, n_test=200, seed=0,
            device=torch.device("cpu"),
        )
        target = problem.dag[0][1]
        for adapter_name in BASELINES:
            try:
                kw = {"device": "cpu"} if adapter_name.startswith("nbn") else {}
                adapter = get_adapter(adapter_name, **kw)
                adapter.fit(problem)
            except Exception as e:
                print(f"  [skip {adapter_name} {net_name}] {e}")
                continue
            qps = _bench_throughput(adapter, problem, target, batch)
            rows.append({
                "baseline": adapter_name, "device": "cpu",
                "n_nodes": n_nodes,
                "throughput_qps": qps,
            })
            print(f"  {adapter_name:10s} {net_name:8s} (n={n_nodes:3d}): "
                  f"{qps:>10.1f} q/s")
            adapter.teardown()

    apply_style()
    import matplotlib.pyplot as plt

    out_dir = _repo_root / "examples" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
    by_baseline: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        by_baseline.setdefault(r["baseline"], []).append((r["n_nodes"], r["throughput_qps"]))
    for baseline, pts in by_baseline.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", lw=1.6, ms=6,
                color=NBN_PALETTE.get(baseline.split("_")[0], "#888"),
                label=f"{baseline} (cpu)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n_nodes (log)")
    ax.set_ylabel("queries / second  (log)")
    ax.set_title(f"Batched throughput on bnlearn ladder (B={batch}, CPU)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.text(0.99, 0.005,
             "seed=0 · NBN v0.3 · CPU sandbox · batched query_batch vs serial loop",
             ha="right", va="bottom", fontsize=6, alpha=0.55, family="monospace")
    written = savefig_multi(fig, str(out_dir / "scaling_nodes_batched_throughput"))
    print(f"\nSaved {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
