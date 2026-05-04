"""Render the v0.3 scaling-ablation figure.

Sweeps a discrete bnlearn network ladder ``{cancer(5), asia(8), child(20),
alarm(37)}`` with three baselines on the same axes — `nbn_ve`, `nbn_lw`,
`pgmpy` — so the page-2 figure can show the **cross-over** trend
(pgmpy ahead at small n; NBN catches up as n grows). Saves
``examples/figures/scaling_nodes_speed.{pdf,svg,png}``.

Usage:
    python examples/render_scaling_figure.py [--smoke]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch  # noqa: E402

from benchmarking.baselines import get_adapter  # noqa: E402
from benchmarking.domains import get_domain  # noqa: E402
from benchmarking.plotter import plot_scaling_ablation  # noqa: E402
from nbn import seed_all  # noqa: E402

# Discrete bnlearn network ladder, ordered by node count.
LADDER_FULL = [("cancer", 5), ("asia", 8), ("child", 20), ("alarm", 37)]
LADDER_SMOKE = [("cancer", 5), ("asia", 8)]

BASELINES = ["nbn_ve", "nbn_lw", "pgmpy"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seed_all(0)

    ladder = LADDER_SMOKE if args.smoke else LADDER_FULL
    print(f"Sweeping bnlearn ladder ∈ {[n for n, _ in ladder]}…")

    rows = []
    for net_name, n_nodes in ladder:
        domain = get_domain("bnlearn")
        problem = domain.load_problem(
            net_name, n_train=1000, n_test=200, seed=0,
            device=torch.device("cpu"),
        )
        # Scaling-grid figure parses problem name as
        # "hybrid_n{N}_d{D}_k{K}_r{R}".  We don't need the full encoding,
        # just the n_nodes column — emit it directly into the row dict.
        for adapter_name in BASELINES:
            try:
                kw = {"device": "cpu"} if adapter_name.startswith("nbn") else {}
                adapter = get_adapter(adapter_name, **kw)
                adapter.fit(problem)
            except Exception as e:
                print(f"  [skip {adapter_name} {net_name}] {e}")
                continue
            marg_qs = [q for q in problem.queries if q.kind == "marginal"][:5]
            t0 = time.perf_counter()
            for q in marg_qs:
                try:
                    adapter.query(q)
                except Exception:
                    pass
            elapsed = time.perf_counter() - t0
            ms_per_q = elapsed / max(len(marg_qs), 1) * 1000
            rows.append({
                "baseline": adapter_name, "device": "cpu",
                "problem": f"hybrid_n{n_nodes}_d020_k4_r0",   # parser-friendly
                "ms_per_query": ms_per_q,
                "time_s": elapsed,
            })
            print(f"  {adapter_name:8s} {net_name:8s} (n={n_nodes:3d}): "
                  f"{ms_per_q:7.3f} ms/q ({len(marg_qs)} q)")
            adapter.teardown()

    out_dir = _repo_root / "examples" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = plot_scaling_ablation(
        rows, axis="n_nodes", metric="ms_per_query",
        out_stem=str(out_dir / "scaling_nodes_speed"),
    )
    if written:
        print(f"Saved {written}")
        return 0
    print("No figure produced — likely no rows from the sweep.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
