"""Render a minimal scaling-ablation figure.

Sweeps ``n_nodes ∈ {10, 50, 200}`` on the synthetic-hybrid scaling domain
with one replicate, fits ``nbn_lw`` and ``nbn_hybrid``, and saves a
log-log throughput-vs-N plot under
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
from benchmarking.domains.synthetic_hybrid import make_scaling_grid  # noqa: E402
from benchmarking.plotter import plot_scaling_ablation  # noqa: E402
from nbn import seed_all  # noqa: E402

SIZES_FULL = [10, 50, 200]
SIZES_SMOKE = [10, 50]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seed_all(0)

    sizes = SIZES_SMOKE if args.smoke else SIZES_FULL
    print(f"Sweeping n_nodes ∈ {sizes}…")

    grid = make_scaling_grid(
        n_nodes=sizes, n_replicates=1, n_train=200, n_test=50,
        device=torch.device("cpu"),
    )
    rows = []
    for problem in grid:
        for adapter_name in ["nbn_lw", "nbn_hybrid"]:
            try:
                adapter = get_adapter(adapter_name, device="cpu")
                adapter.fit(problem)
            except Exception as e:
                print(f"  [skip {adapter_name} {problem.name}] {e}")
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
                "problem": problem.name,
                "ms_per_query": ms_per_q,
                "time_s": elapsed,
            })
            print(f"  {adapter_name:12s} {problem.name:30s} "
                  f"{ms_per_q:6.2f} ms/q ({len(marg_qs)} q)")
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
