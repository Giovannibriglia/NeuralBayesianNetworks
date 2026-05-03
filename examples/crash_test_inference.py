"""Inference-throughput crash test (v0.2).

Where ``examples/crash_test.py`` measures *parameter learning + a handful of
serial queries*, this script targets the **batched-inference** regime and
showcases NBN's headline feature: ``model.query_batch(...)``.

Setup:
    1. Fit each baseline once on the ``alarm`` BN (same training data, same
       seed). Fit time is *not* reported here — we want to isolate inference.
    2. Generate B random conditional queries with the same target and
       randomised evidence on a fixed pair of evidence nodes.
    3. For NBN: call ``model.query_batch(target, evidence={node: [B]-tensor})``
       — one GPU launch.
       For pgmpy / pomegranate / pyro: a Python loop over the same B queries
       (none of these libraries expose a batched query API).

Output:
    * console table of total-time / latency / throughput per (baseline, B)
    * ``examples/figures/crash_test_inference_throughput.{pdf,png,svg}``
      log-log throughput curve

Usage:
    python examples/crash_test_inference.py [--smoke] [--no-figures]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Self-bootstrap so this script runs without `pip install -e .` first.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch  # noqa: E402

from benchmarking.baselines import get_adapter  # noqa: E402
from benchmarking.domains import get_domain  # noqa: E402
from benchmarking.domains.base import Query  # noqa: E402
from benchmarking.style import NBN_PALETTE, apply_style, savefig_multi  # noqa: E402
from nbn import seed_all  # noqa: E402

# Batch sizes to sweep. Smoke mode shrinks the sweep + uses CPU.
SWEEP = (1, 16, 256, 1024, 4096)
SMOKE_SWEEP = (1, 16, 64)


def _devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def _make_query_batch(problem, target, B: int, seed: int = 0) -> Query:
    """B different conditional queries on the same target, randomised evidence."""
    g = torch.Generator().manual_seed(seed)
    nodes = [n for n in problem.variables if n != target]
    if len(nodes) < 2:
        raise ValueError("Need at least 2 non-target nodes for conditional queries.")
    e_nodes = nodes[:2]
    cards = {n: int(problem.variables[n][1]) for n in e_nodes}
    evidence = {
        n: torch.randint(0, cards[n], (B,), generator=g)
        for n in e_nodes
    }
    return Query(targets=(target,), evidence=evidence, kind="conditional")


def _bench_baseline(adapter_name, problem, target, sweep, *, device):
    """For each B in sweep, time the queries and return rows."""
    rows = []
    kw = {"device": device} if adapter_name in {"nbn", "nbn_lw", "nbn_hybrid",
                                                 "gpytorch"} else {}
    try:
        adapter = get_adapter(adapter_name, **kw)
        adapter.fit(problem)
    except Exception as e:
        print(f"  [skip {adapter_name}] {e}")
        return rows

    is_nbn = adapter_name.startswith("nbn")
    for B in sweep:
        q = _make_query_batch(problem, target, B)
        try:
            if is_nbn and hasattr(adapter, "query_batch"):
                # Single launch path
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = adapter.query_batch(q)
                if device == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                method = "batched"
            else:
                # Serial loop (no batched API)
                t0 = time.perf_counter()
                for i in range(B):
                    ev_i = {k: v[i:i + 1] if v.dim() >= 1 else v
                            for k, v in q.evidence.items()}
                    adapter.query(Query(targets=q.targets, evidence=ev_i,
                                         kind=q.kind))
                dt = time.perf_counter() - t0
                method = "loop"
        except Exception as e:
            print(f"  [error {adapter_name} B={B}] {e}")
            continue
        rows.append({
            "baseline": adapter_name, "device": device, "B": B,
            "time_s": dt, "ms_per_query": dt / B * 1e3,
            "throughput_qps": B / max(dt, 1e-9),
            "method": method,
        })
        print(f"  {adapter_name:22s} {device:4s} B={B:5d}  "
              f"{dt:6.3f} s   {dt / B * 1000:7.3f} ms/q   "
              f"{B / max(dt, 1e-9):>10.1f} q/s   ({method})")
    adapter.teardown()
    return rows


def _plot_throughput(rows, out_stem):
    apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.2))
    by_label = {}
    for r in rows:
        label = f"{r['baseline']} ({r['device']})"
        by_label.setdefault(label, []).append(r)
    for label, pts in by_label.items():
        pts = sorted(pts, key=lambda r: r["B"])
        xs = [p["B"] for p in pts]
        ys = [p["throughput_qps"] for p in pts]
        ax.plot(xs, ys, marker="o", label=label,
                color=NBN_PALETTE.get(pts[0]["baseline"].split("_")[0], "#888"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Batch size B  (log)")
    ax.set_ylabel("Throughput  (queries / s, log)")
    ax.set_title("Inference throughput vs batch size")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return savefig_multi(fig, str(out_stem))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    seed_all(0)

    sweep = SMOKE_SWEEP if args.smoke else SWEEP
    devices = ["cpu"] if args.smoke else _devices()

    print("==== NBN inference-throughput crash test ====")
    domain = get_domain("bnlearn")
    problem = domain.load_problem("alarm", n_train=1000, n_test=200,
                                  seed=0, device=torch.device("cpu"))
    target = problem.dag[0][1]  # any non-root node from the DAG

    rows = []
    # Bottom of the lineup: NBN-batched (the win) + serial-loop competitors.
    # pomegranate is excluded here: its predict_proba in v1.1.x has a bug
    # with evidence on masked tensors (works for marginals only) — see the
    # PomegranateAdapter docstring. pyro is slow and skipped in --smoke.
    nbn_baselines = ["nbn_lw", "nbn_hybrid"]
    others = ["pgmpy"]
    if not args.smoke:
        others.append("pyro")

    for device in devices:
        print(f"-- device={device}")
        for b_name in nbn_baselines + others:
            rows.extend(_bench_baseline(b_name, problem, target, sweep, device=device))

    if rows and not args.no_figures:
        out_dir = _repo_root / "examples" / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            _plot_throughput(rows, out_dir / "crash_test_inference_throughput")
            print(f"\nFigure saved to {out_dir}/crash_test_inference_throughput.{{pdf,png,svg}}")
        except Exception as e:
            print(f"  [figure-skip] {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
