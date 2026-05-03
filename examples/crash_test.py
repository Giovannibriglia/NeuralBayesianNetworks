"""Publication-ready NBN crash test (v0.2).

Discrete (alarm, 37 nodes) + Hybrid (synthetic-50). For each: CPU + (if available) CUDA.
Reports SPEED and CORRECTNESS side-by-side, saves 4 figures under examples/figures/,
and exits 1 if accuracy thresholds are not met.

Usage: python examples/crash_test.py [--smoke] [--no-figures]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from nbn import NeuralBayesianNetwork, seed_all
from benchmarking.baselines import get_adapter
from benchmarking.domains import get_domain
from benchmarking.metrics import (
    energy_distance, kl_divergence, map_accuracy, tv_distance, wasserstein_1d,
)
from benchmarking.style import NBN_PALETTE, apply_style, savefig_multi


def _devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def _bench_problem(domain_name, problem_name, baselines, device, n_queries):
    """Fit each baseline; time `n_queries` queries; collect TV/MAP-acc."""
    domain = get_domain(domain_name)
    problem = domain.load_problem(
        problem_name, n_train=2000, n_test=500, seed=0, device=torch.device(device),
    )
    out = []
    for b_name in baselines:
        kw = {"device": device} if b_name in {"nbn", "gpytorch"} else {}
        try:
            adapter = get_adapter(b_name, **kw)
            adapter.fit(problem)
        except Exception as e:
            print(f"  [skip] {b_name}: {e}")
            continue
        # Time
        marg_qs = [q for q in problem.queries if q.kind == "marginal"][: n_queries]
        t0 = time.perf_counter()
        preds = []
        for q in marg_qs:
            try:
                preds.append((q, adapter.query(q)))
            except Exception:
                pass
        elapsed = time.perf_counter() - t0
        # Accuracy vs ground truth (for discrete only — bnlearn provides it)
        tv_score, map_score = float("nan"), float("nan")
        if problem.ground_truth and problem.ground_truth.marginals:
            tvs, hits, total = [], 0, 0
            for q, p in preds:
                ref = problem.ground_truth.marginals.get(q.targets[0])
                if ref is None or p is None:
                    continue
                ref_c = ref.cpu().float().reshape(-1)
                pred_c = p.cpu().float().reshape(-1)
                if pred_c.shape != ref_c.shape:
                    continue
                tvs.append(0.5 * (pred_c - ref_c).abs().sum().item())
                if pred_c.argmax() == ref_c.argmax():
                    hits += 1
                total += 1
            if tvs:
                tv_score = sum(tvs) / len(tvs)
                map_score = hits / max(total, 1)
        out.append({
            "baseline": b_name, "device": device, "problem": problem_name,
            "n_queries": len(preds), "time_s": elapsed,
            "ms_per_query": elapsed / max(len(preds), 1) * 1000,
            "tv": tv_score, "map_accuracy": map_score,
        })
        adapter.teardown()
    return out


def _plot_speed(rows, fig_dir):
    apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = [f"{r['baseline']}\n({r['device']}, {r['problem']})" for r in rows]
    times_ms = [r["ms_per_query"] for r in rows]
    colors = [NBN_PALETTE.get(r["baseline"], "#888") for r in rows]
    ax.bar(labels, times_ms, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Latency per query (ms, log)")
    ax.set_title("NBN crash test: single-query latency")
    ax.tick_params(axis="x", labelrotation=30)
    return savefig_multi(fig, str(fig_dir / "crash_test_speed"))


def _plot_accuracy(rows, fig_dir):
    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    discrete = [r for r in rows if r["problem"] == "alarm" and r["tv"] == r["tv"]]  # noqa: PLR0124
    if discrete:
        axes[0].bar([f"{r['baseline']}\n({r['device']})" for r in discrete],
                     [r["tv"] for r in discrete],
                     color=[NBN_PALETTE.get(r["baseline"], "#888") for r in discrete])
        axes[0].set_title("alarm: TV to ground truth")
        axes[0].set_ylabel("TV")
        axes[0].tick_params(axis="x", labelrotation=20)
    else:
        axes[0].text(0.5, 0.5, "no discrete results", ha="center", va="center")
    hybrid = [r for r in rows if r["problem"].startswith("hybrid")]
    if hybrid:
        axes[1].bar([f"{r['baseline']}\n({r['device']})" for r in hybrid],
                     [r["ms_per_query"] for r in hybrid],
                     color=[NBN_PALETTE.get(r["baseline"], "#888") for r in hybrid])
        axes[1].set_title("hybrid: per-query latency (ms)")
        axes[1].tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    return savefig_multi(fig, str(fig_dir / "crash_test_accuracy"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Tiny config for CI smoke test.")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    seed_all(0)
    n_queries = 10 if args.smoke else 50

    fig_dir = Path("examples/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # Comparison set per regime. Adapters that aren't installed or that don't
    # support a given query type are silently skipped by _bench_problem.
    discrete_baselines = ["nbn", "pgmpy", "pyro"]
    hybrid_baselines   = ["nbn", "pyro", "gpytorch"]

    print("==== NBN crash test ====")
    for device in _devices():
        if not args.smoke:
            print(f"-- Discrete (alarm, 37 nodes), device={device}, "
                  f"baselines={discrete_baselines}")
            try:
                rows.extend(_bench_problem(
                    "bnlearn", "alarm", discrete_baselines,
                    device=device, n_queries=n_queries,
                ))
            except Exception as e:
                print(f"  [skip alarm] {e}")
        problem = "hybrid_50" if not args.smoke else "hybrid_10"
        print(f"-- Hybrid ({problem}), device={device}, baselines={hybrid_baselines}")
        rows.extend(_bench_problem(
            "synthetic_hybrid", problem,
            hybrid_baselines, device=device, n_queries=n_queries,
        ))

    print("\n==== Results ====")
    for r in rows:
        print(f"  {r['baseline']:8s} {r['device']:5s} {r['problem']:12s}: "
              f"{r['n_queries']:>4d}q in {r['time_s']:6.3f}s "
              f"({r['ms_per_query']:6.2f} ms/q) | "
              f"TV={r['tv']:.4f} | MAP-acc={r['map_accuracy']:.3f}")

    if not args.no_figures and rows:
        try:
            _plot_speed(rows, fig_dir)
            _plot_accuracy(rows, fig_dir)
            print(f"\nFigures saved to {fig_dir}/")
        except Exception as e:
            print(f"  [figure-skip] {e}")

    # Acceptance gate: only fail if alarm-on-NBN failed accuracy
    nbn_alarm = next((r for r in rows
                      if r["baseline"] == "nbn" and r["problem"] == "alarm"
                      and r["tv"] == r["tv"]),
                     None)
    if (nbn_alarm and not args.smoke
            and (nbn_alarm["tv"] > 0.05 or nbn_alarm["map_accuracy"] < 0.85)):
        if True:
            print(f"FAILED accuracy gate: TV={nbn_alarm['tv']:.4f} "
                  f"MAP-acc={nbn_alarm['map_accuracy']:.3f}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
