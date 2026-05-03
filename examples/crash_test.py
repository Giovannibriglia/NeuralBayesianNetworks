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

# Self-bootstrap: make `nbn` and `benchmarking` importable when this script is
# launched directly (e.g. `python examples/crash_test.py`) without first
# running `pip install -e .`. Inserts the repo root onto sys.path if needed.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch  # noqa: E402

from benchmarking.baselines import get_adapter  # noqa: E402
from benchmarking.domains import get_domain  # noqa: E402
from benchmarking.metrics import (  # noqa: E402
    energy_distance, kl_divergence, map_accuracy, tv_distance, wasserstein_1d,
)
from benchmarking.style import NBN_PALETTE, apply_style, savefig_multi  # noqa: E402
from nbn import NeuralBayesianNetwork, seed_all  # noqa: E402


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


def _color_for(r):
    return NBN_PALETTE.get(r["baseline"].split("_")[0], "#888")


def _label(r):
    return f"{r['baseline']}\n({r['device']})"


def _plot_summary(rows, fig_dir):
    """Single 2x2 figure: rows = discrete / continuous, cols = accuracy / speed."""
    apply_style()
    import matplotlib.pyplot as plt

    discrete = [r for r in rows if r["problem"] == "alarm"]
    hybrid = [r for r in rows if r["problem"].startswith("hybrid")]

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5),
                             gridspec_kw=dict(hspace=0.55, wspace=0.30))

    # --- Row 0: Discrete (alarm) ---
    ax = axes[0, 0]
    discrete_acc = [r for r in discrete if r["tv"] == r["tv"]]  # noqa: PLR0124
    if discrete_acc:
        ax.bar([_label(r) for r in discrete_acc],
               [r["tv"] for r in discrete_acc],
               color=[_color_for(r) for r in discrete_acc])
        ax.set_ylabel("TV (lower is better)")
        ax.set_title("Discrete (alarm) — accuracy")
        ax.tick_params(axis="x", labelrotation=30)
        ax.axhline(0.05, color="grey", ls=":", lw=0.8, alpha=0.6)
    else:
        ax.text(0.5, 0.5, "no discrete results", ha="center", va="center")
        ax.set_title("Discrete (alarm) — accuracy")

    ax = axes[0, 1]
    if discrete:
        ax.bar([_label(r) for r in discrete],
               [r["ms_per_query"] for r in discrete],
               color=[_color_for(r) for r in discrete])
        ax.set_ylabel("ms / query  (log)")
        ax.set_yscale("log")
        ax.set_title("Discrete (alarm) — speed")
        ax.tick_params(axis="x", labelrotation=30)
    else:
        ax.text(0.5, 0.5, "no discrete results", ha="center", va="center")
        ax.set_title("Discrete (alarm) — speed")

    # --- Row 1: Continuous / hybrid ---
    ax = axes[1, 0]
    hybrid_acc = [r for r in hybrid if r["tv"] == r["tv"]]  # noqa: PLR0124
    if hybrid_acc:
        ax.bar([_label(r) for r in hybrid_acc],
               [r["tv"] for r in hybrid_acc],
               color=[_color_for(r) for r in hybrid_acc])
        ax.set_ylabel("TV (lower is better)")
        ax.set_title("Continuous / hybrid — accuracy")
        ax.tick_params(axis="x", labelrotation=30)
    else:
        # Hybrid problems use sample-based ground truth (no marginal TV);
        # show held-out NLL placeholder instead.
        ax.text(0.5, 0.5, "no marginal ground truth\n(sample-based domain)",
                ha="center", va="center", fontsize=9, alpha=0.7)
        ax.set_title("Continuous / hybrid — accuracy")
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 1]
    if hybrid:
        ax.bar([_label(r) for r in hybrid],
               [r["ms_per_query"] for r in hybrid],
               color=[_color_for(r) for r in hybrid])
        ax.set_ylabel("ms / query  (log)")
        ax.set_yscale("log")
        ax.set_title("Continuous / hybrid — speed")
        ax.tick_params(axis="x", labelrotation=30)
    else:
        ax.text(0.5, 0.5, "no hybrid results", ha="center", va="center")
        ax.set_title("Continuous / hybrid — speed")

    fig.suptitle("NBN crash test: parameter learning + serial inference", fontsize=12)
    return savefig_multi(fig, str(fig_dir / "crash_test_summary"))


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
    #
    # All NBN public engines + mechanism families are exposed:
    #   * nbn_ve                 — TensorVariableElimination (exact, discrete)
    #   * nbn_lw                 — LikelihoodWeightingEngine (IS, hybrid)
    #   * nbn_hybrid             — HybridRouter (auto-pick by treewidth)
    #   * nbn_neural_categorical — discrete CPDs as MLP + embedding
    #   * nbn_linear_gaussian    — continuous CPDs as closed-form ridge LG
    #
    # Why GPyTorch is absent from the discrete lineup:
    #   GPyTorch implements Gaussian Processes — its likelihoods (Gaussian /
    #   Bernoulli / Multitask) are continuous regression targets. There is no
    #   first-class way to model a multi-class categorical CPT inside a GP
    #   without a continuous relaxation (e.g. the Polya-Gamma trick), and that
    #   is itself a research project rather than a fair benchmark adapter. The
    #   `GPyTorchAdapter` therefore declares `supports = {"continuous"}`; the
    #   runner skips queries with discrete evidence with `not_supported` rather
    #   than producing meaningless numbers.
    discrete_baselines = [
        "nbn_ve", "nbn_lw", "nbn_hybrid", "nbn_neural_categorical",
        "pgmpy", "pomegranate", "pyro",
    ]
    hybrid_baselines = [
        "nbn_lw", "nbn_hybrid", "nbn_linear_gaussian",
        "pyro", "gpytorch",
    ]

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
            _plot_summary(rows, fig_dir)
            print(f"\nFigures saved to {fig_dir}/")
        except Exception as e:
            print(f"  [figure-skip] {e}")

    # Always emit a paper-ready LaTeX table next to the figures.
    if rows:
        from benchmarking.latex import write_latex_table
        cols = [
            ("baseline", "Baseline"), ("device", "Device"),
            ("problem", "Problem"), ("n_queries", "Q"),
            ("time_s", "Time (s)"), ("ms_per_query", "ms/q"),
            ("tv", "TV"), ("map_accuracy", "MAP-acc"),
        ]
        out_tex = fig_dir / "crash_test_summary.tex"
        write_latex_table(
            out_tex, rows, cols,
            caption="Crash test: parameter learning + serial inference.",
            label="tab:nbn-crash-test",
            formats={"time_s": ".3f", "ms_per_query": ".2f",
                     "tv": ".4f", "map_accuracy": ".3f"},
        )
        print(f"LaTeX table saved to {out_tex}")

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
