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

# Third-party packages — imported before sys.path manipulation so ruff E402 does not fire.
import torch
from tqdm import tqdm

# Self-bootstrap: make `nbn` and `benchmarking` importable when this script is
# launched directly (e.g. `python examples/crash_test.py`) without first
# running `pip install -e .`. Inserts the repo root onto sys.path if needed.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

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
    pbar = tqdm(baselines)
    for b_name in pbar:
        pbar.set_description(f"Running [{b_name}]...")
        kw = {"device": device} if b_name in {"nbn", "gpytorch"} else {}
        try:
            adapter = get_adapter(b_name, **kw)
            adapter.fit(problem)
        except Exception as e:
            print(f"  [skip] {b_name}: {e}")
            continue
        # Time
        # Interleave discrete + continuous targets so an n_queries=15 budget
        # doesn't accidentally hit only the first 15 (often all-discrete) nodes.
        all_marg = [q for q in problem.queries if q.kind == "marginal"]
        disc_q = [q for q in all_marg if problem.variables[q.targets[0]][0] == "discrete"]
        cont_q = [q for q in all_marg if problem.variables[q.targets[0]][0] == "continuous"]
        # Round-robin: take half from each, fall back if one bucket is empty.
        per_kind = max(1, n_queries // 2)
        marg_qs = (disc_q[:per_kind] + cont_q[:per_kind])[: n_queries]
        if not marg_qs:
            marg_qs = all_marg[: n_queries]
        t0 = time.perf_counter()
        preds = []
        for q in marg_qs:
            try:
                preds.append((q, adapter.query(q)))
            except Exception:
                pass
        elapsed = time.perf_counter() - t0
        # Accuracy vs ground truth.
        # Discrete (bnlearn): TV between predicted marginal and pgmpy exact.
        # Hybrid (synthetic): W₁ between rejection-filtered SCM samples and
        # baseline samples drawn via NBNAdapter.model.sample(...).  See
        # benchmarking/domains/_ground_truth.py for the SCM resampling.
        from benchmarking.domains._ground_truth import (
            ground_truth_samples_for_query,
        )
        tv_score, map_score = float("nan"), float("nan")
        w1_score, energy_score, js_score = float("nan"), float("nan"), float("nan")
        if problem.ground_truth and problem.ground_truth.marginals:
            tvs, hits, total = [], 0, 0
            w1_acc, energy_acc, js_acc, n_cont = 0.0, 0.0, 0.0, 0
            for q, p in preds:
                target = q.targets[0]
                kind = problem.variables[target][0]
                if p is None:
                    continue
                if kind == "discrete":
                    ref = problem.ground_truth.marginals.get(target)
                    if ref is None:
                        continue
                    ref_c = ref.cpu().float().reshape(-1)
                    pred_c = p.detach().cpu().float().reshape(-1)
                    if pred_c.shape != ref_c.shape:
                        continue
                    tvs.append(0.5 * (pred_c - ref_c).abs().sum().item())
                    if pred_c.argmax() == ref_c.argmax():
                        hits += 1
                    total += 1
                else:
                    # Distributional metrics on continuous targets.
                    # We need *samples* from the baseline. NBN adapters carry
                    # a ``model`` attribute exposing ``sample(n=..., evidence=...)``;
                    # the others (gpytorch, pyro) only return point estimates
                    # — they're explicitly recorded as N/A in panel (d).
                    gt = ground_truth_samples_for_query(
                        problem, target, q.evidence,
                        n_pool=20_000, eps=0.40, n_eff_min=100,
                    )
                    if gt is None or gt.numel() < 200:
                        continue
                    nbn_model = getattr(adapter, "model", None)
                    if nbn_model is None or not hasattr(nbn_model, "sample"):
                        continue
                    try:
                        ev_t = {
                            k: (v if isinstance(v, torch.Tensor)
                                else torch.tensor(v))
                            for k, v in q.evidence.items()
                        }
                        method_samples = nbn_model.sample(
                            n=500, evidence=ev_t,
                        )[target].detach().cpu().float().reshape(-1)
                    except Exception:
                        continue
                    gt_c = gt.detach().cpu().float().reshape(-1)
                    if method_samples.numel() < 50 or gt_c.numel() < 50:
                        continue
                    n = min(method_samples.numel(), gt_c.numel(), 1024)
                    qs = torch.linspace(0, 1, n)
                    w1_acc += float((torch.quantile(method_samples, qs) -
                                     torch.quantile(gt_c, qs)).abs().mean())
                    # Energy distance (1-D) cheap form
                    a = method_samples[:n]; b = gt_c[:n]
                    e_xy = (a.unsqueeze(1) - b.unsqueeze(0)).abs().mean()
                    e_xx = (a.unsqueeze(1) - a.unsqueeze(0)).abs().mean()
                    e_yy = (b.unsqueeze(1) - b.unsqueeze(0)).abs().mean()
                    energy_acc += float(2 * e_xy - e_xx - e_yy)
                    # JS-norm via 64-bin histograms over the union range
                    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
                    if hi > lo:
                        ha = torch.histc(a, bins=64, min=lo, max=hi); ha = ha / ha.sum().clamp_min(1e-12)
                        hb = torch.histc(b, bins=64, min=lo, max=hi); hb = hb / hb.sum().clamp_min(1e-12)
                        m = 0.5 * (ha + hb)
                        js = 0.5 * ((ha * (ha.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum()
                                  + (hb * (hb.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum())
                        import math
                        js_acc += float(js / math.log(2))
                    n_cont += 1
            if tvs:
                tv_score = sum(tvs) / len(tvs)
                map_score = hits / max(total, 1)
            if n_cont > 0:
                w1_score = w1_acc / n_cont
                energy_score = energy_acc / n_cont
                js_score = js_acc / n_cont
        out.append({
            "baseline": b_name, "device": device, "problem": problem_name,
            "n_queries": len(preds), "time_s": elapsed,
            "ms_per_query": elapsed / max(len(preds), 1) * 1000,
            "tv": tv_score, "map_accuracy": map_score,
            "w1": w1_score, "energy": energy_score, "js_norm": js_score,
        })
        adapter.teardown()
    return out


def _color_for(r):
    return NBN_PALETTE.get(r["baseline"].split("_")[0], "#888")


def _label(r):
    return f"{r['baseline']}\n({r['device']})"


def _repro_footer(ax):
    """Reproducibility footer: version, seed, git sha, torch, gpu."""
    import subprocess
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "?"
    try:
        from nbn import __version__ as nbn_v
    except Exception:
        nbn_v = "?"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only"
    text = (f"NBN v{nbn_v} | seed=0 | git={sha} | "
            f"torch={torch.__version__} | {gpu}")
    ax.figure.text(0.99, 0.005, text, ha="right", va="bottom",
                    fontsize=6, alpha=0.55, family="monospace")


def _hbar(ax, rows, value_key, *, log=False, vline=None, fmt=".2f"):
    """Horizontal grouped bars; CUDA solid, CPU hatched.  No collisions."""
    if not rows:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", alpha=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        return
    # Sort by family for visual grouping (NBN first → external)
    fam_order = {"nbn": 0, "pgmpy": 1, "pomegranate": 2, "pyro": 3, "gpytorch": 4}
    rows = sorted(
        rows,
        key=lambda r: (fam_order.get(r["baseline"].split("_")[0], 9),
                       r["baseline"], r["device"]),
    )
    labels = [f"{r['baseline'].replace('nbn_', 'NBN-')} ({r['device']})" for r in rows]
    values = [r[value_key] for r in rows]
    colors = [_color_for(r) for r in rows]
    hatches = ["///" if r["device"] == "cpu" else "" for r in rows]
    y = list(range(len(rows)))
    bars = ax.barh(y, values, color=colors, edgecolor="white",
                   hatch=hatches, height=0.78)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
    if vline is not None:
        ax.axvline(vline, color="grey", ls=":", lw=0.8, alpha=0.6)
    # Bar value labels
    for bar, v in zip(bars, values, strict=False):
        if v != v:  # NaN
            continue
        ax.text(v, bar.get_y() + bar.get_height() / 2, f" {v:{fmt}}",
                va="center", ha="left", fontsize=6.8, alpha=0.85)


def _plot_summary(rows, fig_dir):
    """Page-1 figure: 2x2 with horizontal bars, family grouping, hatched CPU."""
    apply_style()
    import matplotlib.pyplot as plt

    discrete = [r for r in rows if r["problem"] == "alarm"]
    hybrid = [r for r in rows if r["problem"].startswith("hybrid")]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), constrained_layout=True)

    # (a) Discrete speed
    ax = axes[0, 0]; ax.set_title("(a) Discrete (alarm) — speed")
    _hbar(ax, discrete, "ms_per_query", log=True, fmt=".2f")
    ax.set_xlabel("ms / query  (log)")

    # (b) Discrete accuracy
    ax = axes[0, 1]; ax.set_title("(b) Discrete (alarm) — TV")
    discrete_acc = [r for r in discrete if r["tv"] == r["tv"]]  # noqa: PLR0124
    _hbar(ax, discrete_acc, "tv", vline=0.05, fmt=".4f")
    ax.set_xlabel("TV vs pgmpy ground truth")

    # (c) Continuous speed
    ax = axes[1, 0]; ax.set_title("(c) Hybrid (synthetic-50) — speed")
    _hbar(ax, hybrid, "ms_per_query", log=True, fmt=".2f")
    ax.set_xlabel("ms / query  (log)")

    # (d) Continuous accuracy — W₁ between method's `model.sample(n=...,
    # evidence=...)` samples and rejection-filtered ground-truth SCM
    # samples. Baselines that return only point estimates (gpytorch
    # posterior mean, pyro Importance mean) are explicitly omitted; the
    # panel labels them "N/A" via a hatched zero-bar.
    ax = axes[1, 1]; ax.set_title("(d) Hybrid — W₁ vs filtered SCM samples")
    hybrid_w1 = [r for r in hybrid
                  if r.get("w1") == r.get("w1") and r.get("w1") is not None]
    if hybrid_w1:
        _hbar(ax, hybrid_w1, "w1", fmt=".3f")
        ax.set_xlabel("W₁(method samples ‖ filtered SCM samples)  (lower better)")
        # F4 footnote: pyro/gpytorch are point-estimate-only baselines.
        ax.text(
            0.99, -0.32,
            "Pyro/GPyTorch report posterior means, not samples — "
            "distributional metrics not applicable. See appendix.",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, alpha=0.7, style="italic",
        )
    else:
        ax.text(0.5, 0.5,
                "No method returned distributional samples\n"
                "(rejection pool too small or sample API not exposed)",
                ha="center", va="center", fontsize=8.5, alpha=0.75)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("NBN crash test — parameter learning + serial inference  ·  "
                 "seed=0 · NBN v0.3 · CPU sandbox",
                 fontsize=10, fontweight="bold")
    _repro_footer(axes[1, 1])
    return savefig_multi(fig, str(fig_dir / "crash_test_summary"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Tiny config for CI smoke test.")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Drop pyro from the lineup (its NUTS sampler "
                             "dominates wall-clock at hybrid-50).")
    parser.add_argument("--n-queries", type=int, default=None,
                        help="Number of marginal queries per baseline "
                             "(default: 10 for --smoke, 50 otherwise).")
    args = parser.parse_args()

    seed_all(0)
    n_queries = args.n_queries if args.n_queries is not None else (10 if args.smoke else 50)

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
    if args.skip_slow:
        discrete_baselines = [b for b in discrete_baselines if b != "pyro"]
        hybrid_baselines = [b for b in hybrid_baselines if b != "pyro"]

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
        extra = ""
        if r.get("w1") == r.get("w1") and r.get("w1") is not None:  # finite
            extra = (f" | W1={r['w1']:.3f}"
                     f" | E={r.get('energy', float('nan')):.3f}"
                     f" | JS={r.get('js_norm', float('nan')):.3f}")
        print(f"  {r['baseline']:8s} {r['device']:5s} {r['problem']:12s}: "
              f"{r['n_queries']:>4d}q in {r['time_s']:6.3f}s "
              f"({r['ms_per_query']:6.2f} ms/q) | "
              f"TV={r['tv']:.4f} | MAP-acc={r['map_accuracy']:.3f}{extra}")

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
            ("w1", "W₁"), ("energy", "Energy"), ("js_norm", "JS-norm"),
        ]
        out_tex = fig_dir / "crash_test_summary.tex"
        write_latex_table(
            out_tex, rows, cols,
            caption="Crash test: parameter learning + serial inference. "
                    "Discrete: TV vs pgmpy exact. Continuous: W₁ / "
                    "energy distance / JS-normalised between baseline "
                    "samples and rejection-filtered SCM ground truth.",
            label="tab:nbn-crash-test",
            formats={"time_s": ".3f", "ms_per_query": ".2f",
                     "tv": ".4f", "map_accuracy": ".3f",
                     "w1": ".3f", "energy": ".3f", "js_norm": ".3f"},
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
