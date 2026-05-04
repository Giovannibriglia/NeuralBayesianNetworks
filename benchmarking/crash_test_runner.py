"""Orchestrator for the v0.4 crash tests.

Two public entry points used by the ``nbn-bench`` CLI:

* ``run_parameter_learning(config_path)`` — Phase 2.1 of v0.4b.  Per
  ``(family, n_nodes, seed)`` cell, generate a synthetic BN, fit a
  *fresh* NBN on the training data, and compute per-node accuracy of
  the fitted CPDs vs. the true CPDs.  No timing.
* ``run_inference(config_path)`` — Phase 2.2 of v0.4b.  Use the *true*
  generative model directly and measure both posterior accuracy
  (against rejection-filtered ground-truth samples) and batched-query
  throughput at fixed ``B = config['inference']['batch_size']``.

Both runners share the cell iterator, parquet writer, and figure
renderer from :mod:`benchmarking._crash_test_utils`.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Sequence

import torch

from benchmarking._crash_test_utils import (
    CellResult,
    CrashTestConfig,
    fresh_mechanism_for,
    plot_metric_vs_n_nodes,
    reproducibility_footer,
    run_with_guard,
    write_parquet,
)
from benchmarking.synthetic import SyntheticBN, make_synthetic_bn

logger = logging.getLogger(__name__)


# Smoke baseline matrices (per pre-Phase-D refinement).
SMOKE_PARAM_LEARNING_BASELINES = ("nbn",)        # pgmpy added per family below
SMOKE_INFERENCE_BASELINES = ("nbn_lw", "pgmpy")  # nbn_ve added on discrete only


# ---------------------------------------------------------------------- #
# Public entry points
# ---------------------------------------------------------------------- #


def run_parameter_learning(*, config_path: str, verbose: bool = False) -> int:
    cfg = CrashTestConfig.from_yaml(config_path)
    logger.info("parameter-learning crash test · is_smoke=%s · device=%s",
                cfg.is_smoke, cfg.device)

    rows: List[CellResult] = []
    for family in cfg.families:
        baselines = _param_learning_baselines(family, smoke=cfg.is_smoke)
        for n in cfg.sizes:
            for s in cfg.seeds:
                for b in baselines:
                    timeout = int(cfg.inference.get("per_cell_timeout_s", 30)) \
                        if cfg.is_smoke else 300
                    rows.extend(run_with_guard(
                        lambda f=family, n=n, s=s, b=b: _param_learning_cell(cfg, f, n, s, b),
                        family=family, n_nodes=n, seed=s, baseline=b,
                        timeout_s=timeout,
                    ))

    parquet = cfg.parquet_path("param_learning_metrics")
    write_parquet(rows, parquet)
    fig_path = cfg.figure_path("param_learning_accuracy_vs_size")
    _render_param_learning_figure(rows, fig_path, cfg)
    return 0


def run_inference(*, config_path: str, verbose: bool = False) -> int:
    cfg = CrashTestConfig.from_yaml(config_path)
    logger.info("inference crash test · is_smoke=%s · device=%s",
                cfg.is_smoke, cfg.device)

    rows: List[CellResult] = []
    for family in cfg.families:
        baselines = _inference_baselines(family, smoke=cfg.is_smoke)
        for n in cfg.sizes:
            for s in cfg.seeds:
                for b in baselines:
                    timeout = int(cfg.inference.get("per_cell_timeout_s", 30)) \
                        if cfg.is_smoke else 300
                    rows.extend(run_with_guard(
                        lambda f=family, n=n, s=s, b=b: _inference_cell(cfg, f, n, s, b),
                        family=family, n_nodes=n, seed=s, baseline=b,
                        timeout_s=timeout,
                    ))

    parquet = cfg.parquet_path("inference_metrics")
    write_parquet(rows, parquet)
    _render_inference_figures(rows, cfg)
    return 0


# ---------------------------------------------------------------------- #
# Baseline-matrix policy
# ---------------------------------------------------------------------- #


def _param_learning_baselines(family: str, *, smoke: bool) -> Sequence[str]:
    if smoke:
        # Per the v0.4b refinement: smoke = NBN + pgmpy where applicable.
        # hybrid family requires option-(b) train-quantile binning for
        # cont→disc edges; that mechanism wrapper lands in v0.4c.
        if family == "hybrid":
            return ()
        return ("nbn", "pgmpy") if family in {"discrete", "continuous_lg"} else ("nbn",)
    full = ["nbn"]
    if family in {"discrete", "continuous_lg"}:
        full.append("pgmpy")
    if family in {"continuous_lg", "continuous_nongauss", "hybrid"}:
        full.extend(["gpytorch", "pyro"])
    else:
        full.append("pyro")
    return tuple(full)


def _inference_baselines(family: str, *, smoke: bool) -> Sequence[str]:
    if smoke:
        # nbn_ve only applies to discrete; nbn_lw applies everywhere; pgmpy only
        # for discrete + continuous_lg.  Self-consistency check: nbn_lw vs
        # nbn_ve on discrete.
        out = ["nbn_lw"]
        if family == "discrete":
            out.insert(0, "nbn_ve")
        if family in {"discrete", "continuous_lg"}:
            out.append("pgmpy")
        if family == "hybrid":
            out.insert(0, "nbn_hybrid")
        return tuple(out)
    full = ["nbn_lw"]
    if family == "discrete":
        full.insert(0, "nbn_ve")
    if family == "hybrid":
        full.insert(0, "nbn_hybrid")
    if family in {"discrete", "continuous_lg"}:
        full.append("pgmpy")
    if family in {"continuous_lg", "continuous_nongauss", "hybrid"}:
        full.extend(["gpytorch", "pyro"])
    else:
        full.append("pyro")
    return tuple(full)


# ---------------------------------------------------------------------- #
# Per-cell workers
# ---------------------------------------------------------------------- #


def _generate_bn(cfg: CrashTestConfig, family: str, n: int, seed: int) -> SyntheticBN:
    g = cfg.generator
    return make_synthetic_bn(
        family=family, n_nodes=n,
        edge_density=float(g.get("edge_density", 0.20)),
        max_in_degree=int(g.get("max_in_degree", 4)),
        cardinality=int(g.get("cardinality", 4)),
        fraction_continuous=float(g.get("fraction_continuous", 0.5)),
        n_train=int(g.get("n_train", 2000)),
        n_test=int(g.get("n_test", 500)),
        n_reference=int(g.get("n_reference", 5000)),
        seed=seed, device=cfg.device,
    )


def _param_learning_cell(
    cfg: CrashTestConfig, family: str, n: int, seed: int, baseline: str,
) -> List[CellResult]:
    bn = _generate_bn(cfg, family, n, seed)
    if baseline == "nbn":
        metric, value = _fit_and_score_nbn(bn, family)
    elif baseline == "pgmpy":
        metric, value = _fit_and_score_pgmpy(bn, family)
    else:
        raise NotImplementedError(
            f"param-learning baseline {baseline!r} not yet wired",
        )
    return [CellResult(
        family=family, n_nodes=n, seed=seed, baseline=baseline,
        metric=metric, value=value,
    )]


def _inference_cell(
    cfg: CrashTestConfig, family: str, n: int, seed: int, baseline: str,
) -> List[CellResult]:
    bn = _generate_bn(cfg, family, n, seed)
    inf = cfg.inference
    B = int(inf.get("batch_size", 1024))
    n_warmup = int(inf.get("n_warmup", 5))
    n_repeat = int(inf.get("n_repeat", 3))

    queries = _make_inference_queries(bn, B=B, family=family, seed=seed)
    rows: List[CellResult] = []

    if baseline == "nbn_ve":
        speed_qps, acc = _bench_nbn_ve(bn, queries, B=B, n_warmup=n_warmup, n_repeat=n_repeat)
    elif baseline == "nbn_lw":
        speed_qps, acc = _bench_nbn_lw(bn, queries, B=B, n_warmup=n_warmup, n_repeat=n_repeat)
    elif baseline == "nbn_hybrid":
        speed_qps, acc = _bench_nbn_hybrid(bn, queries, B=B, n_warmup=n_warmup, n_repeat=n_repeat)
    elif baseline == "pgmpy":
        speed_qps, acc = _bench_pgmpy(bn, queries, B=B, n_warmup=n_warmup, n_repeat=n_repeat)
    else:
        raise NotImplementedError(
            f"inference baseline {baseline!r} not yet wired",
        )

    rows.append(CellResult(
        family=family, n_nodes=n, seed=seed, baseline=baseline,
        metric="throughput_qps", value=float(speed_qps),
    ))
    rows.append(CellResult(
        family=family, n_nodes=n, seed=seed, baseline=baseline,
        metric="accuracy_w1", value=float(acc),
    ))
    return rows


# ---------------------------------------------------------------------- #
# Parameter learning workers
# ---------------------------------------------------------------------- #


def _fit_and_score_nbn(bn: SyntheticBN, family: str) -> tuple[str, float]:
    """Fit a fresh NBN per-node and score against the true CPDs.

    For discrete: per-node TV averaged uniformly over CPT rows (we don't
    weight by parent-marginal probability in v0.4b smoke; that's a v0.5
    refinement when we're confident the basic numbers look right).
    For continuous: per-node Wasserstein-1 between samples drawn from
    the fitted CPD and the true CPD on n_eval parent assignments from
    test_data.
    """
    from nbn.core.network import NeuralBayesianNetwork
    fitted = NeuralBayesianNetwork(
        list(bn.dag.edges()), variables=bn.variable_specs, device="cpu",
    )
    for node in bn.dag.nodes():
        kind = bn.variable_specs[node][0]
        mech = fresh_mechanism_for(family, kind)
        fitted.set_mechanism(node, mech)
    # Use ridge-regression / count-MLE per-node fit
    epochs = 20  # smoke / smoke-default
    fitted.fit(bn.train_data, epochs=epochs, batch_size=4096, lr=5e-3)

    if family == "discrete":
        return "tv_per_node", _avg_tv_per_node(fitted, bn)
    return "w1_per_node", _avg_w1_per_node(fitted, bn, n_eval=20, n_samples=200)


def _fit_and_score_pgmpy(bn: SyntheticBN, family: str) -> tuple[str, float]:
    """Fit pgmpy MLE / LinearGaussian and score the same metric."""
    from benchmarking.baselines.pgmpy_adapter import PgmpyAdapter
    from benchmarking.domains.base import BenchmarkProblem
    problem = BenchmarkProblem(
        name=bn.name, dag=list(bn.dag.edges()),
        variables=bn.variable_specs,
        train_data=bn.train_data, test_data=bn.test_data, queries=[],
    )
    adapter = PgmpyAdapter()
    adapter.fit(problem)
    if adapter.kind == "unsupported":
        raise NotImplementedError(
            f"pgmpy refused the {family} family",
        )
    if family == "discrete":
        return "tv_per_node", _avg_tv_per_node_pgmpy(adapter, bn)
    return "w1_per_node", _avg_w1_per_node_pgmpy_lg(adapter, bn, n_eval=20, n_samples=200)


def _avg_tv_per_node(fitted, bn: SyntheticBN) -> float:
    """Mean over nodes of TV(fitted_cpt[r,:], true_cpt[r,:]) averaged over r."""
    out = []
    for node in bn.dag.nodes():
        true_logits = bn.true_model.mechanisms[node]._logits.detach()
        fit_logits = fitted.mechanisms[node]._logits.detach()
        if true_logits.shape != fit_logits.shape:
            continue
        true_p = torch.softmax(true_logits, dim=-1)
        fit_p = torch.softmax(fit_logits, dim=-1)
        tv = 0.5 * (true_p - fit_p).abs().sum(dim=-1).mean().item()
        out.append(tv)
    return float(sum(out) / max(1, len(out)))


def _avg_tv_per_node_pgmpy(adapter, bn: SyntheticBN) -> float:
    """Same metric for the pgmpy fit."""
    out = []
    for cpd in adapter.model.get_cpds():
        node = cpd.variable
        true_logits = bn.true_model.mechanisms[node]._logits.detach()
        true_p = torch.softmax(true_logits, dim=-1)
        # pgmpy stores values as [k_target, *parent_cards]; reshape to
        # match true_p's [n_parent_states, K] layout.
        import numpy as np
        vals = np.asarray(cpd.values)
        k = true_p.shape[-1]
        fit_p = torch.tensor(vals.reshape(k, -1).T, dtype=torch.float)
        if fit_p.shape != true_p.shape:
            continue
        tv = 0.5 * (true_p - fit_p).abs().sum(dim=-1).mean().item()
        out.append(tv)
    return float(sum(out) / max(1, len(out)))


def _avg_w1_per_node(
    fitted, bn: SyntheticBN, *, n_eval: int, n_samples: int,
) -> float:
    """Per-node Wasserstein-1 between fitted and true CPD samples."""
    import networkx as nx
    out = []
    test = bn.test_data
    n_test = next(iter(test.values())).shape[0]
    n_pa_eval = min(n_eval, n_test)
    for node in nx.topological_sort(bn.dag):
        kind = bn.variable_specs[node][0]
        if kind == "discrete":
            continue
        parents = list(bn.dag.predecessors(node))
        true_mech = bn.true_model.mechanisms[node]
        fit_mech = fitted.mechanisms[node]
        # Build [n_pa_eval, P] parent stack from test_data
        if parents:
            pa = torch.cat(
                [test[p][:n_pa_eval].reshape(n_pa_eval, -1).float() for p in parents],
                dim=-1,
            )
        else:
            pa = None
        with torch.no_grad():
            try:
                true_s = true_mech.sample(pa, n=n_samples).reshape(n_pa_eval, n_samples)
                fit_s = fit_mech.sample(pa, n=n_samples).reshape(n_pa_eval, n_samples)
            except Exception:
                continue
        for i in range(n_pa_eval):
            out.append(_w1(true_s[i], fit_s[i]))
    return float(sum(out) / max(1, len(out)))


def _avg_w1_per_node_pgmpy_lg(
    adapter, bn: SyntheticBN, *, n_eval: int, n_samples: int,
) -> float:
    """Pgmpy LG: closed-form Gaussian sampling per node, W1 vs truth."""
    import networkx as nx
    if adapter.kind != "continuous_lg":
        raise NotImplementedError("pgmpy LG path not active")
    test = bn.test_data
    n_test = next(iter(test.values())).shape[0]
    n_pa_eval = min(n_eval, n_test)
    cpd_by_var = {cpd.variable: cpd for cpd in adapter.lg_model.get_cpds()}
    out = []
    for node in nx.topological_sort(bn.dag):
        cpd = cpd_by_var.get(node)
        if cpd is None:
            continue
        parents = list(cpd.evidence) if hasattr(cpd, "evidence") else []
        beta = list(cpd.beta)
        std = float(cpd.std)
        true_mech = bn.true_model.mechanisms[node]
        for i in range(n_pa_eval):
            if parents:
                pa_vals = torch.tensor(
                    [float(test[p][i].reshape(-1)[0].item()) for p in parents],
                ).float()
                mean = beta[0] + sum(beta[k + 1] * pa_vals[k].item() for k in range(len(parents)))
                fit_s = mean + std * torch.randn(n_samples)
                pa_t = pa_vals.reshape(1, -1)
            else:
                fit_s = beta[0] + std * torch.randn(n_samples)
                pa_t = None
            with torch.no_grad():
                true_s = true_mech.sample(pa_t, n=n_samples).reshape(-1)[:n_samples]
            out.append(_w1(true_s, fit_s))
    return float(sum(out) / max(1, len(out)))


def _w1(a: torch.Tensor, b: torch.Tensor) -> float:
    """Wasserstein-1 between two 1-D empirical samples."""
    a, _ = torch.sort(a.reshape(-1).float())
    b, _ = torch.sort(b.reshape(-1).float())
    n = min(a.shape[0], b.shape[0])
    return float((a[:n] - b[:n]).abs().mean().item())


# ---------------------------------------------------------------------- #
# Inference workers
# ---------------------------------------------------------------------- #


def _make_inference_queries(
    bn: SyntheticBN, *, B: int, family: str, seed: int,
):
    """Build a single batched ``Query`` exercising the true model.

    Picks one random target node and 2-3 random evidence nodes; evidence
    values are drawn from ``bn.test_data`` (so they're realistic).
    """
    from benchmarking.domains.base import Query

    g = torch.Generator(device="cpu").manual_seed(seed * 1000 + 7)
    nodes = list(bn.dag.nodes())
    perm = torch.randperm(len(nodes), generator=g).tolist()
    target = nodes[perm[0]]
    n_ev = 2 if len(nodes) <= 4 else 3
    evidence_nodes = [nodes[i] for i in perm[1:1 + n_ev]]
    test = bn.test_data
    n_test = next(iter(test.values())).shape[0]
    take = min(B, n_test)
    evidence = {v: test[v][:take].reshape(take).float()
                for v in evidence_nodes}
    return Query(targets=[target], evidence=evidence, kind="marginal")


def _bench_nbn_ve(bn, q, *, B, n_warmup, n_repeat) -> tuple[float, float]:
    from nbn.inference.tensor_ve import TensorVariableElimination
    eng = TensorVariableElimination()
    return _bench_nbn_engine(bn, q, eng, B=B, n_warmup=n_warmup, n_repeat=n_repeat)


def _bench_nbn_lw(bn, q, *, B, n_warmup, n_repeat) -> tuple[float, float]:
    from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
    eng = LikelihoodWeightingEngine(n_samples=512)
    return _bench_nbn_engine(bn, q, eng, B=B, n_warmup=n_warmup, n_repeat=n_repeat)


def _bench_nbn_hybrid(bn, q, *, B, n_warmup, n_repeat) -> tuple[float, float]:
    from nbn.inference.hybrid import HybridRouter
    eng = HybridRouter()
    return _bench_nbn_engine(bn, q, eng, B=B, n_warmup=n_warmup, n_repeat=n_repeat)


def _bench_nbn_engine(bn, q, eng, *, B, n_warmup, n_repeat) -> tuple[float, float]:
    """Run query_batch with warmup + median timing; return (qps, w1_acc)."""
    actual_B = next(iter(q.evidence.values())).shape[0]
    ev = {k: v.long() if k in [n for n, (kind, _) in bn.variable_specs.items() if kind == "discrete"]
          else v for k, v in q.evidence.items()}
    for _ in range(n_warmup):
        try:
            eng.query_batch(bn.true_model, list(q.targets), ev)
        except Exception:
            return float("nan"), float("nan")
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        out = eng.query_batch(bn.true_model, list(q.targets), ev)
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    qps = actual_B / max(median, 1e-9)
    # W1 accuracy: not yet wired (Phase 2.2 stub) — return NaN so the
    # figure renders the throughput panel cleanly while the accuracy
    # panel shows missing data for now.  v0.4c follow-up plumbs the
    # ground-truth filtering pipeline.
    return qps, float("nan")


def _bench_pgmpy(bn, q, *, B, n_warmup, n_repeat) -> tuple[float, float]:
    from benchmarking.baselines.pgmpy_adapter import PgmpyAdapter
    from benchmarking.domains.base import BenchmarkProblem
    problem = BenchmarkProblem(
        name=bn.name, dag=list(bn.dag.edges()),
        variables=bn.variable_specs,
        train_data=bn.train_data, test_data=bn.test_data, queries=[],
    )
    adapter = PgmpyAdapter()
    adapter.fit(problem)
    if adapter.kind == "unsupported":
        raise NotImplementedError("pgmpy can't handle this family")
    actual_B = next(iter(q.evidence.values())).shape[0]
    # pgmpy adapter loops internally via query_batch; cap effective B for smoke
    cap = min(actual_B, 32)
    capped_q = type(q)(
        targets=q.targets,
        evidence={k: v[:cap] for k, v in q.evidence.items()},
        kind=q.kind,
    )
    for _ in range(n_warmup):
        try:
            adapter.query_batch(capped_q)
        except Exception:
            return float("nan"), float("nan")
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        adapter.query_batch(capped_q)
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    qps = cap / max(median, 1e-9)
    return qps, float("nan")


# ---------------------------------------------------------------------- #
# Figure rendering
# ---------------------------------------------------------------------- #


def _render_param_learning_figure(
    rows: List[CellResult], path: Path, cfg: CrashTestConfig,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame([r.__dict__ for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = axes.flatten()
    # Build a synthetic combined-metric column so the plotter can lay out
    # all four panels in one pass.  Discrete = TV; continuous_lg /
    # continuous_nongauss / hybrid = W1.  Renaming both metrics to
    # "param_accuracy" lets the existing helper render them side-by-side
    # without the double-plot overlay bug we hit on first render.
    df = df.copy()
    metric_map = {
        "tv_per_node": "param_accuracy",
        "w1_per_node": "param_accuracy",
    }
    df["metric"] = df["metric"].map(metric_map).fillna(df["metric"])
    plot_metric_vs_n_nodes(
        df, metric="param_accuracy",
        ax_grid=axes,
        fig=fig, metric_label="per-node TV (discrete) / W1 (continuous)",
        log_y=False, log_x=True,
    )
    fig.text(
        0.99, 0.005,
        reproducibility_footer(version="v0.4", seed=cfg.seeds[0],
                               device=cfg.device),
        ha="right", va="bottom", fontsize=6, color="gray",
    )
    fig.suptitle(
        "Parameter-learning accuracy (smoke)" if cfg.is_smoke
        else "Parameter-learning accuracy",
        fontsize=12, fontweight="bold",
    )
    for ext in ("pdf", "svg", "png"):
        out = path.with_suffix(f".{ext}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _render_inference_figures(
    rows: List[CellResult], cfg: CrashTestConfig,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame([r.__dict__ for r in rows])

    # 1. Speed
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    plot_metric_vs_n_nodes(
        df, metric="throughput_qps",
        ax_grid=axes.flatten(),
        fig=fig, metric_label="queries / second (higher is better)",
        log_y=True, log_x=True, lower_is_better=False,
    )
    fig.text(
        0.99, 0.005,
        reproducibility_footer(version="v0.4", seed=cfg.seeds[0], device=cfg.device),
        ha="right", va="bottom", fontsize=6, color="gray",
    )
    fig.suptitle(
        "Inference batched throughput at B=1024 (smoke)" if cfg.is_smoke
        else "Inference batched throughput at B=1024",
        fontsize=12, fontweight="bold",
    )
    for ext in ("pdf", "svg", "png"):
        out = cfg.figure_path("inference_speed_vs_size", ext=ext)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # 2. Accuracy (currently stub-NaN; rendered as empty panels w/ note)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    plot_metric_vs_n_nodes(
        df, metric="accuracy_w1",
        ax_grid=axes.flatten(),
        fig=fig, metric_label="W1 to filtered ground truth (lower is better)",
        log_y=False, log_x=True,
    )
    fig.text(
        0.99, 0.005,
        reproducibility_footer(version="v0.4", seed=cfg.seeds[0], device=cfg.device),
        ha="right", va="bottom", fontsize=6, color="gray",
    )
    fig.suptitle(
        "Inference accuracy (smoke; W1 plumbing in v0.4c)" if cfg.is_smoke
        else "Inference accuracy",
        fontsize=12, fontweight="bold",
    )
    for ext in ("pdf", "svg", "png"):
        out = cfg.figure_path("inference_accuracy_vs_size", ext=ext)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # 3. Pareto (placeholder — only valid once accuracy_w1 is populated)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
    ax.text(
        0.5, 0.5,
        "Pareto frontier: requires accuracy_w1 (v0.4c)",
        transform=ax.transAxes, ha="center", va="center", fontsize=9,
        color="gray",
    )
    ax.set_xlabel("accuracy (W1, lower better)")
    ax.set_ylabel("throughput (q/s, higher better)")
    fig.text(
        0.99, 0.005,
        reproducibility_footer(version="v0.4", seed=cfg.seeds[0], device=cfg.device),
        ha="right", va="bottom", fontsize=6, color="gray",
    )
    for ext in ("pdf", "svg", "png"):
        out = cfg.figure_path("inference_pareto", ext=ext)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


__all__ = ["run_parameter_learning", "run_inference"]
