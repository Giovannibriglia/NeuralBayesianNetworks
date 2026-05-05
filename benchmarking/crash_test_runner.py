"""Orchestrator for the v0.5 synthetic-BN crash tests.

Two public entry points used by the ``nbn-bench`` CLI:

* ``run_parameter_learning(config_path)`` — per ``(family, n_nodes,
  seed)`` cell, generate a synthetic BN, fit a fresh NBN on its
  training data, and compute per-node accuracy of the fitted CPDs vs
  the true CPDs.  Speed is *not* measured.
* ``run_inference(config_path)`` — per ``(family, n_nodes, seed)``
  cell, build a query battery against the *true* generative model and
  time how long it takes each baseline to answer all
  ``n_queries_per_cell`` queries.

Workload contract (PR-A §3.5)
-----------------------------
NBN engines use ``query_batch(targets, evidence_batch)`` once with
``B = nbn_batch_size``; non-NBN baselines loop ``B`` times calling
``query`` per row in Python.  Both report ``total_time_s`` for the
same workload; NBN gets to amortise via batched einsum, baselines pay
per-call Python overhead.

Data-layer correctness (NaN-with-ok rows, accuracy plumbing for
inference) is intentionally left as v0.5b PR-B work; this runner is
the *structure* PR.
"""
from __future__ import annotations

import logging
import statistics
import time
from typing import Dict, List

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


# ---------------------------------------------------------------------- #
# Public entry points
# ---------------------------------------------------------------------- #


def run_parameter_learning(
    config_path: str, *, device: str = "auto", verbose: bool = False,
) -> int:
    cfg = CrashTestConfig.from_yaml(config_path)
    if device != "auto":
        cfg.device = device
    logger.info("parameter-learning crash test · prefix=%s · device=%s",
                cfg.output_prefix, cfg.device)

    rows: List[CellResult] = []
    for family in cfg.families:
        for n in cfg.n_nodes:
            for s in cfg.seeds:
                for b in cfg.baselines:
                    rows.extend(run_with_guard(
                        lambda f=family, n=n, s=s, b=b:
                            _param_learning_cell(cfg, f, n, s, b),
                        family=family, n_nodes=n, seed=s, baseline=b,
                        timeout_s=cfg.per_cell_timeout_s,
                    ))

    write_parquet(rows, cfg.parquet_path())
    _render_two_figures(rows, cfg)
    return 0


def run_inference(
    config_path: str, *, device: str = "auto", verbose: bool = False,
) -> int:
    cfg = CrashTestConfig.from_yaml(config_path)
    if device != "auto":
        cfg.device = device
    logger.info("inference crash test · prefix=%s · device=%s",
                cfg.output_prefix, cfg.device)

    rows: List[CellResult] = []
    for family in cfg.families:
        for n in cfg.n_nodes:
            for s in cfg.seeds:
                for b in cfg.baselines:
                    rows.extend(run_with_guard(
                        lambda f=family, n=n, s=s, b=b:
                            _inference_cell(cfg, f, n, s, b),
                        family=family, n_nodes=n, seed=s, baseline=b,
                        timeout_s=cfg.per_cell_timeout_s,
                    ))

    write_parquet(rows, cfg.parquet_path())
    _render_two_figures(rows, cfg)
    return 0


# ---------------------------------------------------------------------- #
# Per-cell workers
# ---------------------------------------------------------------------- #


def _generate_bn(cfg: CrashTestConfig, family: str, n: int, seed: int) -> SyntheticBN:
    return make_synthetic_bn(
        family=family, n_nodes=n,
        edge_density=cfg.edge_density,
        max_in_degree=cfg.max_in_degree,
        cardinality=cfg.cardinality,
        fraction_continuous=cfg.fraction_continuous,
        n_train=cfg.n_train,
        n_test=cfg.n_test,
        n_reference=cfg.n_reference,
        seed=seed, device=cfg.device,
    )


def _param_learning_cell(
    cfg: CrashTestConfig, family: str, n: int, seed: int, baseline: str,
) -> List[CellResult]:
    """Fit a fresh model with `baseline`, measure accuracy vs truth."""
    bn = _generate_bn(cfg, family, n, seed)
    if baseline == "nbn":
        metric, value = _fit_and_score_nbn(cfg, bn, family)
    elif baseline == "pgmpy":
        metric, value = _fit_and_score_pgmpy(bn, family)
    else:
        # pomegranate / gpytorch / pyro accuracy plumbing lands in v0.5b.
        raise NotImplementedError(
            f"param-learning baseline {baseline!r} accuracy wiring is v0.5b",
        )
    return [CellResult(
        family=family, n_nodes=n, seed=seed, baseline=baseline,
        metric=metric, value=value,
    )]


def _inference_cell(
    cfg: CrashTestConfig, family: str, n: int, seed: int, baseline: str,
) -> List[CellResult]:
    """Time `B` queries against the *true* model under the workload contract."""
    bn = _generate_bn(cfg, family, n, seed)
    B = cfg.nbn_batch_size or cfg.n_queries_per_cell
    queries_batch = _build_query_batch(bn, B=B, seed=seed)

    if baseline.startswith("nbn"):
        total_time_s = _time_nbn_inference(bn, baseline, queries_batch)
    elif baseline in {"pgmpy", "pomegranate", "gpytorch", "pyro"}:
        total_time_s = _time_loop_inference(bn, baseline, queries_batch)
    else:
        raise NotImplementedError(
            f"inference baseline {baseline!r} not registered",
        )

    return [
        CellResult(
            family=family, n_nodes=n, seed=seed, baseline=baseline,
            metric="total_time_s", value=float(total_time_s),
        ),
        # accuracy plumbing is v0.5b; emit NaN so the figure renders the
        # panel honestly (PR-B fills in real values).
        CellResult(
            family=family, n_nodes=n, seed=seed, baseline=baseline,
            metric="accuracy", value=float("nan"),
        ),
    ]


# ---------------------------------------------------------------------- #
# Parameter-learning workers
# ---------------------------------------------------------------------- #


def _fit_and_score_nbn(
    cfg: CrashTestConfig, bn: SyntheticBN, family: str,
) -> tuple[str, float]:
    from nbn.core.network import NeuralBayesianNetwork
    fitted = NeuralBayesianNetwork(
        list(bn.dag.edges()), variables=bn.variable_specs, device="cpu",
    )
    for node in bn.dag.nodes():
        kind = bn.variable_specs[node][0]
        fitted.set_mechanism(node, fresh_mechanism_for(family, kind))
    fitted.fit(
        bn.train_data, epochs=cfg.fit_epochs, batch_size=cfg.batch_size, lr=5e-3,
    )
    if family == "discrete":
        return "tv_per_node", _avg_tv_per_node(fitted, bn)
    return "w1_per_node", _avg_w1_per_node(fitted, bn, n_eval=20, n_samples=200)


def _fit_and_score_pgmpy(bn: SyntheticBN, family: str) -> tuple[str, float]:
    from benchmarking.baselines.pgmpy_adapter import PgmpyAdapter
    from benchmarking.domains.base import BenchmarkProblem
    problem = BenchmarkProblem(
        name=bn.name, dag=list(bn.dag.edges()), variables=bn.variable_specs,
        train_data=bn.train_data, test_data=bn.test_data, queries=[],
    )
    adapter = PgmpyAdapter()
    adapter.fit(problem)
    if adapter.kind == "unsupported":
        raise NotImplementedError(f"pgmpy refused the {family} family")
    if family == "discrete":
        return "tv_per_node", _avg_tv_per_node_pgmpy(adapter, bn)
    return "w1_per_node", _avg_w1_per_node_pgmpy_lg(adapter, bn, n_eval=20, n_samples=200)


def _avg_tv_per_node(fitted, bn: SyntheticBN) -> float:
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
    import numpy as np
    out = []
    for cpd in adapter.model.get_cpds():
        node = cpd.variable
        true_logits = bn.true_model.mechanisms[node]._logits.detach()
        true_p = torch.softmax(true_logits, dim=-1)
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
    import networkx as nx
    out = []
    test = bn.test_data
    n_test = next(iter(test.values())).shape[0]
    n_pa_eval = min(n_eval, n_test)
    for node in nx.topological_sort(bn.dag):
        if bn.variable_specs[node][0] == "discrete":
            continue
        parents = list(bn.dag.predecessors(node))
        true_mech = bn.true_model.mechanisms[node]
        fit_mech = fitted.mechanisms[node]
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
    a, _ = torch.sort(a.reshape(-1).float())
    b, _ = torch.sort(b.reshape(-1).float())
    n = min(a.shape[0], b.shape[0])
    return float((a[:n] - b[:n]).abs().mean().item())


# ---------------------------------------------------------------------- #
# Inference workers (workload contract per §3.5)
# ---------------------------------------------------------------------- #


def _build_query_batch(bn: SyntheticBN, *, B: int, seed: int):
    """Construct a single ``Query`` with ``[B]`` evidence values per evidence node."""
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
    return Query(targets=(target,), evidence=evidence, kind="marginal")


def _time_nbn_inference(bn: SyntheticBN, baseline: str, q) -> float:
    """One batched call per timed run; report median total time."""
    from nbn.inference.hybrid import HybridRouter
    from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
    from nbn.inference.tensor_ve import TensorVariableElimination
    if baseline == "nbn_ve":
        eng = TensorVariableElimination()
    elif baseline == "nbn_lw":
        eng = LikelihoodWeightingEngine(n_samples=512)
    elif baseline == "nbn_hybrid":
        eng = HybridRouter()
    else:
        raise NotImplementedError(f"unknown nbn variant {baseline!r}")

    discrete_nodes = {n for n, (k, _) in bn.variable_specs.items() if k == "discrete"}
    ev: Dict[str, torch.Tensor] = {}
    for k, v in q.evidence.items():
        ev[k] = v.long() if k in discrete_nodes else v

    # Warmup
    for _ in range(3):
        eng.query_batch(bn.true_model, list(q.targets), ev)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        eng.query_batch(bn.true_model, list(q.targets), ev)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _time_loop_inference(bn: SyntheticBN, baseline: str, q) -> float:
    """Loop ``B`` times with per-row ``query``; report median total time."""
    from benchmarking.baselines import get_adapter
    from benchmarking.domains.base import BenchmarkProblem, Query
    problem = BenchmarkProblem(
        name=bn.name, dag=list(bn.dag.edges()), variables=bn.variable_specs,
        train_data=bn.train_data, test_data=bn.test_data, queries=[],
    )
    adapter = get_adapter(baseline)
    adapter.fit(problem)
    if hasattr(adapter, "kind") and adapter.kind == "unsupported":
        raise NotImplementedError(
            f"{baseline} cannot handle this family",
        )

    B = next(iter(q.evidence.values())).shape[0]

    def _row_query(i: int) -> Query:
        return Query(
            targets=q.targets,
            evidence={k: v[i] for k, v in q.evidence.items()},
            kind=q.kind,
        )

    # Warmup
    for i in range(min(3, B)):
        adapter.query(_row_query(i))

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for i in range(B):
            adapter.query(_row_query(i))
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ---------------------------------------------------------------------- #
# Figure rendering
# ---------------------------------------------------------------------- #


def _render_two_figures(rows: List[CellResult], cfg: CrashTestConfig) -> None:
    """Render the two canonical figures: total_time_vs_size + accuracy_vs_size."""
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame([r.__dict__ for r in rows])

    if cfg.mode == "parameter_learning":
        accuracy_metrics = ("tv_per_node", "w1_per_node")
        accuracy_label = "per-node TV (discrete) / W1 (continuous)  (lower better)"
    else:
        accuracy_metrics = ("accuracy",)
        accuracy_label = "Wasserstein-1 / TV (lower is better)"

    # Figure 1: total time (only inference mode has timing rows; for
    # parameter-learning we still render an empty figure so the file
    # exists with the canonical name).
    fig1, ax_grid1 = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    plot_metric_vs_n_nodes(
        df, metric="total_time_s",
        ax_grid=ax_grid1.flatten(), fig=fig1,
        metric_label="total time for B queries (s, lower better)",
        log_y=True, log_x=True,
    )
    fig1.text(
        0.99, 0.005,
        reproducibility_footer(version="v0.5", seed=cfg.seeds[0], device=cfg.device),
        ha="right", va="bottom", fontsize=6, color="gray",
    )
    fig1.suptitle(
        f"{cfg.mode} · total time vs network size",
        fontsize=12, fontweight="bold",
    )
    for ext in ("pdf", "svg", "png"):
        out = cfg.figure_path("total_time_vs_size", ext=ext)
        fig1.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig1)

    # Figure 2: accuracy
    df_acc = df[df["metric"].isin(accuracy_metrics)].copy()
    df_acc["metric"] = "accuracy"
    fig2, ax_grid2 = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    plot_metric_vs_n_nodes(
        df_acc, metric="accuracy",
        ax_grid=ax_grid2.flatten(), fig=fig2,
        metric_label=accuracy_label,
        log_y=False, log_x=True,
    )
    fig2.text(
        0.99, 0.005,
        reproducibility_footer(version="v0.5", seed=cfg.seeds[0], device=cfg.device),
        ha="right", va="bottom", fontsize=6, color="gray",
    )
    fig2.suptitle(
        f"{cfg.mode} · accuracy vs network size",
        fontsize=12, fontweight="bold",
    )
    for ext in ("pdf", "svg", "png"):
        out = cfg.figure_path("accuracy_vs_size", ext=ext)
        fig2.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig2)


__all__ = ["run_parameter_learning", "run_inference"]
