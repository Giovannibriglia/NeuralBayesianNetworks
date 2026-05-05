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

import networkx as nx
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


# PR-B §A.4: Structurally-invalid (family, baseline) combinations.
# Listed here so the runner skips the cell up-front with a single
# ``not_supported`` row rather than dispatching a doomed call that
# would surface as a ValueError or NotImplementedError downstream.
_NOT_APPLICABLE: set[tuple[str, str]] = {
    ("discrete", "gpytorch"),                      # GP is continuous
    ("continuous_lg", "pomegranate"),              # pomegranate adapter is discrete-only
    ("continuous_nongauss", "pomegranate"),
    ("continuous_lg", "nbn_ve"),                   # exact VE is discrete-only
    ("continuous_nongauss", "nbn_ve"),
    ("continuous_nongauss", "pgmpy"),              # pgmpy LG cannot do non-Gaussian
    ("hybrid", "gpytorch"),                        # hybrid has discrete components
    ("hybrid", "pomegranate"),                     # hybrid has continuous components
    ("hybrid", "pgmpy"),                           # conservative skip; LG mix unsupported
    ("hybrid", "nbn_ve"),                          # hybrid has continuous components
}


# v0.5c bug 2: baselines that produce a valid throughput measurement but
# cannot meaningfully report posterior-quality accuracy because they do
# not condition on evidence at the BN level.  The speed measurement
# runs normally; only the accuracy row is gated.
#
# GPyTorch SVGPs return ``posterior.sample(target_inputs)`` which is the
# *prior* marginal at the target — independent of ``q.evidence``.  Pre-fix,
# this scored as W₁ ≈ 1.0 flat-line on continuous_lg / continuous_nongauss
# (same signature as the v0.5b round-1 LW-uniform-weights bug we
# eliminated, but here it is the baseline's own structural limit, not a
# runner bug).  Plotting that line would mislead a reader; reporting
# ``not_supported`` for accuracy is honest.
_ACCURACY_NOT_APPLICABLE: set[tuple[str, str]] = {
    ("continuous_lg", "gpytorch"),
    ("continuous_nongauss", "gpytorch"),
}


def _not_applicable_row(
    family: str, n_nodes: int, seed: int, baseline: str,
) -> List[CellResult]:
    """Single-row early-skip with a clear reason, written for both
    parameter-learning (single metric) and inference (two metrics)."""
    msg = f"{baseline} not applicable to {family}"
    return [
        CellResult(
            family=family, n_nodes=n_nodes, seed=seed, baseline=baseline,
            metric="status", value=float("nan"), status="not_supported",
            extra={"error_msg": msg},
        ),
    ]


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
    if (family, baseline) in _NOT_APPLICABLE:
        return _not_applicable_row(family, n, seed, baseline)
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
    if (family, baseline) in _NOT_APPLICABLE:
        return _not_applicable_row(family, n, seed, baseline)
    bn = _generate_bn(cfg, family, n, seed)
    # PR-B-round-2 §3 fix: discrete family has no ``ground_truth_samples``
    # by design; synthesise one reference pool per cell so the accuracy
    # filter has something to work with.  Cached on the SyntheticBN so
    # subsequent queries within the cell reuse it (the dataclass is
    # frozen, so we use object.__setattr__).
    if bn.ground_truth_samples is None:
        try:
            n_ref = min(5000, max(1000, cfg.n_reference // 2))
            with torch.no_grad():
                ref = bn.true_model.sample(n=n_ref)
            # v0.6a: cat columns in ``bn.column_order`` (the canonical
            # topological-sort order established in
            # ``make_synthetic_bn``).  PR #14 round-4 documented why
            # this must not be ``list(bn.dag.nodes())`` (insertion
            # order ≠ topological sort for non-trivial DAGs).
            cached = torch.cat(
                [ref[nm].reshape(n_ref, -1).float().cpu() for nm in bn.column_order],
                dim=-1,
            )
            object.__setattr__(bn, "ground_truth_samples", cached)
        except Exception:  # pragma: no cover  (best-effort)
            pass

    B = cfg.nbn_batch_size or cfg.n_queries_per_cell
    queries_batch = _build_query_batch(bn, B=B, seed=seed)

    if baseline.startswith("nbn"):
        total_time_s = _time_nbn_inference(
            bn, baseline, queries_batch, n_lw_samples=cfg.nbn_lw_n_samples,
        )
    elif baseline in {"pgmpy", "pomegranate", "gpytorch", "pyro"}:
        total_time_s = _time_loop_inference(bn, baseline, queries_batch)
    else:
        raise NotImplementedError(
            f"inference baseline {baseline!r} not registered",
        )

    # PR-B §B.2 — accuracy plumbing.  Compute distributional accuracy
    # against ground-truth samples filtered by evidence.  Best-effort:
    # if a baseline doesn't expose a usable posterior on this family
    # (or filtering leaves too few effective samples), emit a
    # ``no_result`` status row rather than NaN-with-ok (which the
    # acceptance gate forbids).
    rows: list[CellResult] = [
        CellResult(
            family=family, n_nodes=n, seed=seed, baseline=baseline,
            metric="total_time_s", value=float(total_time_s),
        ),
    ]
    # v0.5c bug 2: gate accuracy for baselines that are speed-valid but
    # cannot condition on evidence at the BN level (currently gpytorch
    # on continuous families).  Speed measurement above already ran;
    # accuracy is honest ``not_supported`` rather than a misleading
    # W₁ ≈ 1.0 flat-line that looks like a real benchmark result.
    if (family, baseline) in _ACCURACY_NOT_APPLICABLE:
        rows.append(CellResult(
            family=family, n_nodes=n, seed=seed, baseline=baseline,
            metric="accuracy", value=float("nan"), status="not_supported",
            extra={"error_msg": (
                f"{baseline} cannot condition on evidence at the BN level; "
                f"accuracy comparison is not meaningful"
            )},
        ))
        return rows

    accuracy = _compute_inference_accuracy(
        bn, baseline, queries_batch, family,
        n_lw_samples=cfg.nbn_lw_n_samples,
    )
    if isinstance(accuracy, float) and accuracy == accuracy:  # not NaN
        rows.append(CellResult(
            family=family, n_nodes=n, seed=seed, baseline=baseline,
            metric="accuracy", value=float(accuracy),
        ))
    else:
        rows.append(CellResult(
            family=family, n_nodes=n, seed=seed, baseline=baseline,
            metric="accuracy", value=float("nan"), status="no_result",
            extra={"error_msg": "accuracy: filter or posterior unavailable"},
        ))
    return rows


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
        parents = list(bn.dag.predecessors(node))
        parent_kinds = [bn.variable_specs[p] for p in parents]
        fitted.set_mechanism(node, fresh_mechanism_for(
            family, kind,
            parent_kinds=parent_kinds, cardinality=cfg.cardinality,
        ))
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
        # device-align both tensors to CPU before subtraction (Bug 1):
        # NBN's fitted model lives on cuda when device='auto'; the truth
        # model may live on cpu (or vice-versa).  Normalise to cpu for
        # the metric — these are tiny tensors so the move is free.
        true_logits = bn.true_model.mechanisms[node]._logits.detach().cpu()
        fit_logits = fitted.mechanisms[node]._logits.detach().cpu()
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
        # Both tensors normalised to cpu (Bug 1).
        true_logits = bn.true_model.mechanisms[node]._logits.detach().cpu()
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
        # Place the parent tensor on the FITTED model's device so
        # mech.sample(pa) doesn't trip on device mismatch (Bug 1).
        # _w1 then operates on cpu copies of the resulting samples.
        fit_device = next(
            (p.device for p in fit_mech.parameters() if p is not None),
            torch.device("cpu"),
        )
        if parents:
            pa = torch.cat(
                [test[p][:n_pa_eval].reshape(n_pa_eval, -1).float()
                 for p in parents], dim=-1,
            ).to(fit_device)
        else:
            pa = None
        with torch.no_grad():
            try:
                true_pa = pa.to(next(true_mech.parameters()).device) if pa is not None else None
                true_s = true_mech.sample(true_pa, n=n_samples).reshape(n_pa_eval, n_samples)
                fit_s = fit_mech.sample(pa, n=n_samples).reshape(n_pa_eval, n_samples)
            except Exception:
                continue
        for i in range(n_pa_eval):
            out.append(_w1(true_s[i], fit_s[i]))
    return float(sum(out) / max(1, len(out)))


def _avg_w1_per_node_pgmpy_lg(
    adapter, bn: SyntheticBN, *, n_eval: int, n_samples: int,
) -> float:
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
        true_device = next(true_mech.parameters()).device
        for i in range(n_pa_eval):
            if parents:
                # parents come from bn.test_data (may be on cuda).
                # Move to cpu for the pgmpy-side arithmetic, then put
                # the tensor for true_mech.sample on its native device.
                pa_vals = torch.tensor(
                    [float(test[p][i].reshape(-1)[0].cpu().item()) for p in parents],
                ).float()
                mean = beta[0] + sum(beta[k + 1] * pa_vals[k].item() for k in range(len(parents)))
                fit_s = mean + std * torch.randn(n_samples)
                pa_t = pa_vals.reshape(1, -1).to(true_device)
            else:
                fit_s = beta[0] + std * torch.randn(n_samples)
                pa_t = None
            with torch.no_grad():
                true_s = true_mech.sample(pa_t, n=n_samples).reshape(-1)[:n_samples]
            out.append(_w1(true_s, fit_s))
    return float(sum(out) / max(1, len(out)))


def _w1(a: torch.Tensor, b: torch.Tensor) -> float:
    # Bug 1: device-align before subtraction.
    a, _ = torch.sort(a.reshape(-1).float().cpu())
    b, _ = torch.sort(b.reshape(-1).float().cpu())
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


def _time_nbn_inference(
    bn: SyntheticBN, baseline: str, q, *, n_lw_samples: int = 512,
) -> float:
    """One batched call per timed run; report median total time."""
    from nbn.inference.hybrid import HybridRouter
    from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
    from nbn.inference.tensor_ve import TensorVariableElimination
    if baseline == "nbn_ve":
        eng = TensorVariableElimination()
    elif baseline == "nbn_lw":
        eng = LikelihoodWeightingEngine(n_samples=n_lw_samples)
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
    """Render the canonical figure(s) for the current mode.

    PR-B-round-2 §1: parameter-learning is *accuracy only* per the v0.4
    spec ("don't check the speed, just check metrics about accuracy").
    Inference renders both total-time and accuracy.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame([r.__dict__ for r in rows])

    if cfg.mode == "parameter_learning":
        accuracy_metrics = ("tv_per_node", "w1_per_node")
        accuracy_label = "per-node TV / W1"
        accuracy_suffix = "(lower better)"
    else:
        accuracy_metrics = ("accuracy",)
        accuracy_label = "TV / Wasserstein-1"
        accuracy_suffix = "(lower better)"

    # Figure 1: total time (inference mode only).
    if cfg.mode == "inference":
        fig1, ax_grid1 = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        plot_metric_vs_n_nodes(
            df, metric="total_time_s",
            ax_grid=ax_grid1.flatten(), fig=fig1,
            metric_label="total time for B queries (s), lower better",
            log_y=True, log_x=True,
        )
        fig1.text(
            0.99, 0.005,
            reproducibility_footer(version="v0.5", seed=cfg.seeds[0], device=cfg.device),
            ha="right", va="bottom", fontsize=7, color="gray",
            transform=fig1.transFigure,
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
    df_acc = df[df["metric"].isin(accuracy_metrics)
                | (df["metric"] == "status")].copy()
    # Renaming non-status accuracy metrics to a single 'accuracy' column
    # lets the plotter pick them up uniformly across families.
    metric_map = dict.fromkeys(accuracy_metrics, "accuracy")
    df_acc["metric"] = df_acc["metric"].map(metric_map).fillna(df_acc["metric"])
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
        ha="right", va="bottom", fontsize=7, color="gray",
        transform=fig2.transFigure,
    )
    fig2.suptitle(
        f"{cfg.mode} · accuracy vs network size  {accuracy_suffix}",
        fontsize=12, fontweight="bold",
    )
    for ext in ("pdf", "svg", "png"):
        out = cfg.figure_path("accuracy_vs_size", ext=ext)
        fig2.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig2)


# ---------------------------------------------------------------------- #
# Inference accuracy (PR-B §B.2)
# ---------------------------------------------------------------------- #


def _compute_inference_accuracy(
    bn: SyntheticBN, baseline: str, q, family: str,
    *, n_samples: int = 200, eps_factor: float = 0.50, n_eff_min: int = 10,
    n_lw_samples: int = 512, n_oracle_samples: int = 2000,
) -> float:
    """Average distributional metric over the query battery.

    Round-4 oracle change (PR #14)
    ------------------------------
    Continuous + hybrid targets now use **forward-with-clamp** ancestral
    sampling from ``bn.true_model`` as the ground-truth oracle.  The
    previous ε-ball rejection filter was correct for evidence at the
    marginal mean but *biased* on multi-evidence queries with non-zero
    evidence — the round-3 closed-form LG proof showed the analytic
    posterior agreed with LW (W₁ < 0.02) while ε-ball rejection
    disagreed by ≈1.4σ.  Forward-with-clamp generates exact samples
    from ``p(target | evidence)`` for every SCM family this generator
    produces.

    Discrete targets continue to use exact-match rejection on
    ``bn.ground_truth_samples`` (synthesising the pool on demand for
    the discrete family); that path is unaffected by the bias because
    discrete-evidence rejection is exact-equality, not band-filtering.
    """
    target = q.targets[0]
    target_kind = bn.variable_specs[target][0]
    # v0.6a: ``bn.column_index`` is the canonical name → column-index
    # map (single source of truth for the topological-sort schema
    # established by ``make_synthetic_bn``).  PR #14 round-4 root cause
    # was indexing via insertion order — never recompute the topo sort
    # from ``bn.dag.nodes()`` for column lookups.
    target_idx = bn.column_index(target)
    B = next(iter(q.evidence.values())).shape[0]

    # Ensure we have a discrete-family ground-truth pool for exact-match
    # rejection; the round-2 fix synthesised this on demand.
    if target_kind == "discrete" and bn.ground_truth_samples is None:
        try:
            n_ref = 5000
            with torch.no_grad():
                ref = bn.true_model.sample(n=n_ref)
            cached = torch.cat(
                [ref[nm].reshape(n_ref, -1).float().cpu() for nm in bn.column_order],
                dim=-1,
            )
            object.__setattr__(bn, "ground_truth_samples", cached)
        except Exception:  # pragma: no cover
            return float("nan")

    accs: list[float] = []
    for i in range(min(B, 16)):  # cap accuracy work to keep smoke fast
        ev_row = {k: v[i] for k, v in q.evidence.items()}
        try:
            pred = _baseline_posterior_for_query(
                bn, baseline, target, ev_row, target_kind,
                n_samples=n_samples, n_lw_samples=n_lw_samples,
            )
        except Exception:
            continue
        if pred is None:
            continue

        if target_kind == "discrete":
            # Discrete: exact-match rejection on the cached pool.
            gt_target = _filter_ground_truth(
                bn, ev_row, target_idx,
                eps_factor=eps_factor, n_eff_min=n_eff_min,
            )
            if gt_target is None:
                continue
            k = bn.variable_specs[target][1]
            empirical = torch.zeros(k)
            empirical.scatter_add_(
                0, gt_target.long().clamp(0, k - 1),
                torch.ones_like(gt_target, dtype=torch.float),
            )
            empirical = empirical / empirical.sum().clamp_min(1e-12)
            tv = 0.5 * (pred.cpu().reshape(-1)[:k] - empirical).abs().sum().item()
            accs.append(tv)
        else:
            # Continuous / hybrid-continuous-target: forward-with-clamp.
            try:
                oracle = _forward_with_clamp_posterior_samples(
                    bn, [target], ev_row, n_samples=n_oracle_samples,
                )
            except Exception:
                continue
            if oracle is None or oracle.shape[0] < 100:
                continue
            accs.append(_w1(pred, oracle.reshape(-1)))
    if not accs:
        return float("nan")
    return float(sum(accs) / len(accs))


def _forward_with_clamp_posterior_samples(
    bn: SyntheticBN,
    targets: list[str],
    evidence: Dict[str, torch.Tensor],
    *, n_samples: int = 2000,
) -> torch.Tensor | None:
    """Generate posterior samples by ancestral sampling with evidence clamped.

    For SCMs of the form ``X_j = f_j(X_{pa(j)}) + ε_j`` with ε_j
    independent of ``{X_k : k ≠ j}``, ancestral sampling that *clamps*
    evidence nodes to their observed values (rather than re-sampling
    them) yields exact samples from ``p(T | E=e)``.  This is the same
    machinery ``bn.true_model.sample(n, evidence=...)`` already
    implements — we just reshape evidence to match the engine's
    expected ``[B] / [B, D]`` contract and stack the resulting
    target columns into ``[n_samples, |targets|]``.

    Replaces the v0.5b ε-ball rejection oracle (PR #14 round-3 finding:
    ε-ball is biased on multi-evidence non-zero-mean continuous queries).
    """
    ev = {
        k: (v.reshape(1) if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
        for k, v in evidence.items()
    }
    with torch.no_grad():
        try:
            samples = bn.true_model.sample(n=n_samples, evidence=ev)
        except Exception:
            return None
    cols = []
    for t in targets:
        col = samples[t]
        if col.dim() >= 2 and col.shape[-1] == 1:
            col = col.squeeze(-1)
        cols.append(col.float().cpu().reshape(n_samples, -1))
    return torch.cat(cols, dim=-1)


def _filter_ground_truth(
    bn: SyntheticBN, evidence_row: Dict[str, torch.Tensor], target_idx: int,
    *, eps_factor: float, n_eff_min: int,
) -> torch.Tensor | None:
    """Exact-match rejection on ``bn.ground_truth_samples`` (discrete only).

    Round-4: continuous + hybrid families no longer call this path —
    ``_forward_with_clamp_posterior_samples`` is used instead.  Kept
    here for the discrete exact-equality path (which is correct).
    """
    samples = bn.ground_truth_samples
    if samples is None or samples.numel() == 0:
        return None
    # v0.6a: name → column lookups go through ``bn.column_index`` (the
    # canonical topological-sort schema established in
    # ``make_synthetic_bn``).  See _compute_inference_accuracy.
    # v0.5c bug 1: ``mask`` and per-row arithmetic must live on the
    # samples tensor's device.  When ``bn.ground_truth_samples`` is on
    # cuda (hybrid family during a cuda smoke run) and ``q.evidence``
    # values arrive on cpu (the runner's default for continuous
    # evidence literals built in ``_build_query_batch``), the previous
    # ``mask &= cuda_bool`` raised ``RuntimeError: Expected all tensors
    # to be on the same device``.  Two cells in PR #14's cuda smoke
    # were classified as ``status='error'`` because of this.
    mask = torch.ones(
        samples.shape[0], dtype=torch.bool, device=samples.device,
    )
    for node, val in evidence_row.items():
        idx = bn.column_index(node)
        col = samples[:, idx]
        kind = bn.variable_specs[node][0]
        raw = val.item() if isinstance(val, torch.Tensor) else val
        if kind == "discrete":
            mask &= (col.long() == int(raw))
        else:
            v = torch.as_tensor(raw, device=col.device, dtype=col.dtype)
            sigma = col.std().clamp_min(1e-3)
            mask &= (col - v).abs() < eps_factor * sigma
    if int(mask.sum().item()) < n_eff_min:
        return None
    return samples[mask, target_idx].cpu().float()


def _baseline_posterior_for_query(
    bn: SyntheticBN, baseline: str, target: str,
    ev_row: Dict[str, torch.Tensor], target_kind: str, *, n_samples: int,
    n_lw_samples: int = 512,
) -> torch.Tensor | None:
    """Return posterior samples (continuous) or probability vector (discrete)
    for a single query, per baseline."""
    discrete_nodes = {n for n, (k, _) in bn.variable_specs.items() if k == "discrete"}
    if baseline.startswith("nbn"):
        from nbn.inference.hybrid import HybridRouter
        from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
        from nbn.inference.tensor_ve import TensorVariableElimination
        eng_map = {
            "nbn_ve": TensorVariableElimination(),
            "nbn_lw": LikelihoodWeightingEngine(n_samples=n_lw_samples),
            "nbn_hybrid": HybridRouter(),
        }
        eng = eng_map.get(baseline)
        if eng is None:
            return None
        ev = {
            k: (v.long().reshape(1) if k in discrete_nodes
                else v.float().reshape(1))
            for k, v in ev_row.items()
        }
        with torch.no_grad():
            try:
                out = eng.query(bn.true_model, [target], ev)
            except Exception:
                return None
        if isinstance(out, tuple):
            # PR-B-round-2 §3 (Hypothesis B fix): LW returns
            # ``(normalised_weights, prior_samples)``; the *posterior*
            # is obtained by importance-resampling samples according to
            # weights.  Pre-fix we used ``samples`` directly, which
            # ignored the conditioning and gave prior samples — the
            # source of the flat W₁≈4 we saw on continuous_lg.
            w, samp = out
            w_flat = w.detach().cpu().float().reshape(-1)
            samp_flat = samp.detach().cpu().float().reshape(-1)
            n = min(samp_flat.shape[0], w_flat.shape[0])
            w_flat = w_flat[:n].clamp_min(1e-12)
            w_flat = w_flat / w_flat.sum()
            n_resample = min(2000, n)
            try:
                idx = torch.multinomial(w_flat, n_resample, replacement=True)
                return samp_flat[idx]
            except Exception:
                return samp_flat
        out = out.detach().cpu().float().reshape(-1)
        if target_kind == "discrete":
            return out
        return None
    if baseline in {"pgmpy", "pomegranate"}:
        from benchmarking.baselines import get_adapter
        from benchmarking.domains.base import BenchmarkProblem, Query
        problem = BenchmarkProblem(
            name=bn.name, dag=list(bn.dag.edges()),
            variables=bn.variable_specs,
            train_data=bn.train_data, test_data=bn.test_data, queries=[],
        )
        adapter = get_adapter(baseline)
        adapter.fit(problem)
        if hasattr(adapter, "kind") and adapter.kind == "unsupported":
            return None
        try:
            out = adapter.query(Query(targets=(target,), evidence=ev_row, kind="marginal"))
        except Exception:
            return None
        out = out.detach().cpu().float().reshape(-1)
        if target_kind == "discrete":
            return out
        return None
    if baseline == "gpytorch":
        from benchmarking.baselines import get_adapter
        from benchmarking.domains.base import BenchmarkProblem, Query
        problem = BenchmarkProblem(
            name=bn.name, dag=list(bn.dag.edges()),
            variables=bn.variable_specs,
            train_data=bn.train_data, test_data=bn.test_data, queries=[],
        )
        adapter = get_adapter(baseline, device="cpu")
        adapter.fit(problem)
        try:
            samples = adapter.query_batch_samples(
                Query(targets=(target,),
                      evidence={k: v.reshape(1) for k, v in ev_row.items()},
                      kind="marginal"),
                n_samples=n_samples,
            )
        except Exception:
            return None
        if samples.shape[0] == 0 or torch.isnan(samples).any():
            return None
        return samples.detach().cpu().float().reshape(-1)
    return None


__all__ = ["run_parameter_learning", "run_inference"]
