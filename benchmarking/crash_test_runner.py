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
    fresh_mechanism_for_spec,
    run_with_guard,
    write_parquet,
)
from benchmarking._baseline_registry import (
    BaselineSpec,
    _label_from_spec,
    is_applicable,
)
from benchmarking._run_logging import finalise_run_logging, setup_run_logging
from benchmarking.synthetic import SyntheticBN, make_synthetic_bn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# v0.6c-C-1a — schema-v2 spec → legacy adapter shim
# ---------------------------------------------------------------------- #
#
# v0.6c-C-1a introduces structured baseline specs in YAML
# (``{library, mechanism, param_method[, inference_method]}``) and a
# new method-keyed label scheme via ``_label_from_spec``.
#
# The runner refactor that consumes these specs natively (fit-then-
# query semantics, per-method dispatch) lands in v0.6c-C-1b.  In 1a
# we keep the runner's existing string-keyed dispatch + the legacy
# ``_NOT_APPLICABLE`` table; ``_legacy_adapter_for_spec`` translates
# each spec to its legacy-adapter string.  After the cell runs, the
# resulting rows are *relabeled* to use the new
# ``_label_from_spec(spec)`` label in the parquet ``baseline`` column.
#
# This means smoke STATUS counts are invariant in 1a (the dispatch
# mapping is 1-to-1 with v0.6c-B's flat list); only the labels in
# the parquet change.
def _legacy_adapter_for_spec(spec) -> str:
    """Map a v0.6c-C-1a spec to the legacy adapter string used by the
    existing string-keyed dispatch in :func:`_inference_cell` /
    :func:`_param_learning_cell`.

    Accepts either a dict (legacy YAML-loaded form) or a
    :class:`BaselineSpec` (v0.8-#51 method-keyed form).  Raises
    ``ValueError`` on unknown specs — every spec listed in the
    shipping YAML configs has a defined mapping; new specs added in
    v0.6c-C-1b use spec-keyed dispatch and don't go through this shim.
    """
    if isinstance(spec, BaselineSpec):
        library = spec.library
        inference_method = spec.inference_method
    else:
        library = str(spec["library"])
        inference_method = spec.get("inference_method")
    if library == "nbn":
        if inference_method == "ve":
            return "nbn_ve"
        if inference_method == "lw":
            return "nbn_lw"
        if inference_method == "router":
            return "nbn_hybrid"
        if inference_method is None:
            return "nbn"        # parameter-learning
        raise ValueError(
            f"v0.6c-C-1a: unknown inference_method "
            f"{inference_method!r} in nbn spec {spec!r}",
        )
    if library in {"pgmpy", "pomegranate", "gpytorch", "pyro"}:
        return library
    raise ValueError(
        f"v0.6c-C-1a: no legacy adapter mapping for spec {spec!r}; "
        f"new libraries land in v0.6c-C-1b's spec-keyed dispatch.",
    )


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


# v0.8 Pass-9: parametrized per-cell metric scoring config.
_PARAM_LEARNING_METRICS = ("tv", "jsd", "w1")
_DISCRETE_FAMILIES = {"discrete"}
_CONTINUOUS_FAMILIES = {"continuous_lg", "continuous_nongauss"}
_SKIP_REASONS = {
    ("discrete", "w1"): (
        "W1 not meaningful on unordered-categorical discrete CPDs "
        "(collapses to 1/2 * TV under Hamming metric)"
    ),
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
    run_meta = setup_run_logging(cfg)
    logger.info("parameter-learning crash test · prefix=%s · device=%s",
                cfg.output_prefix, cfg.device)

    rows: List[CellResult] = []
    for family in cfg.families:
        for n in cfg.n_nodes:
            for s in cfg.seeds:
                for spec in cfg.baselines:
                    label = _label_from_spec(spec)
                    # v0.6c-C-1b: registry-based applicability gate
                    # (BEFORE adapter dispatch).  Method-keyed labels
                    # like ``pgmpy-mle-ve`` are discrete-only per the
                    # C-1a registry; the legacy ``_NOT_APPLICABLE``
                    # table inside the cell functions is keyed by the
                    # *polymorphic* legacy adapter string ("pgmpy",
                    # "nbn_lw") which gates a different (less strict)
                    # set of combinations.  Without this gate, e.g.,
                    # ``pgmpy-mle-ve`` would dispatch on continuous_lg
                    # via the legacy adapter's lg path and emit a
                    # nonsense ``ok`` row in the parquet.
                    if not is_applicable(label, family):
                        rows.append(_not_applicable_row(
                            family, n, s, label,
                        )[0])
                        continue
                    # v0.8-#51: dispatch on the method-keyed spec, not
                    # the legacy polymorphic string.  The cell now
                    # reads spec.library / spec.mechanism / spec.param_method
                    # directly and produces label-keyed rows; the post-cell
                    # relabel is no longer needed (cell already labels
                    # with the canonical _label_from_spec(spec)).
                    cell_rows = run_with_guard(
                        lambda f=family, nn=n, ss=s, sp=spec:
                            _param_learning_cell(cfg, f, nn, ss, sp),
                        family=family, n_nodes=n, seed=s, baseline=label,
                        timeout_s=cfg.per_cell_timeout_s,
                    )
                    rows.extend(cell_rows)

    write_parquet(rows, cfg.parquet_path())
    _render_two_figures(rows, cfg)
    status_counts: dict[str, int] = {}
    for r in rows:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    finalise_run_logging(run_meta, status_summary=status_counts)
    return 0


def run_inference(
    config_path: str, *, device: str = "auto", verbose: bool = False,
) -> int:
    cfg = CrashTestConfig.from_yaml(config_path)
    if device != "auto":
        cfg.device = device
    run_meta = setup_run_logging(cfg)
    logger.info("inference crash test · prefix=%s · device=%s",
                cfg.output_prefix, cfg.device)

    rows: List[CellResult] = []
    for family in cfg.families:
        for n in cfg.n_nodes:
            for s in cfg.seeds:
                for spec in cfg.baselines:
                    label = _label_from_spec(spec)
                    # v0.6c-C-1b: registry-based applicability gate
                    # BEFORE adapter dispatch.  See run_parameter_learning
                    # for the rationale.  Without this, ``pgmpy-mle-ve``
                    # would dispatch on continuous_lg via the legacy
                    # adapter's lg path and emit a nonsense ``ok`` row;
                    # ``nbn-cat-lw`` would dispatch on hybrid and trip
                    # a cuda device-side assert during sampling.
                    if not is_applicable(label, family):
                        rows.append(_not_applicable_row(
                            family, n, s, label,
                        )[0])
                        continue
                    legacy = _legacy_adapter_for_spec(spec)
                    cell_rows = run_with_guard(
                        lambda f=family, nn=n, ss=s, lg=legacy:
                            _inference_cell(cfg, f, nn, ss, lg),
                        family=family, n_nodes=n, seed=s, baseline=label,
                        timeout_s=cfg.per_cell_timeout_s,
                    )
                    # v0.6c-C-1a relabel: cells use the legacy adapter
                    # string internally; replace it with the registry's
                    # canonical label for the parquet ``baseline`` column.
                    for r in cell_rows:
                        r.baseline = label
                    rows.extend(cell_rows)

    write_parquet(rows, cfg.parquet_path())
    _render_two_figures(rows, cfg)
    status_counts: dict[str, int] = {}
    for r in rows:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    finalise_run_logging(run_meta, status_summary=status_counts)
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
    cfg: CrashTestConfig, family: str, n: int, seed: int,
    spec: BaselineSpec,
) -> List[CellResult]:
    """Fit a fresh model per ``spec``, measure accuracy vs truth.

    v0.8-#51: dispatches on the method-keyed ``spec`` (library +
    mechanism + param_method) rather than the legacy polymorphic
    string.  Pre-fix, every ``nbn-*`` baseline collapsed to a single
    fit path that ignored ``spec.mechanism`` (so ``nbn-cat`` and
    ``nbn-neuralcat`` produced bit-identical TV); see
    ``docs/audits/v0.7-43-fit-path-audit.md`` for the audit trail.

    v0.6c-C-1b: reports ``total_time_s`` alongside the accuracy
    metric.  Speed includes the ``adapter.fit(...)`` call AND the
    ``_avg_*_per_node`` direct-CPD comparison loop (collectively
    "learn the CPDs and evaluate them" — what a paper figure
    timing column should report).
    """
    # YAML-loaded specs are dicts; tests pass BaselineSpec dataclasses.
    # Normalise once at the boundary so downstream attribute access is
    # uniform.
    if not isinstance(spec, BaselineSpec):
        spec = BaselineSpec(
            library=str(spec["library"]),
            mechanism=str(spec["mechanism"]),
            param_method=str(spec["param_method"]),
            inference_method=spec.get("inference_method"),
        )
    label = _label_from_spec(spec)
    legacy = _legacy_adapter_for_spec(spec)
    if (family, legacy) in _NOT_APPLICABLE:
        return _not_applicable_row(family, n, seed, label)
    bn = _generate_bn(cfg, family, n, seed)
    t0 = time.perf_counter()
    if spec.library == "nbn":
        # v0.8 Pass-9: returns list of (metric, value, status, error_msg)
        # so each cell can emit one row per metric (tv_per_node,
        # jsd_per_node, w1_per_node).  Skipped metrics get
        # status="not_supported".
        metric_rows = _fit_and_score_nbn(cfg, bn, family, spec)
    elif spec.library == "pgmpy":
        metric_rows = _fit_and_score_pgmpy(bn, family, spec)
    else:
        raise NotImplementedError(
            f"param-learning library {spec.library!r} accuracy wiring is v0.5b",
        )
    total_time = time.perf_counter() - t0
    rows: List[CellResult] = []
    for metric_name, value, status, error_msg in metric_rows:
        rows.append(CellResult(
            family=family, n_nodes=n, seed=seed, baseline=label,
            metric=metric_name, value=value, status=status,
            extra={"error_msg": error_msg} if error_msg else {},
        ))
    rows.append(CellResult(
        family=family, n_nodes=n, seed=seed, baseline=label,
        metric="total_time_s", value=float(total_time),
    ))
    return rows


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

    # v0.8 Pass-9: compute the full {tv, jsd, w1} metric dict once;
    # derive the family-dispatched 'accuracy' value AND emit explicit
    # per-metric rows (tv_per_node, jsd_per_node, w1_per_node).  The
    # 'accuracy' row preserves backward compat with existing
    # aggregator/plotter consumers; explicit rows are additive.
    metric_dict = _compute_inference_metrics(
        bn, baseline, queries_batch, family,
        n_lw_samples=cfg.nbn_lw_n_samples,
    )
    target_kind = bn.variable_specs[queries_batch.targets[0]][0]
    accuracy_key = "tv_per_node" if target_kind == "discrete" else "w1_per_node"
    accuracy = metric_dict.get(accuracy_key)
    if accuracy is not None:
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
    # Explicit per-metric rows.  Same _format_metric_rows shape as
    # Phase 2's param-learning emit, with skip reasons mapped via
    # _SKIP_REASONS (e.g. W1 on pure-discrete-target cells).
    for metric_name, value, status, error_msg in _format_metric_rows(
        metric_dict, family,
    ):
        rows.append(CellResult(
            family=family, n_nodes=n, seed=seed, baseline=baseline,
            metric=metric_name, value=value, status=status,
            extra={"error_msg": error_msg} if error_msg else {},
        ))
    return rows


# ---------------------------------------------------------------------- #
# Parameter-learning workers
# ---------------------------------------------------------------------- #


def _fit_and_score_nbn(
    cfg: CrashTestConfig, bn: SyntheticBN, family: str,
    spec: BaselineSpec,
) -> "list[tuple[str, float, str, str | None]]":
    from nbn.core.network import NeuralBayesianNetwork
    fitted = NeuralBayesianNetwork(
        list(bn.dag.edges()), variables=bn.variable_specs, device="cpu",
    )
    for node in bn.dag.nodes():
        kind = bn.variable_specs[node][0]
        parents = list(bn.dag.predecessors(node))
        parent_kinds = [bn.variable_specs[p] for p in parents]
        # v0.8-#51: spec-aware mechanism factory routes on
        # spec.mechanism (cat / neuralcat / lg / mdn / flow / hybrid)
        # so each method-keyed NBN baseline gets its promised
        # mechanism class.
        fitted.set_mechanism(node, fresh_mechanism_for_spec(
            spec, family, kind,
            parent_kinds=parent_kinds, cardinality=cfg.cardinality,
        ))
    fitted.fit(
        bn.train_data, epochs=cfg.fit_epochs, batch_size=cfg.batch_size, lr=5e-3,
    )
    return _score_param_learning_metrics_nbn(fitted, bn, family)


def _fit_and_score_pgmpy(
    bn: SyntheticBN, family: str, spec: BaselineSpec,
) -> "list[tuple[str, float, str, str | None]]":
    from benchmarking.baselines.pgmpy_adapter import PgmpyAdapter
    from benchmarking.domains.base import BenchmarkProblem
    problem = BenchmarkProblem(
        name=bn.name, dag=list(bn.dag.edges()), variables=bn.variable_specs,
        train_data=bn.train_data, test_data=bn.test_data, queries=[],
    )
    # v0.8-#53: pre-fix constructed PgmpyAdapter() with no kwargs,
    # which defaults param_method="mle" — so pgmpy-bayes silently ran
    # the MLE branch and produced rows bit-identical to pgmpy-mle in
    # the v0.6c-d parquet.  Now we plumb spec.param_method through so
    # the BayesianEstimator branch (BDeu prior, ESS=5) actually fires
    # for pgmpy-bayes specs.  Audit: docs/audits/v0.7-43-fit-path-audit.md.
    adapter = PgmpyAdapter(param_method=spec.param_method)
    adapter.fit(problem)
    if adapter.kind == "unsupported":
        raise NotImplementedError(f"pgmpy refused the {family} family")
    return _score_param_learning_metrics_pgmpy(adapter, bn, family)


# ---------------------------------------------------------------------- #
# v0.8 Pass-9: parametrized metric scoring (TV / JSD / W1 per cell)
# ---------------------------------------------------------------------- #
#
# _score_param_learning_metrics_nbn / _pgmpy build (true_logits,
# fit_logits) cpd_pairs for discrete nodes via Mechanism.tabulate()
# (Pass 8 #59/#26) and (true_samples, fit_samples) sample_pairs for
# continuous nodes via mechanism.sample().  Both lists are passed to
# benchmarking.metrics._compute_metrics_per_node which averages each
# requested metric over its contributing pairs.
#
# Returns list[tuple[metric_name, value, status, error_msg]] —
# one per requested metric.  Skipped metrics (e.g. W1 on pure
# discrete-unordered) emit (metric, NaN, "not_supported", reason).
#
# The four legacy helpers (_avg_tv_per_node, _avg_tv_per_node_pgmpy,
# _avg_w1_per_node, _avg_w1_per_node_pgmpy_lg) are kept as thin
# wrappers around the new scoring path — backward compat for any
# external callers / tests that key on the old names.


def _build_param_learning_pairs_nbn(
    fitted, bn: SyntheticBN, *, n_eval: int = 20, n_samples: int = 200,
) -> "tuple[list, list]":
    """Build (cpd_pairs, sample_pairs) from fitted vs bn.true_model.

    Iterates topologically over bn.dag; for each discrete node appends
    one (true_logits, fit_logits) pair via Mechanism.tabulate(); for
    each continuous node appends ``n_pa_eval`` (true_samples,
    fit_samples) pairs via mechanism.sample().  Hybrid families
    naturally produce both lists populated.
    """
    cpd_pairs = []
    sample_pairs = []
    n_test = next(iter(bn.test_data.values())).shape[0]
    n_pa_eval = min(n_eval, n_test)
    for node in nx.topological_sort(bn.dag):
        kind = bn.variable_specs[node][0]
        parents = list(bn.dag.predecessors(node))
        true_mech = bn.true_model.mechanisms[node]
        fit_mech = fitted.mechanisms[node]
        if kind == "discrete":
            parent_cards = [bn.true_model.mechanisms[p].n_classes for p in parents]
            true_logits = true_mech.tabulate(parent_cards).detach().cpu()
            fit_logits = fit_mech.tabulate(parent_cards).detach().cpu()
            cpd_pairs.append((true_logits, fit_logits))
        else:
            # Continuous: place pa tensor on fitted device for
            # fit_mech.sample(); _w1 / helper run on cpu copies.
            fit_device = next(
                (p.device for p in fit_mech.parameters() if p is not None),
                torch.device("cpu"),
            )
            if parents:
                pa = torch.cat(
                    [bn.test_data[p][:n_pa_eval].reshape(n_pa_eval, -1).float()
                     for p in parents], dim=-1,
                ).to(fit_device)
            else:
                pa = None
            with torch.no_grad():
                try:
                    true_pa = (
                        pa.to(next(true_mech.parameters()).device)
                        if pa is not None else None
                    )
                    true_s = true_mech.sample(true_pa, n=n_samples).reshape(
                        n_pa_eval, n_samples,
                    )
                    fit_s = fit_mech.sample(pa, n=n_samples).reshape(
                        n_pa_eval, n_samples,
                    )
                except Exception:
                    continue
            for i in range(n_pa_eval):
                sample_pairs.append((true_s[i].cpu(), fit_s[i].cpu()))
    return cpd_pairs, sample_pairs


def _build_param_learning_pairs_pgmpy(
    adapter, bn: SyntheticBN, *, n_eval: int = 20, n_samples: int = 200,
) -> "tuple[list, list]":
    """Build (cpd_pairs, sample_pairs) for pgmpy-fitted models.

    Discrete (adapter.kind == "discrete"): truth uses tabulate(), pgmpy
    side reads cpd.values and reshapes to [*parent_cards, K] then
    converts probabilities to logits via log() for shape parity with
    the discrete cpd_pairs convention (helper softmaxes anyway, so
    log-probs and logits are equivalent under shift-invariance).

    Continuous LG (adapter.kind == "continuous_lg"): hand-rolled LG
    sampling from the pgmpy CPD, matching the previous
    _avg_w1_per_node_pgmpy_lg shape.
    """
    import numpy as np
    cpd_pairs = []
    sample_pairs = []
    n_test = next(iter(bn.test_data.values())).shape[0]
    n_pa_eval = min(n_eval, n_test)
    if adapter.kind == "discrete":
        for cpd in adapter.model.get_cpds():
            node = cpd.variable
            parents = list(bn.dag.predecessors(node))
            parent_cards = [bn.true_model.mechanisms[p].n_classes for p in parents]
            true_logits = (
                bn.true_model.mechanisms[node].tabulate(parent_cards).detach().cpu()
            )
            vals = np.asarray(cpd.values)
            # pgmpy CPD shape [K, *pa]; movedim → [*pa, K] to match
            # tabulate()'s convention.
            fit_p = torch.tensor(vals, dtype=torch.float).movedim(0, -1).contiguous()
            # Helper softmaxes its inputs; pass log-probs so softmax
            # recovers fit_p.  clamp_min(1e-12) to avoid log(0).
            fit_logits = torch.log(fit_p.clamp_min(1e-12))
            cpd_pairs.append((true_logits, fit_logits))
    elif adapter.kind == "continuous_lg":
        cpd_by_var = {cpd.variable: cpd for cpd in adapter.lg_model.get_cpds()}
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
                    pa_vals = torch.tensor(
                        [float(bn.test_data[p][i].reshape(-1)[0].cpu().item())
                         for p in parents],
                    ).float()
                    mean = beta[0] + sum(
                        beta[k + 1] * pa_vals[k].item() for k in range(len(parents))
                    )
                    fit_s = mean + std * torch.randn(n_samples)
                    pa_t = pa_vals.reshape(1, -1).to(true_device)
                else:
                    fit_s = beta[0] + std * torch.randn(n_samples)
                    pa_t = None
                with torch.no_grad():
                    try:
                        true_s = true_mech.sample(pa_t, n=n_samples).reshape(-1)[:n_samples]
                    except Exception:
                        continue
                sample_pairs.append((true_s.cpu(), fit_s.cpu()))
    return cpd_pairs, sample_pairs


def _format_metric_rows(
    metric_dict: "dict[str, float | None]", family: str,
) -> "list[tuple[str, float, str, str | None]]":
    """Convert helper-output dict to per-metric (name, value, status, error_msg) rows.

    None values (skipped metrics) emit NaN with status='not_supported'
    and a human-readable reason.  Computed values emit status='ok'.
    """
    rows = []
    for short_name, value in metric_dict.items():
        # short_name is like "tv_per_node" (already suffixed by helper)
        metric_name = short_name
        short_kind = short_name.replace("_per_node", "")
        if value is None:
            reason = _SKIP_REASONS.get(
                (family, short_kind),
                f"{short_kind} not applicable to family {family!r} "
                f"(no contributing pairs)",
            )
            rows.append((metric_name, float("nan"), "not_supported", reason))
        else:
            rows.append((metric_name, float(value), "ok", None))
    return rows


def _score_param_learning_metrics_nbn(
    fitted, bn: SyntheticBN, family: str,
) -> "list[tuple[str, float, str, str | None]]":
    from benchmarking.metrics import _compute_metrics_per_node
    cpd_pairs, sample_pairs = _build_param_learning_pairs_nbn(fitted, bn)
    metric_dict = _compute_metrics_per_node(
        cpd_pairs=cpd_pairs, sample_pairs=sample_pairs,
        metrics=_PARAM_LEARNING_METRICS,
    )
    return _format_metric_rows(metric_dict, family)


def _score_param_learning_metrics_pgmpy(
    adapter, bn: SyntheticBN, family: str,
) -> "list[tuple[str, float, str, str | None]]":
    from benchmarking.metrics import _compute_metrics_per_node
    cpd_pairs, sample_pairs = _build_param_learning_pairs_pgmpy(adapter, bn)
    metric_dict = _compute_metrics_per_node(
        cpd_pairs=cpd_pairs, sample_pairs=sample_pairs,
        metrics=_PARAM_LEARNING_METRICS,
    )
    return _format_metric_rows(metric_dict, family)


# ---------------------------------------------------------------------- #
# Backward-compat thin wrappers around the Pass-9 scoring path.
# Pre-Pass-9 callers (tests, external scripts) may still key on these
# names; each returns a single float for its specific metric.
# ---------------------------------------------------------------------- #


def _avg_tv_per_node(fitted, bn: SyntheticBN) -> float:
    """Backward-compat wrapper: average per-node TV on a NBN-fitted model."""
    from benchmarking.metrics import _compute_metrics_per_node
    cpd_pairs, _ = _build_param_learning_pairs_nbn(fitted, bn)
    out = _compute_metrics_per_node(cpd_pairs=cpd_pairs, metrics=("tv",))
    return float(out["tv_per_node"]) if out["tv_per_node"] is not None else float("nan")


def _avg_tv_per_node_pgmpy(adapter, bn: SyntheticBN) -> float:
    """Backward-compat wrapper: average per-node TV on a pgmpy-fitted model."""
    from benchmarking.metrics import _compute_metrics_per_node
    cpd_pairs, _ = _build_param_learning_pairs_pgmpy(adapter, bn)
    out = _compute_metrics_per_node(cpd_pairs=cpd_pairs, metrics=("tv",))
    return float(out["tv_per_node"]) if out["tv_per_node"] is not None else float("nan")


def _avg_w1_per_node(
    fitted, bn: SyntheticBN, *, n_eval: int = 20, n_samples: int = 200,
) -> float:
    """Backward-compat wrapper: average per-node W1 on a NBN-fitted model."""
    from benchmarking.metrics import _compute_metrics_per_node
    _, sample_pairs = _build_param_learning_pairs_nbn(
        fitted, bn, n_eval=n_eval, n_samples=n_samples,
    )
    out = _compute_metrics_per_node(sample_pairs=sample_pairs, metrics=("w1",))
    return float(out["w1_per_node"]) if out["w1_per_node"] is not None else float("nan")


def _avg_w1_per_node_pgmpy_lg(
    adapter, bn: SyntheticBN, *, n_eval: int = 20, n_samples: int = 200,
) -> float:
    """Backward-compat wrapper: average per-node W1 on pgmpy-LG-fitted model."""
    from benchmarking.metrics import _compute_metrics_per_node
    if adapter.kind != "continuous_lg":
        raise NotImplementedError("pgmpy LG path not active")
    _, sample_pairs = _build_param_learning_pairs_pgmpy(
        adapter, bn, n_eval=n_eval, n_samples=n_samples,
    )
    out = _compute_metrics_per_node(sample_pairs=sample_pairs, metrics=("w1",))
    return float(out["w1_per_node"]) if out["w1_per_node"] is not None else float("nan")


def _w1(a: torch.Tensor, b: torch.Tensor) -> float:
    """Empirical 1-D Wasserstein-1 distance between two sample arrays.

    v0.7-#37 fix: previous implementation did ``min(len(a), len(b))``
    truncation post-sort, which compares mismatched quantiles when the
    arrays differ in length (e.g. lower half of one vs all of the other).
    This is not a Wasserstein estimator.  The continuous_lg accuracy
    cell hit this because the runner's predicted-vs-oracle paths
    produced unequal-length arrays (pgmpy budget=4000 vs oracle=2000),
    inflating pgmpy's W₁ by ≈10× — see ``scripts/diagnostics/diagnose_issue_37.py``.

    Use the standard quantile-interpolation estimator: take both sorted
    arrays at the same evenly-spaced quantile grid (length ``max(len)``
    so neither is downsampled), then average the absolute differences.
    Matches ``scipy.stats.wasserstein_distance`` for equal-length input
    and degrades gracefully when the arrays differ.
    """
    a = a.reshape(-1).float().cpu()
    b = b.reshape(-1).float().cpu()
    n = max(a.shape[0], b.shape[0])
    qs = torch.linspace(0.0, 1.0, n)
    a_q = torch.quantile(a, qs)
    b_q = torch.quantile(b, qs)
    return float((a_q - b_q).abs().mean().item())


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
    """One batched call per timed run; report median total time.

    v0.6c-C-1b: routes through ``BaselineAdapterV2Shim`` so the engine
    queries a *fitted* NBN model (trained on ``bn.train_data``), not
    ``bn.true_model``.  Pre-PR this function called
    ``eng.query_batch(bn.true_model, ...)`` directly, which measured
    "engine quality on the canonical SCM" rather than "library quality
    after fitting on data" — wrong semantics for paper figures.
    """
    from benchmarking._baseline_registry import BaselineSpec
    from benchmarking.baselines._adapter_v2 import build_adapter_v2

    legacy_to_inference_method = {
        "nbn_ve": "ve",
        "nbn_lw": "lw",
        "nbn_hybrid": "router",
    }
    inf_method = legacy_to_inference_method.get(baseline)
    if inf_method is None:
        raise NotImplementedError(f"unknown nbn variant {baseline!r}")

    # Pick a default mechanism per family — these match the v1 NBNAdapter's
    # auto-routing.  v0.6c-C-1b's runner refactor is per-spec dispatch in
    # principle, but the smoke YAML still uses the C-1a 5-baseline set
    # mapped 1-to-1 to legacy names; per-mechanism expansion lands in
    # the YAML of this PR for the labels we have adapter support for.
    family_kinds = {k for (k, _) in bn.variable_specs.values()}
    if family_kinds == {"discrete"}:
        spec = BaselineSpec(library="nbn", mechanism="cat",
                            param_method="mle", inference_method=inf_method)
    elif family_kinds == {"continuous"}:
        # continuous_lg vs continuous_nongauss: pick LG when the
        # name encodes _lg (the synthetic generator's convention).
        if "_lg" in bn.name:
            spec = BaselineSpec(library="nbn", mechanism="lg",
                                param_method="mle", inference_method=inf_method)
        else:
            spec = BaselineSpec(library="nbn", mechanism="mdn",
                                param_method="mle", inference_method=inf_method)
    else:
        # hybrid → HybridRouter handles per-node dispatch.
        spec = BaselineSpec(library="nbn", mechanism="hybrid",
                            param_method="mle", inference_method=inf_method)

    adapter = build_adapter_v2(spec, device=str(bn.true_model.device))
    fitted = adapter.fit(bn.train_data, bn.dag, bn.variable_specs)

    # v0.7-#35: warmup and timed loop now propagate exceptions so that
    # genuine engine failures land as ``status='error'`` via
    # ``run_with_guard``.  Pre-fix this function silently swallowed any
    # exception from ``query_batch_samples`` and returned ``float('nan')``
    # — combined with NBNAdapter not implementing ``query_batch_samples``
    # at all (so the base class always raised), the parquet's
    # ``total_time_s`` rows for every NBN inference baseline came out as
    # NaN-with-status='ok' and disappeared from the figures.
    for _ in range(3):
        adapter.query_batch_samples(fitted, q, n_samples=n_lw_samples)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        adapter.query_batch_samples(fitted, q, n_samples=n_lw_samples)
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
    """v0.6c-C-3: render figures + write summary tables.

    Reads the parquet that ``write_parquet`` just produced and pipes
    it through the new aggregator + table writers + plotter v2:

    * 2 figures (accuracy_vs_size + total_time_vs_size) via
      :mod:`benchmarking._plot_v2.render_figures` — per-panel
      applicability filter, b&w-safe markers, mean ± std bands.
    * 4 tables (CSV / markdown / LaTeX / parquet) via
      :func:`benchmarking._tables.write_all`.

    The old plotter (``plot_metric_vs_n_nodes``) is no longer called
    — it was tied to the pre-C-1a flat baseline list.  The new
    plotter consumes the registry's per-baseline applicability matrix
    so legends correctly filter per panel even with 13 method-keyed
    baselines.

    Best-effort: any failure in the aggregator / tables / plotter
    pipeline logs a warning and lets the run complete (the parquet
    is already on disk by the time we get here).
    """
    from benchmarking._aggregate import aggregate
    from benchmarking._plot_v2 import render_figures
    from benchmarking._tables import write_all

    parquet_path = cfg.parquet_path()
    output_prefix = cfg.output_prefix
    output_dir = cfg.output_dir

    try:
        aggregated = aggregate(parquet_path)
        write_all(aggregated, output_dir=output_dir, output_prefix=output_prefix)
    except Exception as exc:
        logger.warning(
            "v0.6c-C-3 aggregator/tables pipeline failed: %s "
            "(parquet still written; figures still attempted)",
            exc,
        )

    try:
        render_figures(
            parquet_path=parquet_path,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
    except Exception as exc:
        logger.warning(
            "v0.6c-C-3 plotter v2 failed: %s (parquet + tables still on disk)",
            exc,
        )


# ---------------------------------------------------------------------- #
# Inference accuracy (PR-B §B.2)
# ---------------------------------------------------------------------- #


def _compute_inference_accuracy(
    bn: SyntheticBN, baseline: str, q, family: str,
    *, n_samples: int = 200, eps_factor: float = 0.50, n_eff_min: int = 10,
    n_lw_samples: int = 512, n_oracle_samples: int = 2000,
) -> float:
    """Average distributional metric over the query battery.

    Backward-compat wrapper around _compute_inference_metrics
    (introduced v0.8 Pass-9): family-dispatches the dict to a single
    float — TV for discrete targets, W1 for continuous targets —
    matching the pre-Pass-9 contract so existing callers (the
    accuracy-row emit path in _inference_cell, and external tests
    keying on this signature) continue to work.

    Round-4 oracle change (PR #14): continuous + hybrid targets use
    forward-with-clamp ancestral sampling; discrete targets use
    exact-match rejection on bn.ground_truth_samples.  Both paths are
    delegated to _compute_inference_metrics; this wrapper only does
    family-dispatch and None→NaN mapping.
    """
    target_kind = bn.variable_specs[q.targets[0]][0]
    metric_dict = _compute_inference_metrics(
        bn, baseline, q, family,
        n_samples=n_samples, eps_factor=eps_factor, n_eff_min=n_eff_min,
        n_lw_samples=n_lw_samples, n_oracle_samples=n_oracle_samples,
    )
    # Family-dispatch: TV for discrete-target, W1 for continuous-target.
    # Hybrid family with continuous target falls into the W1 branch
    # (matches pre-Pass-9 behaviour at lines 1052-1062).
    if target_kind == "discrete":
        value = metric_dict.get("tv_per_node")
    else:
        value = metric_dict.get("w1_per_node")
    # All-skipped cell → None → NaN (matches pre-Pass-9 line 1064:
    # "if not accs: return float('nan')").  The aggregator + plotter
    # already filter NaN; emitting NaN here preserves the existing
    # no_result path in _inference_cell.
    return float(value) if value is not None else float("nan")


def _compute_inference_metrics(
    bn: SyntheticBN, baseline: str, q, family: str,
    *, n_samples: int = 200, eps_factor: float = 0.50, n_eff_min: int = 10,
    n_lw_samples: int = 512, n_oracle_samples: int = 2000,
) -> "dict[str, float | None]":
    """Compute {tv, jsd, w1}_per_node for the inference query battery.

    Per row in the up-to-16-row loop, build either a (truth_logits,
    pred_logits) cpd_pair (discrete target) or a (truth_samples,
    pred_samples) sample_pair (continuous target).  Per-row failures
    (pred is None, ground-truth filter empty, oracle too few samples)
    skip that row but don't fail the cell — same masking shape as the
    pre-Pass-9 _compute_inference_accuracy.

    Returns dict[str, float | None] keyed by metric name (tv_per_node,
    jsd_per_node, w1_per_node).  None for any metric whose pair-list
    is empty (caller decides None → NaN via _format_metric_rows or
    _compute_inference_accuracy wrapper).
    """
    from benchmarking.metrics import _compute_metrics_per_node

    target = q.targets[0]
    target_kind = bn.variable_specs[target][0]
    target_idx = bn.column_index(target)
    B = next(iter(q.evidence.values())).shape[0]

    # Discrete ground-truth-pool synthesis (verbatim from
    # pre-Pass-9 _compute_inference_accuracy lines 1009-1020):
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
            return {f"{m}_per_node": None for m in _PARAM_LEARNING_METRICS}

    cpd_pairs = []
    sample_pairs = []
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
            # Both empirical (truth) and pred are probability vectors;
            # log-clamp to logit space for the helper (helper softmaxes,
            # so log-probs and raw logits are equivalent under
            # shift-invariance).
            true_logits = torch.log(empirical.clamp_min(1e-12))
            pred_vec = pred.cpu().reshape(-1)[:k]
            fit_logits = torch.log(pred_vec.clamp_min(1e-12))
            cpd_pairs.append((true_logits, fit_logits))
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
            sample_pairs.append((oracle.reshape(-1).cpu(), pred.cpu().reshape(-1)))

    return _compute_metrics_per_node(
        cpd_pairs=cpd_pairs, sample_pairs=sample_pairs,
        metrics=_PARAM_LEARNING_METRICS,
    )


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
        # v0.6c-C-1b: route NBN posterior queries through a *fitted*
        # NBN model (trained on bn.train_data via the v2 adapter shim),
        # not bn.true_model.  Pre-PR this branch called
        # ``eng.query(bn.true_model, [target], ev)`` directly — wrong
        # semantics for honest paper figures.
        from benchmarking._baseline_registry import BaselineSpec
        from benchmarking.baselines._adapter_v2 import build_adapter_v2
        from nbn.inference.hybrid import HybridRouter
        from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
        from nbn.inference.tensor_ve import TensorVariableElimination
        legacy_to_inference_method = {
            "nbn_ve": "ve", "nbn_lw": "lw", "nbn_hybrid": "router",
        }
        inf_method = legacy_to_inference_method.get(baseline)
        if inf_method is None:
            return None
        # Pick a default mechanism per family.
        family_kinds = {k for (k, _) in bn.variable_specs.values()}
        if family_kinds == {"discrete"}:
            spec = BaselineSpec(library="nbn", mechanism="cat",
                                param_method="mle", inference_method=inf_method)
        elif family_kinds == {"continuous"}:
            mech = "lg" if "_lg" in bn.name else "mdn"
            spec = BaselineSpec(library="nbn", mechanism=mech,
                                param_method="mle", inference_method=inf_method)
        else:
            spec = BaselineSpec(library="nbn", mechanism="hybrid",
                                param_method="mle", inference_method=inf_method)
        try:
            adapter = build_adapter_v2(spec, device=str(bn.true_model.device))
            fitted = adapter.fit(bn.train_data, bn.dag, bn.variable_specs)
        except Exception:
            return None
        # Use the fitted adapter's query path.  The v1 NBNAdapter's
        # ``query`` returns a probability vector for discrete or a
        # mean for continuous; we still need engine-level samples for
        # the accuracy oracle, so call the underlying engine on the
        # *fitted* model (NEVER bn.true_model).
        fitted_model = adapter._v1.model  # NBNAdapter exposes .model post-fit
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
                out = eng.query(fitted_model, [target], ev)
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
        if target_kind == "discrete":
            try:
                out = adapter.query(
                    Query(targets=(target,), evidence=ev_row, kind="marginal"),
                )
            except Exception:
                return None
            return out.detach().cpu().float().reshape(-1)
        # v0.5c bug 3: continuous-target path goes through
        # ``query_batch_samples`` (pgmpy's closed-form Gaussian
        # posterior → reparam samples), matching the gpytorch branch
        # below.  Pre-fix, this branch hard-returned ``None`` for
        # ``target_kind != "discrete"`` and every pgmpy continuous_lg
        # accuracy cell skipped → ``no_result`` on all 3 cells in
        # PR #14's smoke parquet.  pomegranate is structurally excluded
        # from continuous families via ``_NOT_APPLICABLE`` so only
        # pgmpy reaches this branch in practice.
        #
        # Sample-budget parity: use the larger of ``n_samples`` (200
        # default) and ``n_lw_samples`` (4000 in smoke) so pgmpy's
        # closed-form posterior is sampled at the same density as
        # nbn_lw's.  Without this, pgmpy's W₁ averaged over the query
        # battery shows ≈4× higher MC noise than nbn_lw on identical
        # SCMs — making the closed-form-exact baseline look worse
        # than the MC approximation, which is misleading on the figure.
        budget = max(n_samples, n_lw_samples)
        try:
            samples = adapter.query_batch_samples(
                Query(
                    targets=(target,),
                    evidence={k: v.reshape(1) for k, v in ev_row.items()},
                    kind="marginal",
                ),
                n_samples=budget,
            )
        except Exception:
            return None
        if samples.shape[0] == 0 or torch.isnan(samples).any():
            return None
        return samples.detach().cpu().float().reshape(-1)
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
