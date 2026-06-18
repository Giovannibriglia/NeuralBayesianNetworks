"""Torch-native, device-aware metrics for the NBN benchmark suite.

All metrics return a ``MetricResult`` with name, value, and (when applicable)
bootstrap confidence interval bounds.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MetricResult:
    name: str
    value: float
    ci_low: float | None = None
    ci_high: float | None = None

    def as_dict(self) -> dict:
        return {
            "metric": self.name,
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


# ── discrete marginal metrics ────────────────────────────────────────────────

def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)


def kl_marginals(p: torch.Tensor, q: torch.Tensor) -> MetricResult:
    return MetricResult("kl_marginals", float(kl_divergence(p, q).mean()))


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m, eps) + kl_divergence(q, m, eps))


def js_marginals(p: torch.Tensor, q: torch.Tensor) -> MetricResult:
    return MetricResult("js_marginals", float(js_divergence(p, q).mean()))


def tv_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 0.5 * (p - q).abs().sum(dim=-1)


def tv_marginals(p: torch.Tensor, q: torch.Tensor) -> MetricResult:
    return MetricResult("tv_marginals", float(tv_distance(p, q).mean()))


def mae_marginals(p: torch.Tensor, q: torch.Tensor) -> MetricResult:
    return MetricResult("mae_marginals", float((p - q).abs().mean()))


# ── continuous metrics ──────────────────────────────────────────────────────

def wasserstein_1d(x: torch.Tensor, y: torch.Tensor) -> MetricResult:
    """1-D Wasserstein-1 between two empirical 1-D samples.

    Uses quantile-interpolation at a grid of length ``max(len(x), len(y))``
    so neither series is downsampled when inputs have unequal length (issue
    #37 fix: the previous ``min`` approach compared mismatched quantiles and
    inflated W₁ by ≈10× for unequal-budget inputs).

    Both inputs are moved to CPU before computation for device safety (PR-B
    §A.1 fix: mixed-device subtraction raised ``RuntimeError`` when one
    tensor was on CUDA and the other on CPU).
    """
    x_cpu = x.reshape(-1).float().cpu()
    y_cpu = y.reshape(-1).float().cpu()
    n = max(x_cpu.shape[0], y_cpu.shape[0])
    if n == 0:
        return MetricResult("wasserstein_1d", float("nan"))
    qs = torch.linspace(0.0, 1.0, n)
    x_q = torch.quantile(x_cpu, qs)
    y_q = torch.quantile(y_cpu, qs)
    return MetricResult("wasserstein_1d", float((x_q - y_q).abs().mean()))


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> MetricResult:
    """E[|X-Y|] - 0.5 * (E[|X-X'|] + E[|Y-Y'|]) via subsampled Monte Carlo."""
    nx, ny = x.shape[0], y.shape[0]
    if nx == 0 or ny == 0:
        return MetricResult("energy_distance", float("nan"))
    # Cap pairs for tractability
    cap = min(1024, nx, ny)
    xi = x[torch.randperm(nx, device=x.device)[:cap]]
    yi = y[torch.randperm(ny, device=y.device)[:cap]]
    d_xy = torch.cdist(xi.float().reshape(cap, -1), yi.float().reshape(cap, -1)).mean()
    d_xx = torch.cdist(xi.float().reshape(cap, -1), xi.float().reshape(cap, -1)).mean()
    d_yy = torch.cdist(yi.float().reshape(cap, -1), yi.float().reshape(cap, -1)).mean()
    return MetricResult("energy_distance", float(2 * d_xy - d_xx - d_yy))


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float | None = None) -> MetricResult:
    """Unbiased squared MMD with an RBF kernel, median-heuristic bandwidth."""
    nx, ny = x.shape[0], y.shape[0]
    if nx < 2 or ny < 2:
        return MetricResult("mmd_rbf", float("nan"))
    cap = min(512, nx, ny)
    xi = x[torch.randperm(nx, device=x.device)[:cap]].float().reshape(cap, -1)
    yi = y[torch.randperm(ny, device=y.device)[:cap]].float().reshape(cap, -1)
    if sigma is None:
        with torch.no_grad():
            d = torch.cdist(xi, yi)
            sigma = float(d.median().clamp_min(1e-6))
    def _kk(a, b):
        return torch.exp(-torch.cdist(a, b).pow(2) / (2 * sigma * sigma))
    kxx = _kk(xi, xi); kyy = _kk(yi, yi); kxy = _kk(xi, yi)
    n = cap
    mmd = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1)) \
        + (kyy.sum() - kyy.diag().sum()) / (n * (n - 1)) \
        - 2 * kxy.mean()
    return MetricResult("mmd_rbf", float(mmd))


# ── prediction metrics ──────────────────────────────────────────────────────

def log_likelihood(log_probs: torch.Tensor) -> MetricResult:
    """Mean held-out joint log-probability (higher is better).

    Parameters
    ----------
    log_probs : torch.Tensor
        Per-ROW joint log-probs of held-out test rows under the fitted
        model — each element is already summed over the graph's nodes.
        The topological per-node assembly over ``test_data`` lives in the
        ParamLearningMeasurement (issue #109), not here: this primitive
        stays free of model/graph internals, mirroring the purity of the
        other functions in this module.

    Returns
    -------
    MetricResult
        ``name == "log_likelihood"`` (no ``_per_node`` suffix — this is a
        single joint score, not a per-node average), ``value`` the POSITIVE
        mean log-prob. Higher-is-better, matching
        ``_paper_figures.HIGHER_IS_BETTER``.
    """
    return MetricResult("log_likelihood", float(log_probs.mean()))


# ── parameter-recovery metrics (#109 PR 2) ──────────────────────────────────
#
# How well a fitted model recovers the TRUE discrete CPDs. Per discrete node we
# compare the true CPT P_true(node | parents) against the learned CPT, one row
# per parent configuration, frequency-weighted by the TRUE parent-config
# probability (so configs the data never visits — where everyone estimates
# badly — don't dominate). The per-cell scalar is the simple mean over nodes.
# Inputs are PURE tensors (no model/graph objects): the per-node CPT extraction
# and the canonical [n_parent_configs, K] layout live in ParamLearningMeasurement
# / the adapters. ``true_cpts``/``learned_cpts``/``weights`` are aligned lists,
# one entry per discrete node; entry i is [n_configs_i, K_i] / [n_configs_i].

def frequency_weights(counts: torch.Tensor) -> torch.Tensor:
    """Normalize parent-config counts to weights that sum to 1.

    ``counts[c]`` is the number of TRUE-model samples landing in parent config
    ``c`` for one node. Returns ``counts / counts.sum()``. The denominator is
    clamped at 1 purely to avoid division by zero in the degenerate all-zero
    case (no samples) — with any real sample every row carries a count, so the
    clamp never bites and the weights sum to exactly 1.
    """
    counts = counts.to(torch.float64)
    return counts / counts.sum().clamp_min(1.0)


def _kl_rows_unclamped(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-row KL(p ‖ q), summed over the last (class) dim. q is NOT clamped.

    Conventions (locked #109): ``0 * log(0) = 0`` for true zeros; ``p`` (the
    TRUE distribution) is clamped at 1e-12 ONLY so ``log(p)`` is finite, never
    ``q`` — a learned zero on a class the truth supports must yield ``+inf``,
    not a clamped finite number (that is the whole point of the companion KL,
    and the signal pgmpy MLE hard zeros will produce in a later PR).
    """
    p = p.to(torch.float64)
    q = q.to(torch.float64)
    log_p = torch.log(p.clamp_min(1e-12))          # safe log(p); masked where p==0
    log_q = torch.log(q)                            # UNCLAMPED: q==0 -> -inf -> +inf KL
    per_class = p * (log_p - log_q)
    # Enforce 0*log(0)=0 exactly: where p==0 the term is 0 regardless of q
    # (kills the 0*inf -> nan that arises when both p and q are 0).
    per_class = torch.where(p > 0, per_class, torch.zeros_like(per_class))
    return per_class.sum(dim=-1)


def _weighted_node_reduce(
    rows: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Frequency-weighted sum of per-config ``rows`` for one node.

    ``where(weights > 0, ...)`` so a zero-weight config contributes exactly 0
    even if its row is ``+inf`` (a 0*inf would otherwise NaN). A config with
    POSITIVE weight whose row is ``+inf`` correctly propagates ``+inf``.
    """
    weights = weights.to(torch.float64)
    contrib = torch.where(
        weights > 0, weights * rows.to(torch.float64), torch.zeros_like(rows, dtype=torch.float64)
    )
    return contrib.sum()


def param_recovery_tv(
    true_cpts: list[torch.Tensor],
    learned_cpts: list[torch.Tensor],
    weights: list[torch.Tensor],
) -> MetricResult:
    """Frequency-weighted total-variation parameter-recovery error (headline).

    Per discrete node: weighted mean of ``tv_distance`` over parent configs,
    weighted by the true parent-config probability. Per-cell value: simple
    mean over nodes. Lower is better; TV ∈ [0, 1] so this is bounded and
    robust cross-baseline (the reason it, not KL, is the headline).

    ``true_cpts[i]`` and ``learned_cpts[i]`` are ``[n_configs_i, K_i]`` aligned
    in the canonical layout; ``weights[i]`` is ``[n_configs_i]`` summing to 1.
    Empty input (no discrete nodes) returns ``nan`` — callers gate that as
    ``status="not_applicable"`` upstream.
    """
    if not true_cpts:
        return MetricResult("param_recovery_tv", float("nan"))
    node_vals = [
        _weighted_node_reduce(tv_distance(pt, pl), w)
        for pt, pl, w in zip(true_cpts, learned_cpts, weights)
    ]
    return MetricResult(
        "param_recovery_tv", float(torch.stack(node_vals).mean())
    )


def param_recovery_kl(
    true_cpts: list[torch.Tensor],
    learned_cpts: list[torch.Tensor],
    weights: list[torch.Tensor],
) -> MetricResult:
    """Frequency-weighted KL(true ‖ learned) parameter-recovery error (companion).

    ASYMMETRIC by construction — ``KL(true ‖ learned)``, i.e. expectation under
    the TRUE CPD of the log-ratio. Per discrete node: weighted mean of the
    unclamped per-config KL (see ``_kl_rows_unclamped``); per-cell value: simple
    mean over nodes. Lower is better.

    May be ``+inf``: a learned CPT that puts zero mass on a class the truth
    supports (e.g. MLE hard zeros) diverges. This is a real signal, emitted as
    ``value=float('inf')`` — NOT clamped. Downstream aggregation across cells
    must therefore use a MEDIAN or clipped-mean, never a plain mean, or one
    diverged cell poisons the statistic.

    Same input contract as ``param_recovery_tv``.
    """
    if not true_cpts:
        return MetricResult("param_recovery_kl", float("nan"))
    node_vals = [
        _weighted_node_reduce(_kl_rows_unclamped(pt, pl), w)
        for pt, pl, w in zip(true_cpts, learned_cpts, weights)
    ]
    return MetricResult(
        "param_recovery_kl", float(torch.stack(node_vals).mean())
    )


# ── calibration metrics (#109 PR 7, continuous-only) ────────────────────────
#
# How well a fitted model's PREDICTIVE distribution for a continuous node is
# calibrated. Both are sample-based — the measurement / adapter supply S
# predictive samples per test row per continuous node; PIT-KS additionally uses
# the held-out test value, sd_ratio additionally uses oracle samples from
# problem.true_model. The two are ORTHOGONAL axes of miscalibration: PIT-KS
# (distributional uniformity) and sd_ratio (sharpness vs truth). A model can be
# PIT-uniform yet over/under-confident at the variance scale, and vice versa.
# Inputs are PURE tensors; per-node extraction + the canonical layout live in
# the adapter / ParamLearningMeasurement. The per-cell scalar is the mean over
# continuous nodes (mirrors the parameter-recovery aggregation).

def _pit_ks_single(predictive_samples: torch.Tensor, y: torch.Tensor) -> float:
    """KS distance of one node's PIT distribution from Uniform(0, 1).

    ``predictive_samples`` is ``[N, S]`` (S predictive draws per test row); ``y``
    is ``[N]`` held-out values. The probability-integral transform is the
    sample-empirical CDF ``PIT_i = mean_s(samples[i, s] <= y[i])``; the statistic
    is ``max_i |sorted(PIT) - i/N|`` (one-sample KS vs the uniform grid). 0 is
    perfectly calibrated; larger is worse.
    """
    samples = predictive_samples.detach().double()
    yv = y.detach().double().reshape(-1, 1)
    pit = (samples <= yv).double().mean(dim=1)          # [N]
    pit_sorted = torch.sort(pit).values
    n = pit_sorted.shape[0]
    unif = torch.arange(1, n + 1, dtype=torch.float64) / n
    return float((pit_sorted - unif).abs().max())


def _sd_ratio_single(
    predictive_samples: torch.Tensor, oracle_samples: torch.Tensor
) -> float:
    """Mean per-row ratio of predictive SD to oracle (true-conditional) SD.

    Both ``[N, S]``. Per row: ``std(predictive) / std(oracle)``; reduced by the
    mean over rows (matching the calibration_diagnostic harness). ~1 is the right
    sharpness, <1 over-confident (too narrow), >1 under-confident. The oracle SD
    is clamped at 1e-12 purely as a divide-by-zero numerical guard (a degenerate
    point-mass true conditional) — NOT a distributional smoothing clamp.
    """
    pred_std = predictive_samples.detach().double().std(dim=1)
    true_std = oracle_samples.detach().double().std(dim=1).clamp_min(1e-12)
    return float((pred_std / true_std).mean())


def calibration_pit_ks(
    predictive_samples: list[torch.Tensor],
    y: list[torch.Tensor],
) -> MetricResult:
    """PIT-KS calibration error, mean over continuous nodes (headline).

    ``predictive_samples[i]`` is node i's ``[N, S]`` predictive draws and
    ``y[i]`` its ``[N]`` held-out values. Per-cell value = mean over nodes of the
    per-node PIT-KS. Lower is better (0 = perfectly calibrated). Empty input (no
    continuous nodes) returns ``nan`` — callers gate that as
    ``status="not_applicable"`` upstream.
    """
    if not predictive_samples:
        return MetricResult("calibration_pit_ks", float("nan"))
    node_vals = [_pit_ks_single(s, yy) for s, yy in zip(predictive_samples, y)]
    return MetricResult(
        "calibration_pit_ks", float(torch.tensor(node_vals).mean())
    )


def calibration_sd_ratio(
    predictive_samples: list[torch.Tensor],
    oracle_samples: list[torch.Tensor],
) -> MetricResult:
    """Sharpness ratio (predictive SD / true SD), mean over continuous nodes.

    Companion to PIT-KS. ``predictive_samples[i]`` and ``oracle_samples[i]`` are
    node i's ``[N, S]`` fitted and true-conditional draws. Per-cell value = mean
    over nodes of the per-node mean SD ratio. ~1 is right sharpness; <1
    over-confident, >1 under-confident. Empty input returns ``nan``.
    """
    if not predictive_samples:
        return MetricResult("calibration_sd_ratio", float("nan"))
    node_vals = [
        _sd_ratio_single(s, o) for s, o in zip(predictive_samples, oracle_samples)
    ]
    return MetricResult(
        "calibration_sd_ratio", float(torch.tensor(node_vals).mean())
    )


def map_accuracy(pred: torch.Tensor, true: torch.Tensor) -> MetricResult:
    """Exact-match accuracy for argmax MAP predictions."""
    if pred.shape != true.shape:
        true = true.reshape(pred.shape)
    return MetricResult("map_accuracy", float((pred == true).float().mean()))


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> MetricResult:
    """Brier score: Σ (p_pred - 1[y=k])²."""
    n, k = probs.shape
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.long().reshape(n, 1), 1.0)
    return MetricResult("brier_score", float(((probs - one_hot) ** 2).sum(-1).mean()))


def crps(samples: torch.Tensor, observed: torch.Tensor) -> MetricResult:
    """CRPS via empirical quantile form: E|X - y| - 0.5 * E|X - X'|."""
    s = samples.reshape(-1)
    if s.shape[0] < 2:
        return MetricResult("crps", float("nan"))
    y = observed.reshape(-1)
    cap = min(1024, s.shape[0])
    si = s[torch.randperm(s.shape[0], device=s.device)[:cap]]
    sj = s[torch.randperm(s.shape[0], device=s.device)[:cap]]
    e_xy = (si.unsqueeze(1) - y.unsqueeze(0)).abs().mean()
    e_xx = (si - sj).abs().mean()
    return MetricResult("crps", float(e_xy - 0.5 * e_xx))


# ── speed/memory metrics ────────────────────────────────────────────────────

def query_latency_ms(times_s: list[float]) -> MetricResult:
    """Median wall-clock per query."""
    if not times_s:
        return MetricResult("query_latency_ms", float("nan"))
    t = torch.tensor(times_s)
    return MetricResult("query_latency_ms", float(t.median() * 1000))


def batched_throughput(batch_size: int, total_time_s: float) -> MetricResult:
    return MetricResult("batched_throughput", batch_size / max(total_time_s, 1e-9))


def gpu_peak_mb(device: torch.device) -> MetricResult:
    if device.type != "cuda":
        return MetricResult("gpu_peak_mb", 0.0)
    return MetricResult("gpu_peak_mb", torch.cuda.max_memory_allocated() / (1024 * 1024))


# ── new in v0.3: histogram-TV and k-NN KL for continuous samples ────────────


def tv_continuous(x: torch.Tensor, y: torch.Tensor, *, bins: int = 256) -> MetricResult:
    """Histogram-based total-variation distance between two 1-D sample sets."""
    x = x.reshape(-1).float(); y = y.reshape(-1).float()
    if x.numel() == 0 or y.numel() == 0:
        return MetricResult("tv_continuous", float("nan"))
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if hi <= lo:
        return MetricResult("tv_continuous", 0.0)
    px = torch.histc(x, bins=bins, min=lo, max=hi); px = px / px.sum().clamp_min(1e-12)
    py = torch.histc(y, bins=bins, min=lo, max=hi); py = py / py.sum().clamp_min(1e-12)
    return MetricResult("tv_continuous", float(0.5 * (px - py).abs().sum()))


def kl_div_continuous_knn(
    p_samples: torch.Tensor, q_samples: torch.Tensor, *, k: int = 5,
) -> MetricResult:
    """Kozachenko–Leonenko k-NN estimator of KL(p || q).

    Wang–Kulkarni–Verdú (2009).  Works on n-D continuous samples.  Small
    batches are fine; assumes no exact-tied distances.
    """
    import math
    x = p_samples.float().reshape(p_samples.shape[0], -1)
    y = q_samples.float().reshape(q_samples.shape[0], -1)
    n, d = x.shape
    m = y.shape[0]
    if n < k + 1 or m < k:
        return MetricResult("kl_div_continuous_knn", float("nan"))
    d_xx = torch.cdist(x, x)
    d_xy = torch.cdist(x, y)
    rho_k = d_xx.topk(k + 1, largest=False).values[:, -1].clamp_min(1e-12)
    nu_k = d_xy.topk(k, largest=False).values[:, -1].clamp_min(1e-12)
    kl = float(d * (torch.log(nu_k) - torch.log(rho_k)).mean()
               + math.log(m / max(n - 1, 1)))
    return MetricResult("kl_div_continuous_knn", kl)


def js_normalized(p: torch.Tensor, q: torch.Tensor) -> MetricResult:
    """JS divergence normalised to [0, 1] by dividing by log(2)."""
    import math
    return MetricResult(
        "js_normalized", float(js_divergence(p, q).mean() / math.log(2)),
    )


# ── back-compat aliases (kept for v0.1 examples) ────────────────────────────

def marginal_mae(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p - q).abs().mean()


# ── Pass-9 parametrized per-node aggregator ─────────────────────────────────

def _compute_metrics_per_node(
    *,
    cpd_pairs: list[tuple[torch.Tensor, torch.Tensor]] | tuple = (),
    sample_pairs: list[tuple[torch.Tensor, torch.Tensor]] | tuple = (),
    metrics: tuple[str, ...] = ("tv", "jsd", "w1"),
    n_bins: int = 256,
) -> dict[str, float | None]:
    """Average TV / JSD / W₁ across a collection of CPD or sample pairs.

    Single helper for both pipelines (param-learning + inference) and
    both family classes (discrete + continuous), with hybrid handled
    via per-target dispatch by the caller (discrete sub-targets go in
    ``cpd_pairs``; continuous sub-targets go in ``sample_pairs``).

    Parameters
    ----------
    cpd_pairs : sequence of (true_logits, fit_logits)
        Per-node pairs of tabulated CPDs in logit space (shape
        ``[*parent_cards, K]`` or ``[K]``).  Used for discrete metrics
        (TV, JSD); each pair contributes one TV and one JSD value
        (averaged over parent configurations).  W₁ is **skipped** on
        these pairs — the synthetic generator's discrete categories are
        unordered, so W₁ collapses to ½·TV under any sensible metric
        and is not informative.
    sample_pairs : sequence of (true_samples, fit_samples)
        Per-evaluation-row pairs of 1-D sample tensors (shape
        ``[n_samples]``).  Used for continuous metrics: W₁ via
        ``wasserstein_1d`` (sliced 1-D quantile-CDF) plus binned-TV
        and binned-JSD via ``torch.histc`` with ``n_bins`` bins.
    metrics : tuple of {"tv", "jsd", "w1"}
        Which metrics to compute.  Output keys are
        ``"<name>_per_node"`` matching the long-form parquet's
        ``metric`` column convention.
    n_bins : int, default 256
        Histogram bin count for binned-TV / binned-JSD on
        ``sample_pairs``.  Matches the default in
        :func:`tv_continuous`.

    Returns
    -------
    dict[str, float | None]
        Mapping ``"<metric>_per_node"`` → average value.  ``None`` for
        a metric whose contributing-pair list is empty (e.g. ``"w1"``
        on a pure-discrete cell, or ``"tv"`` on an empty fixture).
        The caller decides whether ``None`` becomes a ``not_supported``
        row, a skipped row, or a NaN.

    Notes
    -----
    Caller responsibility:

    * Build ``cpd_pairs`` for discrete nodes only (use
      ``Mechanism.tabulate(parent_cards)`` from Pass 8).
    * Build ``sample_pairs`` for continuous nodes only (sample from
      both true and fitted mechanisms at matched evidence rows).
    * Hybrid families: send each node's contribution to the right
      list based on ``variable_specs[node][0]``.

    The shape-mismatch guard in the discrete branch (``true_logits.shape
    != fit_logits.shape``) silently drops pairs to match the existing
    ``_avg_tv_per_node`` behaviour from Pass 8 — these correspond to
    cases where the fitted mechanism's parent cardinalities don't
    match the truth (e.g. a categorical didn't see all parent
    configurations in train_data and tabulated a smaller CPT).
    """
    import math

    out: dict[str, list[float]] = {m: [] for m in metrics}

    for true_logits, fit_logits in cpd_pairs:
        if true_logits.shape != fit_logits.shape:
            continue
        true_p = torch.softmax(true_logits, dim=-1)
        fit_p = torch.softmax(fit_logits, dim=-1)
        if "tv" in metrics:
            out["tv"].append(
                float(0.5 * (true_p - fit_p).abs().sum(dim=-1).mean())
            )
        if "jsd" in metrics:
            # js_divergence returns shape [*parent_cards]; mean → scalar.
            # Normalise by log(2) so the value lives in [0, 1].
            jsd_per_row = js_divergence(true_p, fit_p)
            out["jsd"].append(float(jsd_per_row.mean() / math.log(2)))
        # "w1" intentionally skipped: discrete-unordered.

    for true_s, fit_s in sample_pairs:
        true_s = true_s.reshape(-1).float()
        fit_s = fit_s.reshape(-1).float()
        if true_s.numel() == 0 or fit_s.numel() == 0:
            continue
        if "w1" in metrics:
            out["w1"].append(wasserstein_1d(true_s, fit_s).value)
        if "tv" in metrics:
            out["tv"].append(tv_continuous(true_s, fit_s, bins=n_bins).value)
        if "jsd" in metrics:
            lo = float(min(true_s.min(), fit_s.min()))
            hi = float(max(true_s.max(), fit_s.max()))
            if hi <= lo:
                # Degenerate range: identical samples → JSD = 0.
                out["jsd"].append(0.0)
                continue
            true_h = torch.histc(true_s, bins=n_bins, min=lo, max=hi)
            true_h = true_h / true_h.sum().clamp_min(1e-12)
            fit_h = torch.histc(fit_s, bins=n_bins, min=lo, max=hi)
            fit_h = fit_h / fit_h.sum().clamp_min(1e-12)
            out["jsd"].append(
                float(js_divergence(true_h, fit_h) / math.log(2))
            )

    return {
        f"{m}_per_node": (float(sum(vals) / len(vals)) if vals else None)
        for m, vals in out.items()
    }
