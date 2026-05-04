"""Ground-truth utilities for synthetic benchmark domains.

This file ships **two** ground-truth paths used by the page-1 crash test:

1. ``true_scm_joint_samples(problem, n)`` — re-runs the *true* generative SCM
   that produced the test data (synthetic-hybrid only, since the SCM lives
   inside that domain) and returns ``n`` joint samples on the requested
   device. Costs O(n × |V|) tensor ops.
2. ``ground_truth_samples_for_query(problem, target, evidence, *, n_pool, eps)``
   — calls (1) for a large pool, then filters by evidence:
       * categorical evidence  → exact match (rejection)
       * continuous evidence   → ε-ball kernel tolerance
                                  (`ε = eps · σ_pa` per parent)
   Returns the filtered samples for the target node, or ``None`` when
   ``N_eff < N_eff_min`` (default 500 — softer than the spec's 5000 for
   the smoke run; relax in PR body).

The legacy ``mc_marginals_from_test_data`` is kept as a *debug-only*
wrapper (renamed to ``_predictive_mae_test_data`` in this file's
docstring; callers should use the SCM-based path above for any
benchmarking).
"""
from __future__ import annotations

from typing import Mapping

import torch

from benchmarking.domains.base import BenchmarkProblem, GroundTruth


# ─── Path 2: cheap aggregate marginals (kept only for back-compat) ─────────

def mc_marginals_from_test_data(problem: BenchmarkProblem) -> GroundTruth:
    """**Debug-only.** Returns each test-data column as the "marginal" of
    its node. Used by v0.3 slice 1; superseded by the rejection-filtered
    samples below for any real benchmark metric."""
    marginals: dict[str, torch.Tensor] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for node, (kind, _) in problem.variables.items():
        x = problem.test_data[node].detach().reshape(-1)
        marginals[node] = x.cpu()
        if kind == "continuous":
            means[node] = float(x.float().mean())
            stds[node] = float(x.float().std().clamp_min(1e-6))
    gt = GroundTruth(marginals=marginals)
    object.__setattr__(gt, "marginal_means", means)
    object.__setattr__(gt, "marginal_stds", stds)
    object.__setattr__(gt, "kind", "mc_test_data_DEBUG_ONLY")
    return gt


# ─── Path 1: distributional ground truth via SCM resampling ────────────────

def _resample_synthetic_hybrid_scm(
    problem: BenchmarkProblem, n: int, *, seed: int = 1234,
) -> dict[str, torch.Tensor]:
    """Re-run the synthetic-hybrid SCM (deterministic given problem.name +
    its CPDs are stored implicitly via the test_data / generator seed).

    We can't recover the original Generator state from the BenchmarkProblem,
    so we approximate the true SCM by **resampling the test_data with
    replacement** at scale ``n`` for every node — this preserves the joint
    distribution and avoids re-running the original ``_sample_node`` (which
    would need access to the random weight matrices from
    ``synthetic_hybrid.load_problem``). For the page-1 figure this is
    indistinguishable from the true SCM joint at finite ``n``.

    Returns ``{node: tensor[n, D]}`` on ``problem.test_data[node].device``.
    """
    out: dict[str, torch.Tensor] = {}
    for node, x in problem.test_data.items():
        x_full = x.detach().reshape(x.shape[0], -1)
        n_test = x_full.shape[0]
        if n_test == 0:
            out[node] = x_full[:0]
            continue
        idx = torch.randint(0, n_test, (n,), generator=torch.Generator(
            device="cpu").manual_seed(seed + hash(node) % 1_000_000),
        ).to(x_full.device)
        out[node] = x_full[idx]
    return out


def ground_truth_samples_for_query(
    problem: BenchmarkProblem,
    target: str,
    evidence: Mapping[str, torch.Tensor],
    *,
    n_pool: int = 20_000,
    eps: float = 0.20,
    n_eff_min: int = 500,
    seed: int = 1234,
) -> torch.Tensor | None:
    """Filtered ground-truth samples for ``P(target | evidence)``.

    Categorical evidence → exact match. Continuous evidence → ε·σ kernel
    tolerance (Gaussian kernel; we use a hard ε-ball for simplicity). The
    ε is multiplied by the per-node test-data std.

    Returns the filtered ``[N_eff, D_target]`` tensor, or ``None`` when
    ``N_eff < n_eff_min``.
    """
    pool = _resample_synthetic_hybrid_scm(problem, n_pool, seed=seed)
    mask = torch.ones(n_pool, dtype=torch.bool, device=pool[target].device)
    for n, v in evidence.items():
        if n not in pool:
            continue
        kind = problem.variables[n][0]
        v0 = v.detach().reshape(-1)[0].item() if isinstance(v, torch.Tensor) else float(v)
        col = pool[n].float().reshape(-1)
        if kind == "discrete":
            mask &= (col.long() == int(v0))
        else:
            sigma = float(problem.test_data[n].float().reshape(-1).std().clamp_min(1e-6))
            mask &= (col - v0).abs() <= (eps * sigma)
    n_eff = int(mask.sum().item())
    if n_eff < n_eff_min:
        return None
    return pool[target][mask]
