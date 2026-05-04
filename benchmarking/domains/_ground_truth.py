"""Ground-truth utilities for synthetic benchmark domains.

v0.3 ships a single, cheap, honest ground-truth source: **MC empirical
marginals** taken from the test-set samples that ``synthetic_hybrid``
already draws from the true SCM.  For every node it records:

    GroundTruth.marginals[node]                # 1-D tensor of test samples
    GroundTruth.marginal_means[node]           # scalar mean of those samples
    GroundTruth.marginal_stds[node]            # scalar std

These are the cheapest meaningful ground truth: they cost nothing extra
(``problem.test_data`` is already cached), they are honest about being
finite-MC estimates, and they let every continuous accuracy metric in
``benchmarking/metrics.py`` (W₁, MAE, energy, KL-knn) score against a
common reference.

The deferred Lauritzen–Jensen analytic-CG sampler and NUTS gold-standard
paths are tracked in v0.3.x follow-up issues.  Wiring is identical
because they all populate the same ``GroundTruth`` fields — only the
*source* of the samples changes.
"""
from __future__ import annotations

from typing import Mapping

import torch

from benchmarking.domains.base import BenchmarkProblem, GroundTruth


def mc_marginals_from_test_data(problem: BenchmarkProblem) -> GroundTruth:
    """Return a populated ``GroundTruth`` whose ``marginals[node]`` holds the
    flattened test-data column for that node.  Both discrete and continuous
    nodes are supported; discrete nodes carry int labels that the metric
    code can histogram.

    For continuous nodes additional ``marginal_means`` / ``marginal_stds``
    fields are attached as side dicts (the dataclass is frozen, so we use a
    simple wrapper)."""
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
    # Stash the continuous summary on the dataclass without breaking frozen.
    object.__setattr__(gt, "marginal_means", means)
    object.__setattr__(gt, "marginal_stds", stds)
    object.__setattr__(gt, "kind", "mc_test_data")
    return gt


def per_query_reference_means(
    problem: BenchmarkProblem,
    target: str,
    evidence: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor | None:
    """Return ``[B]`` reference means for a continuous target under a batch
    of evidence assignments.

    Cheap implementation: the target's overall MC mean from the test set.
    Doesn't condition on evidence — that's the v0.3.x analytic-CG path.
    Sufficient to populate the page-1 figure with an honest "mean abs
    error vs. MC reference" bar.
    """
    if target not in problem.test_data:
        return None
    x = problem.test_data[target].detach().reshape(-1).float()
    B = 1
    if evidence:
        any_v = next(iter(evidence.values()))
        if isinstance(any_v, torch.Tensor) and any_v.dim() >= 1:
            B = any_v.shape[0]
    return x.mean().expand(B).cpu()
