"""Ground-truth oracle helpers for v0.13 accuracy measurements.

Extracted from ``nbn/bench/crash_test_runner.py`` (v0.12 private helpers
``_filter_ground_truth`` and ``_forward_with_clamp_posterior_samples``),
adapted to work against ``BenchmarkProblem`` instead of ``SyntheticBN``.

These helpers are *core* capabilities shared by ``AccuracyAndTiming`` and
potentially by audit scripts, diagnostics, and future measurements.  Placing
them in ``nbn/bench/core/`` rather than inside the measurement package
ensures they remain accessible without importing the full measurement stack.

Reference: docs/v0.13-benchmark-redesign.md §4.1, §5.2
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from nbn.bench.domains.base import BenchmarkProblem


# ---------------------------------------------------------------------------
# Column-order helper (used by both oracle functions)
# ---------------------------------------------------------------------------

def _column_order(problem: BenchmarkProblem) -> tuple[str, ...]:
    """Return the topological-sort column order for ``problem``.

    Mirrors the ``SyntheticBN.column_order`` convention established in
    ``make_synthetic_bn``: columns in ``ground_truth.samples`` are written
    in this order, so any column-indexed access must use this same order.

    Implementation notes
    --------------------
    * ``problem.dag`` may be a ``list[tuple[str, str]]`` (edges only) **or**
      an ``nx.DiGraph`` (the test-helper convention).  Both are handled by
      ``add_edges_from``.
    * Isolated nodes (no edges) are only present in ``problem.variables``,
      *not* in the edge list.  ``add_nodes_from(problem.variables)`` ensures
      they are included before ``topological_sort`` is called.  Omitting
      this step would give a different (shorter) sort than the one used when
      ``ground_truth.samples`` was written.
    * ``nx.topological_sort`` on a fixed graph is deterministic (DFS-based),
      so the result is stable across calls given the same graph.

    Complexity: O(|V| + |E|), negligible for benchmark-scale DAGs.
    """
    import networkx as nx

    dag_nx = nx.DiGraph()
    dag_nx.add_nodes_from(problem.variables)
    dag_nx.add_edges_from(problem.dag)
    return tuple(nx.topological_sort(dag_nx))


# ---------------------------------------------------------------------------
# Oracle 1: discrete — exact-match rejection filter
# ---------------------------------------------------------------------------

def filter_ground_truth(
    problem: BenchmarkProblem,
    evidence_row: dict[str, torch.Tensor],
    target: str,
    *,
    eps_factor: float = 0.50,
    n_eff_min: int = 10,
) -> torch.Tensor | None:
    """Exact-match rejection filter on ``problem.ground_truth.samples``.

    Used for **discrete** targets.  Filters the joint reference-sample pool
    to rows matching the evidence, then returns the target column of the
    surviving rows.

    Adapted from ``crash_test_runner._filter_ground_truth`` (v0.12).
    The main structural change: accepts ``BenchmarkProblem`` instead of
    ``SyntheticBN``; column index is looked up via ``_column_order(problem)``
    rather than ``bn.column_index(name)``.

    Parameters
    ----------
    problem:
        A ``BenchmarkProblem`` whose ``ground_truth.samples`` holds the
        reference pool in ``_column_order(problem)`` column order.
    evidence_row:
        A ``{node_name: scalar_tensor}`` dict for one query row.
    target:
        The target node name.
    eps_factor:
        For continuous evidence nodes: accept reference rows within
        ``eps_factor * std(col)`` of the evidence value.
    n_eff_min:
        Minimum surviving rows required; return ``None`` if fewer survive.

    Returns
    -------
    Filtered target-column samples, shape ``[n_eff]``, on CPU, float32.
    ``None`` if the ground-truth pool is absent or too few rows survive.
    """
    if problem.ground_truth is None or problem.ground_truth.samples is None:
        return None
    samples = problem.ground_truth.samples
    if samples.numel() == 0:
        return None

    col_order = _column_order(problem)
    col_idx = {name: i for i, name in enumerate(col_order)}

    # v0.5c device-safety fix (preserved): mask must live on the same device
    # as samples.  Evidence values may arrive on CPU while samples are on GPU
    # (hybrid/cuda smoke runs).
    mask = torch.ones(samples.shape[0], dtype=torch.bool, device=samples.device)

    for node, val in evidence_row.items():
        if val is None:
            continue  # empty-mode evidence: marginalize, don't filter (Phase 3)
        if node not in col_idx:
            continue  # node not in column map — skip safely
        idx = col_idx[node]
        col = samples[:, idx]
        kind = problem.variables[node][0]
        raw = val.item() if isinstance(val, torch.Tensor) else val
        if kind == "discrete":
            mask &= (col.long() == int(raw))
        else:
            v = torch.as_tensor(raw, device=col.device, dtype=col.dtype)
            sigma = col.std().clamp_min(1e-3)
            mask &= (col - v).abs() < eps_factor * sigma

    if int(mask.sum().item()) < n_eff_min:
        return None

    target_idx = col_idx[target]
    return samples[mask, target_idx].cpu().float()


# ---------------------------------------------------------------------------
# Oracle 2: continuous/hybrid — forward-with-clamp ancestral sampling
# ---------------------------------------------------------------------------

def forward_with_clamp_posterior_samples(
    problem: BenchmarkProblem,
    targets: list[str],
    evidence: dict[str, torch.Tensor],
    *,
    n_samples: int = 2000,
) -> torch.Tensor | None:
    """Posterior samples via ancestral sampling with evidence clamped.

    Used for **continuous** and **hybrid-continuous-target** queries.

    For SCMs of the form ``X_j = f_j(X_{pa(j)}) + ε_j`` with ε_j
    independent of ``{X_k : k ≠ j}``, ancestral sampling that *clamps*
    evidence nodes to their observed values yields exact samples from
    ``p(T | E=e)``.  This is what ``true_model.sample(n, evidence=...)``
    implements.

    Adapted from ``crash_test_runner._forward_with_clamp_posterior_samples``
    (v0.12).  Structural change: accepts ``BenchmarkProblem`` instead of
    ``SyntheticBN``; uses ``problem.true_model`` instead of ``bn.true_model``.

    Parameters
    ----------
    problem:
        A ``BenchmarkProblem`` with ``true_model`` populated.  Returns
        ``None`` immediately if ``problem.true_model`` is ``None`` (bnlearn
        networks, Phase 4).
    targets:
        List of target node names whose posterior to sample.
    evidence:
        A ``{node: scalar_tensor_or_value}`` dict for one query row.
        Scalars and 0-dim tensors are both handled.
    n_samples:
        Number of posterior samples to draw.

    Returns
    -------
    Tensor of shape ``[n_samples, len(targets)]`` on CPU, float32.
    ``None`` on any exception or if ``true_model`` is unavailable.
    """
    if problem.true_model is None:
        return None

    # Normalise 0-dim tensors to [1]-shaped for the engine contract.
    # Empty-mode evidence (None values, Phase 3) is skipped so the engine
    # marginalizes over those variables rather than conditioning on them.
    ev = {
        k: (v.reshape(1) if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
        for k, v in evidence.items()
        if v is not None
    }
    with torch.no_grad():
        try:
            samples = problem.true_model.sample(n=n_samples, evidence=ev)
        except Exception:
            return None

    cols = []
    for t in targets:
        col = samples[t]
        if col.dim() >= 2 and col.shape[-1] == 1:
            col = col.squeeze(-1)
        cols.append(col.float().cpu().reshape(n_samples, -1))
    return torch.cat(cols, dim=-1)
