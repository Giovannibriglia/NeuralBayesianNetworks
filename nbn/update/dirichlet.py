"""Conjugate Dirichlet-count accumulation for tabular categorical CPDs.

The posterior over a categorical CPD under a Dirichlet prior is itself a
Dirichlet whose parameters are prior pseudo-counts plus observed counts.
Persisting the count table as sufficient statistics therefore makes
incremental update exact: ``accumulate`` adds the new-data counts to the
stored table, and a chunked update reproduces a single fit on the pooled
data (no rehearsal, posterior-as-prior).
"""
from __future__ import annotations

import torch


def counts_to_logits(counts: torch.Tensor) -> torch.Tensor:
    """Normalise a ``[R, K]`` count table into log-probabilities."""
    probs = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.log(probs.clamp_min(1e-12))


def new_counts(x_idx, parent_row_idx, n_parent_states, n_classes):
    """Vectorized ``[R, K]`` count table of raw new-data observations."""
    device = x_idx.device
    flat = parent_row_idx * n_classes + x_idx
    counts = torch.zeros(n_parent_states * n_classes, device=device, dtype=torch.float)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
    return counts.reshape(n_parent_states, n_classes)


def accumulate(prior_counts, batch_counts, *, forgetting: float = 1.0):
    """Posterior counts = ``forgetting * prior + batch`` (recursive Bayes)."""
    if prior_counts.shape != batch_counts.shape:
        raise ValueError(
            f"count-table shape mismatch: prior {tuple(prior_counts.shape)} vs "
            f"batch {tuple(batch_counts.shape)}; declare parent/class cardinalities "
            "via the network Variable specs so the table is fixed across fit/update.")
    return forgetting * prior_counts + batch_counts
