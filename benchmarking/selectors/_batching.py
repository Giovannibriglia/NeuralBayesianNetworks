"""Shared n_batch_queries expansion + chunking for selectors (v0.14, #148).

Implements design doc §2.1-§2.3: for each query position a selector picks,
produce ``n_batch_queries`` evidence-value variants by sampling rows from
``problem.train_data``, then chunk the variants by ``batch_size`` into
inner lists. Every selector's ``select_groups`` delegates here — the
position-selection logic stays in each selector's existing ``select()``.

At ``n_batch_queries=1`` the original Query objects pass through untouched
(each wrapped in a length-1 inner list) — identity behavior for all
existing benchmarks.

Reference: docs/v0.14-batched-queries-design.md §2.
"""
from __future__ import annotations

import torch

from benchmarking.domains.base import BenchmarkProblem, Query


def make_query_groups(
    base_queries: list[Query],
    problem: BenchmarkProblem,
    *,
    n_batch_queries: int = 1,
    batch_size: int = 1,
    seed: int = 0,
) -> list[list[Query]]:
    """Expand base queries into grouped evidence-value variants.

    For each query position (one entry of ``base_queries``):
      1. Sample ``n_batch_queries`` rows from ``problem.train_data``
         (without replacement when n_train >= N, with replacement
         otherwise — design doc §2.2)
      2. Build one Query variant per sampled row: same targets / kind /
         role metadata, evidence values read off the sampled row for the
         same evidence keys. ``None`` (empty-mode) evidence values are
         preserved as ``None`` — variants keep the position's mode.
      3. Chunk the N variants into ``ceil(N / batch_size)`` inner lists
         of length <= batch_size, emitted consecutively (§2.3)

    With ``n_batch_queries=1``: returns ``[[q] for q in base_queries]``
    with the original Query objects (bit-identical, no sampling).

    Deterministic given ``seed`` (§2.5): the RNG is derived from the
    cell seed and the position index.
    """
    if n_batch_queries < 1:
        raise ValueError(f"n_batch_queries must be >= 1, got {n_batch_queries}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    if n_batch_queries == 1:
        # Identity at defaults: original objects, length-1 inner lists.
        return [[q] for q in base_queries]

    train = problem.train_data
    n_train = next(iter(train.values())).shape[0] if train else 0
    if n_train == 0:
        raise ValueError(
            "make_query_groups requires problem.train_data to sample "
            "evidence-value variants (design doc §2.1)."
        )

    groups: list[list[Query]] = []
    for pos_idx, base in enumerate(base_queries):
        # Per-position derived seed: same cell seed -> same rows, but
        # positions don't all reuse identical row indices.
        gen = torch.Generator(device="cpu").manual_seed(
            seed * 1000 + 13 + pos_idx
        )
        if n_train >= n_batch_queries:
            idx = torch.randperm(n_train, generator=gen)[:n_batch_queries]
        else:
            idx = torch.randint(
                0, n_train, (n_batch_queries,), generator=gen
            )

        variants: list[Query] = []
        for i in idx.tolist():
            evidence = {
                k: (None if v is None
                    else train[k][i].reshape(()).float())
                for k, v in base.evidence.items()
            }
            variants.append(
                Query(
                    targets=base.targets,
                    evidence=evidence,
                    kind=base.kind,
                    query_role=base.query_role,
                    query_kind=base.query_kind,
                    evidence_strategy=base.evidence_strategy,
                )
            )

        for start in range(0, n_batch_queries, batch_size):
            groups.append(variants[start:start + batch_size])

    return groups
