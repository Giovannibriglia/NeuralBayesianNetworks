"""Shared Phase-1 group dispatch for measurements (PR 4 stage b, #148).

Runs the selector's query groups through the adapter, timing each group
call and amortizing per-query time. Used by both ``AccuracyAndTiming``
and ``TimingOnly`` so the dispatch / failure / budget semantics stay in
one place.

Dispatch rule:
  * B == 1 → ``adapter.query(q)`` — the exact pre-batching code path,
    so n_batch_queries=1 runs are identical to pre-PR-4 master.
  * B > 1  → ``adapter.query_batch(group)`` (single library call);
    adapters without query_batch (legacy test fixtures) fall back to
    looping ``query()``.

Failure semantics: the batch is atomic. A raise
from query_batch marks all B queries with the classified status. The
per-query time for a failed batched call is NaN (§4.7); a failed B=1
call keeps the measured elapsed time — the pre-batching convention.

Budget semantics: the soft cumulative ``query_budget_s`` is checked
before each group call (group granularity, matching the pre-batching
per-query granularity at B=1). Groups that never start are returned for
per-group timeout-row emission — one row-set per group, not per query,
per the §4.3 sentinel principle (sentinels mark "didn't complete", not
one row per planned evidence value).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from nbn.bench.core.runner import _classify_exception
from nbn.bench.domains.base import Query


def _device_sync(device: Any) -> None:
    """``torch.cuda.synchronize`` / ``torch.mps.synchronize`` for an adapter
    on an accelerator, so per-query wall-clock includes the queued kernels."""
    dev = str(device or "cpu")
    try:
        import torch

        if dev.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(dev)
        elif dev.startswith("mps"):
            backend = getattr(torch.backends, "mps", None)
            if backend is not None and backend.is_available():
                torch.mps.synchronize()
    except Exception:  # pragma: no cover - a sync failure must not fail a query
        pass


def pass_batch_size(groups: list[list[Query]]) -> int:
    """The batch size of a measure() pass: its largest group (1 if empty)."""
    return max((len(g) for g in groups), default=1)


def stamped_batch_size(group: list[Query], pass_b: int) -> int:
    """``batch_size`` recorded on the rows of ``group``.

    ``len(group)`` (queries per library call), except for an evidence-free
    singleton: since D7 such a position is ONE query in every pass instead of
    ``pass_b`` identical copies, and its rows are stamped with the pass's
    batch size so they stay in that pass's ``batch_size`` slice. Stamping 1
    pooled the copies from every pass of a sweep into the B=1 slice (twice
    the evidence-free rows at B=1, with clashing ``query_idx``).
    """
    evidence = getattr(group[0], "evidence", None) if len(group) == 1 else None
    if isinstance(evidence, dict) and all(v is None for v in evidence.values()):
        return pass_b
    return len(group)


@dataclass
class QueryOutcome:
    """Per-query outcome from group dispatch (flat order)."""

    posterior: Any | None    # None on failure
    error_msg: str | None
    status: str              # "ok" or classified failure
    query_time_s: float      # amortized batch_time/B; NaN for failed B>1
    batch_size: int          # stamped batch size (see stamped_batch_size)
    flat_index: int          # index into the flattened query list


def run_query_groups(
    adapter: Any,
    groups: list[list[Query]],
    *,
    query_budget_s: float = float("inf"),
) -> tuple[list[QueryOutcome], list[tuple[list[Query], int]]]:
    """Dispatch groups through the adapter.

    Returns ``(outcomes, unstarted)``:
      * outcomes — one ``QueryOutcome`` per query that was part of a
        started group, in flat order
      * unstarted — ``(group, flat_start_index)`` for every group that
        never started because the cumulative budget was exhausted
    """
    outcomes: list[QueryOutcome] = []
    unstarted: list[tuple[list[Query], int]] = []
    cumulative = 0.0
    qi = 0
    pass_b = pass_batch_size(groups)

    for group in groups:
        b = len(group)
        b_stamp = stamped_batch_size(group, pass_b)
        if cumulative >= query_budget_s:
            unstarted.append((group, qi))
            qi += b
            continue

        t0 = time.perf_counter()
        posteriors: list[Any] | None
        try:
            if b == 1:
                # Pre-batching code path — identical to master at B=1.
                posteriors = [adapter.query(group[0])]
            else:
                query_batch = getattr(adapter, "query_batch", None)
                if query_batch is not None:
                    posteriors = query_batch(group)
                else:
                    # Legacy fixture without query_batch: sequential.
                    posteriors = [adapter.query(q) for q in group]
                if posteriors is None or len(posteriors) != b:
                    raise RuntimeError(
                        f"query_batch returned "
                        f"{0 if posteriors is None else len(posteriors)} "
                        f"posteriors for a batch of {b}"
                    )
            # Asynchronous accelerators: wait for the kernels the call queued
            # before stopping the clock (no-op on CPU / CPU-only adapters).
            _device_sync(getattr(adapter, "device", "cpu"))
            status = "ok"
            err_msg = None
        except Exception as exc:
            posteriors = None
            status = _classify_exception(exc)
            err_msg = str(exc)
        elapsed = time.perf_counter() - t0
        cumulative += elapsed

        for j in range(b):
            if posteriors is None:
                # Atomic batch failure (§4.2/§4.4). B=1 keeps the
                # measured elapsed (pre-batching convention); B>1 is NaN
                # per §4.7 (a failed batch has no meaningful per-query time).
                q_time = elapsed if b == 1 else float("nan")
                outcomes.append(QueryOutcome(
                    posterior=None, error_msg=err_msg, status=status,
                    query_time_s=q_time, batch_size=b_stamp, flat_index=qi + j,
                ))
            else:
                outcomes.append(QueryOutcome(
                    posterior=posteriors[j], error_msg=None, status="ok",
                    query_time_s=elapsed / b, batch_size=b_stamp, flat_index=qi + j,
                ))
        qi += b

    return outcomes, unstarted
