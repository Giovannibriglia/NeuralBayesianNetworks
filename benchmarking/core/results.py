"""v0.13 parquet schema (v3) data classes.

Reference: docs/v0.13-benchmark-redesign.md §6
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CellResult:
    """One parquet row in the v3 schema.

    One CellResult per (cell, query, metric). With N queries per cell
    and M metrics per query, a single cell produces N * M rows.
    fit_time_s and metrics_time_s are repeated identically across all
    rows of a given cell (per-cell costs, repeated for query-row
    joinability).

    Mode-aware semantics:
    - Inference mode: `query` is a posterior query p(target | evidence).
    - Parameter learning mode: `query` is a per-CPD scoring call.
      fit_time_s is the fitter time (not gated by timeout).

    Reference: docs/v0.13-benchmark-redesign.md §3, §6
    """

    # Identity columns
    benchmark: str      # "synthetic" | "scalability" | "bnlearn"
    family: str         # "discrete" | "continuous_lg" | "continuous_nongauss" | "hybrid"
    problem_id: str     # n_nodes as string (synthetic) | network name (bnlearn)
    seed: int           # RNG seed (1 for bnlearn; DAG is fixed, training data resampled)
    baseline: str       # e.g., "nbn-cat-lw"
    query_role: str     # "hub" | "cut" | "terminal" | "random"
                        # | "heaviest_hub" | "heaviest_cut" | "longest_propagation"

    # Measurement columns
    metric: str         # e.g., "tv_per_node", "w1_per_node", "log_likelihood"
    value: float        # the metric value
    status: str         # "ok" | "timeout" | "oom" | "error" | "not_supported"

    # Timing columns (the #107 fix)
    fit_time_s: float       # adapter.fit() wall-clock, NOT counted toward timeout
    query_time_s: float     # adapter.query() wall-clock, what timeout wraps
    metrics_time_s: float   # accuracy/score computation, NOT counted toward timeout

    # Error context
    error_msg: str | None = None

    # Phase 2 per-query metadata (defaults keep pre-Phase-2 selectors working)
    query_kind: str = "prediction"          # "prediction" | "diagnosis"
    evidence_strategy: str = "random"       # "longest_path" | "mb_neighbors" | "random"

    # Phase 3 per-query metadata (default "full" keeps pre-Phase-3 selectors working)
    evidence_mode: str = "full"             # "full" | "empty"


# Valid status values (for validation in implementations)
VALID_STATUSES = frozenset({"ok", "timeout", "oom", "error", "not_supported"})

# Valid benchmark names (for validation)
VALID_BENCHMARKS = frozenset({"synthetic", "scalability", "bnlearn"})

# Valid evidence modes (Phase 3): "full" = concrete evidence values (V1);
# "empty" = evidence variables marginalized over None values (V2).
VALID_EVIDENCE_MODES = frozenset({"full", "empty"})

# Valid query roles (for validation; future-extensible)
VALID_QUERY_ROLES_SYNTHETIC = frozenset({"hub", "cut", "terminal", "random"})
VALID_QUERY_ROLES_SCALABILITY = frozenset({
    "heaviest_hub", "heaviest_cut", "longest_propagation",
})
VALID_QUERY_ROLES_ALL = VALID_QUERY_ROLES_SYNTHETIC | VALID_QUERY_ROLES_SCALABILITY
