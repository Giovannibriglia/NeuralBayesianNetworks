"""Protocol definitions for v0.13 benchmark architecture.

Composition-based: a Benchmark is the product of a ProblemSource,
a QuerySelector, a Measurement, and a set of BaselineAdapters.

Reference: docs/v0.13-benchmark-redesign.md §4.1
"""
from __future__ import annotations

from typing import Any, Iterator, Protocol, runtime_checkable

from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.domains.posterior import Posterior
from benchmarking.core.results import CellResult


@runtime_checkable
class ProblemSource(Protocol):
    """Produces BenchmarkProblems with ground-truth structure + CPDs.

    Each problem includes a `true_model` from which `train_data` is
    sampled. Concrete implementations: SyntheticProblemSource (Phase 1b),
    BnlearnProblemSource (Phase 4).

    The training data sampling lives on the implementation, not on the
    runner. This is uniform across synthetic and bnlearn problem sources.
    """

    def iter_problems(self, cfg: Any) -> Iterator[BenchmarkProblem]:
        """Yield BenchmarkProblems matching the config.

        `cfg` is a config object specific to the problem source
        (e.g., SyntheticConfig has n_nodes_list, BnlearnConfig has
        networks). Each problem has problem.train_data already
        sampled from problem.true_model.
        """
        ...


@runtime_checkable
class QuerySelector(Protocol):
    """Selects queries given a DAG and a budget.

    Concrete implementations: UniformRandomSelector (Phase 1c,
    placeholder), TopologicalAllocator (Phase 2), HeaviestQueryByRole
    (Phase 3).

    If `n_queries` exceeds the number of unique queries possible on
    the problem, the selector caps at the maximum unique queries
    available (no duplicates).
    """

    def select(
        self,
        problem: BenchmarkProblem,
        n_queries: int,
        seed: int,
    ) -> list[Query]:
        """Return a list of queries, ordered. Deterministic given seed."""
        ...


@runtime_checkable
class Measurement(Protocol):
    """Computes per-query measurements (timing, accuracy, or both).

    Concrete implementations: AccuracyAndTiming (Phase 1c),
    TimingOnly (Phase 3).

    The measurement runs `adapter.query(q)` per query, timing each
    call individually. Fit time and metrics time are recorded but
    NOT counted toward the per-cell timeout.
    """

    def measure(
        self,
        problem: BenchmarkProblem,
        adapter: BaselineAdapter,
        queries: list[Query],
        *,
        query_budget_s: float = float("inf"),
    ) -> list[CellResult]:
        """Return one CellResult per query, with timing and (optionally) accuracy.

        ``query_budget_s`` is a soft cumulative timeout on query_time_s. When
        the cumulative total of adapter.query() wall-clock exceeds this budget,
        remaining unstarted queries receive status="timeout" rows with
        query_time_s=NaN. Fit and metrics time are NOT gated by this budget.
        Default float("inf") disables the timeout (existing behaviour).
        """
        ...


@runtime_checkable
class BaselineAdapter(Protocol):
    """Stateful fit-then-query contract.

    Replaces BaselineAdapterV2 (which was functional/stateless) and
    removes the BaselineAdapterV2Shim. State is stored on `self`
    after fit(); query() reads from self.

    Per-cell lifecycle: each cell creates a fresh adapter instance,
    calls fit() once, then queries N times. Adapter instances are
    NOT reused across cells.

    Concrete implementations: NBNAdapter, PgmpyAdapter, PyroAdapter,
    GpytorchAdapter, PomegranateAdapter (all Phase 1b).
    """

    name: str  # e.g., "nbn-cat-lw", "pgmpy-mle-ve"

    def fit(self, problem: BenchmarkProblem, **kwargs: Any) -> None:
        """Fit on problem.train_data. State is stored on self.

        `**kwargs` is for adapter-specific knobs (e.g., epochs for
        NBN, device for pyro). Common patterns flow through the
        runner; per-adapter specifics are set at instantiation.
        """
        ...

    def query(self, query: Query) -> Posterior:
        """Query the fitted model. This call is what gets timed."""
        ...

    def is_applicable(self, problem: BenchmarkProblem) -> bool:
        """Return False if this adapter cannot handle problem.family.

        Replaces the static _BASELINE_APPLICABILITY table from the
        v0.12 registry. The runner consults this method to skip
        inapplicable (adapter, problem) pairs cleanly.
        """
        ...
