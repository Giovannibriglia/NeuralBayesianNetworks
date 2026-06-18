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

    def select_groups(
        self,
        problem: BenchmarkProblem,
        n_queries: int,
        seed: int,
        *,
        batch_size: int = 1,
    ) -> list[list[Query]]:
        """Return grouped queries (v0.14 batched queries, #148).

        Outer list: groups whose queries share (targets, evidence_keys),
        chunked to length <= ``batch_size``. With the selector's
        ``n_batch_queries=1`` (default), every inner list has length 1
        and wraps the exact Query objects ``select()`` returns —
        identity behavior for existing benchmarks.

        Deterministic given seed.
        See docs/v0.14-batched-queries-design.md §1.3, §2.3.
        """
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

    Optional capability — parameter-learning scoring (#109)
    -------------------------------------------------------
    Adapters MAY additionally support parameter-learning (PL) mode, in
    which a fitted model is scored on held-out ``problem.test_data``
    rather than queried. This capability is OPT-IN and deliberately NOT a
    structural member of this protocol — adding a required method would
    break ``isinstance(adapter, BaselineAdapter)`` for adapters that do
    not (yet) implement it. It mirrors the ``supports_batched_queries``
    flag precedent (a concrete-class attribute checked via ``getattr``,
    never declared here).

    An adapter opts in by declaring BOTH, as concrete-class members:

        supports_scoring: bool = True

        def score_data(self, test_data: dict[str, torch.Tensor]) -> torch.Tensor:
            '''Per-row joint log-prob of held-out rows under the fitted
            model. Returns a 1-D tensor of shape ``[B]`` (one joint
            log-probability per test row, summed over the graph's nodes).
            ``ParamLearningMeasurement`` feeds this straight into
            ``metrics.log_likelihood`` (mean over rows).'''

    ``ParamLearningMeasurement`` gates on the flag:
    ``getattr(adapter, "supports_scoring", False)``. Adapters that do not
    set it (the default — flag absent → ``False``) are reported with
    ``status="not_supported"`` and ``score_data`` is never called on
    them. NBNAdapter is the first implementer (#109 PR 1); pgmpy /
    pomegranate / pyro scoring land in later PRs of the series.

    Optional capability — parameter recovery (#109 PR 2)
    ----------------------------------------------------
    Adapters MAY additionally expose their LEARNED discrete CPTs so the PL
    benchmark can measure how well parameter learning recovers the true
    CPDs (``param_recovery_tv`` headline, ``param_recovery_kl`` companion).
    Same OPT-IN / non-structural design as scoring above (mirrors
    ``supports_batched_queries``). An adapter opts in by declaring BOTH, as
    concrete-class members:

        supports_param_recovery: bool = True

        def extract_learned_cpts(self) -> dict[str, torch.Tensor]:
            '''Learned discrete CPTs, one entry per DISCRETE node whose
            parents are all discrete. Each value is a dense probability
            tensor of shape ``[n_parent_configs, K]`` in a CANONICAL,
            adapter-internal-order-INDEPENDENT layout so tables from any
            adapter (and the true model) compare cell-by-cell:

              * parents sorted lexicographically by name;
              * parent configs in row-major order (first parent slowest),
                each parent ranging over ``0..card-1``;
              * columns are classes ``0..K-1``.

            Each row is a proper distribution (sums to 1). Nodes that are
            continuous, or discrete-with-continuous-parents, are OMITTED —
            recovery is a fully-discrete-network metric.'''

    ``ParamLearningMeasurement`` gates on ``getattr(adapter,
    "supports_param_recovery", False)``: adapters without it get
    ``param_recovery_*`` rows with ``status="not_supported"`` and
    ``extract_learned_cpts`` is never called. Even when supported, the
    rows are ``status="not_applicable"`` for non-fully-discrete problems
    (continuous / hybrid, or a problem with no NBN ``true_model``). The
    measurement extracts the TRUE CPTs from ``problem.true_model`` with the
    SAME canonical layout, so the two align. NBNAdapter is the first
    implementer (#109 PR 2); pgmpy / pomegranate / pyro land in later PRs.
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

    def query_batch(self, queries: list[Query]) -> list[Posterior]:
        """Process a list of queries; return Posteriors in input order.

        Adapters that support library-level batching SHOULD override
        this. The default implementation (the module-level
        ``default_query_batch`` helper) loops ``query()`` sequentially.

        Queries within a single call SHOULD share
        (target, frozenset(evidence_keys)). The selector (when
        ``n_batch_queries > 1``) emits such groups; adapters MAY assume
        this and exploit it for batching. The default helper makes no
        such assumption.

        Failure semantics: this method does NOT promise atomic
        semantics. A failure partway through MUST raise;
        partial-success returns are not supported.
        See docs/v0.14-batched-queries-design.md §1.2, §4.4.
        """
        ...

    def is_applicable(self, problem: BenchmarkProblem) -> bool:
        """Return False if this adapter cannot handle problem.family.

        Replaces the static _BASELINE_APPLICABILITY table from the
        v0.12 registry. The runner consults this method to skip
        inapplicable (adapter, problem) pairs cleanly.
        """
        ...


# ---------------------------------------------------------------------------
# Default query_batch helper
# ---------------------------------------------------------------------------

def default_query_batch(
    adapter: BaselineAdapter,
    queries: list[Query],
) -> list[Posterior]:
    """Sequential fallback for adapters without library-level batching.

    Adapters that don't implement library-level batching opt in
    explicitly by delegating to this helper::

        def query_batch(self, queries):
            return default_query_batch(self, queries)

    Posteriors are returned in input order. A failure partway through
    raises (no partial-success returns) — see the protocol docstring.

    See docs/v0.14-batched-queries-design.md §1.2, §3.3.
    """
    return [adapter.query(q) for q in queries]
