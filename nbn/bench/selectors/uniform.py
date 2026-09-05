"""UniformRandomSelector — uniform random query selection.

Implements the v0.13 ``QuerySelector`` protocol: selects a list of
``Query`` objects (one per row, scalar evidence values) by uniform
random target + evidence assignment over the test data.

This is the placeholder selector used by all v0.13 Phase 1 benchmarks.
``TopologicalAllocator`` (Phase 2, #74) will replace it for paper runs
and expose ``query_role`` values beyond ``"random"``.

Reference: docs/v0.13-benchmark-redesign.md §4.1, §7.1
"""
from __future__ import annotations

import torch

from nbn.bench.domains.base import BenchmarkProblem, Query


class UniformRandomSelector:
    """Select queries by uniform random target + evidence assignment.

    For each of the ``n_queries`` requested queries, picks ONE target node
    and up to ``n_evidence`` evidence nodes from the problem's node set
    (uniformly at random, without replacement within a single query), then
    uses one row from ``problem.test_data`` as the evidence values.

    All returned queries have ``kind="marginal"``.

    Cap policy (n_queries > |unique queries|)
    -----------------------------------------
    Note: this implementation caps at ``n_test`` (test-set size), which is
    a pragmatic proxy for the ``|unique queries|`` concept from issue #74.
    True unique-query enumeration — counting distinct
    ``(target, evidence_nodes, evidence_values)`` combinations — is
    intractable for large DAGs because the evidence-value space is
    continuous.  Capping at ``n_test`` ensures each query uses a distinct
    evidence row from the held-out set, which approximates uniqueness
    sufficiently for benchmark purposes (each test row is an independent
    draw from the joint distribution, so collisions are rare in practice).
    Future ``TopologicalAllocator`` (Phase 2) may refine this cap per role
    bucket.

    Invariants preserved
    --------------------
    * Target node is never in the evidence dict (permutation guarantee).
    * Evidence node set is a strict subset of the non-target nodes.
    * Result is deterministic given the same ``seed``.

    Parameters
    ----------
    n_evidence:
        Number of evidence nodes to condition on.  If the problem has
        ``<= 4`` nodes the count is automatically capped at
        ``min(n_evidence, n_nodes - 1)`` (matching the v0.12 convention
        of ``n_ev = 2`` for small DAGs).  For very small DAGs (n <= 2)
        evidence may be empty.
    """

    def __init__(self, n_evidence: int = 3, n_batch_queries: int = 1) -> None:
        self.n_evidence = n_evidence
        # v0.14 (#148): evidence-value variants per query position,
        # design doc §1.1, §2.4. 1 = existing single-variant behavior.
        self.n_batch_queries = int(n_batch_queries)

    def select(
        self,
        problem: BenchmarkProblem,
        n_queries: int,
        seed: int,
    ) -> list[Query]:
        """Return a list of ``n_queries`` ``Query`` objects (or fewer).

        Returns
        -------
        list[Query]
            One ``Query`` per row, with scalar evidence values.  The list
            has length ``min(n_queries, n_test)``.  All queries have
            ``query_role = "random"`` (set on the selector; callers read
            this via ``selector.query_role`` to populate ``CellResult``).
        """
        nodes = list(problem.variables.keys())
        n_nodes = len(nodes)
        if n_nodes < 1:
            return []

        # Cap n_evidence to leave at least one node for the target.
        # v0.12 convention: use 2 evidence nodes for small DAGs (n <= 4).
        if n_nodes <= 4:
            effective_n_ev = min(self.n_evidence, max(0, n_nodes - 1), 2)
        else:
            effective_n_ev = min(self.n_evidence, n_nodes - 1)

        # Cap queries at test-set size — see docstring for cap policy.
        test_values = problem.test_data
        n_test = next(iter(test_values.values())).shape[0] if test_values else 0
        take = min(n_queries, n_test)
        if take <= 0:
            return []

        # Deterministic generator — seed * 1000 + 7 matches v0.12 convention
        # in _build_query_batch so behaviour is unchanged when both use the
        # same seed.
        gen = torch.Generator(device="cpu").manual_seed(seed * 1000 + 7)
        perm = torch.randperm(n_nodes, generator=gen).tolist()

        target = nodes[perm[0]]
        evidence_nodes = [nodes[perm[i]] for i in range(1, 1 + effective_n_ev)]

        queries: list[Query] = []
        for row_idx in range(take):
            ev: dict[str, torch.Tensor] = {
                v: test_values[v][row_idx].reshape(()).float()
                for v in evidence_nodes
            }
            queries.append(Query(targets=(target,), evidence=ev, kind="marginal"))

        return queries

    def select_groups(
        self,
        problem: BenchmarkProblem,
        n_queries: int,
        seed: int,
        *,
        batch_size: int = 1,
    ) -> list[list[Query]]:
        """Grouped form of ``select()`` (v0.14, #148; design doc §1.3, §2.3).

        Expands each selected query position into ``self.n_batch_queries``
        evidence-value variants sampled from ``problem.train_data``,
        chunked into inner lists of length <= ``batch_size``. At
        ``n_batch_queries=1`` this is ``[[q] for q in select(...)]`` with
        the original Query objects.
        """
        from nbn.bench.selectors._batching import make_query_groups

        return make_query_groups(
            self.select(problem, n_queries, seed),
            problem,
            n_batch_queries=self.n_batch_queries,
            batch_size=batch_size,
            seed=seed,
        )

    # -----------------------------------------------------------------------
    # Selector role label (for CellResult.query_role)
    # -----------------------------------------------------------------------

    #: All queries from this selector have role ``"random"``.
    query_role: str = "random"
