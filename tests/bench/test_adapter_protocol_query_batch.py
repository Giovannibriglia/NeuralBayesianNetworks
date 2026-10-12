"""Tests for the ``query_batch`` protocol addition + default helper (PR 1, #148).

PR 1 of the batched-queries sequence: the protocol gains
``query_batch(queries) -> list[Posterior]`` and a module-level
``default_query_batch`` sequential helper that every existing adapter
opts into.  These tests verify only that the default helper is a
faithful passthrough on a recording mock — batched-vs-sequential
equivalence on the real adapters (including pgmpy through this helper)
lives in test_query_batch_contract.py.

Reference: docs/v0.14-batched-queries-design.md §1.2, §3.3, §7.1.
"""
from __future__ import annotations

import pytest
import torch

from nbn.bench.core.interfaces import default_query_batch
from nbn.bench.domains.base import Query
from nbn.bench.domains.posterior import Posterior


# ---- Helpers ----------------------------------------------------------------

class _RecordingAdapter:
    """Mock adapter whose query() echoes the query's target name.

    The returned Posterior carries a probs tensor whose length encodes
    the call order, and the target is recorded so order preservation is
    checkable from the outside.
    """

    name = "mock-recording"

    def __init__(self) -> None:
        self.seen_targets: list[str] = []

    def query(self, q: Query) -> Posterior:
        self.seen_targets.append(q.targets[0])
        # Encode the target identity in the posterior for order checks.
        idx = int(q.targets[0].lstrip("X"))
        probs = torch.zeros(4)
        probs[idx] = 1.0
        return Posterior(probs=probs)


# ---- Default helper: passthrough semantics ----------------------------------

class TestDefaultQueryBatch:
    def test_preserves_input_order(self):
        """B=4 queries → posteriors come back in input order."""
        adapter = _RecordingAdapter()
        targets = ["X2", "X0", "X3", "X1"]
        queries = [
            Query(targets=(t,), evidence={}, kind="marginal") for t in targets
        ]

        posteriors = default_query_batch(adapter, queries)

        assert len(posteriors) == 4
        assert adapter.seen_targets == targets  # called sequentially, in order
        for t, p in zip(targets, posteriors):
            assert int(p.probs.argmax()) == int(t.lstrip("X"))

    def test_empty_list_returns_empty(self):
        adapter = _RecordingAdapter()
        assert default_query_batch(adapter, []) == []
        assert adapter.seen_targets == []

    def test_mid_batch_failure_raises(self):
        """§4.4: no partial-success returns — a failure partway raises."""

        class _FailsOnSecond(_RecordingAdapter):
            def query(self, q: Query) -> Posterior:
                if len(self.seen_targets) == 1:
                    raise RuntimeError("boom")
                return super().query(q)

        adapter = _FailsOnSecond()
        queries = [
            Query(targets=(t,), evidence={}, kind="marginal")
            for t in ["X0", "X1", "X2"]
        ]
        with pytest.raises(RuntimeError, match="boom"):
            default_query_batch(adapter, queries)
