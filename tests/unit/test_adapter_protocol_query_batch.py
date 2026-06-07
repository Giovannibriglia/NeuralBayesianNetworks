"""Tests for the ``query_batch`` protocol addition + default helper (PR 1, #148).

PR 1 of the batched-queries sequence: the protocol gains
``query_batch(queries) -> list[Posterior]`` and a module-level
``default_query_batch`` sequential helper that every existing adapter
opts into.  These tests verify only that the default helper is a
faithful passthrough — batched-vs-sequential equivalence for the real
library-batched overrides lives in PR 2/3 test files.

Reference: docs/v0.14-batched-queries-design.md §1.2, §3.3, §7.1.
"""
from __future__ import annotations

import pytest
import torch

from benchmarking.adapters import (
    NBNAdapter,
    PgmpyAdapter,
    PomegranateAdapter,
    PyroAdapter,
)
from benchmarking.core.interfaces import default_query_batch
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.domains.posterior import Posterior


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


def _make_small_discrete_problem(
    n_samples: int = 300, seed: int = 0,
) -> BenchmarkProblem:
    """3-node binary BN: X0 → X1 → X2 (same shape as the pgmpy adapter tests)."""
    torch.manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2")]
    train_data = {
        "X0": torch.randint(0, 2, (n_samples,)),
        "X1": torch.randint(0, 2, (n_samples,)),
        "X2": torch.randint(0, 2, (n_samples,)),
    }
    variables = {
        "X0": ("discrete", 2),
        "X1": ("discrete", 2),
        "X2": ("discrete", 2),
    }
    return BenchmarkProblem(
        name="test_discrete",
        dag=dag,
        variables=variables,
        train_data=train_data,
        test_data=train_data,
        queries=[],
    )


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


# ---- Default helper: equivalence on a real adapter ---------------------------

@pytest.mark.slow
class TestDefaultQueryBatchRealAdapter:
    def test_equivalent_to_sequential_query_calls_pgmpy(self):
        """pgmpy on a 3-node BN: helper output == looped query(), bitwise."""
        adapter = PgmpyAdapter(param_method="mle", inference_method="ve")
        problem = _make_small_discrete_problem(n_samples=500, seed=42)
        adapter.fit(problem)

        queries = [
            Query(
                targets=("X2",),
                evidence={"X0": torch.tensor(v)},
                kind="marginal",
            )
            for v in (0, 1, 0)
        ]

        batched = default_query_batch(adapter, queries)
        sequential = [adapter.query(q) for q in queries]

        assert len(batched) == len(sequential) == 3
        for b, s in zip(batched, sequential):
            assert torch.equal(b.probs, s.probs)  # bitwise identical (VE is exact)


# ---- Protocol surface: every adapter has query_batch -------------------------

class TestAdaptersExposeQueryBatch:
    @pytest.mark.parametrize(
        "adapter_cls, kwargs",
        [
            (NBNAdapter, {"mechanism": "cat", "engine": "ve", "device": "cpu"}),
            (PgmpyAdapter, {"param_method": "mle", "inference_method": "ve"}),
            (PomegranateAdapter, {"device": "cpu"}),
            (
                PyroAdapter,
                {"mechanism": "empirical", "inference_method": "importance"},
            ),
        ],
        ids=["nbn", "pgmpy", "pomegranate", "pyro"],
    )
    def test_has_callable_query_batch(self, adapter_cls, kwargs):
        adapter = adapter_cls(**kwargs)
        assert hasattr(adapter, "query_batch")
        assert callable(adapter.query_batch)
