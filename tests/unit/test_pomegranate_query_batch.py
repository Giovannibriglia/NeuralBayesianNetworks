"""Tests for PomegranateAdapter.query_batch — library-level batching (PR 3, #148).

Pomegranate's ``predict_proba`` is loopy BP — deterministic given the
same input — so batched-vs-sequential equivalence is checked at
atol=1e-5 (allowing tiny numerical differences between the [B, n] and
[1, n] message-passing paths).

Mirrors tests/unit/test_nbn_query_batch.py's structure: equivalence,
engine-path spy, B=1 / empty / heterogeneous-fallback edge cases,
all-None evidence, per-row independence.

Reference: docs/v0.14-batched-queries-design.md §3.2, §7.1.
"""
from __future__ import annotations

import pytest
import torch

from nbn.bench.adapters import PomegranateAdapter
from nbn.bench.domains.base import BenchmarkProblem, Query
from nbn.bench.domains.posterior import Posterior


# ---- Fixtures ----------------------------------------------------------------

def _make_small_discrete_problem(
    n_samples: int = 500, seed: int = 0,
) -> BenchmarkProblem:
    """4-node binary BN: X0 → X1 → X2, X0 → X3 (correlated, non-degenerate)."""
    g = torch.Generator().manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2"), ("X0", "X3")]
    x0 = torch.randint(0, 2, (n_samples,), generator=g)
    flip = lambda x, p: torch.where(  # noqa: E731 — local helper
        torch.rand(n_samples, generator=g) < p, 1 - x, x
    )
    train_data = {
        "X0": x0,
        "X1": flip(x0, 0.3),
        "X2": flip(flip(x0, 0.3), 0.3),
        "X3": flip(x0, 0.2),
    }
    variables = dict.fromkeys(train_data, ("discrete", 2))
    return BenchmarkProblem(
        name="test_discrete",
        dag=dag,
        variables=variables,
        train_data=train_data,
        test_data=train_data,
        queries=[],
    )


def _queries(b: int = 4) -> list[Query]:
    """B queries sharing (target=X2, evidence_keys={X0}) with varying values."""
    return [
        Query(
            targets=("X2",),
            evidence={"X0": torch.tensor(v % 2)},
            kind="marginal",
        )
        for v in range(b)
    ]


@pytest.fixture(scope="module")
def adapter() -> PomegranateAdapter:
    a = PomegranateAdapter(device="cpu")
    a.fit(_make_small_discrete_problem())
    return a


# ---- 1. Batch-vs-sequential equivalence ---------------------------------------

@pytest.mark.slow
class TestBatchEquivalence:
    def test_batch_matches_sequential(self, adapter):
        queries = _queries(b=4)
        batched = adapter.query_batch(queries)
        sequential = [adapter.query(q) for q in queries]

        assert len(batched) == 4
        for bp, sp in zip(batched, sequential):
            assert bp.probs is not None and sp.probs is not None
            assert bp.probs.shape == sp.probs.shape
            assert torch.allclose(bp.probs, sp.probs, atol=1e-5)

    def test_batch_order_follows_input(self, adapter):
        """Reversing the input order reverses the output posteriors."""
        queries = _queries(b=4)
        fwd = adapter.query_batch(queries)
        rev = adapter.query_batch(list(reversed(queries)))
        for f, r in zip(fwd, reversed(rev)):
            assert torch.allclose(f.probs, r.probs, atol=1e-6)


# ---- 2. Engine path spy --------------------------------------------------------

@pytest.mark.slow
class TestEnginePath:
    def test_predict_proba_called_once_for_homogeneous_batch(self, adapter):
        """A homogeneous B=4 batch makes exactly one predict_proba call —
        without this the equivalence test could pass while silently
        falling back to sequential."""
        calls = []
        orig = adapter.model.predict_proba

        def spy(x):
            calls.append(tuple(x.shape))
            return orig(x)

        adapter.model.predict_proba = spy
        try:
            out = adapter.query_batch(_queries(b=4))
        finally:
            adapter.model.predict_proba = orig

        assert len(out) == 4
        assert calls == [(4, 4)]  # one call, [B=4, n=4]


# ---- 3-7. Edge cases -----------------------------------------------------------

@pytest.mark.slow
class TestQueryBatchEdgeCases:
    def test_b1_identical_to_query(self, adapter):
        q = _queries(b=1)
        [batched] = adapter.query_batch(q)
        single = adapter.query(q[0])
        assert torch.equal(batched.probs, single.probs)  # same code path

    def test_empty_list(self, adapter):
        assert adapter.query_batch([]) == []

    def test_heterogeneous_targets_fall_back(self, adapter):
        queries = [
            Query(targets=("X2",), evidence={"X0": torch.tensor(0)}, kind="marginal"),
            Query(targets=("X3",), evidence={"X0": torch.tensor(1)}, kind="marginal"),
        ]
        batched = adapter.query_batch(queries)
        sequential = [adapter.query(q) for q in queries]
        for bp, sp in zip(batched, sequential):
            assert isinstance(bp, Posterior)
            assert torch.allclose(bp.probs, sp.probs, atol=1e-6)

    def test_heterogeneous_evidence_keys_fall_back(self, adapter):
        queries = [
            Query(targets=("X2",), evidence={"X0": torch.tensor(0)}, kind="marginal"),
            Query(targets=("X2",), evidence={"X1": torch.tensor(1)}, kind="marginal"),
        ]
        batched = adapter.query_batch(queries)
        sequential = [adapter.query(q) for q in queries]
        for bp, sp in zip(batched, sequential):
            assert torch.allclose(bp.probs, sp.probs, atol=1e-6)

    def test_all_none_evidence_key_marginalizes(self, adapter):
        """A key that is None in every query is dropped (empty mode) —
        matches query()'s behavior of leaving the column masked."""
        queries = [
            Query(targets=("X2",), evidence={"X0": None}, kind="marginal")
            for _ in range(3)
        ]
        batched = adapter.query_batch(queries)
        sequential = [adapter.query(q) for q in queries]
        for bp, sp in zip(batched, sequential):
            assert torch.allclose(bp.probs, sp.probs, atol=1e-5)

    def test_mixed_none_evidence_falls_back(self, adapter):
        """Mixed None/concrete values for a key → sequential fallback."""
        queries = [
            Query(targets=("X2",), evidence={"X0": torch.tensor(0)}, kind="marginal"),
            Query(targets=("X2",), evidence={"X0": None}, kind="marginal"),
        ]
        batched = adapter.query_batch(queries)
        sequential = [adapter.query(q) for q in queries]
        for bp, sp in zip(batched, sequential):
            assert torch.allclose(bp.probs, sp.probs, atol=1e-6)


# ---- 8. Per-row independence ----------------------------------------------------

@pytest.mark.slow
class TestPerRowIndependence:
    def test_different_evidence_gives_different_posteriors(self, adapter):
        """X2 is correlated with X0 in the fixture, so P(X2 | X0=0) and
        P(X2 | X0=1) must differ — guards against rows accidentally
        sharing one evidence value."""
        queries = _queries(b=4)  # evidence alternates 0, 1, 0, 1
        batched = adapter.query_batch(queries)
        assert not torch.allclose(batched[0].probs, batched[1].probs, atol=1e-3)
        # Rows with equal evidence must agree exactly.
        assert torch.allclose(batched[0].probs, batched[2].probs, atol=1e-6)
        assert torch.allclose(batched[1].probs, batched[3].probs, atol=1e-6)
