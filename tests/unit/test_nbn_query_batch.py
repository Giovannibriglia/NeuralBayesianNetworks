"""Tests for NBNAdapter.query_batch — library-level batching (PR 2, #148).

Covers the three engine paths through the adapter's batched unpack:

  - VE discrete:  exact — batched [B, K] must match looped query() to 1e-6
  - LW discrete:  stochastic — batched weighted-histogram probs must match
                  looped query() within TV ≤ 0.05 at S=4096
  - LW continuous: stochastic — batched per-row multinomial resample must
                  match looped query() within KS ≤ 0.10 at S=4096.
                  This is the equivalence test the V1 investigation flagged
                  as missing (test_vectorized_query_batch_correctness.py
                  covers the engine's VE path only).

Plus the protocol edge cases: B=1, empty list, heterogeneous fallback.

Reference: docs/v0.14-batched-queries-design.md §3.1, §7.1.
"""
from __future__ import annotations

import pytest
import torch

from nbn.bench.adapters import NBNAdapter
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


def _make_small_continuous_lg_problem(
    n_samples: int = 400, seed: int = 0,
) -> BenchmarkProblem:
    """3-node linear-Gaussian chain: X0 → X1 → X2."""
    torch.manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2")]
    x0 = torch.randn(n_samples)
    x1 = 0.5 * x0 + 0.5 * torch.randn(n_samples)
    x2 = 0.5 * x1 + 0.5 * torch.randn(n_samples)
    train_data = {"X0": x0, "X1": x1, "X2": x2}
    variables = dict.fromkeys(train_data, ("continuous", None))
    return BenchmarkProblem(
        name="test_continuous_lg",
        dag=dag,
        variables=variables,
        train_data=train_data,
        test_data=train_data,
        queries=[],
    )


def _discrete_queries(b: int = 4) -> list[Query]:
    """B queries sharing (target=X2, evidence_keys={X0}) with varying values."""
    return [
        Query(
            targets=("X2",),
            evidence={"X0": torch.tensor(v % 2)},
            kind="marginal",
        )
        for v in range(b)
    ]


def _continuous_queries(b: int = 4) -> list[Query]:
    """B queries sharing (target=X2, evidence_keys={X0}) with varying values."""
    return [
        Query(
            targets=("X2",),
            evidence={"X0": torch.tensor(-1.0 + 0.5 * v)},
            kind="marginal",
        )
        for v in range(b)
    ]


@pytest.fixture(scope="module")
def cat_ve_adapter() -> NBNAdapter:
    adapter = NBNAdapter(mechanism="cat", engine="ve", device="cpu")
    adapter.fit(_make_small_discrete_problem(), epochs=5)
    return adapter


@pytest.fixture(scope="module")
def cat_lw_adapter() -> NBNAdapter:
    adapter = NBNAdapter(
        mechanism="cat", engine="lw", device="cpu", n_samples=4096
    )
    adapter.fit(_make_small_discrete_problem(), epochs=5)
    return adapter


@pytest.fixture(scope="module")
def lg_lw_adapter() -> NBNAdapter:
    adapter = NBNAdapter(
        mechanism="lg", engine="lw", device="cpu", n_samples=4096
    )
    adapter.fit(_make_small_continuous_lg_problem(), epochs=5)
    return adapter


def _ks_statistic(a: torch.Tensor, b: torch.Tensor) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (sup |ECDF_a − ECDF_b|)."""
    combined = torch.cat([a, b]).sort().values
    cdf_a = torch.searchsorted(a.sort().values, combined, right=True) / len(a)
    cdf_b = torch.searchsorted(b.sort().values, combined, right=True) / len(b)
    return float((cdf_a - cdf_b).abs().max())


# ---- 1. VE discrete: exact equivalence ----------------------------------------

@pytest.mark.slow
class TestVEDiscreteBatch:
    def test_batch_matches_sequential_exactly(self, cat_ve_adapter):
        queries = _discrete_queries(b=4)
        batched = cat_ve_adapter.query_batch(queries)
        sequential = [cat_ve_adapter.query(q) for q in queries]

        assert len(batched) == 4
        for bp, sp in zip(batched, sequential):
            assert bp.probs is not None and sp.probs is not None
            assert torch.allclose(bp.probs, sp.probs, atol=1e-6)

    def test_batch_order_follows_input(self, cat_ve_adapter):
        """Reversing the input order reverses the output posteriors."""
        queries = _discrete_queries(b=4)
        fwd = cat_ve_adapter.query_batch(queries)
        rev = cat_ve_adapter.query_batch(list(reversed(queries)))
        for f, r in zip(fwd, reversed(rev)):
            assert torch.allclose(f.probs, r.probs, atol=1e-6)


# ---- 2. LW discrete: statistical equivalence ----------------------------------

@pytest.mark.slow
class TestLWDiscreteBatch:
    def test_batch_matches_sequential_tv(self, cat_lw_adapter):
        queries = _discrete_queries(b=4)
        torch.manual_seed(7)
        batched = cat_lw_adapter.query_batch(queries)
        torch.manual_seed(7)
        sequential = [cat_lw_adapter.query(q) for q in queries]

        for bp, sp in zip(batched, sequential):
            assert bp.probs is not None and sp.probs is not None
            tv = 0.5 * (bp.probs - sp.probs).abs().sum()
            assert tv <= 0.05, f"TV {tv:.4f} > 0.05"


# ---- 3. LW continuous: statistical equivalence (V1-flagged gap) ---------------

@pytest.mark.slow
class TestLWContinuousBatch:
    def test_batch_matches_sequential_ks(self, lg_lw_adapter):
        queries = _continuous_queries(b=4)
        torch.manual_seed(7)
        batched = lg_lw_adapter.query_batch(queries)
        torch.manual_seed(7)
        sequential = [lg_lw_adapter.query(q) for q in queries]

        for i, (bp, sp) in enumerate(zip(batched, sequential)):
            assert bp.samples is not None and sp.samples is not None
            assert bp.samples.shape == sp.samples.shape == (4096,)
            ks = _ks_statistic(bp.samples, sp.samples)
            assert ks <= 0.10, f"query {i}: KS {ks:.4f} > 0.10"

    def test_rows_are_independent(self, lg_lw_adapter):
        """Different evidence values in one batch → different posteriors.

        Guards the V1 finding (per-row independent particle pools): if the
        engine ever regressed to a shared pool conditioned on one row's
        evidence, posterior means across rows would collapse together.
        """
        queries = _continuous_queries(b=4)  # evidence −1.0 … +0.5
        torch.manual_seed(7)
        batched = lg_lw_adapter.query_batch(queries)
        means = torch.tensor([p.samples.mean() for p in batched])
        # X2 | X0=v has mean 0.25·v — the spread across rows must survive.
        assert means[-1] - means[0] > 0.15, f"posterior means collapsed: {means}"


# ---- 4-6. Protocol edge cases --------------------------------------------------

@pytest.mark.slow
class TestQueryBatchEdgeCases:
    def test_b1_identical_to_query(self, cat_ve_adapter):
        q = _discrete_queries(b=1)
        [batched] = cat_ve_adapter.query_batch(q)
        single = cat_ve_adapter.query(q[0])
        assert torch.equal(batched.probs, single.probs)  # same code path

    def test_empty_list(self, cat_ve_adapter):
        assert cat_ve_adapter.query_batch([]) == []

    def test_heterogeneous_targets_fall_back(self, cat_ve_adapter):
        """Different targets in one batch → sequential fallback, same results."""
        queries = [
            Query(targets=("X2",), evidence={"X0": torch.tensor(0)}, kind="marginal"),
            Query(targets=("X3",), evidence={"X0": torch.tensor(1)}, kind="marginal"),
        ]
        batched = cat_ve_adapter.query_batch(queries)
        sequential = [cat_ve_adapter.query(q) for q in queries]
        for bp, sp in zip(batched, sequential):
            assert torch.allclose(bp.probs, sp.probs, atol=1e-6)
            assert isinstance(bp, Posterior)

    def test_heterogeneous_evidence_keys_fall_back(self, cat_ve_adapter):
        """Different evidence keys in one batch → sequential fallback."""
        queries = [
            Query(targets=("X2",), evidence={"X0": torch.tensor(0)}, kind="marginal"),
            Query(targets=("X2",), evidence={"X1": torch.tensor(1)}, kind="marginal"),
        ]
        batched = cat_ve_adapter.query_batch(queries)
        sequential = [cat_ve_adapter.query(q) for q in queries]
        for bp, sp in zip(batched, sequential):
            assert torch.allclose(bp.probs, sp.probs, atol=1e-6)

    def test_all_empty_mode_batch_falls_back(self, cat_ve_adapter):
        """All-None evidence (empty mode, e.g. heaviest V2 queries):
        engines infer B from evidence tensors, so an evidence-free batch
        must take the sequential fallback and still return B posteriors.
        Regression for the PR 5 speed-smoke failure (shape (1, K) for B=4).
        """
        queries = [
            Query(targets=("X2",), evidence={"X0": None}, kind="marginal")
            for _ in range(4)
        ]
        batched = cat_ve_adapter.query_batch(queries)
        sequential = [cat_ve_adapter.query(q) for q in queries]
        assert len(batched) == 4
        for bp, sp in zip(batched, sequential):
            assert torch.allclose(bp.probs, sp.probs, atol=1e-6)
