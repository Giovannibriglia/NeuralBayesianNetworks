"""Tests for v0.13 PomegranateAdapter.

Three test classes:
  - TestProtocolConformance: isinstance checks + name/pre-fit state
  - TestApplicability: is_applicable() per family
  - TestBehavioral: end-to-end fit + query on a small discrete BN
    (marked @pytest.mark.slow — excluded from the fast gate)

The dtype=torch.long bug fix (v0.5b / issue #31) is verified by a
dedicated structural test in TestBehavioral.

Reference: docs/v0.13-benchmark-redesign.md §4.1
"""
from __future__ import annotations

import pytest
import torch

from nbn.bench.adapters import PomegranateAdapter
from nbn.bench.core.interfaces import BaselineAdapter
from nbn.bench.domains.base import BenchmarkProblem, Query
from nbn.bench.domains.posterior import Posterior


# ---- Helpers ----------------------------------------------------------------

def _make_small_discrete_problem(
    n_samples: int = 300, seed: int = 0,
) -> BenchmarkProblem:
    """3-node binary BN: X0 → X1 → X2."""
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


def _make_small_continuous_problem(
    n_samples: int = 200, seed: int = 0,
) -> BenchmarkProblem:
    """3-node continuous BN (for applicability rejection tests)."""
    torch.manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2")]
    x0 = torch.randn(n_samples)
    x1 = 0.5 * x0 + 0.5 * torch.randn(n_samples)
    x2 = 0.5 * x1 + 0.5 * torch.randn(n_samples)
    train_data = {"X0": x0, "X1": x1, "X2": x2}
    variables = dict.fromkeys(train_data, ("continuous", None))
    return BenchmarkProblem(
        name="test_continuous",
        dag=dag,
        variables=variables,
        train_data=train_data,
        test_data=train_data,
        queries=[],
    )


# ---- Tests ------------------------------------------------------------------

class TestProtocolConformance:
    """PomegranateAdapter satisfies the v0.13 BaselineAdapter protocol."""

    def test_pomegranate_adapter_satisfies_protocol(self):
        adapter = PomegranateAdapter()
        assert isinstance(adapter, BaselineAdapter)

    def test_name_is_correct(self):
        assert PomegranateAdapter().name == "pomegranate-discrete-ve"

    def test_pre_fit_state_is_none(self):
        adapter = PomegranateAdapter()
        assert adapter.model is None
        assert adapter.problem is None
        assert adapter._node_to_idx == {}
        assert adapter._cards == {}
        assert adapter._topo == []

    def test_kwargs_silently_accepted(self):
        """Constructor accepts **kwargs for runner API compatibility."""
        adapter = PomegranateAdapter(some_future_kwarg=42)
        assert adapter.name == "pomegranate-discrete-ve"


class TestApplicability:
    """is_applicable() returns correct results per family."""

    def test_applicable_to_discrete(self):
        adapter = PomegranateAdapter()
        assert adapter.is_applicable(_make_small_discrete_problem()) is True

    def test_not_applicable_to_continuous(self):
        adapter = PomegranateAdapter()
        assert adapter.is_applicable(_make_small_continuous_problem()) is False

    def test_unknown_name_returns_false(self):
        adapter = PomegranateAdapter()
        adapter.name = "pomegranate-not-in-registry"  # type: ignore[misc]
        assert adapter.is_applicable(_make_small_discrete_problem()) is False


@pytest.mark.slow
class TestBehavioral:
    """End-to-end fit + query on a small discrete problem.

    Verifies shape, value range, Posterior contract, and the critical
    dtype=torch.long bug fix for pomegranate v1.1.x conditional queries.
    """

    def test_fit_stores_state(self):
        adapter = PomegranateAdapter()
        problem = _make_small_discrete_problem(n_samples=300, seed=0)
        assert adapter.model is None

        adapter.fit(problem)

        assert adapter.model is not None
        assert adapter.problem is problem
        assert len(adapter._topo) == 3
        assert set(adapter._node_to_idx.keys()) == {"X0", "X1", "X2"}
        assert adapter._cards == {"X0": 2, "X1": 2, "X2": 2}

    def test_marginal_query_returns_valid_posterior(self):
        """Marginal query (no evidence): probs sums to 1."""
        adapter = PomegranateAdapter()
        problem = _make_small_discrete_problem(n_samples=500, seed=42)
        adapter.fit(problem)

        q = Query(targets=("X2",), evidence={}, kind="marginal")
        posterior = adapter.query(q)

        assert isinstance(posterior, Posterior)
        assert posterior.probs is not None
        assert posterior.samples is None
        assert posterior.probs.shape == (2,)
        assert torch.isclose(posterior.probs.sum(), torch.tensor(1.0), atol=1e-4)
        assert (posterior.probs >= 0).all()
        assert (posterior.probs <= 1).all()

    def test_conditional_query_returns_valid_posterior(self):
        """Conditional query with evidence: probs sums to 1.

        This exercises the dtype=torch.long code path (v0.5b fix).
        A float row tensor would raise IndexError in pomegranate v1.1.x.
        """
        adapter = PomegranateAdapter()
        problem = _make_small_discrete_problem(n_samples=500, seed=7)
        adapter.fit(problem)

        q = Query(
            targets=("X2",),
            evidence={"X0": torch.tensor(1)},
            kind="marginal",
        )
        posterior = adapter.query(q)

        assert isinstance(posterior, Posterior)
        assert posterior.probs is not None
        assert posterior.samples is None
        assert posterior.probs.shape == (2,)
        assert torch.isclose(posterior.probs.sum(), torch.tensor(1.0), atol=1e-4)
        assert (posterior.probs >= 0).all()
        assert (posterior.probs <= 1).all()

    def test_conditional_query_with_scalar_int_evidence(self):
        """Evidence as plain Python int (not tensor) works correctly."""
        adapter = PomegranateAdapter()
        problem = _make_small_discrete_problem(n_samples=400, seed=3)
        adapter.fit(problem)

        q = Query(
            targets=("X2",),
            evidence={"X0": 0},   # plain int, not tensor
            kind="marginal",
        )
        posterior = adapter.query(q)

        assert isinstance(posterior, Posterior)
        assert posterior.probs is not None
        assert posterior.probs.shape == (2,)
        assert torch.isclose(posterior.probs.sum(), torch.tensor(1.0), atol=1e-4)

    def test_fit_rejects_continuous_nodes(self):
        """fit() raises ValueError if problem contains continuous nodes."""
        adapter = PomegranateAdapter()
        problem = _make_small_continuous_problem()
        with pytest.raises(ValueError, match="discrete-only"):
            adapter.fit(problem)

    def test_query_before_fit_raises_runtime_error(self):
        adapter = PomegranateAdapter()
        q = Query(targets=("X0",), evidence={}, kind="marginal")
        with pytest.raises(RuntimeError, match="not fitted"):
            adapter.query(q)

    def test_epochs_kwarg_accepted(self):
        """fit() silently accepts epochs kwarg (runner API compatibility)."""
        adapter = PomegranateAdapter()
        problem = _make_small_discrete_problem(n_samples=200, seed=1)
        adapter.fit(problem, epochs=5)   # should not raise
        assert adapter.model is not None
