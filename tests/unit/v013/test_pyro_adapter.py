"""Tests for v0.13 PyroAdapter.

Three test classes:
  - TestProtocolConformance: isinstance checks + name/pre-fit state
  - TestApplicability: is_applicable() per family
  - TestBehavioral: end-to-end fit + query on small synthetic BNs
    (marked @pytest.mark.slow — excluded from the fast gate)

Behavioral tests cover all three applicable families:
  - discrete  (3-node binary BN)
  - continuous_lg (3-node LG chain)
  - hybrid (2-node: discrete root → continuous child)

Key check for continuous targets: query() returns Posterior(samples=...)
with shape (n_samples,), NOT a collapsed point estimate.  This verifies
the behaviour change from the old adapter's marg.mean(0) return.

Behavioral tests use n_samples=20 (reduced from default 50) so each
importance sampler run completes in well under a second for 3-node BNs.

Reference: docs/v0.13-benchmark-redesign.md §4.1
"""
from __future__ import annotations

import pytest
import torch

from benchmarking.adapters import PyroAdapter
from benchmarking.core.interfaces import BaselineAdapter
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.domains.posterior import Posterior


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
    n_samples: int = 300, seed: int = 0,
) -> BenchmarkProblem:
    """3-node LG chain: X0 → X1 → X2."""
    torch.manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2")]
    x0 = torch.randn(n_samples)
    x1 = 0.5 * x0 + 0.5 * torch.randn(n_samples)
    x2 = 0.5 * x1 + 0.5 * torch.randn(n_samples)
    train_data = {"X0": x0, "X1": x1, "X2": x2}
    variables = dict.fromkeys(train_data, ("continuous", None))
    return BenchmarkProblem(
        name="continuous_lg_test",
        dag=dag,
        variables=variables,
        train_data=train_data,
        test_data=train_data,
        queries=[],
    )


def _make_small_hybrid_problem(
    n_samples: int = 300, seed: int = 0,
) -> BenchmarkProblem:
    """2-node hybrid BN: X0 (discrete, k=2) → X1 (continuous).

    Exercises the hybrid path: continuous node with a discrete parent
    triggers _fit_lg_leaf's one-hot encoding branch.  The target X1
    has a clear discrete-dependent mean (≈0 when X0=0, ≈1 when X0=1).
    """
    torch.manual_seed(seed)
    dag = [("X0", "X1")]
    x0 = torch.randint(0, 2, (n_samples,))
    x1 = x0.float() + 0.5 * torch.randn(n_samples)
    train_data = {"X0": x0, "X1": x1}
    variables: dict = {
        "X0": ("discrete", 2),
        "X1": ("continuous", None),
    }
    return BenchmarkProblem(
        name="hybrid_test",
        dag=dag,
        variables=variables,
        train_data=train_data,
        test_data=train_data,
        queries=[],
    )


# ---- Tests ------------------------------------------------------------------

class TestProtocolConformance:
    """PyroAdapter satisfies the v0.13 BaselineAdapter protocol."""

    def test_pyro_adapter_satisfies_protocol(self):
        adapter = PyroAdapter(mechanism="empirical", inference_method="importance")
        assert isinstance(adapter, BaselineAdapter)

    def test_name_derived_from_args(self):
        adapter = PyroAdapter(mechanism="empirical", inference_method="importance")
        assert adapter.name == "pyro-empirical-importance"

    def test_pre_fit_state(self):
        adapter = PyroAdapter(mechanism="empirical", inference_method="importance")
        assert not adapter._fitted
        assert adapter._cpts == {}
        assert adapter._gaussian == {}
        assert adapter._cpt_parents == {}
        assert adapter._parents == {}
        assert adapter._cards == {}
        assert adapter._topo == []
        assert adapter.problem is None

    def test_invalid_mechanism_raises(self):
        with pytest.raises(ValueError, match="Unknown mechanism"):
            PyroAdapter(mechanism="bogus", inference_method="importance")

    def test_invalid_inference_method_raises(self):
        with pytest.raises(ValueError, match="Unknown inference_method"):
            PyroAdapter(mechanism="empirical", inference_method="bogus")

    def test_n_samples_default(self):
        adapter = PyroAdapter(mechanism="empirical", inference_method="importance")
        assert adapter.n_samples == 50

    def test_n_samples_override(self):
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", n_samples=20,
        )
        assert adapter.n_samples == 20

    def test_device_default_is_cpu(self):
        adapter = PyroAdapter(mechanism="empirical", inference_method="importance")
        assert adapter.device == "cpu"

    def test_device_auto_resolves_at_init(self):
        """device='auto' must be replaced with 'cpu' or 'cuda' at __init__ time."""
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", device="auto",
        )
        assert adapter.device in {"cpu", "cuda"}
        assert adapter.device != "auto"

    def test_kwargs_silently_accepted(self):
        """Constructor accepts **kwargs for runner API compatibility."""
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance",
            some_future_kwarg=True,
        )
        assert adapter.name == "pyro-empirical-importance"


class TestApplicability:
    """is_applicable() returns correct results per family."""

    def test_applicable_to_discrete(self):
        adapter = PyroAdapter(mechanism="empirical", inference_method="importance")
        assert adapter.is_applicable(_make_small_discrete_problem()) is True

    def test_applicable_to_continuous_lg(self):
        adapter = PyroAdapter(mechanism="empirical", inference_method="importance")
        assert adapter.is_applicable(_make_small_continuous_problem()) is True

    def test_applicable_to_hybrid(self):
        adapter = PyroAdapter(mechanism="empirical", inference_method="importance")
        assert adapter.is_applicable(_make_small_hybrid_problem()) is True

    def test_unknown_name_returns_false(self):
        adapter = PyroAdapter(mechanism="empirical", inference_method="importance")
        adapter.name = "pyro-not-in-registry"
        assert adapter.is_applicable(_make_small_discrete_problem()) is False


@pytest.mark.slow
class TestBehavioral:
    """End-to-end fit + query on small synthetic problems.

    Uses n_samples=20 throughout to keep importance sampling fast
    (each run: 20 particles × 20 marg() calls per 3-node BN ≈ ms).
    Assertions check shape, type, finiteness — not distributional accuracy.
    """

    # -- Discrete path --------------------------------------------------------

    def test_discrete_fit_stores_state(self):
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", n_samples=20,
        )
        problem = _make_small_discrete_problem(n_samples=300, seed=0)
        assert not adapter._fitted

        adapter.fit(problem)

        assert adapter._fitted
        assert adapter.problem is problem
        assert len(adapter._topo) == 3
        assert set(adapter._cards.keys()) == {"X0", "X1", "X2"}
        assert set(adapter._cpts.keys()) == {"X0", "X1", "X2"}
        assert adapter._gaussian == {}

    def test_discrete_fit_and_query_returns_valid_posterior(self):
        """pyro-empirical-importance on discrete: probs tensor sums to 1."""
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", n_samples=20,
        )
        problem = _make_small_discrete_problem(n_samples=400, seed=42)
        adapter.fit(problem)

        q = Query(
            targets=("X2",),
            evidence={"X0": torch.tensor(0)},
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

    def test_discrete_marginal_query_no_evidence(self):
        """Marginal query (no evidence) returns valid probs."""
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", n_samples=20,
        )
        problem = _make_small_discrete_problem(n_samples=300, seed=1)
        adapter.fit(problem)

        q = Query(targets=("X0",), evidence={}, kind="marginal")
        posterior = adapter.query(q)

        assert isinstance(posterior, Posterior)
        assert posterior.probs is not None
        assert posterior.probs.shape == (2,)
        assert torch.isclose(posterior.probs.sum(), torch.tensor(1.0), atol=1e-4)

    # -- Continuous path -------------------------------------------------------

    def test_continuous_fit_stores_state(self):
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", n_samples=20,
        )
        problem = _make_small_continuous_problem(n_samples=300, seed=0)
        assert not adapter._fitted

        adapter.fit(problem)

        assert adapter._fitted
        assert set(adapter._gaussian.keys()) == {"X0", "X1", "X2"}
        assert adapter._cpts == {}   # no discrete nodes

    def test_continuous_query_returns_samples_not_point_estimate(self):
        """query() returns Posterior(samples=...) for continuous targets.

        Verifies the behaviour change from the old adapter's point estimate.
        The returned tensor must have shape (n_samples,) with all values
        finite — NOT a collapsed scalar.
        """
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", n_samples=20,
        )
        problem = _make_small_continuous_problem(n_samples=400, seed=42)
        adapter.fit(problem)

        q = Query(
            targets=("X2",),
            evidence={"X0": torch.tensor(0.5)},
            kind="marginal",
        )
        posterior = adapter.query(q)

        assert isinstance(posterior, Posterior)
        assert posterior.samples is not None
        assert posterior.probs is None
        # Full sample tensor, not a point estimate
        assert posterior.samples.ndim == 1
        assert posterior.samples.shape[0] == 20   # n_samples
        assert torch.isfinite(posterior.samples).all()

    def test_continuous_marginal_query_no_evidence(self):
        """Continuous marginal query (no evidence): finite samples."""
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", n_samples=20,
        )
        problem = _make_small_continuous_problem(n_samples=300, seed=3)
        adapter.fit(problem)

        q = Query(targets=("X2",), evidence={}, kind="marginal")
        posterior = adapter.query(q)

        assert isinstance(posterior, Posterior)
        assert posterior.samples is not None
        assert posterior.samples.shape == (20,)
        assert torch.isfinite(posterior.samples).all()

    # -- Hybrid path -----------------------------------------------------------

    def test_hybrid_fit_stores_both_cpts_and_gaussians(self):
        """Hybrid fit populates both _cpts (discrete) and _gaussian (continuous)."""
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", n_samples=20,
        )
        problem = _make_small_hybrid_problem(n_samples=300, seed=0)

        adapter.fit(problem)

        assert adapter._fitted
        assert "X0" in adapter._cpts       # discrete root
        assert "X1" in adapter._gaussian   # continuous child

    def test_hybrid_query_continuous_target_given_discrete_evidence(self):
        """Hybrid BN: query continuous node given discrete root evidence."""
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", n_samples=20,
        )
        problem = _make_small_hybrid_problem(n_samples=400, seed=7)
        adapter.fit(problem)

        q = Query(
            targets=("X1",),
            evidence={"X0": torch.tensor(0)},
            kind="marginal",
        )
        posterior = adapter.query(q)

        assert isinstance(posterior, Posterior)
        assert posterior.samples is not None
        assert posterior.probs is None
        assert posterior.samples.shape == (20,)
        assert torch.isfinite(posterior.samples).all()

    def test_hybrid_lg_leaf_discrete_parent_mean_shift(self):
        """The fitted LG conditional should encode the X0→X1 mean shift.

        X1 = X0 + ε, so E[X1|X0=0] ≈ 0 and E[X1|X0=1] ≈ 1.
        We verify the sample means shift in the expected direction
        (soft check; IS with 20 samples is noisy).
        """
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", n_samples=100,
        )
        problem = _make_small_hybrid_problem(n_samples=600, seed=99)
        adapter.fit(problem)

        q0 = Query(targets=("X1",), evidence={"X0": torch.tensor(0)}, kind="marginal")
        q1 = Query(targets=("X1",), evidence={"X0": torch.tensor(1)}, kind="marginal")
        post0 = adapter.query(q0)
        post1 = adapter.query(q1)

        mean0 = post0.samples.mean().item()
        mean1 = post1.samples.mean().item()
        # E[X1|X0=1] should exceed E[X1|X0=0] by roughly 1
        assert mean1 > mean0, (
            f"Expected mean1 ({mean1:.3f}) > mean0 ({mean0:.3f}); "
            f"the discrete-parent mean shift was not captured."
        )

    # -- Pre-fit guard --------------------------------------------------------

    def test_query_before_fit_raises_runtime_error(self):
        adapter = PyroAdapter(mechanism="empirical", inference_method="importance")
        q = Query(targets=("X0",), evidence={}, kind="marginal")
        with pytest.raises(RuntimeError, match="not fitted"):
            adapter.query(q)

    def test_epochs_kwarg_accepted(self):
        """fit() silently accepts epochs kwarg (runner API compatibility)."""
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", n_samples=20,
        )
        problem = _make_small_discrete_problem(n_samples=200, seed=1)
        adapter.fit(problem, epochs=5)   # should not raise
        assert adapter._fitted
