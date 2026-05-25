"""Tests for v0.13 PgmpyAdapter.

Three test classes:
  - TestProtocolConformance: isinstance checks + name/pre-fit state
  - TestApplicability: is_applicable() per family
  - TestBehavioral: end-to-end fit + query on small synthetic BNs
    (marked @pytest.mark.slow — excluded from the fast gate)

Behavioral tests cover all three label variants:
  - pgmpy-mle-ve   on a 3-node binary BN
  - pgmpy-bayes-ve on the same 3-node binary BN
  - pgmpy-lg-predict on a 3-node linear-Gaussian chain

For pgmpy-lg-predict the test verifies:
  - Posterior.samples has shape (n_samples,) and is finite
  - Sample mean is close to the adapter's own analytical posterior mean
    (self-consistency sanity check; tolerance 0.5)

Reference: docs/v0.13-benchmark-redesign.md §4.1
"""
from __future__ import annotations

import pytest
import torch

from benchmarking.adapters import PgmpyAdapter
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


def _make_small_continuous_lg_problem(
    n_samples: int = 400, seed: int = 0,
) -> BenchmarkProblem:
    """3-node linear-Gaussian chain: X0 → X1 → X2.

    True params:
        X0 ~ N(0, 1)
        X1 = 0.5·X0 + ε,  ε ~ N(0, 0.5²)
        X2 = 0.5·X1 + ε,  ε ~ N(0, 0.5²)
    """
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


# ---- Tests ------------------------------------------------------------------

class TestProtocolConformance:
    """PgmpyAdapter satisfies the v0.13 BaselineAdapter protocol."""

    def test_pgmpy_adapter_satisfies_protocol(self):
        adapter = PgmpyAdapter(param_method="mle", inference_method="ve")
        assert isinstance(adapter, BaselineAdapter)

    def test_name_derived_from_args(self):
        assert PgmpyAdapter(param_method="mle",   inference_method="ve").name     == "pgmpy-mle-ve"
        assert PgmpyAdapter(param_method="bayes",  inference_method="ve").name    == "pgmpy-bayes-ve"
        assert PgmpyAdapter(param_method="lg",    inference_method="predict").name == "pgmpy-lg-predict"

    def test_pre_fit_state(self):
        adapter = PgmpyAdapter(param_method="mle", inference_method="ve")
        assert adapter._kind == "unfit"
        assert adapter._model is None
        assert adapter._infer is None
        assert adapter._lg_topo is None
        assert adapter.problem is None

    def test_invalid_param_method_raises(self):
        with pytest.raises(ValueError, match="Unknown param_method"):
            PgmpyAdapter(param_method="bogus", inference_method="ve")

    def test_invalid_inference_method_raises(self):
        with pytest.raises(ValueError, match="Unknown inference_method"):
            PgmpyAdapter(param_method="mle", inference_method="bogus")

    def test_n_samples_default(self):
        adapter = PgmpyAdapter(param_method="lg", inference_method="predict")
        assert adapter.n_samples == 1024

    def test_n_samples_override(self):
        adapter = PgmpyAdapter(
            param_method="lg", inference_method="predict", n_samples=512,
        )
        assert adapter.n_samples == 512


class TestApplicability:
    """is_applicable() returns correct results per (adapter, family)."""

    def test_mle_ve_applicable_to_discrete(self):
        adapter = PgmpyAdapter(param_method="mle", inference_method="ve")
        assert adapter.is_applicable(_make_small_discrete_problem()) is True

    def test_bayes_ve_applicable_to_discrete(self):
        adapter = PgmpyAdapter(param_method="bayes", inference_method="ve")
        assert adapter.is_applicable(_make_small_discrete_problem()) is True

    def test_lg_predict_applicable_to_continuous(self):
        adapter = PgmpyAdapter(param_method="lg", inference_method="predict")
        assert adapter.is_applicable(_make_small_continuous_lg_problem()) is True

    def test_mle_ve_not_applicable_to_continuous(self):
        adapter = PgmpyAdapter(param_method="mle", inference_method="ve")
        assert adapter.is_applicable(_make_small_continuous_lg_problem()) is False

    def test_lg_predict_not_applicable_to_discrete(self):
        adapter = PgmpyAdapter(param_method="lg", inference_method="predict")
        assert adapter.is_applicable(_make_small_discrete_problem()) is False

    def test_unknown_name_returns_false(self):
        adapter = PgmpyAdapter(param_method="mle", inference_method="ve")
        adapter.name = "pgmpy-not-in-registry"
        assert adapter.is_applicable(_make_small_discrete_problem()) is False


@pytest.mark.slow
class TestBehavioral:
    """End-to-end fit + query on small synthetic problems.

    These tests fit real pgmpy models on small data.  They do not assert
    exact posterior values but verify shape, value range, Posterior type
    contract, and (for LG) self-consistency of the sample mean.
    """

    def test_mle_ve_fit_stores_state(self):
        adapter = PgmpyAdapter(param_method="mle", inference_method="ve")
        problem = _make_small_discrete_problem(n_samples=300, seed=0)
        assert adapter._model is None

        adapter.fit(problem)

        assert adapter._model is not None
        assert adapter._infer is not None
        assert adapter._kind == "discrete"
        assert adapter.problem is problem

    def test_mle_ve_fit_and_query_returns_valid_posterior(self):
        """pgmpy-mle-ve: query returns normalised probs vector."""
        adapter = PgmpyAdapter(param_method="mle", inference_method="ve")
        problem = _make_small_discrete_problem(n_samples=500, seed=42)
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

    def test_bayes_ve_fit_and_query_returns_valid_posterior(self):
        """pgmpy-bayes-ve: query returns normalised probs vector."""
        adapter = PgmpyAdapter(param_method="bayes", inference_method="ve")
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

    def test_lg_predict_fit_stores_state(self):
        adapter = PgmpyAdapter(
            param_method="lg", inference_method="predict", n_samples=1024,
        )
        problem = _make_small_continuous_lg_problem(n_samples=400, seed=0)
        assert adapter._model is None

        adapter.fit(problem)

        assert adapter._model is not None
        assert adapter._lg_topo is not None
        assert adapter._kind == "continuous_lg"
        assert adapter.problem is problem

    def test_lg_predict_fit_and_query_returns_valid_posterior(self):
        """pgmpy-lg-predict: samples finite, correct shape, mean near analytical."""
        adapter = PgmpyAdapter(
            param_method="lg", inference_method="predict", n_samples=1024,
        )
        problem = _make_small_continuous_lg_problem(n_samples=500, seed=42)
        adapter.fit(problem)

        q = Query(
            targets=("X2",),
            evidence={"X0": torch.tensor(1.0)},
            kind="marginal",
        )
        posterior = adapter.query(q)

        # Contract checks
        assert isinstance(posterior, Posterior)
        assert posterior.samples is not None
        assert posterior.probs is None
        assert posterior.samples.ndim == 1
        assert posterior.samples.shape[0] == 1024
        assert torch.isfinite(posterior.samples).all()

        # Self-consistency: sample mean should be within 0.5 of the adapter's
        # own analytical posterior mean (generous tolerance — this is a MC
        # check with n=1024, not a test of exact posterior values).
        ref_mean, _ = adapter._lg_posterior_moments(q)
        diff = abs(posterior.samples.mean().item() - ref_mean.item())
        assert diff < 0.5, (
            f"Sample mean {posterior.samples.mean():.3f} deviates from "
            f"analytical posterior mean {ref_mean.item():.3f} by {diff:.3f} "
            f"(tolerance 0.5)"
        )

    def test_lg_predict_marginal_query_no_evidence(self):
        """pgmpy-lg-predict: marginal query (no evidence) returns finite samples."""
        adapter = PgmpyAdapter(
            param_method="lg", inference_method="predict", n_samples=512,
        )
        problem = _make_small_continuous_lg_problem(n_samples=400, seed=3)
        adapter.fit(problem)

        q = Query(targets=("X2",), evidence={}, kind="marginal")
        posterior = adapter.query(q)

        assert isinstance(posterior, Posterior)
        assert posterior.samples is not None
        assert posterior.samples.shape == (512,)
        assert torch.isfinite(posterior.samples).all()

    def test_query_before_fit_raises_runtime_error_mle(self):
        adapter = PgmpyAdapter(param_method="mle", inference_method="ve")
        q = Query(targets=("X0",), evidence={}, kind="marginal")
        with pytest.raises(RuntimeError, match="not fitted"):
            adapter.query(q)

    def test_query_before_fit_raises_runtime_error_lg(self):
        adapter = PgmpyAdapter(param_method="lg", inference_method="predict")
        q = Query(targets=("X0",), evidence={}, kind="marginal")
        with pytest.raises(RuntimeError, match="not fitted"):
            adapter.query(q)

    def test_epochs_kwarg_accepted(self):
        """fit() silently accepts epochs kwarg (runner API compatibility)."""
        adapter = PgmpyAdapter(param_method="mle", inference_method="ve")
        problem = _make_small_discrete_problem(n_samples=200, seed=1)
        adapter.fit(problem, epochs=5)   # should not raise
        assert adapter._model is not None
