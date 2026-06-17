"""Tests for v0.13 NBNAdapter.

Three test classes:
  - TestProtocolConformance: isinstance checks + name/pre-fit state
  - TestApplicability: is_applicable() per family
  - TestBehavioral: end-to-end fit + query on small synthetic BNs
    (marked @pytest.mark.slow — excluded from the fast gate)

Reference: docs/v0.13-benchmark-redesign.md §4.1
"""
from __future__ import annotations

import pytest
import torch

from benchmarking.adapters import NBNAdapter
from benchmarking.core.interfaces import BaselineAdapter
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.domains.posterior import Posterior


# ---- Helpers to build small synthetic problems for testing ----

def _make_small_discrete_problem(n_samples: int = 200, seed: int = 0) -> BenchmarkProblem:
    """Build a small 3-node discrete BN problem.

    Topology: X0 -> X1 -> X2, all binary.
    """
    import networkx as nx
    torch.manual_seed(seed)

    dag = nx.DiGraph()
    dag.add_edges_from([("X0", "X1"), ("X1", "X2")])

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


def _make_small_continuous_problem(n_samples: int = 200, seed: int = 0) -> BenchmarkProblem:
    """Build a small 3-node continuous BN problem (linear Gaussian chain)."""
    import networkx as nx
    torch.manual_seed(seed)

    dag = nx.DiGraph()
    dag.add_edges_from([("X0", "X1"), ("X1", "X2")])

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


# ---- Tests ----

class TestProtocolConformance:
    """NBNAdapter satisfies the v0.13 BaselineAdapter protocol."""

    def test_nbn_adapter_satisfies_protocol(self):
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        assert isinstance(adapter, BaselineAdapter)

    def test_name_derived_from_mechanism_engine(self):
        assert NBNAdapter(mechanism="cat",       engine="lw").name     == "nbn-cat-lw"
        assert NBNAdapter(mechanism="cat",       engine="ve").name     == "nbn-cat-ve"
        assert NBNAdapter(mechanism="neuralcat", engine="lw").name     == "nbn-neuralcat-lw"
        assert NBNAdapter(mechanism="neuralcat", engine="ve").name     == "nbn-neuralcat-ve"
        assert NBNAdapter(mechanism="lg",        engine="lw").name     == "nbn-lg-lw"
        assert NBNAdapter(mechanism="mdn",       engine="lw").name     == "nbn-mdn-lw"
        assert NBNAdapter(mechanism="flow",      engine="lw").name     == "nbn-flow-lw"
        assert NBNAdapter(mechanism="hybrid",    engine="router").name == "nbn-hybrid-router"

    def test_pre_fit_state_is_none(self):
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        assert adapter.model is None
        assert adapter._engine_obj is None
        assert adapter.problem is None

    def test_invalid_mechanism_raises(self):
        with pytest.raises(ValueError, match="Unknown mechanism"):
            NBNAdapter(mechanism="bogus", engine="lw")

    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            NBNAdapter(mechanism="cat", engine="bogus")


class TestApplicability:
    """is_applicable() returns correct results per (adapter, family)."""

    def test_cat_lw_applicable_to_discrete(self):
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        problem = _make_small_discrete_problem()
        assert adapter.is_applicable(problem) is True

    def test_cat_ve_applicable_to_discrete(self):
        adapter = NBNAdapter(mechanism="cat", engine="ve")
        problem = _make_small_discrete_problem()
        assert adapter.is_applicable(problem) is True

    def test_mdn_lw_not_applicable_to_discrete(self):
        adapter = NBNAdapter(mechanism="mdn", engine="lw")
        problem = _make_small_discrete_problem()
        assert adapter.is_applicable(problem) is False

    def test_mdn_lw_applicable_to_continuous(self):
        adapter = NBNAdapter(mechanism="mdn", engine="lw")
        problem = _make_small_continuous_problem()
        assert adapter.is_applicable(problem) is True

    def test_cat_lw_not_applicable_to_continuous(self):
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        problem = _make_small_continuous_problem()
        assert adapter.is_applicable(problem) is False

    def test_unknown_name_returns_false(self):
        # Manually set a name not in the registry
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        adapter.name = "nbn-not-in-registry"
        problem = _make_small_discrete_problem()
        assert adapter.is_applicable(problem) is False


@pytest.mark.slow
class TestBehavioral:
    """End-to-end fit + query on real small problems.

    These tests train a real NBN model on small synthetic data. They
    do not assert exact posterior values (probabilistic output) but do
    verify shape, value range, and Posterior type contracts.
    """

    def test_discrete_fit_stores_state(self):
        """After fit(), model and engine are populated."""
        adapter = NBNAdapter(mechanism="cat", engine="lw", device="cpu")
        problem = _make_small_discrete_problem(n_samples=300, seed=0)
        assert adapter.model is None

        adapter.fit(problem, epochs=3)

        assert adapter.model is not None
        assert adapter._engine_obj is not None
        assert adapter.problem is problem

    def test_discrete_fit_and_query_returns_valid_posterior(self):
        """nbn-cat-lw: query returns probs vector summing to ~1."""
        adapter = NBNAdapter(mechanism="cat", engine="lw", device="cpu")
        problem = _make_small_discrete_problem(n_samples=500, seed=42)
        adapter.fit(problem, epochs=5)

        q = Query(
            targets=("X2",),
            evidence={"X0": torch.tensor(0)},
            kind="marginal",
        )
        posterior = adapter.query(q)

        assert isinstance(posterior, Posterior)
        assert posterior.probs is not None
        assert posterior.samples is None
        assert posterior.probs.shape == (2,)   # binary target
        assert torch.isclose(
            posterior.probs.sum(), torch.tensor(1.0), atol=1e-4
        )
        assert (posterior.probs >= 0).all()
        assert (posterior.probs <= 1).all()

    def test_continuous_fit_and_query_returns_valid_posterior(self):
        """nbn-mdn-lw: query returns finite 1D sample tensor."""
        adapter = NBNAdapter(mechanism="mdn", engine="lw", device="cpu")
        problem = _make_small_continuous_problem(n_samples=500, seed=42)
        adapter.fit(problem, epochs=10)

        q = Query(
            targets=("X2",),
            evidence={"X0": torch.tensor(0.5)},
            kind="marginal",
        )
        posterior = adapter.query(q)

        assert isinstance(posterior, Posterior)
        assert posterior.samples is not None
        assert posterior.probs is None
        assert posterior.samples.ndim == 1
        assert posterior.samples.shape[0] > 0
        assert torch.isfinite(posterior.samples).all()

    def test_query_before_fit_raises_runtime_error(self):
        adapter = NBNAdapter(mechanism="cat", engine="lw")
        q = Query(targets=("X0",), evidence={}, kind="marginal")
        with pytest.raises(RuntimeError, match="not fitted"):
            adapter.query(q)

    def test_epochs_kwarg_accepted(self):
        """fit() accepts epochs as a kwarg without crashing."""
        adapter = NBNAdapter(mechanism="cat", engine="lw", device="cpu")
        problem = _make_small_discrete_problem(n_samples=200, seed=1)
        adapter.fit(problem, epochs=2)   # very short — just checking it runs
        assert adapter.model is not None


# ---------------------------------------------------------------------------
# Regression: NeuralCategoricalMechanism root-node empirical frequency
# Ported from test_param_learning_dispatch.py assertion 4.
# ---------------------------------------------------------------------------

class TestNeuralCatRootNode:
    """NeuralCategoricalMechanism root-node logits must reflect empirical
    log-frequency, not uniform initialisation (issue #52 regression).

    This test has no crash_test_runner dependency — it tests the mechanism
    directly.  Extracted from test_param_learning_dispatch_triplet assertion 4.
    """

    def test_root_node_logits_reflect_empirical_frequency(self) -> None:
        from nbn.mechanisms.parametric.neural_categorical import NeuralCategoricalMechanism

        # Biased training data: 80% class 0, 15% class 1, 5% class 2.
        biased_x = torch.cat([
            torch.zeros(800, dtype=torch.long),
            torch.ones(150, dtype=torch.long),
            torch.full((50,), 2, dtype=torch.long),
        ])
        expected_freq = torch.tensor([0.8, 0.15, 0.05])
        mech = NeuralCategoricalMechanism(n_classes=3)
        mech.fit_local(biased_x, parents=None)
        fitted_probs = torch.softmax(mech._root_logits.detach(), dim=-1)

        max_diff_from_expected = (expected_freq - fitted_probs).abs().max().item()
        max_diff_from_uniform = (
            torch.full((3,), 1 / 3) - fitted_probs
        ).abs().max().item()

        assert max_diff_from_expected < 1e-4, (
            f"Root-node logits must reflect empirical log-frequency (#52); "
            f"got fitted={fitted_probs.tolist()} vs expected={expected_freq.tolist()} "
            f"(max abs diff={max_diff_from_expected:.6e})"
        )
        assert max_diff_from_uniform > 0.1, (
            f"Root-node logits must NOT be uniform (uniform-init was the "
            f"v0.7-#43 audit's #52 finding); got fitted={fitted_probs.tolist()} "
            f"(max diff from uniform={max_diff_from_uniform:.6e})"
        )
