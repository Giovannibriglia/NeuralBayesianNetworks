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

from nbn.bench.adapters import PyroAdapter
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

    def test_device_default_auto_detects(self):
        """Default device (None) auto-detects: cuda-if-available-else-cpu.

        Behavior change (device-autodetect fix, 2026-06-04): the default
        was "cpu"; it now resolves via resolve_device() so the adapter uses
        the GPU when one is present. Pin device="cpu" in config to force CPU.
        """
        import torch

        adapter = PyroAdapter(mechanism="empirical", inference_method="importance")
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert adapter.device == expected

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


# ---------------------------------------------------------------------------
# Regression tests — ported from tests/integration/test_adapters_smoke.py
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRegressions:
    """Regression tests for bugs fixed in v0.5-#32 and v0.7-#61.

    Ported from the v0.12 test_adapters_smoke.py integration tests to use
    the v0.13 PyroAdapter API.
    """

    def test_lg_conditional_32(self) -> None:
        """Issue #32 regression: pyro must condition on parents, not marginal.

        Pre-fix, _fit_gaussian_leaf stored per-node marginals and _pyro_model
        sampled Normal(mean, std) independent of parent values.  Post-fix,
        _fit_lg_leaf regresses each continuous node on its continuous parents.

        Fixture: 2-node chain A → B with B = 2A + N(0, 0.1).
        P(B|A=+2) mean must cluster near +4, P(B|A=-2) near -4.
        Gap must be ~8; a gap near 0 means regression to the marginal path.
        """
        pytest.importorskip("pyro")
        torch.manual_seed(0)
        n = 2000
        a = torch.randn(n)
        b = 2.0 * a + 0.1 * torch.randn(n)
        problem = BenchmarkProblem(
            name="pyro_lg_conditional_32",
            dag=[("A", "B")],
            variables={"A": ("continuous", 1), "B": ("continuous", 1)},
            train_data={"A": a, "B": b},
            test_data={"A": a[:100], "B": b[:100]},
            queries=[],
            ground_truth=None,
        )
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", n_samples=400,
        )
        adapter.fit(problem)

        means = {}
        for a_obs in (2.0, -2.0):
            post = adapter._posterior_samples(
                Query(targets=("B",),
                      evidence={"A": torch.tensor(a_obs)},
                      kind="marginal"),
                n_samples=400,
            )
            means[a_obs] = float(post.mean())

        gap = means[2.0] - means[-2.0]
        assert abs(gap - 8.0) < 1.0, (
            f"P(B|A=+2) mean = {means[2.0]:+.3f}, P(B|A=-2) mean = "
            f"{means[-2.0]:+.3f}, gap = {gap:+.3f} (expected ~8.0). "
            f"A gap near 0 indicates regression to the pre-#32 marginal path."
        )
        assert abs(means[2.0] - 4.0) < 0.5
        assert abs(means[-2.0] - (-4.0)) < 0.5

    def test_hybrid_one_hot_regression_61(self) -> None:
        """Issue #61 regression: discrete parents are one-hot encoded.

        Fixture: 3-node network D → B ← A where D is discrete (k=3) and A is
        continuous.  B = 0.5*A + offset[D] + N(0, 0.1), offset = [0, 1, 2].

        Two checks:
        1. Beta coefficients for the discrete one-hot columns have std > 0.3,
           confirming the discrete-parent dimension was fitted.
        2. Posterior mean of B conditioned on D=2 (offset=2) vs D=0 (offset=0)
           differs by approximately 2.0 (planted category gap).
        """
        pytest.importorskip("pyro")
        torch.manual_seed(42)
        n = 1000
        d = torch.randint(0, 3, (n,))
        a = torch.randn(n)
        offsets = torch.tensor([0.0, 1.0, 2.0])
        b = 0.5 * a + offsets[d] + 0.1 * torch.randn(n)

        problem = BenchmarkProblem(
            name="pyro_hybrid_61",
            dag=[("D", "B"), ("A", "B")],
            variables={
                "D": ("discrete", 3),
                "A": ("continuous", 1),
                "B": ("continuous", 1),
            },
            train_data={"D": d, "A": a, "B": b},
            test_data={"D": d[:100], "A": a[:100], "B": b[:100]},
            queries=[],
            ground_truth=None,
        )
        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", n_samples=200,
        )
        adapter.fit(problem)

        # Check 1: discrete parent in _gaussian fit
        beta, sigma, cont_pa, disc_pa, disc_cards = adapter._gaussian["B"]
        assert disc_pa == ["D"], f"expected disc_pa=['D'], got {disc_pa}"
        assert disc_cards == [3], f"expected disc_cards=[3], got {disc_cards}"
        # beta layout: [intercept, cont_coeff(A), oh_coeff_D0, oh_coeff_D1]
        assert beta.shape[0] == 4, f"expected beta len 4, got {beta.shape[0]}"
        oh_coeffs = beta[2:]
        assert oh_coeffs.std() > 0.3, (
            f"one-hot coefficients {oh_coeffs.tolist()} have near-zero std; "
            "discrete-parent dimension was not fitted."
        )

        # Check 2: posterior mean shifts with discrete parent value
        means = {}
        for d_obs in (0, 2):
            post = adapter._posterior_samples(
                Query(targets=("B",),
                      evidence={"D": torch.tensor(d_obs),
                                 "A": torch.tensor(0.0)},
                      kind="marginal"),
                n_samples=400,
            )
            means[d_obs] = float(post.mean())

        gap = means[2] - means[0]
        assert abs(gap - 2.0) < 0.5, (
            f"P(B|D=2,A=0) mean = {means[2]:+.3f}, P(B|D=0,A=0) mean = "
            f"{means[0]:+.3f}, gap = {gap:+.3f} (expected ~2.0)."
        )

    # -- Bug 5 (#127): topological fit order ----------------------------------

    @pytest.mark.parametrize(
        "net, target",
        [("alarm", "LVFAILURE"), ("sachs", "Erk")],
    )
    def test_fit_succeeds_when_child_precedes_parent_127(self, net, target):
        """Bug 5 regression: fit() must register parent cardinalities in
        topological order, not problem.variables insertion order.

        alarm and sachs both have discrete child nodes appearing *before*
        their parents in insertion order (alarm: 17 such (child, parent)
        pairs, sachs: 15).  Pre-fix, fit() iterated insertion order and
        _fit_discrete_cpt's ``self._cards[p]`` parent lookup raised
        KeyError(parent_name) — 472x KeyError('LVFAILURE') on alarm and
        135x KeyError('Erk') on sachs in the bnlearn_paper run.  These are
        NODE names, not state values.

        Reference: issue #127 (Bug 5),
        docs/v0.13-nbn-cat-ve-investigation.md.
        """
        pytest.importorskip("pyro")
        from nbn.bench.problems.bnlearn import (
            BnlearnConfig,
            BnlearnProblemSource,
        )

        cfg = BnlearnConfig(
            networks=[net], seeds=[0],
            n_train=500, n_test=100, n_reference=100,
        )
        problem = next(BnlearnProblemSource().iter_problems(cfg))

        adapter = PyroAdapter(
            mechanism="empirical", inference_method="importance", n_samples=20,
        )
        # Pre-fix this raised KeyError(target) at fit time.
        adapter.fit(problem)
        assert adapter._fitted
        assert target in adapter._cards

        # A query on the previously-failing node returns a valid posterior.
        post = adapter.query(
            Query(targets=(target,), evidence={}, kind="marginal")
        )
        assert isinstance(post, Posterior)
        assert post.probs is not None
        assert torch.isclose(post.probs.sum(), torch.tensor(1.0), atol=1e-4)
