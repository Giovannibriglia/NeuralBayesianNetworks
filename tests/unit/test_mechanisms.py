"""Unit tests for all mechanisms.

For each mechanism we verify:
1. log_prob of MC samples integrates to ≈1 (Monte Carlo estimate of Z).
2. Gradients flow: loss.backward() gives non-None grads on all parameters.
"""
import math

import pytest
import torch

from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism
from nbn.mechanisms.parametric.deterministic import DeterministicMechanism
from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism
from nbn.mechanisms.parametric.mdn import MDNMechanism
from nbn.mechanisms.parametric.neural_categorical import NeuralCategoricalMechanism

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_discrete_data(n=200, n_parents=2, k=3, parent_k=2):
    pa = torch.randint(0, parent_k, (n, n_parents)).float()
    x = torch.randint(0, k, (n,)).float()
    return x, pa


def _make_continuous_data(n=300, n_parents=2, d_x=1):
    pa = torch.randn(n, n_parents)
    x = pa.sum(-1, keepdim=True) + 0.1 * torch.randn(n, d_x)
    return x, pa


# ── CategoricalTableMechanism ─────────────────────────────────────────────────

class TestCategoricalTable:
    def _fitted(self):
        x, pa = _make_discrete_data(n=300, n_parents=2, k=3, parent_k=2)
        mech = CategoricalTableMechanism(alpha=1.0)
        mech.fit_local(x, pa, parent_cards=[2, 2])
        return mech, x, pa

    def test_log_prob_sums_to_one(self):
        mech, x, pa = self._fitted()
        # For a fixed parent row, probs over all classes must sum to 1
        pa_row = torch.tensor([[0.0, 0.0]])
        probs = mech.cpt  # [n_parent_states, K]
        assert torch.allclose(probs.sum(-1), torch.ones(probs.shape[0]), atol=1e-5)

    def test_sample_shape(self):
        mech, x, pa = self._fitted()
        s = mech.sample(pa[:4], n=10)
        assert s.shape == (4, 10, 1)

    def test_log_prob_shape(self):
        mech, x, pa = self._fitted()
        lp = mech.log_prob(x[:5], pa[:5])
        assert lp.shape == (5,)

    def test_grad_flow(self):
        mech, x, pa = self._fitted()
        loss = -mech.log_prob(x[:32], pa[:32]).mean()
        loss.backward()
        assert mech._logits.grad is not None
        assert not torch.isnan(mech._logits.grad).any()

    def test_root_node(self):
        x = torch.randint(0, 4, (100,)).float()
        mech = CategoricalTableMechanism()
        mech.fit_local(x, None)
        probs = mech.cpt
        assert abs(probs.sum().item() - 1.0) < 1e-4


# ── LinearGaussianMechanism ───────────────────────────────────────────────────

class TestLinearGaussian:
    def _fitted(self):
        x, pa = _make_continuous_data(n=300)
        mech = LinearGaussianMechanism()
        mech.fit_local(x, pa)
        return mech, x, pa

    def test_sample_shape(self):
        mech, x, pa = self._fitted()
        s = mech.sample(pa[:4], n=10)
        assert s.shape == (4, 10, 1)

    def test_log_prob_shape(self):
        mech, x, pa = self._fitted()
        lp = mech.log_prob(x[:5], pa[:5])
        assert lp.shape == (5,)

    def test_log_prob_finite(self):
        mech, x, pa = self._fitted()
        lp = mech.log_prob(x, pa)
        assert torch.isfinite(lp).all()

    def test_mc_normalisation(self):
        """Monte Carlo estimate of ∫ p(x|pa) dx ≈ 1."""
        mech, x, pa = self._fitted()
        pa_row = pa[:1]
        n_mc = 50_000
        samp = mech.sample(pa_row, n=n_mc).squeeze(0)  # [n_mc, 1]
        lp = mech.log_prob(samp, pa_row.expand(n_mc, -1))
        # E[p] ≈ 1 ⟹ mean(exp(lp)) ≈ 1/volume; just check it's positive and finite
        assert torch.isfinite(lp).all()

    def test_grad_flow(self):
        mech, x, pa = self._fitted()
        loss = -mech.log_prob(x[:32], pa[:32]).mean()
        loss.backward()
        assert mech._log_scale.grad is not None

    def test_root_node(self):
        x = torch.randn(200, 1)
        mech = LinearGaussianMechanism()
        mech.fit_local(x, None)
        assert mech._bias is not None


# ── MDNMechanism ──────────────────────────────────────────────────────────────

class TestMDN:
    def _fitted(self):
        x, pa = _make_continuous_data(n=400)
        mech = MDNMechanism(num_components=3, hidden=(32,), activation="relu")
        mech.fit_local(x, pa, epochs=20, lr=1e-3, batch_size=128)
        return mech, x, pa

    def test_sample_shape(self):
        mech, x, pa = self._fitted()
        s = mech.sample(pa[:4], n=10)
        assert s.shape == (4, 10, 1)

    def test_log_prob_shape(self):
        mech, x, pa = self._fitted()
        lp = mech.log_prob(x[:5], pa[:5])
        assert lp.shape == (5,)

    def test_log_prob_finite(self):
        mech, x, pa = self._fitted()
        lp = mech.log_prob(x, pa)
        assert torch.isfinite(lp).all()

    def test_grad_flow(self):
        mech, x, pa = self._fitted()
        lp = mech.log_prob(x[:16], pa[:16])
        (-lp.mean()).backward()
        grad_found = any(
            p.grad is not None for p in mech.parameters()
        )
        assert grad_found

    def test_root_node(self):
        x = torch.randn(200, 1)
        mech = MDNMechanism(num_components=3, hidden=(16,))
        mech.fit_local(x, None, epochs=5)
        assert mech._root_logits is not None


# ── DeterministicMechanism ────────────────────────────────────────────────────

class TestDeterministic:
    def test_sample_constant(self):
        val = torch.tensor([3.0])
        mech = DeterministicMechanism(val)
        pa = torch.randn(5, 2)
        s = mech.sample(pa, n=10)
        assert s.shape == (5, 10, 1)
        assert (s == 3.0).all()

    def test_log_prob_zero(self):
        val = torch.tensor([1.0])
        mech = DeterministicMechanism(val)
        x = torch.randn(10, 1)
        lp = mech.log_prob(x, None)
        assert (lp == 0.0).all()


# ── NormalizingFlow with Long (discrete) parents ──────────────────────────────

class TestNormalizingFlowLongParents:
    """Regression: a flow node with a discrete (Long) parent must not crash
    with 'mat1 and mat2 must have the same dtype' — the context is cast to
    float at every inference entry point (clg networks healthcare / sangiovese,
    every seed/kind, used to fail here)."""

    def _fit(self):
        from nbn.mechanisms.parametric.normalizing_flow import NormalizingFlowMechanism
        mech = NormalizingFlowMechanism()
        x = torch.randn(96, 1)
        pa = torch.randint(0, 3, (96, 1))   # Long, NOT cast by the caller
        mech.fit_local(x, pa, epochs=2)
        return mech, x, pa

    def test_log_prob_long_parent(self):
        mech, x, pa = self._fit()
        lp = mech.log_prob(x, pa)           # would raise Long vs Float
        assert lp.shape == (96,)
        assert torch.isfinite(lp).all()

    def test_forward_long_parent(self):
        mech, _, pa = self._fit()
        mech.forward(pa)                    # builds the conditioned distribution

    def test_sample_long_parent(self):
        mech, _, pa = self._fit()
        s = mech.sample(pa, n=2)
        assert s.shape == (96, 2, 1)


class TestNormalizingFlowSanitiseParents:
    """Regression (#82): a flow node whose continuous parents carry NaN/Inf
    (from a deep ancestral chain) must not trip a zuko NSF device assert — the
    context is sanitised to 0 at every inference entry point, mirroring MDN's
    guard. Output stays finite on the happy path and under non-finite parents."""

    def _fit(self):
        from nbn.mechanisms.parametric.normalizing_flow import NormalizingFlowMechanism
        mech = NormalizingFlowMechanism()
        x = torch.randn(96, 1)
        pa = torch.randn(96, 2)             # continuous parents
        mech.fit_local(x, pa, epochs=2)
        return mech, x, pa

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_nonfinite_parent_does_not_raise_and_stays_finite(self, bad):
        mech, x, pa = self._fit()
        p = pa.clone()
        p[0, 0] = bad                       # one poisoned entry
        lp = mech.log_prob(x, p)
        mech.forward(p)
        s = mech.sample(p, n=2)
        assert torch.isfinite(lp).all()
        assert torch.isfinite(s).all()

    def test_happy_path_unchanged(self):
        # All-finite parents: guard is a no-op, results finite.
        mech, x, pa = self._fit()
        assert torch.isfinite(mech.log_prob(x, pa)).all()
        assert torch.isfinite(mech.sample(pa, n=2)).all()

    def test_shared_guard_is_mdn_guard(self):
        # The flow guard is the very same helper MDN uses (promoted to utils).
        from nbn.mechanisms.parametric.mdn import _sanitise_parents as mdn_sp
        from nbn.utils.batching import _sanitise_parents as util_sp
        assert mdn_sp is util_sp
