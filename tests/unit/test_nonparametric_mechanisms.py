"""Unit tests for the non-parametric mechanisms.

Covers, for each estimator:
1. ``fit_local`` returns a dict and flips ``is_fitted`` to True.
2. Shape contracts: ``log_prob`` -> ``[B]`` (or ``[B, S]``); ``sample`` -> ``[B, n, D_x]``.
3. ``log_prob`` is finite on the training data and on fresh samples.
4. Normalisation: continuous children integrate to ~1 over a grid; discrete
   children produce per-parent rows that sum to 1.
5. Root-node (``parents=None``) handling.
6. Conditional tracking: the estimated mean follows a known linear SCM.

These are deterministic-ish (seeded) and CPU-only so they run in the standard
unit-test gate without a GPU.
"""
import math

import pytest
import torch

from nbn.mechanisms.non_parametric.conditional_kde import ConditionalKDEMechanism
from nbn.mechanisms.non_parametric.knn_conditional import KNNConditionalMechanism
from nbn.mechanisms.non_parametric.flexcode import FlexCodeMechanism
from nbn.mechanisms.non_parametric.smoothed_empirical_categorical import (
    SmoothedEmpiricalCategoricalMechanism,
)

# Public re-export sanity (the network builder resolves classes from here).
from nbn.mechanisms import (
    ConditionalKDEMechanism as _KDE_reexport,
    KNNConditionalMechanism as _KNN_reexport,
    FlexCodeMechanism as _FC_reexport,
    SmoothedEmpiricalCategoricalMechanism as _SEC_reexport,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _seed(s=0):
    torch.manual_seed(s)


def _linear_scm(n=600, n_parents=2, noise=0.1):
    """y = 1.5*x1 - 0.5*x2 + noise; continuous child, continuous parents."""
    pa = torch.randn(n, n_parents)
    coef = torch.tensor([1.5, -0.5][:n_parents])
    x = (pa * coef).sum(-1, keepdim=True) + noise * torch.randn(n, 1)
    return x, pa


def _discrete_data(n=400, n_parents=2, k=3, parent_k=2):
    pa = torch.randint(0, parent_k, (n, n_parents)).float()
    x = torch.randint(0, k, (n,)).float()
    return x, pa


def _grid_integral(mech, pa_row, lo, hi, steps=2000):
    """∫ p(x|pa) dx over [lo, hi] via the trapezoidal rule (1-D child)."""
    xs = torch.linspace(lo, hi, steps).unsqueeze(-1)  # [steps, 1]
    pa_b = pa_row.expand(steps, -1) if pa_row is not None else None
    lp = mech.log_prob(xs, pa_b)
    dens = lp.exp()
    return torch.trapz(dens, xs.squeeze(-1)).item()


# ── ConditionalKDEMechanism ───────────────────────────────────────────────────

class TestConditionalKDE:
    def _fitted(self, **kw):
        _seed(0)
        x, pa = _linear_scm()
        mech = ConditionalKDEMechanism(**kw)
        info = mech.fit_local(x, pa)
        return mech, x, pa, info

    def test_fit_returns_dict_and_marks_fitted(self):
        mech, *_ , info = self._fitted()
        assert isinstance(info, dict)
        assert mech.is_fitted

    def test_log_prob_shape(self):
        mech, x, pa, _ = self._fitted()
        lp = mech.log_prob(x[:5], pa[:5])
        assert lp.shape == (5,)

    def test_sample_shape(self):
        mech, x, pa, _ = self._fitted()
        s = mech.sample(pa[:4], n=10)
        assert s.shape == (4, 10, 1)

    def test_log_prob_finite(self):
        mech, x, pa, _ = self._fitted()
        assert torch.isfinite(mech.log_prob(x, pa)).all()

    def test_integrates_to_one(self):
        mech, x, pa, _ = self._fitted()
        z = _grid_integral(mech, pa[:1], lo=-8.0, hi=8.0)
        assert z == pytest.approx(1.0, abs=0.06)

    def test_not_discrete(self):
        mech, *_ = self._fitted()
        assert mech.is_discrete is False

    def test_root_node(self):
        _seed(0)
        x = torch.randn(400, 1)
        mech = ConditionalKDEMechanism()
        mech.fit_local(x, None)
        assert mech.is_fitted
        s = mech.sample(None, n=8)  # root sample
        assert s.shape[-1] == 1
        z = _grid_integral(mech, None, lo=-8.0, hi=8.0)
        assert z == pytest.approx(1.0, abs=0.06)

    def test_conditional_mean_tracks_scm(self):
        mech, x, pa, _ = self._fitted()
        query = torch.tensor([[1.0, -1.0]])  # true E[y] = 1.5 + 0.5 = 2.0
        s = mech.sample(query, n=4000).squeeze(0).squeeze(-1)
        assert s.mean().item() == pytest.approx(2.0, abs=0.3)


# ── KNNConditionalMechanism ───────────────────────────────────────────────────

class TestKNNContinuous:
    def _fitted(self, **kw):
        _seed(0)
        x, pa = _linear_scm()
        mech = KNNConditionalMechanism(**kw)
        info = mech.fit_local(x, pa)
        return mech, x, pa, info

    def test_fit_returns_dict_and_marks_fitted(self):
        mech, *_, info = self._fitted()
        assert isinstance(info, dict)
        assert mech.is_fitted

    def test_log_prob_shape(self):
        mech, x, pa, _ = self._fitted()
        assert mech.log_prob(x[:5], pa[:5]).shape == (5,)

    def test_sample_shape(self):
        mech, x, pa, _ = self._fitted()
        assert mech.sample(pa[:4], n=10).shape == (4, 10, 1)

    def test_log_prob_finite(self):
        mech, x, pa, _ = self._fitted()
        assert torch.isfinite(mech.log_prob(x, pa)).all()

    def test_integrates_to_one(self):
        mech, x, pa, _ = self._fitted()
        z = _grid_integral(mech, pa[:1], lo=-8.0, hi=8.0)
        assert z == pytest.approx(1.0, abs=0.08)

    def test_conditional_mean_tracks_scm(self):
        mech, x, pa, _ = self._fitted()
        query = torch.tensor([[1.0, -1.0]])
        s = mech.sample(query, n=4000).squeeze(0).squeeze(-1)
        assert s.mean().item() == pytest.approx(2.0, abs=0.3)

    def test_root_node(self):
        _seed(0)
        x = torch.randn(400, 1)
        mech = KNNConditionalMechanism()
        mech.fit_local(x, None)
        assert mech.is_fitted
        assert mech.sample(None, n=8).shape[-1] == 1


class TestKNNDiscrete:
    def _fitted(self, **kw):
        _seed(0)
        x, pa = _discrete_data(n=400, n_parents=2, k=3, parent_k=2)
        mech = KNNConditionalMechanism(discrete_child=True, **kw)
        info = mech.fit_local(x, pa, n_classes=3, parent_cards=[2, 2])
        return mech, x, pa, info

    def test_is_discrete_flag(self):
        mech, *_ = self._fitted()
        assert mech.is_discrete is True

    def test_log_prob_shape(self):
        mech, x, pa, _ = self._fitted()
        assert mech.log_prob(x[:5], pa[:5]).shape == (5,)

    def test_sample_shape(self):
        mech, x, pa, _ = self._fitted()
        assert mech.sample(pa[:4], n=10).shape == (4, 10, 1)

    def test_probs_sum_to_one(self):
        mech, x, pa, _ = self._fitted()
        dist = mech.forward(pa[:6])
        probs = dist.probs
        assert torch.allclose(probs.sum(-1), torch.ones(probs.shape[:-1]), atol=1e-5)

    def test_tabulate_rows_normalised(self):
        mech, x, pa, _ = self._fitted()
        logits = mech.tabulate([2, 2])
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(-1), torch.ones(probs.shape[:-1]), atol=1e-5)


# ── FlexCodeMechanism ─────────────────────────────────────────────────────────

class TestFlexCode:
    """FlexCode's fit is a 60-epoch MLP — by far the most expensive mechanism
    fit in this file.  Every test below asked for the same fit (same seed,
    same kwargs, same data) and got its own, so the class paid for six
    identical trainings.  One class-scoped fixture, six assertions against
    it: identical coverage, a sixth of the wall clock.

    The fitted mechanism is shared, so no test may mutate its parameters.
    ``test_grad_flow`` only populates ``.grad`` (no optimiser step), which
    nothing else reads.
    """

    @pytest.fixture(scope="class")
    def fitted(self):
        return self._fit()

    @staticmethod
    def _fit(**kw):
        _seed(0)
        x, pa = _linear_scm()
        kw.setdefault("epochs", 60)
        kw.setdefault("n_basis", 21)
        mech = FlexCodeMechanism(**kw)
        info = mech.fit_local(x, pa)
        return mech, x, pa, info

    def test_fit_returns_dict_and_marks_fitted(self, fitted):
        mech, *_, info = fitted
        assert isinstance(info, dict)
        assert mech.is_fitted

    def test_log_prob_shape(self, fitted):
        mech, x, pa, _ = fitted
        assert mech.log_prob(x[:5], pa[:5]).shape == (5,)

    def test_sample_shape(self, fitted):
        mech, x, pa, _ = fitted
        assert mech.sample(pa[:4], n=10).shape == (4, 10, 1)

    def test_log_prob_finite(self, fitted):
        mech, x, pa, _ = fitted
        assert torch.isfinite(mech.log_prob(x, pa)).all()

    def test_integrates_to_one(self, fitted):
        mech, x, pa, _ = fitted
        # FlexCode renormalises on its own internal grid; over the support it ≈1.
        lo, hi = mech._y_min.item(), mech._y_max.item()
        z = _grid_integral(mech, pa[:1], lo=lo, hi=hi, steps=4000)
        assert z == pytest.approx(1.0, abs=0.08)

    def test_grad_flow(self, fitted):
        mech, x, pa, _ = fitted
        lp = mech.log_prob(x[:16], pa[:16])
        (-lp.mean()).backward()
        assert any(p.grad is not None for p in mech.parameters())

    def test_multivariate_child_raises(self):
        _seed(0)
        x = torch.randn(200, 2)  # D_x = 2 unsupported
        pa = torch.randn(200, 2)
        mech = FlexCodeMechanism(epochs=2)
        with pytest.raises(NotImplementedError):
            mech.fit_local(x, pa)

    def test_root_node(self):
        _seed(0)
        x = torch.randn(400, 1)
        mech = FlexCodeMechanism(epochs=40, n_basis=21)
        mech.fit_local(x, None)
        assert mech.is_fitted
        assert mech.sample(None, n=8).shape[-1] == 1


# ── SmoothedEmpiricalCategoricalMechanism ─────────────────────────────────────

class TestSmoothedEmpiricalCategorical:
    def _fitted(self, alpha=1.0):
        _seed(0)
        x, pa = _discrete_data(n=400, n_parents=2, k=3, parent_k=2)
        mech = SmoothedEmpiricalCategoricalMechanism(alpha=alpha)
        info = mech.fit_local(x, pa, parent_cards=[2, 2])
        return mech, x, pa, info

    def test_is_discrete(self):
        mech, *_ = self._fitted()
        assert mech.is_discrete is True

    def test_rows_sum_to_one(self):
        mech, *_ = self._fitted()
        probs = mech.cpt
        assert torch.allclose(probs.sum(-1), torch.ones(probs.shape[0]), atol=1e-5)

    def test_log_prob_shape(self):
        mech, x, pa, _ = self._fitted()
        assert mech.log_prob(x[:5], pa[:5]).shape == (5,)

    def test_sample_shape(self):
        mech, x, pa, _ = self._fitted()
        assert mech.sample(pa[:4], n=10).shape == (4, 10, 1)

    def test_alpha_zero_allowed(self):
        # alpha=0 is the unsmoothed MLE; should still fit without error.
        mech, *_ = self._fitted(alpha=0.0)
        assert mech.is_fitted

    def test_negative_alpha_rejected(self):
        with pytest.raises((ValueError, AssertionError)):
            SmoothedEmpiricalCategoricalMechanism(alpha=-1.0)

    def test_matches_parent_categorical_table(self):
        """alpha=1 must reproduce CategoricalTableMechanism(alpha=1) exactly."""
        from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism
        _seed(0)
        x, pa = _discrete_data(n=400, n_parents=2, k=3, parent_k=2)
        a = SmoothedEmpiricalCategoricalMechanism(alpha=1.0)
        a.fit_local(x, pa, parent_cards=[2, 2])
        b = CategoricalTableMechanism(alpha=1.0)
        b.fit_local(x, pa, parent_cards=[2, 2])
        assert torch.allclose(a.cpt, b.cpt, atol=1e-6)


# ── public re-export identity ─────────────────────────────────────────────────

def test_public_reexports_are_identical():
    assert _KDE_reexport is ConditionalKDEMechanism
    assert _KNN_reexport is KNNConditionalMechanism
    assert _FC_reexport is FlexCodeMechanism
    assert _SEC_reexport is SmoothedEmpiricalCategoricalMechanism


# ── auto hyper-parameter selection ────────────────────────────────────────────

class TestAutoSelection:
    def test_kde_auto_resolves_to_float(self):
        _seed(0)
        x, pa = _linear_scm(n=400)
        mech = ConditionalKDEMechanism(bw_factor="auto")
        mech.fit_local(x, pa)
        assert mech.is_fitted
        assert isinstance(mech.bw_factor, float)
        assert mech.bw_factor > 0
        assert torch.isfinite(mech.log_prob(x[:5], pa[:5])).all()
        assert mech.sample(pa[:4], n=8).shape == (4, 8, 1)

    def test_kde_auto_integrates_to_one(self):
        _seed(0)
        x, pa = _linear_scm(n=400)
        mech = ConditionalKDEMechanism(bw_factor="auto")
        mech.fit_local(x, pa)
        z = _grid_integral(mech, pa[:1], lo=-8.0, hi=8.0)
        assert z == pytest.approx(1.0, abs=0.06)

    def test_kde_rejects_bad_bw_factor_string(self):
        with pytest.raises(ValueError):
            ConditionalKDEMechanism(bw_factor="nope")

    def test_kde_auto_root_node(self):
        _seed(0)
        x = torch.randn(400, 1)
        mech = ConditionalKDEMechanism(bw_factor="auto")
        mech.fit_local(x, None)
        assert isinstance(mech.bw_factor, float)
        assert mech.sample(None, n=8).shape[-1] == 1

    def test_flexcode_auto_resolves_to_float(self):
        _seed(0)
        x, pa = _linear_scm(n=400)
        mech = FlexCodeMechanism(sharpen="auto", epochs=40, n_basis=21)
        mech.fit_local(x, pa)
        assert mech.is_fitted
        assert isinstance(mech.sharpen, float)
        assert mech.sharpen >= 1.0
        assert torch.isfinite(mech.log_prob(x[:5], pa[:5])).all()
        assert mech.sample(pa[:4], n=8).shape == (4, 8, 1)

    def test_flexcode_rejects_bad_sharpen_string(self):
        with pytest.raises(ValueError):
            FlexCodeMechanism(sharpen="nope")
