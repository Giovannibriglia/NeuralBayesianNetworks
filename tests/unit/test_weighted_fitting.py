"""Per-sample weights: replication equivalence, zero-weight ablation, refusal.

Weights are multiplicities, so the acceptance criterion throughout is
*replication equivalence*: fitting with integer weights ``w`` on ``D`` must
equal fitting on ``D`` with each row repeated ``w_i`` times.

* Closed-form families (categorical, LinearGaussian) are compared on their
  **fitted parameters**.
* Gradient-trained families are compared on their **full-batch gradients**
  instead.  Comparing end parameters there would confound the weighting with
  the optimiser path and the batch composition; the gradient is the thing the
  weighting is supposed to change, so it is what gets asserted.  The recipe is
  ``epochs=1, lr=0.0, batch_size=huge``: one full-batch step that moves
  nothing, leaving ``.grad`` holding exactly the quantity under test.

The zero-weight tests are separate because "down-weighted to almost nothing"
and "absent" are different claims, and only the second one is the contract.
"""
from __future__ import annotations

import pytest
import torch

from nbn import NeuralBayesianNetwork as NBN
from nbn.learning.weighting import (
    validate_weights,
    weighted_mean,
    weighted_moments,
    weighted_quantile,
)
from nbn.mechanisms import (
    CategoricalTableMechanism,
    LinearGaussianMechanism,
    MDNMechanism,
    NeuralCategoricalMechanism,
)
from nbn.mechanisms.non_parametric.conditional_kde import ConditionalKDEMechanism
from nbn.mechanisms.non_parametric.flexcode import FlexCodeMechanism
from nbn.mechanisms.non_parametric.knn_conditional import KNNConditionalMechanism


def _replicate(w: torch.Tensor) -> torch.Tensor:
    """Row indices repeating row i exactly w_i times."""
    return torch.cat([torch.full((int(k),), i) for i, k in enumerate(w)])


# ==========================================================================
# The reduction convention itself
# ==========================================================================


def test_weighted_mean_reduces_to_mean_when_unweighted():
    v = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(weighted_mean(v, None), v.mean())


def test_weighted_mean_equals_the_replicated_mean():
    v = torch.tensor([1.0, 2.0, 3.0])
    w = torch.tensor([2.0, 1.0, 1.0])
    torch.testing.assert_close(
        weighted_mean(v, w), v[_replicate(w)].mean(), atol=1e-7, rtol=0,
    )


def test_weighted_mean_is_invariant_to_the_weights_magnitude():
    """The reason for a weighted mean over a weighted sum: step-size scale."""
    v = torch.randn(50)
    w = torch.rand(50) + 0.1
    torch.testing.assert_close(
        weighted_mean(v, w), weighted_mean(v, w * 17.0), atol=1e-6, rtol=0,
    )


def test_weighted_moments_uses_the_frequency_weight_convention():
    """``unbiased`` must divide by ``sum(w) - 1``, not ``n - 1``.

    Getting this wrong is a ~2% error on a standardisation statistic: large
    enough to move a fitted model, small enough to read as numerical noise.
    """
    t = torch.randn(30, 2)
    w = torch.randint(1, 4, (30,)).float()
    rep = t[_replicate(w)]
    m_w, s_w = weighted_moments(t, w, unbiased=True)
    torch.testing.assert_close(m_w, rep.mean(0), atol=1e-5, rtol=0)
    torch.testing.assert_close(s_w, rep.std(0, unbiased=True), atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (torch.tensor([1.0, 2.0]), "align"),
        (torch.tensor([1.0, -1.0, 1.0]), "non-negative"),
        (torch.zeros(3), "sum"),
        (torch.tensor([1.0, float("nan"), 1.0]), "NaN"),
    ],
)
def test_validate_weights_rejects_unusable_input(bad, match):
    with pytest.raises(ValueError, match=match):
        validate_weights(bad, 3, where="t")


# ==========================================================================
# Replication equivalence, zero weights, unit weights -- per family
# ==========================================================================
# Closed-form families (and the KDE / FlexCode-root estimators) are compared on
# their fitted parameters; gradient-trained families on their full-batch
# gradients (see the module docstring).  A root FlexCode node's coefficients
# are the closed form beta_j = E[phi_j(Z)]: that branch once computed
# ``targets.mean(0)`` and dropped the validated weights, so
# ``supports_weights = True`` was false for exactly this node type -- the fit
# converged, to the unweighted estimator, silently.  It is pinned here with
# the other estimators for that reason.


def _lg_data(n=60, seed=0):
    torch.manual_seed(seed)
    pa = torch.randn(n, 2)
    x = pa @ torch.tensor([[1.5], [-0.7]]) + 0.3 + 0.2 * torch.randn(n, 1)
    return x, pa, torch.randint(0, 4, (n,)).float()


def _assert_attrs_close(a, b, names, atol):
    for name in names:
        got, want = getattr(a, name), getattr(b, name)
        got, want = (got() if callable(got) else got), (want() if callable(want) else want)
        torch.testing.assert_close(got, want, atol=atol, rtol=0, msg=name)


def _case_lg():
    x, pa, w = _lg_data()
    return x, pa, w, LinearGaussianMechanism, {}


def _case_lg_root():
    x, _, w = _lg_data()
    return x, None, w, LinearGaussianMechanism, {}


def _case_categorical():
    torch.manual_seed(0)
    pa = torch.randint(0, 3, (200, 1))
    x = torch.randint(0, 2, (200,))
    w = torch.randint(0, 4, (200,)).float()
    return x, pa, w, CategoricalTableMechanism, {"parent_cards": [3], "n_classes": 2}


def _case_kde():
    torch.manual_seed(0)
    pa = torch.randn(80, 2)
    y = 0.7 * pa[:, :1] + 0.1 * torch.randn(80, 1)
    w = torch.randint(1, 4, (80,)).float()
    return y, pa, w, lambda: ConditionalKDEMechanism(bw_factor=0.5), {}


def _case_flexcode_root():
    torch.manual_seed(0)
    x = torch.randn(60, 1)
    w = torch.randint(1, 4, (60,)).float()
    return x, None, w, lambda: FlexCodeMechanism(n_basis=7), {}


def _cmp_lg(a, b, atol=1e-5):
    _assert_attrs_close(a, b, ("_weight", "_bias", "_scale"), atol)


def _cmp_lg_root(a, b, atol=1e-5):
    _assert_attrs_close(a, b, ("_bias", "_scale"), atol)


def _cmp_categorical(a, b, atol=1e-6):
    torch.testing.assert_close(a.cpt, b.cpt, atol=atol, rtol=0)


def _cmp_kde(a, b, atol=1e-5):
    """The bandwidths (child and parent), the standardisation and the density."""
    _assert_attrs_close(a, b, ("_b", "_h", "_pa_mean", "_pa_std"), atol)
    qpa, qy = torch.randn(32, 2), torch.randn(32, 1)
    torch.testing.assert_close(a.log_prob(qy, qpa), b.log_prob(qy, qpa), atol=1e-4, rtol=0)


def _cmp_flexcode_root(a, b, atol=1e-5):
    # The basis lives in the z-space fixed by (_y_min, _y_max); replicated
    # rows have the same range, so the coefficients are directly comparable.
    torch.testing.assert_close(a._y_min, b._y_min)
    torch.testing.assert_close(a._y_max, b._y_max)
    torch.testing.assert_close(a._root_coef, b._root_coef, atol=atol, rtol=0)


_CLOSED_FORM = {
    "linear_gaussian": (_case_lg, _cmp_lg),
    "linear_gaussian_root": (_case_lg_root, _cmp_lg_root),
    "categorical": (_case_categorical, _cmp_categorical),
    "kde": (_case_kde, _cmp_kde),
    "flexcode_root": (_case_flexcode_root, _cmp_flexcode_root),
}


@pytest.mark.parametrize("family", sorted(_CLOSED_FORM))
def test_replication_equivalence_closed_form(family):
    """Integer weights == fitting on the rows repeated that many times."""
    case, compare = _CLOSED_FORM[family]
    x, pa, w, factory, kw = case()
    idx = _replicate(w)
    weighted = factory()
    weighted.fit_local(x, pa, weights=w, **kw)
    replicated = factory()
    replicated.fit_local(x[idx], None if pa is None else pa[idx], **kw)
    compare(weighted, replicated)


#: family -> (tolerance, exact?) for weights = ones vs weights = None.
#: LinearGaussian is deliberately not bitwise: the weighted path hands LAPACK
#: a freshly built sqrt(w)-scaled design matrix, and whether ``lstsq`` returns
#: bit-identical coefficients depends on the BLAS thread count (identical
#: single-threaded, ~1e-7 apart at 14 threads) -- a claim about scheduling,
#: not weighting.  The categorical integer-count path scatter-adds the same
#: float64 ones into the same bins and *is* exact.
_UNIT_WEIGHT_TOL = {
    "linear_gaussian": 1e-6,
    "categorical": 0.0,
    "kde": 1e-6,
    "flexcode_root": 1e-6,
}


@pytest.mark.parametrize("family", sorted(_UNIT_WEIGHT_TOL))
def test_unit_weights_reproduce_the_unweighted_fit(family):
    """The unweighted default must stay what it was."""
    case, compare = _CLOSED_FORM[family]
    x, pa, _, factory, kw = case()
    ones = factory()
    ones.fit_local(x, pa, weights=torch.ones(x.shape[0]), **kw)
    plain = factory()
    plain.fit_local(x, pa, **kw)
    tol = _UNIT_WEIGHT_TOL[family]
    if tol == 0.0:
        assert torch.equal(ones.cpt, plain.cpt)
    elif family == "kde":
        _assert_attrs_close(plain, ones, ("_b", "_h", "_pa_mean", "_pa_std"), tol)
    else:
        compare(plain, ones, atol=tol)


def _full_batch_grads(factory, x, parents, weights, seed=7):
    """One full-batch step at lr=0: params unmoved, .grad = the quantity tested."""
    torch.manual_seed(seed)
    mech = factory()
    mech.fit_local(
        x, parents,
        epochs=1, lr=0.0, batch_size=10**6,
        weights=weights, consolidate=False,
    )
    return {k: v.grad.clone() for k, v in mech.named_parameters() if v.grad is not None}


def _max_abs_diff(a, b):
    assert set(a) == set(b) and a, "no gradients captured"
    return max(float((a[k] - b[k]).abs().max()) for k in a)


def _grad_case_mdn():
    torch.manual_seed(0)
    pa = torch.randn(40, 2)
    x = pa @ torch.tensor([[1.0], [-0.5]]) + 0.2 * torch.randn(40, 1)
    return x, pa, torch.randint(1, 4, (40,)).float(), \
        lambda: MDNMechanism(num_components=3, hidden=(8,)), 1e-6


def _grad_case_neural_categorical():
    torch.manual_seed(0)
    pa = torch.randn(40, 2)
    x = torch.randint(0, 3, (40,))
    return x, pa, torch.randint(1, 4, (40,)).float(), \
        lambda: NeuralCategoricalMechanism(n_classes=3, hidden=(8,)), 1e-6


def _grad_case_flow():
    pytest.importorskip("zuko")
    from nbn.mechanisms.parametric.normalizing_flow import NormalizingFlowMechanism

    torch.manual_seed(0)
    x = torch.randn(24, 1)
    return x, None, torch.randint(1, 3, (24,)).float(), \
        lambda: NormalizingFlowMechanism(num_transforms=1, hidden=(8,)), 1e-5


_GRADIENT = {
    "mdn": _grad_case_mdn,
    "neural_categorical": _grad_case_neural_categorical,
    "flow": _grad_case_flow,
}


@pytest.mark.parametrize("family", sorted(_GRADIENT))
def test_replication_equivalence_of_the_full_batch_gradient(family):
    x, pa, w, factory, tol = _GRADIENT[family]()
    idx = _replicate(w)
    assert _max_abs_diff(
        _full_batch_grads(factory, x, pa, w),
        _full_batch_grads(factory, x[idx], None if pa is None else pa[idx], None),
    ) < tol


# Zero weights are absence, not smallness: "down-weighted to almost nothing"
# and "absent" are different claims, and only the second is the contract.


def _zero_case_lg():
    x, pa, w = _lg_data()
    w = w.clone()
    w[:30] = 0.0
    keep = slice(30, None)
    return x, pa, w, keep, LinearGaussianMechanism, {}, \
        lambda a, b: _assert_attrs_close(a, b, ("_weight", "_bias"), 1e-5)


def _zero_case_categorical():
    torch.manual_seed(0)
    pa = torch.randint(0, 3, (120, 1))
    x = torch.randint(0, 2, (120,))
    w = torch.ones(120)
    w[:60] = 0.0
    return x, pa, w, slice(60, None), CategoricalTableMechanism, \
        {"parent_cards": [3], "n_classes": 2}, _cmp_categorical


@pytest.mark.parametrize("family", ["linear_gaussian", "categorical"])
def test_zero_weighted_rows_are_dropped_closed_form(family):
    x, pa, w, keep, factory, kw, compare = {
        "linear_gaussian": _zero_case_lg, "categorical": _zero_case_categorical,
    }[family]()
    withzeros = factory()
    withzeros.fit_local(x, pa, weights=w, **kw)
    dropped = factory()
    dropped.fit_local(x[keep], pa[keep], weights=w[keep], **kw)
    compare(withzeros, dropped)


def test_zero_weighted_rows_do_not_influence_mdn_gradients():
    torch.manual_seed(0)
    n = 30
    pa = torch.randn(n, 2)
    x = torch.randn(n, 1)
    w = torch.ones(n)
    w[:15] = 0.0

    def factory():
        return MDNMechanism(num_components=2, hidden=(8,))

    assert _max_abs_diff(
        _full_batch_grads(factory, x, pa, w),
        _full_batch_grads(factory, x[15:], pa[15:], None),
    ) < 1e-6


# ==========================================================================
# Desynchronisation guard
# ==========================================================================


def test_weighted_fit_is_invariant_to_row_order():
    """The guard against a weights vector drifting out of step with its rows.

    Every gradient-trained mechanism shuffles internally, so a future edit
    could index the batch with one permutation and the weights with another.
    That fit still converges — to the wrong thing — so ordinary tests would
    not catch it.  Permuting the rows and their weights together must not
    change the answer.
    """
    x, pa, w = _lg_data()
    perm = torch.randperm(x.shape[0])
    a = LinearGaussianMechanism()
    a.fit_local(x, pa, weights=w)
    b = LinearGaussianMechanism()
    b.fit_local(x[perm], pa[perm], weights=w[perm])
    torch.testing.assert_close(a._weight, b._weight, atol=1e-5, rtol=0)
    torch.testing.assert_close(a._bias, b._bias, atol=1e-5, rtol=0)


def test_shuffling_weights_alone_changes_the_fit():
    """Sanity: the permutation test above would actually detect a desync."""
    x, pa, w = _lg_data()
    perm = torch.randperm(x.shape[0])
    a = LinearGaussianMechanism()
    a.fit_local(x, pa, weights=w)
    b = LinearGaussianMechanism()
    b.fit_local(x, pa, weights=w[perm])
    assert not torch.allclose(a._weight, b._weight, atol=1e-4)


# ==========================================================================
# Refusal, and the fail-fast path
# ==========================================================================


def test_knn_refuses_weights_explicitly():
    mech = KNNConditionalMechanism()
    with pytest.raises(NotImplementedError, match="does not support per-sample"):
        mech.fit_local(torch.randn(20, 1), torch.randn(20, 1), weights=torch.ones(20))


def test_fit_rejects_unsupported_mechanisms_before_fitting_anything():
    """Fail fast, and name both the class and the node."""
    torch.manual_seed(0)
    model = NBN(
        [("A", "R")],
        variables={"A": ("discrete", 2), "R": ("continuous", 1)},
        device="cpu",
    )
    model.set_mechanism("A", CategoricalTableMechanism())
    model.set_mechanism("R", KNNConditionalMechanism())
    a = torch.bernoulli(torch.full((80,), 0.4))
    data = {"A": a, "R": torch.randn(80, 1)}

    with pytest.raises(NotImplementedError) as exc:
        model.fit(data, weights=torch.ones(80))
    msg = str(exc.value)
    assert "KNNConditionalMechanism" in msg and "'R'" in msg
    # Nothing was fitted: the supported mechanism is untouched.
    assert not model.mechanisms["A"].is_fitted


# ==========================================================================
# Model-level, and coherence with incremental update
# ==========================================================================


def test_model_fit_with_weights_matches_replicated_data():
    torch.manual_seed(0)
    n = 300
    a = torch.bernoulli(torch.full((n,), 0.3))
    b = torch.bernoulli(torch.where(a > 0.5, 0.9, 0.1))
    w = torch.randint(0, 4, (n,)).float()
    idx = _replicate(w)

    def _model():
        m = NBN(
            [("A", "B")],
            variables=dict.fromkeys("AB", ("discrete", 2)),
            device="cpu",
        )
        m.auto_mechanisms()
        return m

    weighted = _model()
    weighted.fit({"A": a, "B": b}, weights=w)
    replicated = _model()
    replicated.fit({"A": a[idx], "B": b[idx]})
    torch.testing.assert_close(
        weighted.mechanisms["B"].cpt, replicated.mechanisms["B"].cpt,
        atol=1e-6, rtol=0,
    )


def test_weighted_fit_persists_weighted_sufficient_statistics():
    """The prior ``update`` folds onto must reflect the weights fit used.

    Snapshotting the normal equations unweighted would make a later update
    start from a posterior that never existed — an inconsistency that shows up
    only in the numbers, long after the fit.  ``N`` is the give-away: it counts
    the weighted mass, not the rows.
    """
    x, pa, w = _lg_data(n=80)
    mech = LinearGaussianMechanism()
    mech.fit_local(x, pa, weights=w)
    torch.testing.assert_close(
        mech._neq_N.reshape(()), w.sum(), atol=1e-3, rtol=0,
    )

    unweighted = LinearGaussianMechanism()
    unweighted.fit_local(x, pa)
    torch.testing.assert_close(
        unweighted._neq_N.reshape(()), torch.tensor(80.0), atol=1e-6, rtol=0,
    )


def test_update_local_refuses_weights_rather_than_swallowing_them():
    """``update_local`` also ends in **kwargs, so silence was the default.

    Weighting is scoped to the fitting path.  An unimplemented `weights=`
    quietly absorbed there would produce an unweighted update that looks like
    a weighted one — the same failure the capability flag exists to prevent.
    """
    x, pa, w = _lg_data(n=40)
    mech = LinearGaussianMechanism()
    mech.fit_local(x, pa, weights=w)
    with pytest.raises(NotImplementedError, match="fitting path"):
        mech.update_local(x, pa, weights=w)
    # ...and the unweighted update still works.
    assert mech.update_local(x, pa)


# ==========================================================================
# KDE — weighted Nadaraya-Watson
# ==========================================================================


def test_weighted_quantile_equals_torch_quantile_on_replicated_rows():
    """Integer weights, several q, two columns, zero weights: exact match."""
    torch.manual_seed(0)
    x = torch.randn(37, 2)
    w = torch.randint(0, 4, (37,)).float()          # zeros drop rows exactly
    w[0] = 2.0                                       # keep at least one row
    q = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    got = weighted_quantile(x, q, w.double())
    want = torch.quantile(x[_replicate(w)], q, dim=0)
    torch.testing.assert_close(got, want, atol=1e-6, rtol=0)


def test_weighted_quantile_unit_weights_reproduce_torch_quantile():
    torch.manual_seed(1)
    x = torch.randn(50, 3)
    q = torch.tensor([0.25, 0.75])
    torch.testing.assert_close(weighted_quantile(x, q, torch.ones(50).double()),
                               torch.quantile(x, q, dim=0), atol=1e-6, rtol=0)
    torch.testing.assert_close(weighted_quantile(x, q, None), torch.quantile(x, q, dim=0))


def test_kde_zero_weighted_points_leave_the_mixture():
    """log w = -inf removes the point from both logsumexps exactly."""
    torch.manual_seed(0)
    x = torch.cat([torch.zeros(40, 1), torch.full((40, 1), 10.0)])
    w = torch.cat([torch.ones(40), torch.zeros(40)])

    weighted = ConditionalKDEMechanism(bw_factor=1.0)
    weighted.fit_local(x, None, weights=w)
    query = torch.tensor([[10.0]])
    # With the far cluster zero-weighted, density there must be negligible.
    assert float(weighted.log_prob(query, None)) < -10.0


def test_kde_weighting_shifts_density_toward_weighted_points():
    torch.manual_seed(0)
    x = torch.cat([torch.zeros(50, 1), torch.full((50, 1), 5.0)])
    query = torch.tensor([[5.0]])

    even = ConditionalKDEMechanism(bw_factor=1.0)
    even.fit_local(x, None)
    tilted = ConditionalKDEMechanism(bw_factor=1.0)
    tilted.fit_local(x, None, weights=torch.cat([torch.ones(50), 9 * torch.ones(50)]))
    assert float(tilted.log_prob(query, None)) > float(even.log_prob(query, None))


# ==========================================================================
# FlexCode -- root branch: zero weights
# ==========================================================================


def test_flexcode_root_zero_weighted_rows_leave_the_estimate():
    """A zero weight must be indistinguishable from dropping the row.

    The z-space is fixed by ``(_y_min, _y_max)``, which are the data's range
    regardless of weights, so the dropped block is placed strictly *inside*
    the kept block's range: removing it leaves the support -- and therefore
    the basis -- untouched, and the coefficients are directly comparable.
    """
    torch.manual_seed(2)
    kept = torch.cat([
        torch.tensor([[-3.0], [3.0]]), 6.0 * torch.rand(38, 1) - 3.0,
    ])
    dropped = 0.2 * torch.randn(40, 1)          # well inside [-3, 3]
    x = torch.cat([kept, dropped])
    w = torch.cat([torch.ones(40), torch.zeros(40)])

    masked = FlexCodeMechanism(n_basis=7)
    masked.fit_local(x, None, weights=w)
    without = FlexCodeMechanism(n_basis=7)
    without.fit_local(kept, None)

    torch.testing.assert_close(masked._y_min, without._y_min)
    torch.testing.assert_close(masked._y_max, without._y_max)
    torch.testing.assert_close(
        masked._root_coef, without._root_coef, atol=1e-6, rtol=0,
    )

    # And the weights genuinely bite: an unweighted fit on the same rows,
    # which the dropped cluster dominates, lands somewhere else.
    pooled = FlexCodeMechanism(n_basis=7)
    pooled.fit_local(x, None)
    assert not torch.allclose(
        masked._root_coef, pooled._root_coef, atol=1e-3,
    )
