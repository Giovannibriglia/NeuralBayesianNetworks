"""Regression tests for the mechanism-level review findings (fix/mechanisms).

Each section names the finding it pins (A4, A9, ...).  Every test failed on
the pre-fix code and passes after; the comments say what the failure was so a
future regression reads as the original bug, not as a mystery threshold.
"""
from __future__ import annotations

import math

import pytest
import torch

from nbn.mechanisms import LinearGaussianMechanism
from tests.conftest import available_devices


def _replicate(w: torch.Tensor) -> torch.Tensor:
    """Row indices repeating row i exactly w_i times."""
    return torch.cat([torch.full((int(k),), i) for i, k in enumerate(w)])


# ==========================================================================
# A4 -- LinearGaussian: lstsq drops the slope when a parent has a big offset
# ==========================================================================


def _offset_lg_data(n=2000, device="cpu"):
    g = torch.Generator().manual_seed(1)
    pa = 100.0 + torch.randn(n, 1, generator=g)
    x = 0.5 * pa + 100.0 + 0.1 * torch.randn(n, 1, generator=g)
    return x.to(device), pa.to(device)


@pytest.mark.parametrize("device", available_devices())
def test_a4_linear_gaussian_recovers_slope_under_large_parent_offset(device):
    # Pre-fix: the default gelsy driver treated the near-collinear
    # [parent, 1] design as rank deficient and returned w=1.50, b=0.015.
    x, pa = _offset_lg_data(device=device)
    mech = LinearGaussianMechanism()
    mech.fit_local(x, pa)
    assert abs(float(mech._weight.detach().reshape(-1)[0]) - 0.5) < 0.02
    assert abs(float(mech._bias.detach().reshape(-1)[0]) - 100.0) < 2.0
    assert abs(float(mech._scale().detach().reshape(-1)[0]) - 0.1) < 0.01


def test_a4_linear_gaussian_matches_float64_ridge_normal_equations():
    # The standardised solve must keep the ridge semantics: penalty
    # ridge * ||W||^2 on the raw-unit weights, intercept unpenalised.
    g = torch.Generator().manual_seed(2)
    n = 300
    pa = torch.cat([50 + 3 * torch.randn(n, 1, generator=g),
                    -20 + 0.2 * torch.randn(n, 1, generator=g)], dim=1)
    x = pa @ torch.tensor([[1.5], [-0.7]]) + 7.0 + torch.randn(n, 1, generator=g)
    w = torch.randint(0, 4, (n,), generator=g).float()
    ridge = 5.0
    mech = LinearGaussianMechanism(ridge=ridge)
    mech.fit_local(x, pa, weights=w)

    z = torch.cat([pa, torch.ones(n, 1)], 1).double()
    pen = torch.diag(torch.tensor([ridge, ridge, 0.0], dtype=torch.float64))
    a = z.T @ (w.double()[:, None] * z) + pen
    b = z.T @ (w.double()[:, None] * x.double())
    theta = torch.linalg.solve(a, b)
    torch.testing.assert_close(mech._weight.double(), theta[:-1], atol=2e-3, rtol=1e-4)
    torch.testing.assert_close(mech._bias.double(), theta[-1], atol=5e-2, rtol=1e-4)


def test_a4_linear_gaussian_offset_fit_is_replication_and_zero_weight_exact():
    x, pa = _offset_lg_data(n=200)
    w = torch.randint(0, 4, (200,), generator=torch.Generator().manual_seed(3)).float()
    weighted = LinearGaussianMechanism()
    weighted.fit_local(x, pa, weights=w)
    replicated = LinearGaussianMechanism()
    replicated.fit_local(x[_replicate(w)], pa[_replicate(w)])
    for a, b in [(weighted._weight, replicated._weight),
                 (weighted._bias, replicated._bias),
                 (weighted._scale(), replicated._scale())]:
        torch.testing.assert_close(a, b, atol=2e-3, rtol=1e-4)


# ==========================================================================
# B10 -- 3-D parents through LinearGaussian.forward; batched Dirac values
# ==========================================================================


def test_b10_linear_gaussian_forward_accepts_particle_parents():
    # Pre-fix: scale was expanded to [B, D_x] against a [B, S, D_x] loc and
    # Normal() raised a size mismatch.
    g = torch.Generator().manual_seed(0)
    pa = torch.randn(50, 2, generator=g)
    x = pa.sum(1, keepdim=True) + torch.randn(50, 1, generator=g)
    mech = LinearGaussianMechanism()
    mech.fit_local(x, pa)
    p3 = torch.randn(4, 3, 2, generator=g)
    dist = mech.forward(p3)
    assert tuple(dist.batch_shape) == (4, 3)
    xs = torch.randn(4, 3, 1, generator=g)
    torch.testing.assert_close(dist.log_prob(xs), mech.log_prob(xs, p3))


def test_b10_linear_gaussian_root_forward_with_particle_parents():
    mech = LinearGaussianMechanism()
    mech.fit_local(torch.randn(30, 1), None)
    assert tuple(mech.forward(torch.zeros(4, 3, 0)).batch_shape) == (4, 3)


def test_b10_dirac_gaussian_batched_value():
    # Pre-fix: a documented [B, D] value was flattened to [B*D] and every
    # method crashed on the expand.
    from nbn.mechanisms.parametric.dirac_gaussian import DiracGaussianMechanism

    value = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    mech = DiracGaussianMechanism(value, sigma=1e-3)
    pa = torch.zeros(3, 1)
    assert mech.output_dim == 2
    dist = mech.forward(pa)
    assert tuple(dist.batch_shape) == (3,) and tuple(dist.event_shape) == (2,)
    torch.testing.assert_close(dist.mean, value)
    s = mech.sample(pa, 4)
    assert s.shape == (3, 4, 2)
    torch.testing.assert_close(s.mean(1), value, atol=5e-3, rtol=0)
    lp_at = mech.log_prob(value, pa)
    lp_off = mech.log_prob(value + 0.01, pa)
    assert lp_at.shape == (3,) and (lp_at > lp_off).all()
    lp3 = mech.log_prob(value.unsqueeze(1).expand(3, 5, 2), pa)
    assert lp3.shape == (3, 5)
    torch.testing.assert_close(lp3[:, 0], lp_at)
    # A vector value keeps broadcasting over any batch.
    vec = DiracGaussianMechanism(torch.tensor([1.0, 2.0]))
    assert vec.sample(torch.zeros(5, 1), 2).shape == (5, 2, 2)
    assert vec.log_prob(torch.zeros(5, 3, 2), None).shape == (5, 3)


# ==========================================================================
# A13 -- Deterministic delta log_prob was 0 for every x
# ==========================================================================


def test_a13_deterministic_log_prob_is_zero_only_at_the_value():
    from nbn.mechanisms.parametric.deterministic import _LOG_ZERO, DeterministicMechanism

    cont = DeterministicMechanism(torch.tensor([1.5]))
    lp = cont.log_prob(torch.tensor([[1.5], [2.0]]), None)
    assert lp.tolist() == [0.0, _LOG_ZERO]
    dist_lp = cont.forward(None).log_prob(torch.tensor([[1.5], [2.0]]))
    assert dist_lp.tolist() == [0.0, _LOG_ZERO]
    # Particle block [B, S, D].
    lp3 = cont.log_prob(torch.tensor([[[1.5], [0.0]], [[2.0], [1.5]]]), None)
    assert lp3.tolist() == [[0.0, _LOG_ZERO], [_LOG_ZERO, 0.0]]

    disc = DeterministicMechanism(torch.tensor([2.0]), cardinality=3)
    lp_d = disc.log_prob(torch.tensor([0, 1, 2]), None)
    assert lp_d.dtype.is_floating_point
    assert lp_d.tolist() == [_LOG_ZERO, _LOG_ZERO, 0.0]
    # Consistent with the tabulated CPD exact inference uses.
    torch.testing.assert_close(
        lp_d, disc.tabulate()[torch.tensor([0, 1, 2])],
    )


def test_a13_deterministic_batched_value_aligns_with_particles():
    from nbn.mechanisms.parametric.deterministic import _LOG_ZERO, DeterministicMechanism

    mech = DeterministicMechanism(torch.tensor([[0.0], [1.0]]))
    x = torch.tensor([[[0.0], [1.0], [0.0]], [[1.0], [1.0], [0.0]]])  # [2, 3, 1]
    assert mech.log_prob(x, None).tolist() == [
        [0.0, _LOG_ZERO, 0.0], [0.0, 0.0, _LOG_ZERO],
    ]


# ==========================================================================
# A15 -- NeuralCategorical embeddings indexed the sample axis of 3-D parents
# ==========================================================================


def test_a15_neural_categorical_embeddings_accept_particle_parents():
    from nbn.mechanisms import NeuralCategoricalMechanism

    g = torch.Generator().manual_seed(0)
    pa = torch.randint(0, 3, (300, 2), generator=g)
    x = (pa.sum(1) % 3)
    mech = NeuralCategoricalMechanism(n_classes=3, embedding_dim=4)
    mech.fit_local(x, pa, parent_cards=[3, 3], epochs=3, consolidate=False)
    p3 = torch.randint(0, 3, (4, 5, 2), generator=g)
    probs3 = mech.forward(p3).probs
    probs2 = mech.forward(p3.reshape(20, 2)).probs.reshape(4, 5, 3)
    torch.testing.assert_close(probs3, probs2)


# ==========================================================================
# A20 -- zero-weight rows must equal dropping them; #127 pseudo-count mass
# ==========================================================================


def _cat_data(n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    pa = torch.randint(0, 2, (n, 1), generator=g)
    x = (pa[:, 0] + torch.randint(0, 2, (n,), generator=g)) % 2
    return x, pa


def test_a20_categorical_zero_weight_rows_do_not_widen_the_table():
    from nbn.mechanisms import CategoricalTableMechanism

    x, pa = _cat_data()
    # A zero-weight row carrying an unseen class (4) and an unseen parent
    # value (5).  Pre-fix it widened K to 5 and the parent axis to 6, and
    # the widened class drew the #127 pseudo-count.
    x_z = torch.cat([x, torch.tensor([4])])
    pa_z = torch.cat([pa, torch.tensor([[5]])])
    w = torch.cat([torch.ones(len(x)), torch.zeros(1)])
    with_zero = CategoricalTableMechanism()
    with_zero.fit_local(x_z, pa_z, weights=w)
    dropped = CategoricalTableMechanism()
    dropped.fit_local(x, pa)
    assert with_zero.n_classes == dropped.n_classes == 2
    assert with_zero._parent_cards == dropped._parent_cards == [2]
    torch.testing.assert_close(with_zero._logits, dropped._logits)


def test_a20_unobserved_class_pseudo_count_is_one_observation_in_total():
    from nbn.mechanisms import CategoricalTableMechanism

    # Parent row 0: 99 observations; row 1: a single one; row 2: none.
    x = torch.cat([torch.randint(0, 2, (99,)), torch.tensor([0])])
    pa = torch.cat([torch.zeros(99, 1), torch.ones(1, 1)]).long()
    mech = CategoricalTableMechanism(alpha=0.0)
    mech.fit_local(x, pa, n_classes=3, parent_cards=[3])
    cpt = mech.cpt
    # Pre-fix: the sparse row gave the never-seen class 2 half its mass and
    # the empty row gave it all of it.
    assert 0 < float(cpt[1, 2]) < 0.02
    assert 0 < float(cpt[0, 2]) < 0.02
    torch.testing.assert_close(cpt[0, 2], cpt[1, 2], atol=1e-3, rtol=0)
    torch.testing.assert_close(cpt[2], torch.full((3,), 1 / 3))


def test_a20_unobserved_class_pseudo_count_keeps_update_exact():
    from nbn.mechanisms import CategoricalTableMechanism

    x, pa = _cat_data(n=80, seed=4)
    pooled = CategoricalTableMechanism()
    pooled.fit_local(x, pa, n_classes=3, parent_cards=[2])
    chunked = CategoricalTableMechanism()
    chunked.fit_local(x[:30], pa[:30], n_classes=3, parent_cards=[2])
    chunked.update_local(x[30:], pa[30:])
    torch.testing.assert_close(chunked.cpt, pooled.cpt)


def test_a20_flexcode_zero_weight_rows_do_not_move_the_support():
    from nbn.mechanisms.non_parametric.flexcode import FlexCodeMechanism

    g = torch.Generator().manual_seed(0)
    y = torch.randn(200, 1, generator=g)
    # Pre-fix: y_min/y_max spanned the zero-weight outlier at 1e3, so the
    # whole basis was squeezed into a sliver of [0, 1].
    y_z = torch.cat([y, torch.tensor([[1e3]])])
    w = torch.cat([torch.ones(200), torch.zeros(1)])
    with_zero = FlexCodeMechanism()
    with_zero.fit_local(y_z, None, weights=w)
    dropped = FlexCodeMechanism()
    dropped.fit_local(y, None)
    torch.testing.assert_close(with_zero._y_min, dropped._y_min)
    torch.testing.assert_close(with_zero._y_max, dropped._y_max)
    q = torch.linspace(-3, 3, 13).unsqueeze(-1)
    torch.testing.assert_close(with_zero.log_prob(q, None), dropped.log_prob(q, None))


# ==========================================================================
# A21 -- BinningCategoricalTable: parent cards, n_categories, weights, n_bins
# ==========================================================================


def _binning_mech(**kw):
    from nbn.mechanisms.parametric.binning_categorical import BinningCategoricalTable

    return BinningCategoricalTable(**kw)


def test_a21_binning_discrete_parent_keeps_its_cardinality():
    g = torch.Generator().manual_seed(0)
    n = 400
    # The discrete parent declares 5 states; only 0..2 occur in training.
    pa = torch.stack([torch.randint(0, 3, (n,), generator=g).float(),
                      torch.randn(n, generator=g)], dim=-1)
    x = (pa[:, 0] >= 1).long()
    mech = _binning_mech(parent_kinds=[("discrete", 5), ("continuous", 1)],
                         n_bins=3, n_categories=2)
    mech.fit_local(x, pa)
    # Pre-fix: [3, 3] -- the discrete parent was forced to n_bins states, so
    # the declared values 3 and 4 had no rows and their strides aliased other
    # parent configurations.
    assert mech._parent_cards == [5, 3]
    q = torch.tensor([[4.0, 0.0], [4.0, 5.0]])
    probs = mech.forward(q).probs
    # An unseen-but-declared configuration is an empty row: uniform.
    torch.testing.assert_close(probs, torch.full((2, 2), 0.5))


def test_a21_binning_honours_declared_target_cardinality():
    pa = torch.randn(100, 1)
    x = (pa[:, 0] > 0).long()                 # only classes {0, 1} seen
    mech = _binning_mech(parent_kinds=[("continuous", 1)], n_bins=2, n_categories=4)
    mech.fit_local(x, pa)
    assert mech.n_classes == 4
    # Pre-fix: log_prob of the declared, unseen class 3 indexed past K=2.
    lp = mech.log_prob(torch.tensor([3, 3]), pa[:2])
    assert torch.isfinite(lp).all()


def test_a21_binning_weighted_thresholds_are_replication_exact():
    g = torch.Generator().manual_seed(1)
    pa = torch.randn(60, 1, generator=g)
    x = (pa[:, 0] > 0).long()
    w = torch.randint(0, 4, (60,), generator=g).float()
    idx = _replicate(w)
    weighted = _binning_mech(parent_kinds=[("continuous", 1)], n_bins=4, n_categories=2)
    weighted.fit_local(x, pa, weights=w)
    replicated = _binning_mech(parent_kinds=[("continuous", 1)], n_bins=4, n_categories=2)
    replicated.fit_local(x[idx], pa[idx])
    torch.testing.assert_close(weighted._thresholds, replicated._thresholds)
    torch.testing.assert_close(weighted.cpt, replicated.cpt)


def test_a21_binning_rejects_a_single_bin():
    with pytest.raises(ValueError, match="n_bins"):
        _binning_mech(parent_kinds=[("continuous", 1)], n_bins=1, n_categories=2)


def test_a21_binning_accepts_parent_cards_from_fit():
    # nbn.learning.fit threads parent_cards when every parent declares a
    # cardinality; that used to collide with the binner's own keyword.
    pa = torch.randint(0, 3, (50, 1)).float()
    x = pa[:, 0].long() % 2
    mech = _binning_mech(parent_kinds=[("discrete", 3)], n_bins=2, n_categories=2)
    mech.fit_local(x, pa, parent_cards=[4], n_classes=2)
    assert mech._parent_cards == [4]


# ==========================================================================
# A10 / B9 / B11 -- MDN: child standardisation, float64 inputs, one row
# ==========================================================================


def _wide_data(n=2000, seed=0, scale=3000.0, offset=0.0):
    g = torch.Generator().manual_seed(seed)
    pa = torch.randn(n, 1, generator=g)
    x = offset + scale * pa + scale * torch.randn(n, 1, generator=g)
    return x, pa


def test_a10_mdn_fits_a_wide_target_as_well_as_linear_gaussian():
    from nbn.mechanisms import MDNMechanism

    # Pre-fix: log_scale <= 7 capped each component at ~1097 raw units, so
    # a target with spread ~4000 scored NLL 15.0 against LG's 9.4.
    x, pa = _wide_data()
    lg = LinearGaussianMechanism()
    lg.fit_local(x, pa)
    mdn = MDNMechanism(num_components=3, hidden=(32,))
    mdn.fit_local(x, pa, epochs=40, lr=3e-3, batch_size=256, consolidate=False)
    with torch.no_grad():
        nll_lg = float(-lg.log_prob(x, pa).mean())
        nll_mdn = float(-mdn.log_prob(x, pa).mean())
    assert nll_mdn < nll_lg + 0.2, (nll_mdn, nll_lg)
    s = mdn.sample(pa[:500], 4)
    # Samples come back in raw units.
    assert 2000 < float(s.std()) < 7000


def test_a10_mdn_density_integrates_to_one_in_raw_units():
    from nbn.mechanisms import MDNMechanism

    x, pa = _wide_data(n=500, scale=50.0, offset=1000.0)
    mdn = MDNMechanism(num_components=2, hidden=(16,))
    mdn.fit_local(x, pa, epochs=5, consolidate=False)
    grid = torch.linspace(0.0, 2000.0, 20001).unsqueeze(-1)
    with torch.no_grad():
        dens = mdn.log_prob(grid, torch.zeros(len(grid), 1)).exp()
    assert abs(float(torch.trapz(dens, grid[:, 0])) - 1.0) < 1e-2


def test_a10_mdn_child_standardisation_frozen_under_warm_start():
    from nbn.mechanisms import MDNMechanism

    x, pa = _wide_data(n=300, scale=5.0)
    mdn = MDNMechanism(num_components=2, hidden=(8,))
    mdn.fit_local(x, pa, epochs=2, consolidate=False)
    m0, s0 = mdn._x_mean.clone(), mdn._x_std.clone()
    mdn.fit_local(x + 100.0, pa, epochs=2, consolidate=False, warm_start=True)
    torch.testing.assert_close(mdn._x_mean, m0)
    torch.testing.assert_close(mdn._x_std, s0)
    mdn.fit_local(x + 100.0, pa, epochs=2, consolidate=False)
    assert float((mdn._x_mean - m0).abs().max()) > 50.0
    # The buffers are part of the snapshot round trip.
    assert {"_x_mean", "_x_std"} <= set(mdn.state_dict())


def test_a10_mdn_weighted_child_standardisation_is_replication_exact():
    from nbn.mechanisms import MDNMechanism

    x, pa = _wide_data(n=60, scale=5.0, offset=3.0)
    w = torch.randint(0, 4, (60,), generator=torch.Generator().manual_seed(5)).float()
    idx = _replicate(w)
    a = MDNMechanism(num_components=2, hidden=(8,))
    a.fit_local(x, pa, weights=w, epochs=0, consolidate=False)
    b = MDNMechanism(num_components=2, hidden=(8,))
    b.fit_local(x[idx], pa[idx], epochs=0, consolidate=False)
    torch.testing.assert_close(a._x_mean, b._x_mean)
    torch.testing.assert_close(a._x_std, b._x_std)


@pytest.mark.parametrize("root", [True, False])
def test_b9_mdn_accepts_float64_inputs(root):
    from nbn.mechanisms import MDNMechanism

    x, pa = _wide_data(n=100, scale=1.0)
    x, pa = x.double(), (None if root else pa.double())
    mdn = MDNMechanism(num_components=2, hidden=(8,))
    # Pre-fix: "mat1 and mat2 must have the same dtype" (float64 data,
    # float32 MLP).
    mdn.fit_local(x, pa, epochs=2)
    lp = mdn.log_prob(x, pa)
    assert lp.shape == (100,) and torch.isfinite(lp).all()
    assert mdn.sample(pa, 3).shape == (1 if root else 100, 3, 1)


@pytest.mark.parametrize("root", [True, False])
def test_b11_mdn_fits_a_single_row(root):
    from nbn.mechanisms import MDNMechanism

    x, pa = _wide_data(n=2, scale=1.0)
    mdn = MDNMechanism(num_components=2, hidden=(8,))
    # Pre-fix: the unbiased std of one row is NaN and clamp_min passes it
    # through -> Normal(scale=NaN) raised.
    mdn.fit_local(x[:1], None if root else pa[:1], epochs=2, consolidate=False)
    lp = mdn.log_prob(x, None if root else pa)
    assert torch.isfinite(lp).all()


# ==========================================================================
# A9 / B9 -- NormalizingFlow: standardisation, Jacobian, float64 inputs
# ==========================================================================



def _flow(**kw):
    pytest.importorskip("zuko")
    from nbn.mechanisms.parametric.normalizing_flow import NormalizingFlowMechanism

    return NormalizingFlowMechanism(num_transforms=2, hidden=(16, 16), **kw)


def test_a9_flow_root_far_from_the_spline_window():
    # Pre-fix: the splines act on [-5, 5]; data at ~100 was fitted with NLL
    # ~5003 (true ~1.42) and sampled near 0.
    g = torch.Generator().manual_seed(0)
    x = 100.0 + torch.randn(1000, 1, generator=g)
    mech = _flow()
    mech.fit_local(x, None, epochs=100, batch_size=128)
    with torch.no_grad():
        nll = float(-mech.log_prob(x, None).mean())
    assert nll < 1.7, nll
    s = mech.sample(None, 2000)
    assert abs(float(s.mean()) - 100.0) < 0.3
    assert 0.7 < float(s.std()) < 1.3
    # forward() presents the same raw-unit distribution.
    dist = mech.forward(None)
    torch.testing.assert_close(dist.log_prob(x[:5]), mech.log_prob(x[:5], None))
    assert abs(float(dist.sample((500,)).mean()) - 100.0) < 0.5


def test_a9_flow_conditional_on_large_parents_matches_linear_gaussian():
    g = torch.Generator().manual_seed(1)
    pa = 100.0 + 10.0 * torch.randn(2000, 1, generator=g)
    x = 0.5 * pa + torch.randn(2000, 1, generator=g)
    lg = LinearGaussianMechanism()
    lg.fit_local(x, pa)
    mech = _flow()
    mech.fit_local(x, pa, epochs=30, lr=3e-3, batch_size=256)
    with torch.no_grad():
        nll_lg = float(-lg.log_prob(x, pa).mean())
        nll_flow = float(-mech.log_prob(x, pa).mean())
    assert nll_flow < nll_lg + 0.3, (nll_flow, nll_lg)
    s = mech.sample(torch.tensor([[90.0], [110.0]]), 500)
    torch.testing.assert_close(
        s.mean(1).reshape(-1), torch.tensor([45.0, 55.0]), atol=0.5, rtol=0,
    )


def test_a9_flow_density_integrates_to_one_in_raw_units():
    x = 1000.0 + 50.0 * torch.randn(500, 1)
    mech = _flow()
    mech.fit_local(x, None, epochs=3)
    grid = torch.linspace(0.0, 2000.0, 20001).unsqueeze(-1)
    with torch.no_grad():
        dens = mech.log_prob(grid, None).exp()
    assert abs(float(torch.trapz(dens, grid[:, 0])) - 1.0) < 1e-2


def test_a9_flow_standardisation_frozen_under_warm_start_and_clamped():
    g = torch.Generator().manual_seed(2)
    pa = torch.randn(300, 1, generator=g)
    x = pa + 0.1 * torch.randn(300, 1, generator=g)
    mech = _flow()
    mech.fit_local(x, pa, epochs=1)
    bufs = {k: getattr(mech, k).clone() for k in ("_x_mean", "_x_std", "_pa_mean", "_pa_std")}
    mech.fit_local(x + 50.0, pa * 3.0, epochs=1, warm_start=True)
    for k, v in bufs.items():
        torch.testing.assert_close(getattr(mech, k), v)
    assert set(bufs) <= set(mech.state_dict())
    # A context 1e6 sigma out is clamped, not fed to the conditioner raw.
    far = torch.tensor([[1e6], [1e9]])
    torch.testing.assert_close(mech._std_ctx(far), torch.full((2, 1), mech._pa_clamp))
    assert torch.isfinite(mech.log_prob(torch.zeros(2, 1), far)).all()


def test_a9_flow_weighted_standardisation_is_replication_exact():
    g = torch.Generator().manual_seed(3)
    pa = 5 + 2 * torch.randn(40, 1, generator=g)
    x = pa + torch.randn(40, 1, generator=g)
    w = torch.randint(0, 4, (40,), generator=g).float()
    idx = _replicate(w)
    a, b = _flow(), _flow()
    a.fit_local(x, pa, weights=w, epochs=0)
    b.fit_local(x[idx], pa[idx], epochs=0)
    for k in ("_x_mean", "_x_std", "_pa_mean", "_pa_std"):
        torch.testing.assert_close(getattr(a, k), getattr(b, k))


@pytest.mark.parametrize("root", [True, False])
def test_b9_flow_accepts_float64_inputs(root):
    g = torch.Generator().manual_seed(4)
    pa = torch.randn(64, 1, generator=g).double()
    x = (pa + torch.randn(64, 1, generator=g).double())
    mech = _flow()
    # Pre-fix: float64 data against the float32 zuko net crashed.
    mech.fit_local(x, None if root else pa, epochs=1)
    lp = mech.log_prob(x, None if root else pa)
    assert lp.shape == (64,) and torch.isfinite(lp).all()
    assert mech.sample(None if root else pa, 2).shape == (1 if root else 64, 2, 1)


# ==========================================================================
# A11 / B11 / C -- FlexCode: off-support density, degenerate span, sampler
# ==========================================================================


def _flexcode_root(y, **kw):
    from nbn.mechanisms.non_parametric.flexcode import FlexCodeMechanism

    mech = FlexCodeMechanism(**kw)
    mech.fit_local(y, None)
    return mech


@pytest.mark.slow
def test_a11_flexcode_density_is_normalised_over_the_real_line():
    # Pre-fix: z was clamped into [0, 1], so every y off the support scored
    # the boundary density and the integral over [-100, 100] was 4.29.  An
    # exponential target leaves a high series value at the support's edges
    # (1.4-1.9 pre-fix across seeds), which makes the leak visible.
    y = -torch.log(torch.rand(2000, 1, generator=torch.Generator().manual_seed(0)))
    mech = _flexcode_root(y)
    grid = torch.linspace(-100.0, 100.0, 400001).unsqueeze(-1)
    with torch.no_grad():
        dens = mech.log_prob(grid, None).exp()
    assert abs(float(torch.trapz(dens, grid[:, 0])) - 1.0) < 0.02
    # Off the support: exactly the documented floor, finite.
    far = mech.log_prob(torch.tensor([[-50.0], [50.0]]), None)
    assert torch.isfinite(far).all()
    span = mech._y_max - mech._y_min
    coef = mech._coef(None, 1)
    floor = math.log(mech.min_density) - torch.log(mech._normaliser(coef)) - torch.log(span)
    torch.testing.assert_close(far, floor.expand(2))


@pytest.mark.parametrize("value", [1.5, 1000.0, -3e4])
def test_b11_flexcode_constant_target_is_finite(value):
    # Pre-fix: the 1e-6 span floor vanished in float32 next to |y| ~ 1, so
    # z = 0/0 and log_prob was NaN.
    mech = _flexcode_root(torch.full((20, 1), value))
    lp = mech.log_prob(torch.tensor([[value]]), None)
    assert torch.isfinite(lp).all()
    assert torch.isfinite(mech.sample(None, 10)).all()


def test_b11_flexcode_single_row_with_parents_is_finite():
    from nbn.mechanisms.non_parametric.flexcode import FlexCodeMechanism

    mech = FlexCodeMechanism(epochs=2)
    mech.fit_local(torch.tensor([[0.3]]), torch.tensor([[1.0]]))
    lp = mech.log_prob(torch.tensor([[0.3], [0.2]]), torch.tensor([[1.0], [2.0]]))
    assert torch.isfinite(lp).all()


def test_c_flexcode_samples_are_continuous_and_match_the_density():
    y = torch.randn(3000, 1, generator=torch.Generator().manual_seed(1))
    mech = _flexcode_root(y, grid_size=64)
    s = mech.sample(None, 20000).reshape(-1)
    # Pre-fix: inverse-CDF returned grid nodes only (<= grid_size values).
    assert s.unique().numel() > 10000
    # Kolmogorov distance between the samples and the model's own CDF.
    grid = torch.linspace(float(mech._y_min), float(mech._y_max), 20001)
    with torch.no_grad():
        dens = mech.log_prob(grid.unsqueeze(-1), None).exp()
    cdf = torch.cumulative_trapezoid(dens, grid)
    cdf = torch.cat([torch.zeros(1), cdf]) / cdf[-1]
    emp = torch.searchsorted(s.sort().values, grid, right=True).float() / s.numel()
    assert float((emp - cdf).abs().max()) < 0.02


# ==========================================================================
# A12 / B11 / C -- ConditionalKDE: weighted sampling, weighted auto bw,
# effective sample size, one row, cross-device update
# ==========================================================================


def _kde(**kw):
    from nbn.mechanisms.non_parametric.conditional_kde import ConditionalKDEMechanism

    return ConditionalKDEMechanism(**kw)


def test_a12_kde_root_sample_follows_the_weights():
    # Pre-fix: the root sampler used randint, so the zero-weight cluster at 10
    # received half of the samples.
    y = torch.cat([torch.zeros(50, 1), torch.full((50, 1), 10.0)])
    w = torch.cat([torch.ones(50), torch.full((50,), 1e-6)])
    mech = _kde(bw_factor=1.0)
    mech.fit_local(y, None, weights=w)
    s = mech.sample(None, 4000)
    assert float((s > 5).float().mean()) < 0.01


def test_a12_kde_conditional_sample_follows_the_weights():
    # Pre-fix: the exact conditional path dropped _train_logw from the
    # component logits (the truncated path already had it).
    pa = torch.zeros(100, 1) + 0.01 * torch.randn(100, 1)
    y = torch.cat([torch.zeros(50, 1), torch.full((50, 1), 10.0)])
    w = torch.cat([torch.full((50,), 3.0), torch.full((50,), 1.0)])
    mech = _kde(bw_factor=1.0)
    mech.fit_local(y, pa, weights=w)
    s = mech.sample(torch.zeros(2, 1), 4000)
    frac_far = float((s > 5).float().mean())
    assert abs(frac_far - 0.25) < 0.03, frac_far
    # Consistent with the density the mechanism reports.
    with torch.no_grad():
        p0 = mech.log_prob(torch.zeros(1, 1), torch.zeros(1, 1)).exp()
        p10 = mech.log_prob(torch.full((1, 1), 10.0), torch.zeros(1, 1)).exp()
    assert abs(float(p10 / (p0 + p10)) - 0.25) < 0.01


def test_a12_kde_auto_bandwidth_uses_the_weights():
    g = torch.Generator().manual_seed(0)
    pa = torch.randn(200, 1, generator=g)
    y = torch.sin(3 * pa) + 0.05 * torch.randn(200, 1, generator=g)
    junk_pa = torch.randn(200, 1, generator=g)
    junk_y = 5 * torch.randn(200, 1, generator=g)
    # Zero-weight junk must leave the auto-selected factor (and the whole
    # fit) exactly as if it had been dropped.  Pre-fix the selection ignored
    # the weights and scored on the junk rows too: it picked 1.0 where the
    # sharp curve alone picks 0.2.
    x_all, pa_all = torch.cat([y, junk_y]), torch.cat([pa, junk_pa])
    w = torch.cat([torch.ones(200), torch.zeros(200)])
    weighted = _kde(bw_factor="auto")
    weighted.fit_local(x_all, pa_all, weights=w)
    dropped = _kde(bw_factor="auto")
    dropped.fit_local(y, pa)
    assert weighted.bw_factor == dropped.bw_factor == 0.2
    q = torch.tensor([[0.0], [1.0]])
    torch.testing.assert_close(weighted.log_prob(q, q), dropped.log_prob(q, q))


def test_a12_kde_auto_bandwidth_scores_a_weighted_holdout():
    from nbn.mechanisms.non_parametric import conditional_kde as ck

    g = torch.Generator().manual_seed(1)
    y = torch.randn(200, 1, generator=g)
    w = torch.rand(200, generator=g) + 0.5
    seen = {}
    real = ck.weighted_mean

    def spy(values, weights):
        seen["weights"] = weights
        return real(values, weights)

    mech = _kde(bw_factor="auto")
    try:
        ck.weighted_mean = spy
        mech.fit_local(y, None, weights=w)
    finally:
        ck.weighted_mean = real
    assert seen["weights"] is not None and seen["weights"].numel() == 50


def test_kde_bandwidth_sample_size_is_replication_exact_or_kish():
    from nbn.mechanisms.non_parametric.conditional_kde import _bandwidth_n

    assert _bandwidth_n(None, 7) == 7.0
    # Integer multiplicities: the replicated row count.
    assert _bandwidth_n(torch.tensor([2.0, 1.0, 0.0, 3.0]), 4) == 6.0
    # Fractional weights: Kish n_eff -- scale-invariant, N for equal weights,
    # never the (possibly <= 1) raw total.
    assert _bandwidth_n(torch.full((10,), 0.05), 10) == pytest.approx(10.0)
    w = torch.tensor([0.5, 0.25, 0.25])
    assert _bandwidth_n(w, 3) == pytest.approx(1.0 / (0.25 + 0.0625 + 0.0625))
    assert _bandwidth_n(7.3 * w, 3) == pytest.approx(_bandwidth_n(w, 3))


@pytest.mark.parametrize("root", [True, False])
def test_b11_kde_and_knn_fit_a_single_row(root):
    from nbn.mechanisms.non_parametric.knn_conditional import KNNConditionalMechanism

    y = torch.tensor([[0.3], [0.5]])
    pa = None if root else torch.tensor([[1.0], [1.2]])
    for mech in (_kde(), KNNConditionalMechanism()):
        # Pre-fix: the unbiased std of one row is NaN, survived the clamps,
        # and every log_prob was NaN.
        mech.fit_local(y[:1], None if root else pa[:1])
        lp = mech.log_prob(y, pa)
        assert torch.isfinite(lp).all(), type(mech).__name__
        assert torch.isfinite(mech.sample(pa, 3)).all()


@pytest.mark.parametrize("device", [d for d in available_devices() if d != "cpu"])
def test_c_kde_update_local_accepts_cpu_batch_into_accelerator_model(device):
    # Pre-fix: the new weights were validated on x's (CPU) device as float64,
    # then moved to the model's device -- impossible on MPS.
    g = torch.Generator().manual_seed(0)
    y = torch.randn(40, 1, generator=g)
    pa = torch.randn(40, 1, generator=g)
    mech = _kde(bw_factor=1.0)
    mech.fit_local(y[:30].to(device), pa[:30].to(device), weights=torch.ones(30, device=device))
    mech.update_local(y[30:], pa[30:], weights=torch.full((10,), 2.0))
    assert mech._train_y.shape[0] == 40 and mech._train_logw.device.type == device
    assert torch.isfinite(mech.log_prob(y.to(device), pa.to(device))).all()


# ==========================================================================
# R1 -- checkpoints written by origin/master (pre-standardisation MDN/flow)
# ==========================================================================
#
# tests/data/master_mdn_flow_*.pt were produced by *origin/master* code with
# tests/data/make_master_mdn_flow_checkpoint.py: A (root MDN), B | A (MDN),
# C | B (flow), D (root flow).  Those mechanisms predate the _x_*/_pa_*
# standardisation buffers; on load they must behave exactly as on master
# (identity standardisation), not raise AttributeError or silently rescale.

_DATA_DIR = __import__("pathlib").Path(__file__).resolve().parents[1] / "data"


def _master_ckpt():
    pytest.importorskip("zuko")
    from nbn import NeuralBayesianNetwork as NBN

    model = NBN.load(str(_DATA_DIR / "master_mdn_flow_model.pt"))
    ref = torch.load(_DATA_DIR / "master_mdn_flow_ref.pt", weights_only=False)
    return model, ref


def test_r1_master_checkpoint_scores_exactly_as_on_master():
    model, ref = _master_ckpt()
    with torch.no_grad():
        per_node = model.log_prob(ref["data"], per_node=True)
    for node, lp in ref["per_node"].items():
        torch.testing.assert_close(per_node[node], lp, atol=1e-5, rtol=1e-5, msg=node)


def test_r1_master_checkpoint_samples_and_keeps_working():
    model, ref = _master_ckpt()
    s = model.sample(64)
    assert all(torch.isfinite(v).all() for v in s.values())
    for node in ("B", "C"):
        mech = model.mechanisms[node]
        pa = ref["data"]["A" if node == "B" else "B"][:5]
        assert torch.isfinite(mech.forward(pa).sample()).all()
        assert torch.isfinite(mech.log_prob(ref["data"][node][:5], pa)).all()


def test_r1_master_checkpoint_warm_start_continues_in_master_units():
    model, ref = _master_ckpt()
    data = ref["data"]
    model.fit(data, epochs=1, warm_start=True)
    # Warm start freezes standardisation: a master-made model has none, so
    # it keeps the identity map rather than acquiring one mid-trajectory.
    for node in ("B", "C"):
        assert model.mechanisms[node]._x_mean is None
    assert torch.isfinite(model.log_prob(data)).all()


def test_r1_master_checkpoint_update_runs():
    model, ref = _master_ckpt()
    model.update({k: v[:50] for k, v in ref["data"].items()}, epochs=1)
    assert torch.isfinite(model.log_prob(ref["data"])).all()


def test_r1_master_state_dict_loads_strictly_into_a_branch_fit():
    model, ref = _master_ckpt()
    data = ref["data"]
    fresh = __import__("copy").deepcopy(model)
    fresh.fit(data, epochs=1)           # branch fit: buffers populated
    for node in ("B", "C"):
        assert fresh.mechanisms[node]._x_mean is not None
    for node, sd in ref["state_dicts"].items():
        fresh.mechanisms[node].load_state_dict(sd)   # strict
    with torch.no_grad():
        per_node = fresh.log_prob(data, per_node=True)
    for node, lp in ref["per_node"].items():
        torch.testing.assert_close(per_node[node], lp, atol=1e-5, rtol=1e-5, msg=node)
    # ...and a branch state_dict loads back into a master-made mechanism.
    branch = __import__("copy").deepcopy(model)
    branch.fit(data, epochs=1)
    for node in ("B", "C"):
        model.mechanisms[node].load_state_dict(branch.mechanisms[node].state_dict())
        torch.testing.assert_close(
            model.mechanisms[node].log_prob(data[node], data["A" if node == "B" else "B"]),
            branch.mechanisms[node].log_prob(data[node], data["A" if node == "B" else "B"]),
        )
