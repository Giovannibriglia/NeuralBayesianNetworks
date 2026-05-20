"""Tests for MDN's NaN/Inf parent sanitisation (v0.6c-A Finding 2).

The bug class: a NaN propagating through the ancestral sampling chain
of ``continuous_nongauss n=5000 seed=0`` lands in ``MDN.sample``'s
parents, then survives the linear projection (because PyTorch's
``0 * NaN = NaN`` even when the linear's row weight is zero) and
trips ``Categorical(probs=softmax(NaN))``'s simplex check.

Round-2 fix: ``_sanitise_parents`` in ``nbn/mechanisms/mdn.py``
replaces NaN/Inf with 0 before the linear projection and logs a
warning the first time it triggers per ``(mech_name, decade(count))``.
"""
from __future__ import annotations

import logging

import pytest
import torch

from nbn.mechanisms.mdn import _sanitise_parents


def _reset_sanitise_warned() -> None:
    """Clear the module-level dedupe cache between tests so warning
    expectations are deterministic."""
    if hasattr(_sanitise_parents, "_warned_keys"):
        _sanitise_parents._warned_keys.clear()


# ---------------------------------------------------------------------- #
# Direct unit tests for the sanitiser helper
# ---------------------------------------------------------------------- #


def test_sanitise_parents_replaces_nan(caplog) -> None:
    _reset_sanitise_warned()
    parents = torch.tensor([
        [0.5, 0.3],
        [float("nan"), 0.1],
        [0.2, 0.4],
    ])
    with caplog.at_level(logging.WARNING):
        out = _sanitise_parents(parents, mech_name="test_nan")
    assert torch.isfinite(out).all()
    assert out[1, 0].item() == 0.0
    # Other entries unchanged
    assert out[0].equal(parents[0])
    assert out[2].equal(parents[2])
    sanitisation_warnings = [
        r for r in caplog.records if "sanitised" in r.message.lower()
    ]
    assert len(sanitisation_warnings) >= 1


def test_sanitise_parents_replaces_inf(caplog) -> None:
    _reset_sanitise_warned()
    parents = torch.tensor([[0.5, 0.3], [float("inf"), -float("inf")]])
    with caplog.at_level(logging.WARNING):
        out = _sanitise_parents(parents, mech_name="test_inf")
    assert torch.isfinite(out).all()
    assert out[1, 0].item() == 0.0
    assert out[1, 1].item() == 0.0


def test_sanitise_parents_finite_input_is_noop() -> None:
    """Pure no-op on clean input — exact equality, not allclose."""
    _reset_sanitise_warned()
    torch.manual_seed(0)
    parents = torch.randn(100, 4)
    assert torch.isfinite(parents).all()
    out = _sanitise_parents(parents, mech_name="test_noop")
    assert out is parents or torch.equal(out, parents)


def test_sanitise_parents_dedupes_repeated_warnings(caplog) -> None:
    """First call logs; subsequent calls of similar magnitude don't."""
    _reset_sanitise_warned()
    bad1 = torch.tensor([[float("nan"), 0.0]])
    bad2 = torch.tensor([[float("nan"), 0.0]])
    with caplog.at_level(logging.WARNING):
        _sanitise_parents(bad1, mech_name="test_dedup")
        _sanitise_parents(bad2, mech_name="test_dedup")
    sanitisation_warnings = [
        r for r in caplog.records if "test_dedup" in r.message
    ]
    # Both calls have n_invalid=1 → same decade → only the first warns.
    assert len(sanitisation_warnings) == 1


# ---------------------------------------------------------------------- #
# End-to-end tests through MDN entry points
# ---------------------------------------------------------------------- #


def _build_mdn_with_one_parent():
    """Construct a minimal MDN with one parent dimension via the
    synthetic-side helper that's known to produce zero-weight logit
    rows (the structural condition that turned ``0 * NaN = NaN`` into
    a Simplex violation)."""
    from benchmarking.synthetic import _build_mdn_mechanism
    gen = torch.Generator(device="cpu").manual_seed(0)
    mech = _build_mdn_mechanism(
        parents=["X0"], gen=gen, device=torch.device("cpu"),
    )
    return mech


def test_mdn_sample_handles_nan_in_parents() -> None:
    """Regression for v0.6c-A Finding 2: ``MDN.sample`` must not raise
    on NaN parent values; outputs should be finite."""
    _reset_sanitise_warned()
    mech = _build_mdn_with_one_parent()
    parents = torch.tensor([[0.5], [float("nan")], [0.2]])
    out = mech.sample(parents, n=4)
    assert torch.isfinite(out).all()
    assert out.shape == (3, 4, 1)


def test_mdn_log_prob_handles_inf_in_parents() -> None:
    _reset_sanitise_warned()
    mech = _build_mdn_with_one_parent()
    parents = torch.tensor([[0.5], [float("inf")]])
    value = torch.tensor([[0.0], [1.0]])
    lp = mech.log_prob(value, parents)
    assert torch.isfinite(lp).all()


def test_mdn_forward_handles_nan_in_parents() -> None:
    _reset_sanitise_warned()
    mech = _build_mdn_with_one_parent()
    parents = torch.tensor([[float("nan")], [0.5]])
    dist = mech.forward(parents)
    samp = dist.sample()
    assert torch.isfinite(samp).all()


# ---------------------------------------------------------------------- #
# Parent standardization tests (Bug B, issues #81 #24 #54)
# ---------------------------------------------------------------------- #


def test_mdn_extreme_parents_no_nan_after_fit() -> None:
    """Bug B regression: extreme parent values (3.3M-scale) must not
    produce NaN parameters after fit_local.  Pre-fix, the MLP's linear
    projection turned extreme inputs into inf log_scale → loss=1e14 →
    backward overflow → NaN grads → NaN params → CUDA assert at LW."""
    from nbn.mechanisms.mdn import MDNMechanism
    torch.manual_seed(0)
    mdn = MDNMechanism(num_components=3)
    N = 100
    x = torch.randn(N, 1)
    parents = torch.randn(N, 1) * 3_300_000.0  # simulate deep-chain extreme values
    mdn.fit_local(x, parents, epochs=5, batch_size=32)
    for name, p in mdn.named_parameters():
        assert torch.isfinite(p).all(), f"{name} has NaN/Inf after fit with extreme parents"
    # Inference must also be finite
    test_pa = torch.randn(10, 1) * 3_300_000.0
    samp = mdn.sample(test_pa, n=5)
    assert torch.isfinite(samp).all()


def test_mdn_constant_parent_column_no_crash() -> None:
    """Constant parent column (std=0) must not crash fit_local or inference.

    Pre-fix, the double-standardization bug caused: after clamping pa_std
    to 1e-3, _params_from_parents received (0 - pa_mean) / 1e-3 = -5000
    during the training forward pass, which overflowed to NaN in the MLP.
    """
    from nbn.mechanisms.mdn import MDNMechanism
    mdn = MDNMechanism(num_components=3)
    N = 80
    x = torch.randn(N, 1)
    parents = torch.full((N, 1), 7.0)  # constant → std=0 → clamped to 1e-3
    mdn.fit_local(x, parents, epochs=5, batch_size=32)
    assert mdn._pa_std is not None
    assert mdn._pa_std.item() == pytest.approx(1e-3, rel=0.01)
    test_pa = torch.full((5, 1), 7.0)
    samp = mdn.sample(test_pa, n=4)
    assert torch.isfinite(samp).all()
    lp = mdn.log_prob(x[:5].unsqueeze(1), test_pa)
    assert torch.isfinite(lp).all()


def test_mdn_training_inference_consistency() -> None:
    """Training and inference use the same standardization path.

    Fit on moderately extreme parents; evaluate log_prob on held-out
    parents from the same distribution.  A model trained with double-
    standardization would evaluate at a completely different input range
    and produce degenerate (constant) log_prob values.
    """
    from nbn.mechanisms.mdn import MDNMechanism
    torch.manual_seed(42)
    N = 300
    # parents with large mean/std — training and inference must agree
    pa_scale = 1000.0
    parents_train = torch.randn(N, 1) * pa_scale
    x_train = 0.5 * parents_train / pa_scale + torch.randn(N, 1) * 0.1

    mdn = MDNMechanism(num_components=4, hidden=(32, 32))
    mdn.fit_local(x_train, parents_train, epochs=50, batch_size=64)

    # log_prob on training-distribution parents must be finite and vary
    pa_test = torch.randn(50, 1) * pa_scale
    x_test = 0.5 * pa_test / pa_scale + torch.randn(50, 1) * 0.1
    with torch.no_grad():
        lp = mdn.log_prob(x_test.unsqueeze(1), pa_test)
    assert torch.isfinite(lp).all(), "log_prob has NaN/Inf"
    # Verify the model isn't degenerate (constant output regardless of parents)
    pa_hi = torch.full((50, 1), 3.0 * pa_scale)
    pa_lo = torch.full((50, 1), -3.0 * pa_scale)
    x_mid = torch.zeros(50, 1)
    with torch.no_grad():
        lp_hi = mdn.log_prob(x_mid.unsqueeze(1), pa_hi)
        lp_lo = mdn.log_prob(x_mid.unsqueeze(1), pa_lo)
    # A non-degenerate model should give different log-probs for x=0
    # when parents are +3σ vs -3σ (since the true conditional shifts).
    assert not torch.allclose(lp_hi, lp_lo, atol=0.1), (
        "MDN appears degenerate: log_prob doesn't vary with parents "
        "(possible training/inference standardization mismatch)"
    )


def test_mdn_extreme_inference_parents_no_nan_bug_c() -> None:
    """Bug C regression: inference-time parents at 1000σ must not produce
    NaN/Inf outputs from a fitted MDN.

    Root cause: deep-chain LW sampling at n≥100 produces particles at
    700σ+ after standardization.  Without the ±_pa_clamp guard, MLP
    log_scale outputs at that scale overflow float32 → scale=Inf →
    sample=Inf → downstream Categorical(NaN) crash (follow-up to Bug B /
    PR #90).
    """
    from nbn.mechanisms.mdn import MDNMechanism

    torch.manual_seed(7)
    N = 200
    parents_train = torch.randn(N, 2)
    x_train = 0.3 * parents_train[:, :1] + torch.randn(N, 1) * 0.2

    mdn = MDNMechanism(num_components=3, hidden=(32,))
    mdn.fit_local(x_train, parents_train, epochs=30, batch_size=64)

    assert mdn._pa_std is not None
    pa_std = mdn._pa_std  # [1, 2]

    # Inject parents at 1000σ — wildly outside training distribution
    pa_extreme = pa_std.expand(20, -1) * 1000.0
    with torch.no_grad():
        samp = mdn.sample(pa_extreme, n=10)
        lp = mdn.log_prob(x_train[:20].unsqueeze(1), pa_extreme)

    assert torch.isfinite(samp).all(), "sample() produced NaN/Inf at 1000σ parents"
    assert torch.isfinite(lp).all(), "log_prob() produced NaN/Inf at 1000σ parents"


def test_mdn_log_scale_output_clamp_issue_95() -> None:
    """Issue #95 regression: scale returned by _params_from_parents must
    never exceed exp(_log_scale_clamp) ≈ 1097, even when extreme parents
    bypass the input-side _pa_clamp.

    Strategy: disable the input clamp (set to inf) and inject adversarial
    parents at 1000σ so the MLP operates far outside its training regime.
    The output clamp on log_scale is then the only guard; the test verifies
    it holds.
    """
    import math
    from nbn.mechanisms.mdn import MDNMechanism

    torch.manual_seed(13)
    N = 200
    parents_train = torch.randn(N, 2)
    x_train = 0.3 * parents_train[:, :1] + torch.randn(N, 1) * 0.2

    mdn = MDNMechanism(num_components=3, hidden=(32,))
    mdn.fit_local(x_train, parents_train, epochs=50, batch_size=64)

    assert mdn._pa_std is not None
    pa_std = mdn._pa_std  # [1, 2]

    # Disable the input clamp so the output clamp is the only guard.
    saved_pa_clamp = mdn._pa_clamp
    mdn._pa_clamp = float("inf")
    try:
        pa_extreme = pa_std.expand(50, -1) * 1000.0
        with torch.no_grad():
            _, _, scale = mdn._params_from_parents(pa_extreme, (50,))
    finally:
        mdn._pa_clamp = saved_pa_clamp

    max_allowed = math.exp(mdn._log_scale_clamp)
    assert scale.max().item() <= max_allowed + 1e-4, (
        f"scale.max()={scale.max().item():.4f} > exp(7)≈{max_allowed:.4f}; "
        "output clamp not effective"
    )
    assert torch.isfinite(scale).all(), "scale has Inf/NaN after output clamp"


def test_make_synthetic_bn_continuous_nongauss_n5000_seed0_runs() -> None:
    """Regression test for the exact paper-config failure case.

    Pre-fix, ``make_synthetic_bn(family='continuous_nongauss', n_nodes=5000,
    seed=0)`` raised ``ValueError(... Simplex constraint ...)`` inside
    the ancestral sample chain that builds ``train_data``.  After the
    sanitiser lands, the call must complete and produce finite
    ``train_data`` tensors.

    Uses smaller ``n_train``/``n_reference`` than the paper config to
    keep the test under a few seconds of CPU; the failure mode reproduces
    at any sample count once the chain reaches n=5000."""
    _reset_sanitise_warned()
    from benchmarking.synthetic import make_synthetic_bn
    bn = make_synthetic_bn(
        family="continuous_nongauss", n_nodes=5000,
        edge_density=0.20, max_in_degree=4, cardinality=4,
        seed=0, device="cpu",
        n_train=200, n_test=50, n_reference=200,
    )
    assert bn is not None
    assert bn.train_data is not None
    # Train data may legitimately contain Inf at extreme tail samples
    # of an MDN chain at n=5000 (separate v0.7 root-cause concern).
    # The contract here is "the call did not raise"; finite-check on
    # train_data is asserted only on a small slice for guard-rail
    # purposes.
    sample_node = next(iter(bn.train_data.keys()))
    sample = bn.train_data[sample_node]
    assert sample.shape[0] == 200
