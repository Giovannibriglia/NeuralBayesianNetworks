"""``warm_start=True`` continues from the existing parameters; ``False`` refits.

Without this, a caller invoking ``fit_local`` iteratively does not get
successive refinement — it gets an independent refit from a random
initialisation each time, because every gradient-trained mechanism rebuilds
its network and its optimiser on every call.  Downstream that is used as the
M-step of an EM loop, which must *increase* ``Q(theta | theta_old)`` starting
from ``theta_old``; under a fresh initialisation that premise is simply false,
and the machinery built on it inverts.  The observed signature was a
backtracking line search in which each smaller learning rate came back
*worse*, monotonically, because a smaller step size was not producing a
gentler step — it was producing a worse fresh fit.

The contract, in the order the tests below pin it:

1. ``warm_start=True, epochs=0`` leaves the parameters bitwise unchanged;
   ``warm_start=False, epochs=0`` returns a fresh initialisation.
2. A warm second call does not decrease the objective; a cold one is
   independent of the first (checked distributionally, across seeds, not by a
   single comparison that a lucky draw could satisfy either way).
3. Incompatible shapes raise, naming the mechanism and the mismatch — never a
   silent rebuild, which is exactly the failure that produced this feature.
4. Optimiser state is *not* carried over, and data-derived standardisation
   buffers *are* frozen.  Each has a test that fails if the other choice were
   implemented.
5. Every family is covered, including the closed-form ones, so a caller
   passing ``warm_start=True`` uniformly across a network gets defined
   behaviour at every node.

Note the snapshot idiom.  ``copy.deepcopy(state_dict())`` is mandatory, per
``test_parameter_snapshot_contract`` (PR #258): ``state_dict()`` aliases the
live parameters, so the naive spelling compares a snapshot against itself and
passes regardless of the implementation.  Warm-starting makes that requirement
reach further — see ``test_uncopied_snapshot_is_clobbered_by_a_warm_fit``.
"""
from __future__ import annotations

import copy

import pytest
import torch

from nbn import NeuralBayesianNetwork as NBN
from nbn.mechanisms import (
    BinningCategoricalTable,
    CategoricalTableMechanism,
    ConditionalKDEMechanism,
    DeterministicMechanism,
    DiracGaussianMechanism,
    FlexCodeMechanism,
    KNNConditionalMechanism,
    LinearGaussianMechanism,
    MDNMechanism,
    NeuralCategoricalMechanism,
    SmoothedEmpiricalCategoricalMechanism,
)

zuko = pytest.importorskip("zuko", reason="flow cases need the neural extra")
from nbn.mechanisms.parametric.normalizing_flow import (  # noqa: E402
    NormalizingFlowMechanism,
)


# ==========================================================================
# Cases
# ==========================================================================
# Each factory returns (mechanism, x, parents, fit_kwargs).  ``fit_kwargs``
# carries whatever that mechanism needs to fit at all (declared cardinalities)
# plus the cost controls (tiny epochs, consolidate off).  ``consolidate=False``
# matters for the bitwise tests: the EWC Fisher pass writes _ewc_mu/_ewc_fisher
# buffers that appear in state_dict, and it is orthogonal to warm-starting.


def _cont(n=120, d_pa=1, seed=0):
    torch.manual_seed(seed)
    pa = torch.randn(n, d_pa)
    x = 2.0 * pa[:, :1] + 0.1 * torch.randn(n, 1)
    return x, pa


def _disc(n=150, seed=0):
    torch.manual_seed(seed)
    pa = torch.randint(0, 3, (n, 1))
    x = (pa.reshape(-1) % 2).long()
    return x, pa


def _mdn(seed=0):
    x, pa = _cont(seed=seed)
    return (MDNMechanism(num_components=2, hidden=(8,)), x, pa,
            {"epochs": 3, "consolidate": False})


def _mdn_root(seed=0):
    x, _ = _cont(seed=seed)
    return MDNMechanism(num_components=2, hidden=(8,)), x, None, {"consolidate": False}


def _flow(seed=0):
    x, pa = _cont(seed=seed)
    return (NormalizingFlowMechanism(num_transforms=2, hidden=(8,)), x, pa,
            {"epochs": 2})


def _neural_cat(seed=0):
    x, pa = _disc(seed=seed)
    return (NeuralCategoricalMechanism(n_classes=2, hidden=(8,)), x, pa.float(),
            {"epochs": 3, "consolidate": False})


def _neural_cat_root(seed=0):
    x, _ = _disc(seed=seed)
    return (NeuralCategoricalMechanism(n_classes=2, hidden=(8,)), x, None,
            {"epochs": 3, "consolidate": False})


def _flexcode(seed=0):
    x, pa = _cont(seed=seed)
    return (FlexCodeMechanism(n_basis=7, hidden=(8,), epochs=3), x, pa, {})


def _flexcode_root(seed=0):
    x, _ = _cont(seed=seed)
    return FlexCodeMechanism(n_basis=7, hidden=(8,), epochs=3), x, None, {}


def _lg(seed=0):
    x, pa = _cont(d_pa=2, seed=seed)
    return LinearGaussianMechanism(), x, pa, {}


def _cat(seed=0):
    x, pa = _disc(seed=seed)
    return CategoricalTableMechanism(), x, pa, {"parent_cards": [3], "n_classes": 2}


def _smoothed(seed=0):
    x, pa = _disc(seed=seed)
    return (SmoothedEmpiricalCategoricalMechanism(), x, pa,
            {"parent_cards": [3], "n_classes": 2})


def _binning(seed=0):
    torch.manual_seed(seed)
    pa = torch.randn(150, 2)
    x = (pa[:, 0] > 0).long()
    return (BinningCategoricalTable(
        parent_kinds=[("continuous", 0), ("continuous", 0)],
        n_bins=3, n_categories=2,
    ), x, pa, {})


def _kde(seed=0):
    x, pa = _cont(seed=seed)
    return ConditionalKDEMechanism(bw_factor=1.0), x, pa, {}


def _knn(seed=0):
    x, pa = _cont(seed=seed)
    return KNNConditionalMechanism(k=5), x, pa, {}


def _deterministic(seed=0):
    x, pa = _cont(seed=seed)
    return DeterministicMechanism(value=torch.tensor([1.0])), x, pa, {}


def _dirac(seed=0):
    x, pa = _cont(seed=seed)
    return DiracGaussianMechanism(value=1.0), x, pa, {}


#: Mechanisms with a genuine initialisation to continue from.
GRADIENT = {
    "mdn": _mdn,
    "flow": _flow,
    "neural_categorical": _neural_cat,
    "flexcode": _flexcode,
}

#: Branches whose fit is the exact maximiser of the local objective (or the
#: stored sample itself), where warm_start is a documented no-op.  Includes the
#: *root* branches of three families that are gradient-trained when conditioned
#: — that split is the point of ``warm_start_is_noop`` describing the non-root
#: branch only.
CLOSED_FORM = {
    "linear_gaussian": _lg,
    "categorical_table": _cat,
    "smoothed_empirical": _smoothed,
    "binning": _binning,
    "kde": _kde,
    "knn": _knn,
    "deterministic": _deterministic,
    "dirac": _dirac,
    "mdn_root": _mdn_root,
    "neural_categorical_root": _neural_cat_root,
    "flexcode_root": _flexcode_root,
}

ALL = {**GRADIENT, **CLOSED_FORM}


def _snapshot(module: torch.nn.Module) -> dict:
    """The supported idiom: a *copied* state_dict.  See PR #258."""
    return copy.deepcopy(module.state_dict())


def _same(mech, snap) -> bool:
    """Bitwise equality of every tensor in the state_dict."""
    live = mech.state_dict()
    return set(live) == set(snap) and all(
        torch.equal(live[k], v) for k, v in snap.items() if v is not None
    )


def _agrees(mech, snap, atol=1e-6) -> bool:
    """Numerical equality — the right comparison for the closed-form branches.

    ``torch.linalg.lstsq`` and the weighted reductions are not bit-reproducible
    run to run under multithreaded CPU BLAS: measured, six identical
    ``LinearGaussianMechanism`` fits on the same data produced weights differing
    by up to 1.5e-9 in four of six trials.  That jitter predates warm-starting
    and belongs to the solve, so a closed-form branch's "warm equals cold"
    claim is a numerical one.  The *bitwise* contract is asserted where it is
    actually the contract: the gradient families under ``warm_start=True``,
    which copy nothing and recompute nothing.
    """
    live = mech.state_dict()
    if set(live) != set(snap):
        return False
    return all(
        torch.allclose(live[k].float(), v.float(), atol=atol, rtol=0)
        for k, v in snap.items() if v is not None
    )


# ==========================================================================
# 1. The contract, with no room for interpretation
# ==========================================================================


@pytest.mark.parametrize("name", sorted(GRADIENT))
def test_warm_start_with_zero_epochs_leaves_parameters_bitwise_unchanged(name):
    """The headline: warm_start=True, epochs=0 must be a no-op on theta."""
    mech, x, pa, kw = GRADIENT[name]()
    mech.fit_local(x, pa, **kw)
    snap = _snapshot(mech)
    assert snap, "nothing captured -- the comparison would be vacuous"

    info = mech.fit_local(x, pa, **{**kw, "epochs": 0}, warm_start=True)

    assert _same(mech, snap), f"{name}: warm_start=True rebuilt the parameters"
    assert info["warm_started"] is True


@pytest.mark.parametrize("name", sorted(GRADIENT))
def test_cold_start_with_zero_epochs_is_a_fresh_initialisation(name):
    """The complement, and the guard that the test above is not vacuous."""
    mech, x, pa, kw = GRADIENT[name]()
    mech.fit_local(x, pa, **kw)
    snap = _snapshot(mech)

    info = mech.fit_local(x, pa, **{**kw, "epochs": 0}, warm_start=False)

    assert not _same(mech, snap), (
        f"{name}: warm_start=False left the parameters in place -- either the "
        f"default changed or the rebuild is gone"
    )
    assert info["warm_started"] is False


def test_default_is_false_and_reproduces_the_historical_behaviour():
    """Backward compatibility: an unchanged call must be byte-identical."""
    import inspect
    for name, factory in ALL.items():
        mech, _, _, _ = factory()
        sig = inspect.signature(mech.fit_local)
        assert "warm_start" in sig.parameters, f"{name} does not accept warm_start"
        assert sig.parameters["warm_start"].default is False, name

    # And the fit itself, seeded, is unchanged by the presence of the keyword.
    torch.manual_seed(0)
    a, pa = _cont(seed=3)
    torch.manual_seed(11)
    implicit = MDNMechanism(num_components=2, hidden=(8,))
    implicit.fit_local(a, pa, epochs=3, consolidate=False)
    torch.manual_seed(11)
    explicit = MDNMechanism(num_components=2, hidden=(8,))
    explicit.fit_local(a, pa, epochs=3, consolidate=False, warm_start=False)
    assert _same(explicit, _snapshot(implicit))


# ==========================================================================
# 2. The objective
# ==========================================================================


def _nll(mech, x, pa) -> float:
    with torch.no_grad():
        return float(-mech.log_prob(x, pa).mean())


@pytest.mark.parametrize("name", sorted(GRADIENT))
def test_a_warm_second_call_does_not_lose_ground(name):
    """Continuing must not throw away the first call's progress.

    Asserted with a small slack: the inner loop is stochastic minibatch SGD,
    which is not monotone step-to-step.  The failure this guards is not a
    fraction of a nat -- a cold refit lands wherever a random initialisation
    plus a handful of epochs lands, which is a different regime entirely (see
    the companion test below for how far apart the two are).
    """
    mech, x, pa, kw = GRADIENT[name]()
    mech.fit_local(x, pa, **{**kw, "epochs": 40})
    first = _nll(mech, x, pa)
    mech.fit_local(x, pa, **{**kw, "epochs": 40}, warm_start=True)
    second = _nll(mech, x, pa)
    assert second <= first + 0.05, (
        f"{name}: warm second call regressed {first:.4f} -> {second:.4f}"
    )


@pytest.mark.parametrize("name", ["mdn", "neural_categorical", "flexcode"])
def test_cold_calls_are_independent_and_warm_calls_are_not(name):
    """The distributional check the single comparison above cannot make.

    Across seeds, a *cold* second call's change in objective must straddle
    zero — it is an independent draw, so it lands on both sides.  A *warm*
    second call must not: it continues, so it improves or holds, and a
    one-sided sign pattern is what distinguishes the two implementations.
    """
    cold_deltas, warm_deltas = [], []
    for seed in range(8):
        for warm, sink in ((False, cold_deltas), (True, warm_deltas)):
            mech, x, pa, kw = GRADIENT[name](seed=seed)
            mech.fit_local(x, pa, **{**kw, "epochs": 30})
            before = _nll(mech, x, pa)
            mech.fit_local(x, pa, **{**kw, "epochs": 5}, warm_start=warm)
            sink.append(_nll(mech, x, pa) - before)

    assert max(cold_deltas) > 0.0, (
        f"{name}: no cold refit ever came back worse across 8 seeds -- that is "
        f"not an independent draw, so the rebuild is not happening"
    )
    assert max(warm_deltas) <= 0.05, (
        f"{name}: a warm continuation came back materially worse "
        f"(worst delta {max(warm_deltas):.4f})"
    )
    # And the two populations must be distinguishable at all.
    assert max(cold_deltas) > max(warm_deltas)


def test_backtracking_line_search_is_monotone_in_the_step_size():
    """The downstream failure, turned into a regression guard.

    A generalized-EM loop that rejects an M-step retries at a smaller learning
    rate, expecting a *gentler* step — one that lands closer to the incumbent.
    Under a fresh refit that expectation inverts: each smaller lr produces an
    independent worse fit, so the retries walk monotonically away from the
    incumbent instead of towards it.  Here the distance to the incumbent must
    shrink as lr shrinks, and the smallest lr must be a near no-op.
    """
    mech, x, pa, kw = _mdn()
    mech.fit_local(x, pa, **{**kw, "epochs": 60})
    incumbent = _snapshot(mech)

    def _distance_after(lr: float) -> float:
        mech.load_state_dict(copy.deepcopy(incumbent))
        mech.fit_local(x, pa, **{**kw, "epochs": 1}, lr=lr, warm_start=True)
        return max(
            float((mech.state_dict()[k] - v).abs().max())
            for k, v in incumbent.items()
            if v is not None and v.is_floating_point() and v.numel()
        )

    lrs = [1e-3, 5e-4, 2.5e-4, 6.25e-5, 2.44e-7]
    dists = [_distance_after(lr) for lr in lrs]
    assert dists == sorted(dists, reverse=True), (
        f"a smaller step size did not produce a gentler step: {dists}"
    )
    assert dists[-1] < 1e-5, f"lr=2.44e-7 was not a near no-op: {dists[-1]}"


# ==========================================================================
# 3. Shape incompatibility raises -- never a silent rebuild
# ==========================================================================


@pytest.mark.parametrize("name", sorted(GRADIENT))
def test_changing_the_parent_width_raises(name):
    mech, x, pa, kw = GRADIENT[name]()
    mech.fit_local(x, pa, **kw)
    snap = _snapshot(mech)
    wider = torch.randn(x.shape[0], pa.shape[1] + 2)

    with pytest.raises(ValueError, match=r"d_pa"):
        mech.fit_local(x, wider, **{**kw, "epochs": 0}, warm_start=True)

    assert _same(mech, snap), f"{name}: the rejected call still mutated state"


@pytest.mark.parametrize("name", sorted(GRADIENT))
def test_the_error_names_the_mechanism_and_both_values(name):
    mech, x, pa, kw = GRADIENT[name]()
    mech.fit_local(x, pa, **kw)
    wider = torch.randn(x.shape[0], pa.shape[1] + 2)
    with pytest.raises(ValueError) as excinfo:
        mech.fit_local(x, wider, **{**kw, "epochs": 0}, warm_start=True)
    msg = str(excinfo.value)
    assert type(mech).__name__ in msg
    assert str(pa.shape[1]) in msg and str(pa.shape[1] + 2) in msg
    assert "warm_start=False" in msg, "the message must say how to proceed"


@pytest.mark.parametrize("name", ["mdn", "neural_categorical", "flexcode"])
def test_dropping_the_parents_entirely_raises(name):
    """A root fit and a conditioned fit are different sets of parameters.

    A bare width check would catch 1 -> 3 and let 3 -> 0 through, which is why
    the branch guard is separate.
    """
    mech, x, pa, kw = GRADIENT[name]()
    mech.fit_local(x, pa, **kw)
    with pytest.raises(ValueError, match="root"):
        mech.fit_local(x, None, **{**kw, "epochs": 0}, warm_start=True)


@pytest.mark.parametrize("name", ["mdn", "neural_categorical", "flexcode"])
def test_acquiring_parents_raises(name):
    """The reverse flip: fitted as a root, warm-started with parents."""
    factory = {"mdn": _mdn_root, "neural_categorical": _neural_cat_root,
               "flexcode": _flexcode_root}[name]
    mech, x, _, kw = factory()
    mech.fit_local(x, None, **kw)
    pa = torch.randn(x.shape[0], 1)
    with pytest.raises(ValueError, match="root"):
        mech.fit_local(x, pa, **{**kw, "epochs": 0}, warm_start=True)


def test_changing_the_child_width_raises():
    """d_x is checked as well as d_pa (MDN and flow are multivariate-capable)."""
    torch.manual_seed(0)
    pa = torch.randn(120, 1)
    mech = MDNMechanism(num_components=2, hidden=(8,))
    mech.fit_local(torch.randn(120, 2), pa, epochs=2, consolidate=False)
    with pytest.raises(ValueError, match=r"d_x"):
        mech.fit_local(
            torch.randn(120, 3), pa, epochs=0, consolidate=False, warm_start=True,
        )


def test_changing_a_parent_cardinality_raises_for_embedded_inputs():
    x, pa = _disc()
    mech = NeuralCategoricalMechanism(n_classes=2, hidden=(8,), embedding_dim=4)
    mech.fit_local(x, pa, parent_cards=[3], epochs=2, consolidate=False)
    with pytest.raises(ValueError, match=r"parent_cards\[0\]"):
        mech.fit_local(
            x, pa, parent_cards=[5], epochs=0, consolidate=False, warm_start=True,
        )


def test_switching_between_embedded_and_raw_parents_raises():
    """The MLP's input width differs between the two, so neither continues."""
    x, pa = _disc()
    mech = NeuralCategoricalMechanism(n_classes=2, hidden=(8,), embedding_dim=4)
    mech.fit_local(x, pa, parent_cards=[3], epochs=2, consolidate=False)
    with pytest.raises(ValueError, match="embedded"):
        # parent_cards omitted -> the raw-input branch.
        mech.fit_local(x, pa, epochs=0, consolidate=False, warm_start=True)


def test_flexcode_refuses_to_warm_start_into_an_auto_sharpen_search():
    """Selecting the exponent refits from scratch -- refuse, don't do it quietly."""
    x, pa = _cont()
    mech = FlexCodeMechanism(n_basis=7, hidden=(8,), epochs=3, sharpen="auto")
    mech.fit_local(x, pa)
    assert not isinstance(mech.sharpen, str), "the first fit must resolve sharpen"
    mech.sharpen = "auto"  # only reachable by resetting it by hand
    with pytest.raises(ValueError, match="sharpen"):
        mech.fit_local(x, pa, warm_start=True)


def test_hyperparameter_search_is_not_re_run_on_a_warm_second_call():
    """The resolved value is part of the state a warm start continues from."""
    x, pa = _cont()
    mech = FlexCodeMechanism(n_basis=7, hidden=(8,), epochs=3, sharpen="auto")
    mech.fit_local(x, pa)
    resolved = mech.sharpen

    calls = []
    mech._select_sharpen = lambda *a, **k: calls.append(1)  # type: ignore[method-assign]
    mech.fit_local(x, pa, epochs=0, warm_start=True)
    assert not calls, "the search re-ran and would have discarded the warm state"
    assert mech.sharpen == resolved

    kde = ConditionalKDEMechanism(bw_factor="auto")
    kde.fit_local(x, pa)
    assert not isinstance(kde.bw_factor, str)


# ==========================================================================
# 4a. Optimiser state is NOT carried over
# ==========================================================================


def test_optimiser_moments_are_not_carried_across_calls():
    """Each call builds a fresh Adam over the existing parameters.

    Adam's *first* step from zero moments has a closed form: with
    ``m_hat = g`` and ``v_hat = g^2`` after bias correction, the update is
    ``lr * |g| / (|g| + eps)``.  Two consequences hold exactly and are
    asserted here:

    * it can never exceed ``lr`` -- there is no momentum to overshoot with;
    * it sits *at* ``lr`` for every coordinate whose gradient is large
      relative to ``eps = 1e-8``, which at these scales is all of them.

    Carried-over moments obey neither.  Measured on this fixture, a step taken
    with warm moments put only 75% of coordinates within 1e-6 of ``lr`` (versus
    100% from zero moments) and overshot to 1.0012e-3 against ``lr = 1e-3``.

    This is the assertion that would fail if the other choice were
    implemented.  The choice matters because optimiser moments are not in
    ``state_dict()``: a caller reverting a rejected step via
    ``load_state_dict`` reverts theta but could not revert the moments, so a
    rejected step's momentum would survive into the retry.
    """
    lr = 1e-3
    mech, x, pa, kw = _mdn()
    mech.fit_local(x, pa, **{**kw, "epochs": 5})
    before = _snapshot(mech)

    # One full-batch epoch == exactly one optimiser step.
    mech.fit_local(
        x, pa, **{**kw, "epochs": 1}, batch_size=x.shape[0], lr=lr, warm_start=True,
    )

    moved = torch.cat([
        (mech.state_dict()[k] - v).abs().reshape(-1)
        for k, v in before.items()
        if v is not None and k.startswith("net.")
    ])
    # Slack is absolute, at the scale of a float32 ULP on a ~1.0 parameter
    # (the measured excess from rounding theta_new - theta_old is ~2e-9); a
    # tolerance stated relative to lr would be tighter than the arithmetic.
    assert float(moved.max()) <= lr + 1e-7, (
        f"a coordinate moved {float(moved.max()):.6e} > lr={lr}, which a first "
        f"Adam step from zero moments cannot do -- moments were carried over"
    )
    nonzero = moved > 1e-12
    assert int(nonzero.sum()) > moved.numel() // 2, (
        "almost every gradient was zero -- the assertions would pass vacuously"
    )
    at_lr = ((moved - lr).abs() < 1e-6) & nonzero
    frac = float(at_lr.sum()) / int(nonzero.sum())
    assert frac >= 0.95, (
        f"only {frac:.0%} of moving coordinates stepped by exactly lr; from "
        f"zero moments essentially all of them must"
    )


def test_a_second_warm_call_takes_the_same_first_step_as_the_first():
    """Corollary, stated without reference to Adam's internals.

    Two warm calls from the *same* parameter point must take the same step.
    If moments persisted, the second would differ from the first — that is
    what momentum is.
    """
    mech, x, pa, kw = _mdn()
    mech.fit_local(x, pa, **{**kw, "epochs": 5})
    anchor = _snapshot(mech)

    def _step_once():
        mech.load_state_dict(copy.deepcopy(anchor))
        mech.fit_local(
            x, pa, **{**kw, "epochs": 1},
            batch_size=x.shape[0], lr=1e-3, warm_start=True,
        )
        return {k: v.clone() for k, v in mech.state_dict().items() if v is not None}

    first, second = _step_once(), _step_once()
    for key in first:
        assert torch.equal(first[key], second[key]), (
            f"{key}: the second step differed -- optimiser state persisted"
        )


# ==========================================================================
# 4b. Standardisation buffers ARE frozen
# ==========================================================================


@pytest.mark.parametrize("name", ["mdn", "flexcode"])
def test_standardisation_buffers_freeze_under_warm_start(name):
    """The network's weights were trained against the map these buffers define.

    Recomputing them applies every learned weight to a shifted, rescaled
    input, which is a warm start in name only.  Fails if the other choice were
    implemented.
    """
    mech, x, pa, kw = GRADIENT[name]()
    mech.fit_local(x, pa, **kw)
    before = {
        k: v.clone() for k, v in mech.state_dict().items()
        if k in {"_pa_mean", "_pa_std", "_y_min", "_y_max"} and v is not None
    }
    assert before, f"{name} exposes no standardisation buffers to freeze"

    # Data on a wildly different location and scale.
    mech.fit_local(x * 20 + 500, pa * 20 + 500, **{**kw, "epochs": 0}, warm_start=True)
    for key, expected in before.items():
        assert torch.equal(mech.state_dict()[key], expected), (
            f"{name}: {key} was recomputed under warm_start=True"
        )


@pytest.mark.parametrize("name", ["mdn", "flexcode"])
def test_a_cold_fit_does_recompute_them(name):
    """The complement — otherwise the freeze test could pass on dead code."""
    mech, x, pa, kw = GRADIENT[name]()
    mech.fit_local(x, pa, **kw)
    before = {
        k: v.clone() for k, v in mech.state_dict().items()
        if k in {"_pa_mean", "_pa_std", "_y_min", "_y_max"} and v is not None
    }
    mech.fit_local(x * 20 + 500, pa * 20 + 500, **{**kw, "epochs": 0}, warm_start=False)
    assert any(
        not torch.equal(mech.state_dict()[k], v) for k, v in before.items()
    ), f"{name}: a cold refit left the standardisation buffers alone"


def test_frozen_buffers_keep_the_warm_model_coherent():
    """The reason the freeze is the right choice, not merely a choice.

    Warm-starting on reweighted data must not degrade the model on the data it
    was fitted to.  Recomputing the standardisation would shift the network's
    input distribution underneath weights trained on the old scaling, and the
    in-sample fit would fall apart even with zero further training.
    """
    mech, x, pa, kw = _mdn()
    mech.fit_local(x, pa, **{**kw, "epochs": 60})
    baseline = _nll(mech, x, pa)

    w = torch.rand(x.shape[0]) + 0.5          # a plausible E-step's responsibilities
    mech.fit_local(x, pa, **{**kw, "epochs": 5}, weights=w, warm_start=True)
    assert _nll(mech, x, pa) <= baseline + 0.2


# ==========================================================================
# 5. Closed-form branches: accepted, ignored, and still tracking the E-step
# ==========================================================================


@pytest.mark.parametrize("name", sorted(CLOSED_FORM))
def test_warm_start_is_accepted_on_closed_form_branches(name):
    """Uniform ``warm_start=True`` across a network must be defined everywhere."""
    mech, x, pa, kw = CLOSED_FORM[name]()
    mech.fit_local(x, pa, **kw)
    info = mech.fit_local(x, pa, **kw, warm_start=True)
    assert info["warm_started"] is False, (
        f"{name} claims to have carried parameters over; its fit is "
        f"initialisation-independent, so it must report False"
    )


@pytest.mark.parametrize("name", sorted(CLOSED_FORM))
def test_closed_form_warm_and_cold_agree_on_the_same_data(name):
    """Recomputing an exact maximiser *is* the continuation."""
    mech, x, pa, kw = CLOSED_FORM[name]()
    mech.fit_local(x, pa, **kw)
    mech.fit_local(x, pa, **kw, warm_start=True)
    warm = _snapshot(mech)

    other, x2, pa2, kw2 = CLOSED_FORM[name]()
    other.fit_local(x2, pa2, **kw2)
    other.fit_local(x2, pa2, **kw2, warm_start=False)
    assert _agrees(other, warm)


@pytest.mark.parametrize(
    "name",
    ["linear_gaussian", "categorical_table", "smoothed_empirical",
     "mdn_root", "neural_categorical_root", "flexcode_root"],
)
def test_closed_form_branches_still_respond_to_changed_weights(name):
    """The trap this design avoids.

    If ``warm_start=True`` froze a closed-form branch, it would stop tracking
    the E-step: a silent, permanent M-step failure on exactly the nodes nobody
    inspects.  The root branches of MDN, neural-categorical and FlexCode are
    the ones at risk, because their *classes* are gradient-trained.
    """
    mech, x, pa, kw = CLOSED_FORM[name]()
    mech.fit_local(x, pa, **kw)
    unweighted = _snapshot(mech)

    torch.manual_seed(4)
    w = torch.rand(x.shape[0]) * 3.0 + 0.1
    mech.fit_local(x, pa, **kw, weights=w, warm_start=True)
    assert not _same(mech, unweighted), (
        f"{name}: a reweighted warm M-step did not move the parameters"
    )

    # And it lands exactly where the equivalent cold weighted fit lands.
    cold, x2, pa2, kw2 = CLOSED_FORM[name]()
    cold.fit_local(x2, pa2, **kw2, weights=w, warm_start=False)
    assert _agrees(cold, _snapshot(mech))


def test_the_capability_flag_matches_the_implementation():
    for name, factory in GRADIENT.items():
        mech, _, _, _ = factory()
        assert mech.warm_start_is_noop is False, name
    for name in ("linear_gaussian", "categorical_table", "smoothed_empirical",
                 "binning", "kde", "knn", "deterministic", "dirac"):
        mech, _, _, _ = CLOSED_FORM[name]()
        assert mech.warm_start_is_noop is True, name


# ==========================================================================
# The never-fitted case, and the new aliasing hazard
# ==========================================================================


@pytest.mark.parametrize("name", sorted(ALL))
def test_warm_start_on_an_unfitted_mechanism_builds_cold_and_says_so(name):
    """Nothing is being discarded, so this must not raise — but must be visible."""
    mech, x, pa, kw = ALL[name]()
    assert not mech.is_fitted or name in {"deterministic", "dirac"}
    info = mech.fit_local(x, pa, **kw, warm_start=True)
    assert info["warm_started"] is False
    if name != "dirac":
        # DiracGaussianMechanism is a do-intervention CPD with nothing to fit,
        # and does not override the base is_fitted.
        assert mech.is_fitted


def test_uncopied_snapshot_is_clobbered_by_a_warm_fit():
    """Warm-starting widens the reach of the PR #258 deepcopy requirement.

    A *cold* refit allocates new parameter tensors, so an uncopied
    ``state_dict()`` accidentally survived a subsequent ``fit_local``.  Under
    ``warm_start=True`` the objects persist and the fit mutates them in place,
    so the snapshot is clobbered by the very call it was taken to undo — and
    a backtracking revert would restore the already-stepped values.  Nothing
    raises.  Pinned as a trap, alongside the correct idiom.
    """
    mech, x, pa, kw = _mdn()
    mech.fit_local(x, pa, **{**kw, "epochs": 5})

    naive = mech.state_dict()                 # aliases the live parameters
    correct = copy.deepcopy(mech.state_dict())
    mech.fit_local(x, pa, **{**kw, "epochs": 5}, warm_start=True)

    assert torch.equal(naive["net.0.weight"], mech.state_dict()["net.0.weight"]), (
        "the uncopied snapshot did not track the live parameters -- if this "
        "starts passing the trap is gone and this test should be retired"
    )
    assert not torch.equal(correct["net.0.weight"], mech.state_dict()["net.0.weight"])

    # The correct idiom reverts; the naive one is a silent no-op.
    mech.load_state_dict(correct)
    assert _same(mech, correct)


def test_the_same_uncopied_snapshot_survived_a_cold_refit():
    """Why the hazard is new rather than pre-existing."""
    mech, x, pa, kw = _mdn()
    mech.fit_local(x, pa, **{**kw, "epochs": 5})
    naive = mech.state_dict()
    mech.fit_local(x, pa, **{**kw, "epochs": 5}, warm_start=False)
    assert not torch.equal(naive["net.0.weight"], mech.state_dict()["net.0.weight"])


# ==========================================================================
# Network level
# ==========================================================================


def _mixed_network():
    """A gradient-trained node, a closed-form continuous node, and a CPT.

    The discrete pair is its own component: a CategoricalTable child of a
    *continuous* parent is not a well-posed CPD (the parent values index the
    table), so mixing them in one chain would be testing a modelling error
    rather than warm-starting.
    """
    m = NBN(
        [("A", "B"), ("D", "E")],
        variables={"A": ("continuous", 1), "B": ("continuous", 1),
                   "D": ("discrete", 3), "E": ("discrete", 2)},
        device="cpu",
    )
    m.auto_mechanisms()
    m.set_mechanism("B", MDNMechanism(num_components=2, hidden=(8,)))
    return m


def _mixed_data(n=200, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(n)
    b = 2.0 * a + 0.1 * torch.randn(n)
    d = torch.randint(0, 3, (n,))
    e = (d % 2).long()
    return {"A": a, "B": b, "D": d, "E": e}


def test_model_fit_threads_warm_start_to_every_node():
    """Uniform warm_start=True across a mixed network: no raise, nothing lost."""
    model = _mixed_network()
    data = _mixed_data()
    model.fit(data, epochs=5, consolidate=False)
    snap = _snapshot(model)

    model.fit(data, epochs=0, consolidate=False, warm_start=True)
    assert _same(model, snap), "a node was rebuilt despite warm_start=True"

    model.fit(data, epochs=0, consolidate=False, warm_start=False)
    assert not _same(model, snap)


def test_model_fit_default_is_unchanged():
    import inspect
    sig = inspect.signature(NBN.fit)
    assert sig.parameters["warm_start"].default is False


def test_warm_start_composes_with_per_sample_weights():
    """The downstream call shape: an EM outer loop over reweighted M-steps."""
    model = _mixed_network()
    data = _mixed_data()
    model.fit(data, epochs=30, consolidate=False)

    def _joint_nll(w):
        with torch.no_grad():
            lp = model.log_prob(data)
        return float(-(w * lp).sum() / w.sum())

    torch.manual_seed(7)
    w = torch.rand(200) + 0.25
    before = _joint_nll(w)
    for _ in range(3):
        model.fit(data, epochs=10, consolidate=False, weights=w, warm_start=True)
    assert _joint_nll(w) <= before + 0.05
