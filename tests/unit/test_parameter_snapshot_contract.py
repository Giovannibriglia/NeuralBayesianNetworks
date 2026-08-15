"""Snapshot/restore of mechanism parameters via ``state_dict`` round-trips exactly.

Callers running a monotone generalized-EM loop need to *reject* an M-step:
evaluate the objective, and if it went the wrong way, put the parameters back
and retry with a smaller step.  ``state_dict()`` / ``load_state_dict()`` is the
supported idiom for that — no NBN-specific API exists or is needed — but it is
only correct with one precaution, pinned below.

**The snapshot must be copied.**  ``state_dict()`` returns tensors that *share
storage* with the live parameters, and optimisers update parameters in place.
So the obvious spelling::

    snap = mech.state_dict()      # aliases the parameters
    opt.step()                    # mutates them -- and the snapshot with them
    mech.load_state_dict(snap)    # restores the ALREADY-STEPPED values

silently does nothing.  Nothing raises, the parameters simply fail to revert,
and a backtracking M-step would accept every step it meant to reject —
observable only as an EM that does not converge properly.  This is the same
class of silent failure as a detached parent tensor, so it is tested the same
way: both the correct idiom and the trap.

The correct spelling is ``copy.deepcopy(mech.state_dict())`` (or a dict of
``.clone()``d tensors).
"""
from __future__ import annotations

import copy

import pytest
import torch

from nbn import NeuralBayesianNetwork as NBN
from nbn.mechanisms import (
    CategoricalTableMechanism,
    LinearGaussianMechanism,
    MDNMechanism,
    NeuralCategoricalMechanism,
)


def _snapshot(module: torch.nn.Module) -> dict:
    """The supported idiom: a *copied* state_dict."""
    return copy.deepcopy(module.state_dict())


def _fitted(kind: str, *, epochs: int = 3):
    """Return a fitted mechanism plus the data it was fitted on."""
    torch.manual_seed(0)
    if kind == "linear_gaussian":
        pa = torch.randn(200, 1)
        x = 2.0 * pa + 0.1 * torch.randn(200, 1)
        mech = LinearGaussianMechanism()
        mech.fit_local(x, pa)
    elif kind == "mdn":
        pa = torch.randn(200, 1)
        x = 2.0 * pa + 0.1 * torch.randn(200, 1)
        mech = MDNMechanism(num_components=2, hidden=(8,))
        mech.fit_local(x, pa, epochs=epochs, consolidate=False)
    elif kind == "categorical":
        pa = torch.randint(0, 3, (200, 1))
        x = torch.randint(0, 2, (200,))
        mech = CategoricalTableMechanism()
        mech.fit_local(x, pa, parent_cards=[3], n_classes=2)
    elif kind == "neural_categorical":
        pa = torch.randn(200, 1)
        x = torch.randint(0, 2, (200,))
        mech = NeuralCategoricalMechanism(n_classes=2, hidden=(8,))
        mech.fit_local(x, pa, epochs=epochs, consolidate=False)
    else:  # pragma: no cover
        raise AssertionError(kind)
    return mech, x, pa


_KINDS = ["linear_gaussian", "mdn", "categorical", "neural_categorical"]


# --------------------------------------------------------------------------
# The round-trip itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _KINDS)
def test_state_dict_round_trips_every_tensor_bitwise(kind):
    mech, _, _ = _fitted(kind)
    snap = _snapshot(mech)
    assert snap, "nothing was captured -- the snapshot idiom would be a no-op"

    # Perturb everything, then restore.
    with torch.no_grad():
        for p in mech.parameters():
            p.add_(1.0)
    mech.load_state_dict(snap)

    restored = mech.state_dict()
    assert set(restored) == set(snap)
    for key, expected in snap.items():
        assert torch.equal(restored[key], expected), f"{key} did not round-trip"


@pytest.mark.parametrize("kind", ["mdn", "neural_categorical"])
def test_round_trip_holds_after_a_partial_fit(kind):
    """An interrupted / single-epoch fit is exactly the GEM backtracking state."""
    mech, _, _ = _fitted(kind, epochs=1)
    snap = _snapshot(mech)
    with torch.no_grad():
        for p in mech.parameters():
            p.mul_(-2.0)
    mech.load_state_dict(snap)
    for key, expected in snap.items():
        assert torch.equal(mech.state_dict()[key], expected)


@pytest.mark.parametrize("kind", _KINDS)
def test_snapshot_restores_into_a_separate_instance(kind):
    """Backtracking may restore into a fresh object rather than in place."""
    mech, _, _ = _fitted(kind)
    snap = _snapshot(mech)
    other, _, _ = _fitted(kind)
    with torch.no_grad():
        for p in other.parameters():
            p.add_(0.5)
    other.load_state_dict(snap)
    for key, expected in snap.items():
        assert torch.equal(other.state_dict()[key], expected)


# --------------------------------------------------------------------------
# The M-step rejection this exists for
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["linear_gaussian", "mdn", "neural_categorical"])
def test_a_rejected_optimiser_step_reverts_exactly(kind):
    """Snapshot, step, reject, restore: parameters must be pre-step values."""
    mech, x, pa = _fitted(kind)
    before = {k: v.clone() for k, v in mech.state_dict().items()}
    snap = _snapshot(mech)

    opt = torch.optim.SGD(mech.parameters(), lr=0.5)
    loss = -mech.log_prob(x, pa).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    moved = any(
        not torch.equal(mech.state_dict()[k], v) for k, v in before.items()
    )
    assert moved, "the step changed nothing; the test would prove nothing"

    mech.load_state_dict(snap)
    for key, expected in before.items():
        assert torch.equal(mech.state_dict()[key], expected), (
            f"{key} was not reverted -- a rejected M-step would be silently kept"
        )


def test_bare_state_dict_aliases_the_parameters_and_fails_to_revert():
    """The trap the copy exists to avoid, pinned so the docs stay grounded.

    If a future torch release makes ``state_dict()`` copy, this test fails and
    the guidance can be relaxed deliberately rather than drifting.
    """
    mech, x, pa = _fitted("linear_gaussian")
    before = mech._weight.detach().clone()

    naive = mech.state_dict()  # NOT copied -- aliases the live parameters
    assert any(
        v.data_ptr() == p.data_ptr()
        for v in naive.values()
        for p in mech.parameters()
    ), "state_dict no longer aliases parameters; the deepcopy advice can be revisited"

    opt = torch.optim.SGD(mech.parameters(), lr=0.5)
    opt.zero_grad()
    (-mech.log_prob(x, pa).mean()).backward()
    opt.step()
    mech.load_state_dict(naive)

    assert not torch.equal(mech._weight.detach(), before), (
        "the bare snapshot reverted correctly -- if torch changed, update the docs"
    )


# --------------------------------------------------------------------------
# Scope of the contract
# --------------------------------------------------------------------------


def test_buffers_round_trip_alongside_parameters():
    """LinearGaussian's normal equations and MDN's standardisation stats.

    These are buffers, not parameters, and a partial restore that dropped them
    would leave the mechanism internally inconsistent.
    """
    mech, _, _ = _fitted("linear_gaussian")
    snap = _snapshot(mech)
    assert {"_neq_A", "_neq_B", "_neq_c", "_neq_N"} <= set(snap)

    mdn, _, _ = _fitted("mdn")
    assert {"_pa_mean", "_pa_std"} <= set(_snapshot(mdn))


def test_structural_attributes_are_outside_the_snapshot():
    """Documented boundary: state_dict holds tensors, not fit-time structure.

    ``_parent_cards`` / ``_n_classes`` describe the CPT's shape, are set once
    at fit time, and are not touched by an optimiser step — so the idiom is
    sound for M-step backtracking even though it does not capture them.
    """
    mech, _, _ = _fitted("categorical")
    snap = _snapshot(mech)
    assert "_logits" in snap and "_counts" in snap
    assert "_parent_cards" not in snap and "_n_classes" not in snap


def test_whole_model_snapshot_round_trips():
    """Backtracking a joint M-step operates on the model, not one mechanism."""
    torch.manual_seed(0)
    model = NBN(
        [("X", "Y")],
        variables={"X": ("continuous", 1), "Y": ("continuous", 1)},
        device="cpu",
    )
    model.set_mechanism("X", LinearGaussianMechanism())
    model.set_mechanism("Y", LinearGaussianMechanism())
    x = torch.randn(300, 1)
    model.fit({"X": x, "Y": 2.0 * x + 0.1 * torch.randn(300, 1)})

    snap = _snapshot(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    model.load_state_dict(snap)
    for key, expected in snap.items():
        assert torch.equal(model.state_dict()[key], expected), key
