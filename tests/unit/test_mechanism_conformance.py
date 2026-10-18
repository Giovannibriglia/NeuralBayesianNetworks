"""Every exported mechanism honours the ``Mechanism`` design contract.

``nbn/mechanisms/base.py`` documents, for all mechanisms:

* ``forward(parents)`` returns a Distribution whose ``batch_shape`` starts
  with ``[B]`` (``[B, S]`` for 3-D particle parents ``[B, S, D_pa]``);
* ``log_prob(x, parents)`` -> ``[B]`` or ``[B, S]``;
* ``sample(parents, n)`` -> ``[B, n, D_x]``; root nodes take ``parents=None``;
* ``rsample`` either works or raises ``RuntimeError`` (never anything else);
* ``is_fitted`` is False before ``fit_local`` and True after;
* discrete mechanisms ``tabulate`` to logits whose softmax rows sum to 1;
* every parameter / buffer follows ``.to(device)``.

These used to be checked class by class (test_mechanisms.py and
test_nonparametric_mechanisms.py repeated the 2-D shape checks for a few
classes; the 3-D particle shape, rsample, is_fitted and device move were
checked for some mechanisms only).  The registry below is cross-checked
against ``nbn.mechanisms.__all__`` so a newly exported mechanism cannot
escape the contract.
"""
from __future__ import annotations

import pytest
import torch

import nbn.mechanisms as mechanisms
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
    SparseCategoricalTableMechanism,
)
from tests.conftest import available_devices

N, B, S, NS = 160, 5, 3, 4


def _cont(d_pa=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    pa = torch.randn(N, d_pa, generator=g)
    x = pa.sum(-1, keepdim=True) + 0.3 * torch.randn(N, 1, generator=g)
    return x, pa


def _disc(seed=0):
    g = torch.Generator().manual_seed(seed)
    pa = torch.randint(0, 3, (N, 1), generator=g)
    x = (pa[:, 0] + torch.randint(0, 2, (N,), generator=g)) % 2
    return x, pa


def _flow(cls_name):
    def make():
        zuko_mod = pytest.importorskip("zuko")  # noqa: F841
        cls = getattr(mechanisms, cls_name)
        x, pa = _cont()
        return cls(num_transforms=1, hidden=(8,)), x, pa, {"epochs": 2}
    return make


#: name -> factory returning (unfitted mechanism, x, parents, fit kwargs).
#: Small budgets: the contract is about shapes and bookkeeping, not fit quality.
_REGISTRY = {
    "CategoricalTableMechanism": lambda: (
        CategoricalTableMechanism(), *_disc(), {"parent_cards": [3], "n_classes": 2}),
    "SparseCategoricalTableMechanism": lambda: (
        SparseCategoricalTableMechanism(), *_disc(), {"parent_cards": [3], "n_classes": 2}),
    "SmoothedEmpiricalCategoricalMechanism": lambda: (
        SmoothedEmpiricalCategoricalMechanism(), *_disc(),
        {"parent_cards": [3], "n_classes": 2}),
    "BinningCategoricalTable": lambda: (
        BinningCategoricalTable(parent_kinds=[("continuous", 0), ("continuous", 0)],
                                n_bins=3, n_categories=2),
        (_cont()[1][:, 0] > 0).long(), _cont()[1], {}),
    "NeuralCategoricalMechanism": lambda: (
        NeuralCategoricalMechanism(n_classes=2, hidden=(8,)), *_disc(),
        {"epochs": 2, "consolidate": False}),
    "LinearGaussianMechanism": lambda: (LinearGaussianMechanism(), *_cont(), {}),
    "MDNMechanism": lambda: (
        MDNMechanism(num_components=2, hidden=(8,)), *_cont(),
        {"epochs": 2, "consolidate": False}),
    "ConditionalKDEMechanism": lambda: (ConditionalKDEMechanism(), *_cont(), {}),
    "KNNConditionalMechanism": lambda: (KNNConditionalMechanism(k=5), *_cont(), {}),
    "KNNConditionalMechanism[discrete_child]": lambda: (
        KNNConditionalMechanism(k=5, discrete_child=True), *_disc(),
        {"parent_cards": [3], "n_classes": 2}),
    "FlexCodeMechanism": lambda: (
        FlexCodeMechanism(n_basis=7, hidden=(8,), epochs=2), *_cont(), {}),
    "DeterministicMechanism": lambda: (
        DeterministicMechanism(value=torch.tensor([1.0])), *_cont(), {}),
    "DiracGaussianMechanism": lambda: (DiracGaussianMechanism(value=1.0), *_cont(), {}),
    "NormalizingFlowMechanism": _flow("NormalizingFlowMechanism"),
    "ConditionalFlowMechanism": _flow("ConditionalFlowMechanism"),
}

#: Constant CPDs (do-operator), specified at construction: always fitted.
_ALWAYS_FITTED = {"DeterministicMechanism", "DiracGaussianMechanism"}

NAMES = sorted(_REGISTRY)


def test_registry_covers_every_exported_mechanism():
    exported = {n for n in mechanisms.__all__ if n != "Mechanism"}
    assert exported - set(_REGISTRY) == set(), "exported mechanism without a contract entry"


def _fitted(name):
    mech, x, pa, kw = _REGISTRY[name]()
    mech.fit_local(x, pa, **kw)
    return mech, x, pa


def _x2d(mech, x):
    """Child values as ``[rows, D_x]`` (discrete children are ``[rows, 1]``)."""
    return x.reshape(x.shape[0], -1).to(torch.float32)


#: Known contract violations, kept visible as strict xfails (an xpass fails).
#: Empty: DiracGaussianMechanism.is_fitted was fixed (always fitted, like
#: DeterministicMechanism).
_KNOWN_VIOLATIONS: dict = {}


def _maybe_xfail(contract, names):
    return [pytest.param(n, marks=pytest.mark.xfail(
                strict=True, reason=_KNOWN_VIOLATIONS[(contract, n)]))
            if (contract, n) in _KNOWN_VIOLATIONS else n
            for n in names]


@pytest.mark.parametrize("name", _maybe_xfail("is_fitted", NAMES))
def test_is_fitted_flips_on_fit(name):
    mech, x, pa, kw = _REGISTRY[name]()
    assert mech.is_fitted is (name in _ALWAYS_FITTED)
    mech.fit_local(x, pa, **kw)
    assert mech.is_fitted is True


@pytest.mark.parametrize("name", NAMES)
def test_two_dim_shapes(name):
    mech, x, pa = _fitted(name)
    pa_b = pa[:B].to(torch.float32)
    dist = mech.forward(pa_b)
    assert tuple(dist.batch_shape)[:1] == (B,)
    lp = mech.log_prob(_x2d(mech, x)[:B], pa_b)
    assert lp.shape == (B,) and torch.isfinite(lp).all()
    s = mech.sample(pa_b, n=NS)
    assert s.shape == (B, NS, mech.output_dim)


@pytest.mark.parametrize("name", NAMES)
def test_particle_dim_shapes(name):
    """3-D parents ``[B, S, D_pa]`` (the LW / ancestral-sampling layout)."""
    mech, x, pa = _fitted(name)
    pa3 = pa[: B * S].to(torch.float32).reshape(B, S, -1)
    x3 = _x2d(mech, x)[: B * S].reshape(B, S, -1)
    lp = mech.log_prob(x3, pa3)
    assert lp.shape == (B, S) and torch.isfinite(lp).all()


@pytest.mark.parametrize("name", NAMES)
def test_rsample_works_or_raises_runtime_error(name):
    mech, _, pa = _fitted(name)
    try:
        s = mech.rsample(pa[:B].to(torch.float32), n=2)
    except RuntimeError:
        return
    assert s.shape == (B, 2, mech.output_dim)


@pytest.mark.parametrize("name", [n for n in NAMES
                                  if n != "DeterministicMechanism"
                                  and _REGISTRY[n]()[0].is_discrete])
def test_discrete_tabulation_is_a_normalised_cpt(name):
    mech, _, _ = _fitted(name)
    table = mech.tabulate(parent_cards=[3])
    probs = torch.softmax(table, dim=-1)
    assert torch.isfinite(probs).all()
    torch.testing.assert_close(probs.sum(-1), torch.ones(probs.shape[:-1]),
                               atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("name", NAMES)
def test_to_device_moves_every_tensor(name, device):
    mech, x, pa = _fitted(name)
    mech.to(device)
    want = torch.device(device).type
    for key, t in list(mech.named_parameters()) + list(mech.named_buffers()):
        assert t.device.type == want, f"{name}.{key} stayed on {t.device}"
    lp = mech.log_prob(_x2d(mech, x)[:B].to(device), pa[:B].to(torch.float32).to(device))
    assert lp.device.type == want and torch.isfinite(lp).all()


# ---- capability flags vs implementation ---------------------------------------
# ``supports_update`` / ``supports_weights`` are read by nbn.update / nbn.learning
# *instead of* calling the method and seeing what happens, because every
# fit_local / update_local ends in **kwargs and would silently swallow an
# unsupported keyword.  So each flag must match what the mechanism does.
# (``warm_start_is_noop`` is pinned per mechanism in test_warm_start_contract.py.)

#: Constant CPDs: nothing is estimated from data, so weights cannot bite.
_CONSTANT = {"DeterministicMechanism", "DiracGaussianMechanism"}


@pytest.mark.parametrize("name", NAMES)
def test_supports_update_flag_matches_update_local(name):
    mech, x, pa, kw = _REGISTRY[name]()
    if kw.get("consolidate") is False:
        kw = {**kw, "consolidate": True}   # the EWC update needs the snapshot
    mech.fit_local(x, pa, **kw)
    if type(mech).supports_update:
        info = mech.update_local(x[:40], pa[:40])
        assert isinstance(info, dict)
        lp = mech.log_prob(_x2d(mech, x)[:B], pa[:B].to(torch.float32))
        assert torch.isfinite(lp).all()
    else:
        # nbn.update must skip the node rather than call update_local.
        from nbn import NeuralBayesianNetwork as NBN

        kind = ("discrete", 2) if mech.is_discrete else ("continuous", 1)
        model = NBN([], variables={"X": kind}, device="cpu")
        model.set_mechanism("X", mech)
        hist = model.update({"X": x[:40] if mech.is_discrete else _x2d(mech, x)[:40]})
        assert hist.skipped == ["X"], hist


@pytest.mark.parametrize("name", [n for n in NAMES if n not in _CONSTANT])
def test_supports_weights_flag_matches_fit_local(name):
    """True => the weights change the fit (FlexCode-root once claimed support
    and silently fitted unweighted); False => fit_local refuses them loudly."""
    mech, x, pa, kw = _REGISTRY[name]()
    w = torch.zeros(x.shape[0])
    w[: x.shape[0] // 4] = 1.0                       # keep a quarter of the rows
    if not type(mech).supports_weights:
        with pytest.raises(NotImplementedError):
            mech.fit_local(x, pa, weights=w, **kw)
        return
    torch.manual_seed(0)
    mech.fit_local(x, pa, weights=w, **kw)
    other, *_ = _REGISTRY[name]()
    torch.manual_seed(0)
    other.fit_local(x, pa, **kw)
    probe_x, probe_pa = _x2d(mech, x)[-B:], pa[-B:].to(torch.float32)
    assert not torch.allclose(mech.log_prob(probe_x, probe_pa),
                              other.log_prob(probe_x, probe_pa), atol=1e-4), (
        f"{name} advertises supports_weights but the weighted fit equals the unweighted one")
