"""Vectorised per-sample Fisher (``online_laplace._per_sample_fisher_vmap``).

``consolidate`` runs at the end of every neural ``fit_local``; at the default
``fisher_batch_size=1`` it used to be ``sample_cap`` sequential backward
passes, which dominated ``fit()`` on cuda / mps.  The vmap pass must compute
the *same* estimator — the mean of per-sample squared NLL gradients — and fall
back to the loop for a mechanism vmap cannot trace.
"""
import pytest
import torch

from nbn.mechanisms.parametric.mdn import MDNMechanism
from nbn.mechanisms.parametric.neural_categorical import NeuralCategoricalMechanism
from nbn.update import online_laplace as ol
from nbn.utils.batching import _sanitise_parents, assume_finite_parents
from tests.conftest import available_devices


def _loop_fisher(mech, x, parents):
    """The reference estimator: one backward pass per row."""
    params = ol._trainable_params(mech)
    fisher = [torch.zeros_like(p) for p in params]
    mech.eval()
    for i in range(x.shape[0]):
        pb = None if parents is None else parents[i:i + 1]
        grads = torch.autograd.grad(
            -mech.log_prob(x[i:i + 1], pb).sum(), params, allow_unused=True,
        )
        for f, g in zip(fisher, grads):
            if g is not None:
                f.add_(g.pow(2))
    return [f / x.shape[0] for f in fisher]


def _case(kind, root, device):
    g = torch.Generator().manual_seed(0)
    pa = None if root else torch.randn(300, 2, generator=g)
    if kind == "mdn":
        mech = MDNMechanism()
        x = torch.randn(300, 1, generator=g) + (0 if root else pa[:, :1] ** 2)
    else:
        mech = NeuralCategoricalMechanism(n_classes=3)
        x = torch.randint(0, 3, (300,), generator=g)
    x = x.to(device)
    pa = None if pa is None else pa.to(device)
    mech.to(device)
    mech.fit_local(x, pa, epochs=2, consolidate=False)
    return mech, x, pa


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("root", [False, True], ids=["child", "root"])
@pytest.mark.parametrize("kind", ["mdn", "neural_categorical"])
def test_vmap_fisher_matches_per_sample_loop(kind, root, device):
    mech, x, pa = _case(kind, root, device)
    want = _loop_fisher(mech, x, pa)
    got = ol._per_sample_fisher_vmap(mech, x, pa)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.shape == w.shape and g.device == w.device
        torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-7)


def test_estimate_fisher_uses_vmap_by_default(monkeypatch):
    mech, x, pa = _case("mdn", False, "cpu")
    calls = []
    real = ol._per_sample_fisher_vmap
    monkeypatch.setattr(
        ol, "_per_sample_fisher_vmap",
        lambda *a: calls.append(1) or real(*a),
    )
    ol._estimate_fisher(mech, x, pa, fisher_batch_size=1, sample_cap=4096)
    assert calls == [1]


def test_estimate_fisher_falls_back_to_loop_when_vmap_fails(monkeypatch):
    mech, x, pa = _case("mdn", False, "cpu")

    def _boom(*a):
        raise RuntimeError("data-dependent control flow")

    monkeypatch.setattr(ol, "_per_sample_fisher_vmap", _boom)
    _, got = ol._estimate_fisher(mech, x, pa, fisher_batch_size=1, sample_cap=4096)
    for g, w in zip(got, _loop_fisher(mech, x, pa)):
        torch.testing.assert_close(g, w)


def test_vmap_fisher_restores_mode_and_validation():
    mech, x, pa = _case("mdn", False, "cpu")
    mech.train()
    before = torch.distributions.Distribution._validate_args
    ol._per_sample_fisher_vmap(mech, x, pa)
    assert mech.training
    assert torch.distributions.Distribution._validate_args == before


def test_vmap_fisher_sanitises_non_finite_parents():
    mech, x, pa = _case("mdn", False, "cpu")
    bad = pa.clone()
    bad[3, 0] = float("nan")
    got = ol._per_sample_fisher_vmap(mech, x, bad)
    assert all(torch.isfinite(f).all() for f in got)
    clean = bad.clone()
    clean[3, 0] = 0.0
    for g, w in zip(got, ol._per_sample_fisher_vmap(mech, x, clean)):
        torch.testing.assert_close(g, w)


def test_assume_finite_parents_is_scoped():
    bad = torch.tensor([[float("nan")]])
    with assume_finite_parents():
        assert torch.isnan(_sanitise_parents(bad)).all()
    assert _sanitise_parents(bad).item() == 0.0
