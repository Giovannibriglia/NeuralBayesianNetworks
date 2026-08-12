"""fit() training-budget plumbing + opt-out EWC consolidation (PR A).

Two regressions pinned here:

1. ``nbn.learning.fit.fit`` used ``setdefault("epochs"/"lr"/"batch_size", ...)``
   with *function-default* values, so the keys were always present and every
   mechanism's designed budget (flow 300 epochs @ lr 5e-4, MDN 200,
   neural-categorical 100) was silently flattened to the one global default.
   ``None`` (the new default) now means "no override": the mechanism keeps its
   own budget; explicit values still override globally.

2. ``online_laplace.consolidate`` ran unconditionally at the end of every
   neural ``fit_local`` — up to ``sample_cap`` (4096) sequential per-sample
   backward passes per node — even for fit-only workloads that never call
   ``model.update()``.  ``consolidate=False`` now skips it; ``update()`` on
   such a model raises an informative ``RuntimeError``.

Models are pinned to CPU so behaviour matches the CI runners exactly.
"""
import pytest
import torch

from nbn import NeuralBayesianNetwork as NBN
from nbn.mechanisms.parametric.mdn import MDNMechanism
from nbn.mechanisms.parametric.neural_categorical import NeuralCategoricalMechanism


def _tiny_mdn_model(seed: int = 0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(64, 1, generator=g)
    model = NBN([], {"X": ("continuous", 1)}, device="cpu")
    model.set_mechanism("X", MDNMechanism(num_components=2, hidden=(8,)))
    return model, {"X": x}


def _tiny_neuralcat_model(seed: int = 0):
    torch.manual_seed(seed)
    n = 128
    a = torch.randint(0, 3, (n,)).float()
    b = (a.long() % 2).float()  # B depends on A
    model = NBN([("A", "B")], {"A": ("discrete", 3), "B": ("discrete", 2)},
                device="cpu")
    model.set_mechanism("A", NeuralCategoricalMechanism(n_classes=3))
    model.set_mechanism("B", NeuralCategoricalMechanism(n_classes=2, hidden=(8,)))
    return model, {"A": a, "B": b}


# ── 1. budget plumbing: None = mechanism-designed default ────────────────────


class TestMechanismBudgetRespected:
    def _fit_recording(self, monkeypatch, **fit_kwargs):
        """Fit a tiny root MDN, spying on the kwargs fit_local receives."""
        model, data = _tiny_mdn_model()
        mech = model.mechanisms["X"]
        recorded: dict = {}
        real = mech.fit_local

        def spy(x, parents, **kw):
            recorded.update(kw)
            kw["epochs"] = 2  # keep actual training tiny; asserts use `recorded`
            return real(x, parents, **kw)

        monkeypatch.setattr(mech, "fit_local", spy)
        model.fit(data, **fit_kwargs)
        return recorded

    def test_default_fit_sends_no_budget_override(self, monkeypatch):
        # The setdefault bug made these keys ALWAYS present (epochs=100 etc.);
        # without explicit values the mechanism must now keep its own budget.
        kw = self._fit_recording(monkeypatch)
        assert "epochs" not in kw
        assert "lr" not in kw
        assert "batch_size" not in kw

    def test_explicit_budget_reaches_mechanism(self, monkeypatch):
        kw = self._fit_recording(monkeypatch, epochs=7, lr=5e-4, batch_size=32)
        assert kw["epochs"] == 7
        assert kw["lr"] == 5e-4
        assert kw["batch_size"] == 32

    def test_consolidate_flag_is_threaded(self, monkeypatch):
        assert self._fit_recording(monkeypatch)["consolidate"] is True
        assert self._fit_recording(
            monkeypatch, consolidate=False)["consolidate"] is False

    def test_mechanism_class_defaults_survive_unfitted(self):
        # Sanity anchor for the designed budgets this fix protects.
        import inspect
        mdn_defaults = {
            k: v.default
            for k, v in inspect.signature(MDNMechanism.fit_local).parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        assert mdn_defaults["epochs"] == 200
        assert mdn_defaults["batch_size"] == 512


# ── 2. consolidate=False skips the Fisher pass ───────────────────────────────


class TestConsolidateOptOut:
    def test_mdn_consolidate_false_skips_fisher_and_fit_succeeds(self):
        model, data = _tiny_mdn_model()
        model.fit(data, epochs=3, batch_size=32, consolidate=False)
        mech = model.mechanisms["X"]
        assert mech._ewc_mu is None
        assert mech._ewc_fisher is None
        with torch.no_grad():
            lp = mech.log_prob(data["X"], None)
        assert torch.isfinite(lp).all()

    def test_mdn_consolidate_default_populates_fisher(self):
        model, data = _tiny_mdn_model()
        model.fit(data, epochs=3, batch_size=32)
        mech = model.mechanisms["X"]
        assert mech._ewc_mu is not None
        assert mech._ewc_fisher is not None
        assert torch.isfinite(mech._ewc_fisher).all()

    def test_neuralcat_consolidate_false_root_and_nonroot(self):
        model, data = _tiny_neuralcat_model()
        model.fit(data, epochs=3, batch_size=64, consolidate=False)
        for node in ("A", "B"):  # A = root closed-form path, B = MLP path
            mech = model.mechanisms[node]
            assert mech._ewc_mu is None, node
            assert mech._ewc_fisher is None, node

    def test_update_after_consolidate_false_raises_informative(self):
        model, data = _tiny_mdn_model()
        model.fit(data, epochs=3, batch_size=32, consolidate=False)
        g = torch.Generator().manual_seed(1)
        with pytest.raises(RuntimeError, match="consolidate=True"):
            model.update({"X": torch.randn(32, 1, generator=g)},
                         epochs=2, batch_size=16)

    def test_update_after_default_fit_still_works(self):
        model, data = _tiny_mdn_model()
        model.fit(data, epochs=3, batch_size=32)  # consolidate defaults True
        g = torch.Generator().manual_seed(1)
        hist = model.update({"X": torch.randn(32, 1, generator=g)},
                            epochs=2, batch_size=16)
        assert hist.node_methods["X"] == "online_ewc"
