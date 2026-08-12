"""Online-EWC incremental update for the neural mechanisms (PR2).

The neural updaters are *approximate* (unlike PR1's exact ones): training on new
data under a Fisher-weighted quadratic anchor to the post-fit weights should fold
in the new regime while *mitigating* catastrophic forgetting of the old one.
These tests pin that behaviour plus the state-plumbing, orchestration, and
online-refresh contracts.  Models are pinned to CPU so behaviour matches the CI
runners exactly.
"""
import copy

import pytest
import torch

from nbn import NeuralBayesianNetwork as NBN
from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism
from nbn.mechanisms.parametric.deterministic import DeterministicMechanism
from nbn.mechanisms.parametric.mdn import MDNMechanism
from nbn.mechanisms.parametric.neural_categorical import NeuralCategoricalMechanism


def _gauss(mu, n, seed):
    g = torch.Generator().manual_seed(seed)
    return mu + 0.4 * torch.randn(n, 1, generator=g)


# ── core claim: EWC mitigates catastrophic forgetting ────────────────────────


class TestForgettingMitigation:
    def test_ewc_protects_old_regime_while_learning_new(self, capsys):
        # Two well-separated regimes for a root MDN: A ~ N(-3, .4), B ~ N(+3, .4).
        torch.manual_seed(0)
        xA, xB = _gauss(-3.0, 800, 1), _gauss(3.0, 800, 2)
        xA_test, xB_test = _gauss(-3.0, 400, 11), _gauss(3.0, 400, 12)

        base = NBN([], {"X": ("continuous", 1)}, device="cpu")
        base.set_mechanism("X", MDNMechanism(num_components=5, hidden=(16,)))
        base.fit({"X": xA}, epochs=300, batch_size=128)

        def nll(model, data):
            with torch.no_grad():
                return float(-model.mechanisms["X"].log_prob(data, None).mean())

        A_before, B_before = nll(base, xA_test), nll(base, xB_test)

        # EWC update on B at DEFAULTS — no per-call lam, no per-call
        # fisher_batch_size.  This is exactly the path model.update(new_data)
        # gives an out-of-the-box user (DEFAULT_LAM + per-sample Fisher).
        torch.manual_seed(0)
        ewc = copy.deepcopy(base)
        ewc.update({"X": xB}, epochs=300, batch_size=128)
        A_ewc, B_ewc = nll(ewc, xA_test), nll(ewc, xB_test)

        # Naive fine-tune on B (same path, lam=0 → no anchor) as the baseline.
        torch.manual_seed(0)
        naive = copy.deepcopy(base)
        naive.update({"X": xB}, lam=0.0, epochs=300, batch_size=128)
        A_naive, B_naive = nll(naive, xA_test), nll(naive, xB_test)

        # Surfaced so the margins can be eyeballed in the test output.
        print(
            f"\nNLL  A_before={A_before:.3f}  A_ewc={A_ewc:.3f}  A_naive={A_naive:.3f}"
            f"\n     B_before={B_before:.3f}  B_ewc={B_ewc:.3f}  B_naive={B_naive:.3f}"
        )

        # Protection: EWC forgets regime A less than the naive fine-tune.
        assert A_ewc < A_naive - 0.3
        # Plasticity: EWC still learns regime B (vs the frozen post-fit model).
        assert B_ewc < B_before - 50.0


# ── state plumbing: buffers carry through to()/deepcopy/state_dict ────────────


class TestStatePlumbing:
    def _fitted_neural_categorical(self):
        torch.manual_seed(0)
        n = 400
        a = torch.randint(0, 3, (n,)).float()
        # b depends on a, 2 classes
        b = (a.long() % 2).float()
        model = NBN([("A", "B")], {"A": ("discrete", 3), "B": ("discrete", 2)},
                    device="cpu")
        model.set_mechanism("A", CategoricalTableMechanism())
        model.set_mechanism("B", NeuralCategoricalMechanism(n_classes=2, hidden=(16,)))
        model.fit({"A": a, "B": b}, epochs=40, batch_size=128)
        return model

    def test_buffers_exist_finite_and_nonneg(self):
        model = self._fitted_neural_categorical()
        mech = model.mechanisms["B"]
        assert mech._ewc_mu is not None and mech._ewc_fisher is not None
        assert torch.isfinite(mech._ewc_mu).all()
        assert torch.isfinite(mech._ewc_fisher).all()
        assert (mech._ewc_fisher >= 0).all()
        nparams = sum(p.numel() for p in mech.parameters() if p.requires_grad)
        assert mech._ewc_mu.numel() == nparams

    def test_buffers_registered_and_on_model_device(self):
        model = self._fitted_neural_categorical()
        mech = model.mechanisms["B"]
        bufs = dict(mech.named_buffers())
        assert "_ewc_mu" in bufs and "_ewc_fisher" in bufs  # registered, so .to() carries
        model.to("cpu")
        assert mech._ewc_mu.device.type == "cpu"

    def test_deepcopy_and_state_dict_roundtrip_preserve_buffers(self):
        model = self._fitted_neural_categorical()
        mech = model.mechanisms["B"]

        clone = copy.deepcopy(model)
        cm = clone.mechanisms["B"]
        assert torch.allclose(mech._ewc_mu, cm._ewc_mu)
        assert torch.allclose(mech._ewc_fisher, cm._ewc_fisher)

        # state_dict carries the EWC buffers, and load restores them.
        assert any(k.endswith("_ewc_mu") for k in model.state_dict())
        clone.load_state_dict(model.state_dict())
        assert torch.allclose(mech._ewc_fisher, clone.mechanisms["B"]._ewc_fisher)


# ── orchestration: neural + exact + skipped in one network ───────────────────


class TestOrchestration:
    def test_mixed_network_update_routes_per_mechanism(self):
        torch.manual_seed(0)
        n = 400
        a = torch.randint(0, 3, (n,)).float()
        b = a + 0.3 * torch.randn(n)          # continuous child → MDN (neural)
        d = torch.randint(0, 2, (n,)).float()  # discrete root → deterministic (skip)
        model = NBN(
            [("A", "B")],
            {"A": ("discrete", 3), "B": ("continuous", 1), "D": ("discrete", 2)},
            device="cpu",
        )
        model.set_mechanism("A", CategoricalTableMechanism())          # exact
        model.set_mechanism("B", MDNMechanism(num_components=3, hidden=(16,)))  # neural
        model.set_mechanism("D", DeterministicMechanism(torch.tensor([1.0])))  # unsupported
        data = {"A": a, "B": b, "D": d}
        model.fit(data, epochs=40, batch_size=128)

        a2 = torch.randint(0, 3, (200,)).float()
        b2 = a2 + 0.3 * torch.randn(200)
        d2 = torch.randint(0, 2, (200,)).float()
        hist = model.update({"A": a2, "B": b2, "D": d2}, epochs=20, batch_size=128)

        assert hist.node_methods["A"] == "dirichlet_conjugate"
        assert hist.node_methods["B"] == "online_ewc"
        assert "D" in hist.skipped
        assert "D" not in hist.node_methods


# ── online refresh: theta* re-anchors, F stays well-formed ───────────────────


class TestRefresh:
    def test_sequential_updates_reanchor_and_keep_fisher_wellformed(self):
        torch.manual_seed(0)
        model = NBN([], {"X": ("continuous", 1)}, device="cpu")
        model.set_mechanism("X", MDNMechanism(num_components=4, hidden=(16,)))
        model.fit({"X": _gauss(0.0, 400, 1)}, epochs=60, batch_size=128)
        mech = model.mechanisms["X"]

        for seed in (2, 3):
            model.update({"X": _gauss(float(seed), 400, seed)},
                         lam=1.0, fisher_batch_size=4, epochs=30, batch_size=128)
            # F stays finite & non-negative across the running accumulation.
            assert torch.isfinite(mech._ewc_fisher).all()
            assert (mech._ewc_fisher >= 0).all()
            # theta* is re-anchored to the current weights after each update.
            cur = torch.cat([p.detach().reshape(-1)
                             for p in mech.parameters() if p.requires_grad])
            assert torch.allclose(mech._ewc_mu, cur, atol=1e-6)


# ── guard: update before fit ─────────────────────────────────────────────────


class TestGuards:
    def test_update_before_fit_raises(self):
        # RuntimeError (was AssertionError) since consolidation became
        # opt-out: the message must point at refitting with consolidate=True.
        mech = MDNMechanism(num_components=3)
        with pytest.raises(RuntimeError, match="consolidate=True"):
            mech.update_local(torch.randn(8, 1), None)
