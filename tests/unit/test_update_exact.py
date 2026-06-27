"""Exact-updater equivalence and bookkeeping for ``model.update`` (PR1).

The two exact updaters (Dirichlet-conjugate categorical, recursive-least-squares
linear-Gaussian) are *exact*: a chunked ``fit(A)`` then ``update(B)`` must
reproduce a single ``fit(A | B)``.  These tests pin that equivalence end-to-end
through the public ``model.fit`` / ``model.update`` API, plus the forgetting,
skip, range-guard, and buffer-round-trip contracts.
"""
import copy

import pytest
import torch

from nbn import NeuralBayesianNetwork as NBN
from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism
from nbn.mechanisms.parametric.deterministic import DeterministicMechanism

# ── data generators ─────────────────────────────────────────────────────────


def _gen_discrete(n, *, p_seed):
    """Two-node discrete data P -> C with full class coverage in every chunk."""
    g = torch.Generator().manual_seed(p_seed)
    p = torch.randint(0, 2, (n,), generator=g).float()
    c = torch.randint(0, 3, (n,), generator=g).float()
    return {"P": p, "C": c}


def _gen_cont(n, *, p_seed):
    """Two-node continuous data P -> C (linear-Gaussian)."""
    g = torch.Generator().manual_seed(p_seed)
    p = torch.randn(n, generator=g)
    c = 1.5 * p + 0.3 * torch.randn(n, generator=g)
    return {"P": p, "C": c}


def _discrete_model():
    model = NBN([("P", "C")], {"P": ("discrete", 2), "C": ("discrete", 3)})
    model.auto_mechanisms(default_discrete="categorical_table")
    return model


def _cont_model():
    model = NBN([("P", "C")], {"P": ("continuous", 1), "C": ("continuous", 1)})
    # default_continuous != "mdn" selects LinearGaussianMechanism.
    model.auto_mechanisms(default_continuous="lg")
    return model


def _pool(a, b):
    return {k: torch.cat([a[k], b[k]], dim=0) for k in a}


# ── categorical equivalence ─────────────────────────────────────────────────


class TestCategoricalEquivalence:
    def test_fit_then_update_equals_pooled_fit(self):
        a = _gen_discrete(600, p_seed=1)
        b = _gen_discrete(400, p_seed=2)

        chunked = _discrete_model()
        chunked.fit(a)
        hist = chunked.update(b)

        pooled = _discrete_model()
        pooled.fit(_pool(a, b))

        # Root (P) and 1-parent (C) node CPTs both match the pooled refit.
        for node in ("P", "C"):
            assert torch.allclose(
                chunked.mechanisms[node].cpt, pooled.mechanisms[node].cpt, atol=1e-5
            ), f"CPT mismatch at node {node!r}"
        assert hist.node_methods["C"] == "dirichlet_conjugate"
        assert hist.node_methods["P"] == "dirichlet_conjugate"

    def test_class_first_seen_in_update_chunk_equals_pooled(self):
        # The footgun case: K=3 declared, but chunk A only ever observes
        # classes {0, 1}; class 2 first appears in the update chunk B.  Because
        # smoothing is re-applied from raw counts (not baked into _counts at
        # fit time), the chunked CPT must still match a single pooled fit
        # exactly — a class first seen at update time is handled identically.
        x_a = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        x_b = torch.tensor([2.0, 2.0, 0.0, 1.0, 2.0])

        chunked = CategoricalTableMechanism()
        chunked.fit_local(x_a, None, n_classes=3)
        chunked.update_local(x_b, None, n_classes=3)

        pooled = CategoricalTableMechanism()
        pooled.fit_local(torch.cat([x_a, x_b]), None, n_classes=3)

        assert torch.allclose(chunked.cpt, pooled.cpt, atol=1e-6)
        # ... and the raw stats themselves match the pooled raw counts.
        assert torch.allclose(chunked._counts, pooled._counts)


# ── linear-Gaussian equivalence ─────────────────────────────────────────────


class TestLinearGaussianEquivalence:
    def test_fit_then_update_equals_pooled_fit(self):
        a = _gen_cont(600, p_seed=11)
        b = _gen_cont(400, p_seed=12)

        chunked = _cont_model()
        chunked.fit(a)
        hist = chunked.update(b)

        pooled = _cont_model()
        pooled.fit(_pool(a, b))

        for node in ("P", "C"):
            mu = chunked.mechanisms[node]
            mp = pooled.mechanisms[node]
            assert torch.allclose(mu._weight, mp._weight, atol=1e-4), f"W @ {node}"
            assert torch.allclose(mu._bias, mp._bias, atol=1e-4), f"b @ {node}"
            assert torch.allclose(
                torch.exp(mu._log_scale), torch.exp(mp._log_scale), atol=1e-4
            ), f"sigma @ {node}"
        assert hist.node_methods["C"] == "recursive_gaussian"


# ── forgetting ───────────────────────────────────────────────────────────────


class TestForgetting:
    def test_forgetting_changes_result_and_reduces_mass(self):
        x_a = torch.randint(0, 3, (500,)).float()
        x_b = torch.randint(0, 3, (200,)).float()

        full = CategoricalTableMechanism()
        full.fit_local(x_a, None, n_classes=3)
        mass_a = float(full._counts.sum())
        full.update_local(x_b, None, forgetting=1.0)
        mass_full = float(full._counts.sum())

        half = CategoricalTableMechanism()
        half.fit_local(x_a, None, n_classes=3)
        half.update_local(x_b, None, forgetting=0.5)
        mass_half = float(half._counts.sum())

        # forgetting fades the prior counts: half-update keeps less mass.
        assert mass_half < mass_full
        assert mass_half == pytest.approx(0.5 * mass_a + 200.0, abs=1e-4)
        # and the resulting CPD genuinely differs.
        assert not torch.allclose(full.cpt, half.cpt, atol=1e-4)

    def test_forgetting_out_of_range_rejected(self):
        model = _discrete_model()
        model.fit(_gen_discrete(100, p_seed=3))
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                model.update(_gen_discrete(50, p_seed=4), forgetting=bad)


# ── skip non-updatable mechanisms ───────────────────────────────────────────


class TestSkip:
    def test_deterministic_node_is_skipped(self):
        model = NBN([("P", "C")], {"P": ("discrete", 2), "C": ("discrete", 3)})
        model.set_mechanism("P", CategoricalTableMechanism())
        model.set_mechanism("C", DeterministicMechanism(torch.tensor([1.0])))
        data = _gen_discrete(200, p_seed=5)
        model.fit(data)
        hist = model.update(_gen_discrete(100, p_seed=6))
        assert "C" in hist.skipped
        assert "P" in hist.node_methods
        assert "P" not in hist.skipped


# ── declared-range guard ─────────────────────────────────────────────────────


class TestRangeGuard:
    def test_class_index_beyond_cardinality_raises(self):
        mech = CategoricalTableMechanism()
        mech.fit_local(torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]), None, n_classes=3)
        with pytest.raises(ValueError):
            mech.update_local(torch.tensor([0.0, 1.0, 5.0]), None)

    def test_parent_index_beyond_cardinality_raises(self):
        x = torch.randint(0, 3, (60,)).float()
        pa = torch.randint(0, 2, (60, 1)).float()
        mech = CategoricalTableMechanism()
        mech.fit_local(x, pa, parent_cards=[2], n_classes=3)
        with pytest.raises(ValueError):
            mech.update_local(
                torch.tensor([0.0, 1.0]), torch.tensor([[0.0], [7.0]])
            )


# ── buffer round-trip (why _counts / _neq_* are buffers) ─────────────────────


class TestRoundTrip:
    def test_deepcopy_preserves_state_and_update_matches(self):
        a = _gen_discrete(400, p_seed=21)
        b = _gen_discrete(300, p_seed=22)
        model = _discrete_model()
        model.fit(a)

        clone = copy.deepcopy(model)
        # buffers survived the deepcopy
        for node in ("P", "C"):
            assert torch.allclose(
                model.mechanisms[node]._counts, clone.mechanisms[node]._counts
            )

        # updating the clone reproduces updating the original (state intact)
        model.update(b)
        clone.update(b)
        for node in ("P", "C"):
            assert torch.allclose(
                model.mechanisms[node].cpt, clone.mechanisms[node].cpt, atol=1e-6
            )

    def test_linear_gaussian_stats_survive_deepcopy(self):
        model = _cont_model()
        model.fit(_gen_cont(400, p_seed=31))
        clone = copy.deepcopy(model)
        for node in ("P", "C"):
            for buf in ("_neq_A", "_neq_B", "_neq_c", "_neq_N"):
                assert torch.allclose(
                    getattr(model.mechanisms[node], buf),
                    getattr(clone.mechanisms[node], buf),
                )
