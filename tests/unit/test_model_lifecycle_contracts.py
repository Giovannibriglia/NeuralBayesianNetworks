"""Lifecycle contracts: fitted-state reporting, refit invalidation, persistence.

Each test here pins a defect where the library returned a *plausible wrong
answer* rather than raising: mechanisms reporting themselves unfitted after a
successful fit, an engine serving pre-refit posteriors, a checkpoint that
loaded without its mechanisms, and a discrete parent crashing a linear-Gaussian
child on a dtype mismatch.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from nbn import NeuralBayesianNetwork as NBN
from nbn.inference.tensor_ve import TensorVariableElimination
from nbn.mechanisms import (
    CategoricalTableMechanism,
    LinearGaussianMechanism,
    MDNMechanism,
)


def _discrete_model(p=0.1, n=20_000, seed=0):
    torch.manual_seed(seed)
    model = NBN(
        [("A", "B")],
        variables={"A": ("discrete", 2), "B": ("discrete", 2)},
        default_engine="tensor_ve",
        device="cpu",
    )
    model.auto_mechanisms()
    a = torch.bernoulli(torch.full((n,), p))
    b = torch.bernoulli(torch.where(a > 0.5, 0.9, 0.1))
    model.fit({"A": a, "B": b})
    return model, {"A": a, "B": b}


# ---------------------------------------------------------------- is_fitted


@pytest.mark.parametrize("mech_factory", [LinearGaussianMechanism, MDNMechanism])
def test_continuous_mechanisms_report_fitted_after_fit(mech_factory):
    """These three inherited the base ``False`` and never overrode it."""
    torch.manual_seed(0)
    mech = mech_factory()
    assert mech.is_fitted is False
    x = torch.randn(500, 1)
    mech.fit_local(2 * x + 0.1 * torch.randn(500, 1), x, epochs=2)
    assert mech.is_fitted is True


def test_root_mdn_reports_fitted():
    """A root MDN is fitted via _root_logits, not via ``net``."""
    torch.manual_seed(0)
    mech = MDNMechanism()
    mech.fit_local(torch.randn(500, 1), None, epochs=2)
    assert mech.is_fitted is True


def test_flow_reports_fitted_after_fit():
    zuko = pytest.importorskip("zuko")  # noqa: F841
    from nbn.mechanisms.parametric.normalizing_flow import NormalizingFlowMechanism

    torch.manual_seed(0)
    mech = NormalizingFlowMechanism()
    assert mech.is_fitted is False
    mech.fit_local(torch.randn(256, 1), None, epochs=2)
    assert mech.is_fitted is True


# ---------------------------------------------------------------- LG dtype


def test_linear_gaussian_accepts_long_discrete_parents():
    """discrete action -> continuous reward: the parent arrives as ``long``.

    ``pack_parents`` keeps raw dtypes (tabular children need index dtypes), and
    matmul does not type-promote, so this used to die inside ``model.fit`` with
    "expected m1 and m2 to have the same dtype" — after the ridge solve had
    already succeeded.
    """
    torch.manual_seed(0)
    model = NBN(
        [("A", "R")],
        variables={"A": ("discrete", 3), "R": ("continuous", 1)},
        device="cpu",
    )
    model.set_mechanism("A", CategoricalTableMechanism())
    model.set_mechanism("R", LinearGaussianMechanism())
    a = torch.randint(0, 3, (2000,))
    assert a.dtype == torch.long
    r = a.float().unsqueeze(-1) * 1.5 + 0.1 * torch.randn(2000, 1)

    model.fit({"A": a, "R": r})

    weight = model.mechanisms["R"]._weight.reshape(())
    torch.testing.assert_close(weight, torch.tensor(1.5), atol=0.05, rtol=0)
    # Long parents must also survive the inference path, not just fitting.
    mech = model.mechanisms["R"]
    assert torch.isfinite(mech.log_prob(r, a.unsqueeze(-1))).all()
    assert torch.isfinite(mech.sample(a.unsqueeze(-1), n=2)).all()


# ---------------------------------------------------------------- cache invalidation


def test_externally_held_engine_is_not_stale_after_refit():
    """The engine memoises factors per model; a refit must miss that memo."""
    engine = TensorVariableElimination()
    model, _ = _discrete_model(p=0.1)
    before = engine.query(model, ["A"], {})
    assert before[1] < 0.2

    torch.manual_seed(1)
    n = 20_000
    a = torch.bernoulli(torch.full((n,), 0.9))
    b = torch.bernoulli(torch.where(a > 0.5, 0.9, 0.1))
    model.fit({"A": a, "B": b})

    after = engine.query(model, ["A"], {})
    assert after[1] > 0.8, "engine served a pre-refit posterior"


def test_cache_version_advances_on_every_cpd_mutation():
    model, data = _discrete_model()
    v0 = model._cache_version
    model.fit(data)
    assert model._cache_version > v0
    v1 = model._cache_version
    model.set_mechanism("A", CategoricalTableMechanism())
    assert model._cache_version > v1


def test_engine_cache_survives_model_identity_reuse():
    """A recycled ``id()`` must not inherit a dead model's factors."""
    engine = TensorVariableElimination()
    model_a, _ = _discrete_model(p=0.1)
    engine.query(model_a, ["A"], {})
    # Forge the exact collision the weakref guard exists to catch.
    model_b, _ = _discrete_model(p=0.9, seed=3)
    stale = engine._factor_cache[id(model_a)]
    engine._factor_cache[id(model_b)] = stale
    out = engine.query(model_b, ["A"], {})
    assert out[1] > 0.8, "stale cross-model factors were served"


# ---------------------------------------------------------------- persistence


def test_save_load_round_trips_a_queryable_model():
    model, _ = _discrete_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "m.pt")
        model.save(path)
        loaded = NBN.load(path)

    assert sorted(loaded.mechanisms.keys()) == ["A", "B"]
    for ev in (0, 1):
        torch.testing.assert_close(
            loaded.query(["B"], evidence={"A": torch.tensor(ev)}),
            model.query(["B"], evidence={"A": torch.tensor(ev)}),
            atol=1e-6, rtol=1e-6,
        )


def test_save_load_preserves_declared_cardinality_and_graph_order():
    torch.manual_seed(0)
    model = NBN(
        [("A", "B"), ("B", "C"), ("A", "C")],
        variables={"A": ("discrete", 2), "B": ("discrete", 2), "C": ("discrete", 4)},
        device="cpu",
    )
    model.auto_mechanisms()
    n = 5000
    a = torch.bernoulli(torch.full((n,), 0.4))
    b = torch.bernoulli(torch.where(a > 0.5, 0.8, 0.2))
    c = torch.randint(0, 4, (n,)).float()
    model.fit({"A": a, "B": b, "C": c})
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "m.pt")
        model.save(path)
        loaded = NBN.load(path)
    assert loaded.variables["C"].cardinality == 4
    assert loaded.dag.parents("C") == model.dag.parents("C") == ["B", "A"]


def test_loading_a_format_1_checkpoint_still_works(caplog):
    """Old checkpoints carry no mechanisms; they must load, and say so."""
    model, _ = _discrete_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "old.pt")
        payload = {
            "dag_edges": model.dag.ordered_edges(),
            "dag_nodes": model.dag.nodes(),
            "variables": {
                n: (v.kind, v.dim, v.cardinality) for n, v in model.variables.items()
            },
            "state_dict": model.state_dict(),
            "mechanism_types": {n: type(m).__name__ for n, m in model.mechanisms.items()},
        }
        torch.save(payload, path)
        with caplog.at_level("WARNING"):
            loaded = NBN.load(path)
    assert len(loaded.mechanisms) == 0
    assert "no mechanism modules" in caplog.text


# ---------------------------------------------------------------- train history


def test_train_history_is_populated_for_both_fit_methods():
    """``joint`` recorded nothing, so ``mean_ll`` silently returned NaN."""
    torch.manual_seed(0)
    model = NBN(
        [("X", "Y")],
        variables={"X": ("continuous", 1), "Y": ("continuous", 1)},
        device="cpu",
    )
    model.set_mechanism("X", LinearGaussianMechanism())
    model.set_mechanism("Y", LinearGaussianMechanism())
    x = torch.randn(500, 1)
    data = {"X": x, "Y": 2 * x + 0.1 * torch.randn(500, 1)}

    local = model.fit(data)
    assert set(local.node_log_likelihoods) == {"X", "Y"}

    joint = model.fit(data, method="joint", epochs=2)
    assert set(joint.node_log_likelihoods) == {"X", "Y"}
    for node in ("X", "Y"):
        assert joint.mean_ll(node) == joint.mean_ll(node)  # not NaN
