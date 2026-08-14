"""Interventional inference: the do-operator across every engine and entry point.

Before these fixes ``do=`` reached ``TensorVariableElimination.query`` inside
``**kwargs`` and was silently discarded, so an all-discrete network answered
``P(Y | do(X))`` with the *observational* distribution and no warning — while
``LikelihoodWeightingEngine`` on the same model honoured the intervention.
``intervene()`` was no better: it swapped in a mechanism that reported itself
continuous, so VE rejected the whole model.

The tests below pin the fix from both directions: the four independent
interventional paths must agree with each other, and each must differ from
the observational answer exactly where confounding says it should.
"""
from __future__ import annotations

import pytest
import torch

from nbn import NeuralBayesianNetwork as NBN
from nbn.core.dag import DAG


def _confounded_model(engine="tensor_ve", n=200_000, seed=0):
    """A → B → C with a confounding A → C edge.

    P(C | do(B)) and P(C | B) must differ: conditioning on B carries
    information about A, intervening on it does not.
    """
    torch.manual_seed(seed)
    model = NBN(
        [("A", "B"), ("B", "C"), ("A", "C")],
        variables=dict.fromkeys("ABC", ("discrete", 2)),
        default_engine=engine,
        device="cpu",
    )
    model.auto_mechanisms()
    a = torch.bernoulli(torch.full((n,), 0.3))
    b = torch.bernoulli(torch.where(a > 0.5, 0.9, 0.1))
    c = torch.bernoulli(
        torch.where(b > 0.5, 0.8, 0.2) * 0.5 + torch.where(a > 0.5, 0.7, 0.3) * 0.5
    )
    model.fit({"A": a, "B": b, "C": c})
    return model


def test_ve_do_is_not_silently_ignored():
    """The headline regression: VE used to return the prior for any do=."""
    model = _confounded_model()
    prior = model.query(["C"])
    interventional = model.query(["C"], do={"B": torch.tensor(1)})
    assert not torch.allclose(prior, interventional, atol=1e-3), (
        "P(C | do(B=1)) collapsed onto the prior — do= is being dropped again"
    )


def test_ve_do_differs_from_conditioning_under_confounding():
    model = _confounded_model()
    do = model.query(["C"], do={"B": torch.tensor(1)})
    obs = model.query(["C"], evidence={"B": torch.tensor(1)})
    assert not torch.allclose(do, obs, atol=1e-2)


def test_do_on_a_root_equals_conditioning_on_it():
    """A has no parents, so mutilating it changes nothing: P(C|do(A)) == P(C|A)."""
    model = _confounded_model()
    do = model.query(["C"], do={"A": torch.tensor(1)})
    obs = model.query(["C"], evidence={"A": torch.tensor(1)})
    torch.testing.assert_close(do, obs, atol=1e-6, rtol=1e-6)


def test_all_four_interventional_paths_agree():
    """VE do=, LW do=, intervene()+VE, intervene()+LW must answer alike."""
    ve = _confounded_model("tensor_ve")
    lw = _confounded_model("likelihood_weighting")

    ve_do = ve.query(["C"], do={"B": torch.tensor(1)})
    lw_do = lw.query(["C"], do={"B": torch.tensor(1)}, n_samples=200_000)

    cut_ve = ve.intervene({"B": torch.tensor(1)})
    cut_lw = ve.intervene({"B": torch.tensor(1)})
    cut_lw._engine_spec, cut_lw._engine = "likelihood_weighting", None

    # Exact vs exact.
    torch.testing.assert_close(ve_do, cut_ve.query(["C"]), atol=1e-6, rtol=1e-6)
    # Exact vs Monte Carlo (200k particles).
    torch.testing.assert_close(ve_do, lw_do, atol=0.01, rtol=0)
    torch.testing.assert_close(
        ve_do, cut_lw.query(["C"], n_samples=200_000), atol=0.01, rtol=0,
    )


def test_sample_do_matches_exact_interventional_marginal():
    model = _confounded_model()
    exact = model.query(["C"], do={"B": torch.tensor(1)})
    drawn = model.sample(200_000, do={"B": torch.tensor([1.0])})["C"].mean()
    torch.testing.assert_close(drawn, exact[1], atol=0.01, rtol=0)


def test_do_combines_with_evidence():
    ve = _confounded_model("tensor_ve")
    lw = _confounded_model("likelihood_weighting")
    got = ve.query(["C"], do={"A": torch.tensor(1)}, evidence={"B": torch.tensor(0)})
    ref = lw.query(
        ["C"], do={"A": torch.tensor(1)}, evidence={"B": torch.tensor(0)},
        n_samples=200_000,
    )
    torch.testing.assert_close(got, ref, atol=0.01, rtol=0)


def test_batched_do_matches_looping_scalar_do():
    """A per-row intervention sweep in one query_batch call."""
    model = _confounded_model()
    values = [0, 1, 0, 1]
    batched = model.query_batch(["C"], evidence={}, do={"B": torch.tensor(values)})
    looped = torch.stack(
        [model.query(["C"], do={"B": torch.tensor(v)}) for v in values]
    )
    torch.testing.assert_close(batched, looped, atol=1e-6, rtol=1e-6)


def test_batched_do_in_likelihood_weighting():
    """LW derived its batch axis from evidence only; batched do died in expand()."""
    model = _confounded_model("likelihood_weighting")
    out = model.query_batch(["C"], evidence={}, do={"B": torch.tensor([0, 1])})
    assert out.shape == (2, 2)
    # do(B=1) must raise P(C=1) relative to do(B=0) given the generating process.
    assert out[1, 1] > out[0, 1]


def test_do_and_evidence_on_the_same_node_is_rejected():
    model = _confounded_model()
    with pytest.raises(ValueError, match="both evidence and do"):
        model.query(["C"], do={"B": torch.tensor(1)}, evidence={"B": torch.tensor(1)})


def test_unknown_do_target_is_rejected():
    model = _confounded_model()
    with pytest.raises(ValueError, match="Unknown do-intervention target"):
        model.query(["C"], do={"NOPE": torch.tensor(1)})


def test_intervene_rejects_unknown_and_out_of_range_targets():
    model = _confounded_model()
    with pytest.raises(ValueError, match="Unknown intervention target"):
        model.intervene({"NOPE": torch.tensor(1)})
    with pytest.raises(ValueError, match="outside its 2 declared states"):
        model.intervene({"B": torch.tensor(7)})


def test_intervene_mutilates_the_graph():
    model = _confounded_model()
    cut = model.intervene({"B": torch.tensor(1)})
    assert model.dag.parents("B") == ["A"]
    assert cut.dag.parents("B") == []
    assert cut._do_targets == {"B"}
    # The original is untouched.
    assert model._do_targets == set()


def test_intervene_preserves_every_other_nodes_parent_order():
    """Regression: rebuilding the DAG from edges() transposed C's CPT axes.

    networkx yields edges in source-node order, so ``DAG(dag.edges())`` can
    permute an untouched node's parent list.  Mechanism tabulations are laid
    out in fit-time parent order, so the permutation silently mislabels the
    factor axes and inference returns confidently wrong probabilities.
    """
    model = _confounded_model()
    cut = model.intervene({"B": torch.tensor(1)})
    assert model.dag.parents("C") == ["B", "A"]
    assert cut.dag.parents("C") == ["B", "A"]


def test_batched_intervene_and_batched_sample_do_are_rejected_clearly():
    model = _confounded_model()
    with pytest.raises(ValueError, match="Batched intervention value"):
        model.intervene({"B": torch.tensor([[0], [1]])})
    with pytest.raises(ValueError, match="Batched do-value"):
        model.sample(10, do={"B": torch.tensor([[0.0], [1.0]])})


def test_ve_rejects_continuous_do_target():
    torch.manual_seed(0)
    model = NBN(
        [("X", "Y")],
        variables={"X": ("continuous", 1), "Y": ("continuous", 1)},
        default_engine="tensor_ve",
        device="cpu",
    )
    model.auto_mechanisms()
    x = torch.randn(500, 1)
    model.fit({"X": x, "Y": 2 * x + 0.1 * torch.randn(500, 1)}, epochs=2)
    with pytest.raises(ValueError, match="requires discrete"):
        model.query(["Y"], do={"X": torch.tensor(1.0)})


def test_dag_ordered_edges_round_trips_parent_order():
    dag = DAG([("A", "B"), ("B", "C"), ("A", "C")])
    assert dag.parents("C") == ["B", "A"]
    # The documented trap this method exists to avoid:
    assert DAG(dag.edges()).parents("C") == ["A", "B"]
    # ...and the faithful reconstruction.
    assert DAG(dag.ordered_edges()).parents("C") == dag.parents("C")


def test_intervened_model_is_differentiably_severed_but_sample_do_is_not():
    """The deepcopy in intervene() cuts autograd; sample(do=) is the live path."""
    torch.manual_seed(0)
    from nbn.mechanisms import LinearGaussianMechanism

    model = NBN(
        [("X", "Y")],
        variables={"X": ("continuous", 1), "Y": ("continuous", 1)},
        device="cpu",
    )
    model.set_mechanism("X", LinearGaussianMechanism())
    model.set_mechanism("Y", LinearGaussianMechanism())
    x = torch.randn(5000, 1)
    model.fit({"X": x, "Y": 2.0 * x + 0.1 * torch.randn(5000, 1)})
    weight = model.mechanisms["Y"]._weight

    cut = model.intervene({"X": torch.tensor([1.0])})
    cut.sample(2000)["Y"].mean().backward()
    assert weight.grad is None, "intervene() should not backprop into the original"

    torch.manual_seed(1)
    model.sample(50_000, do={"X": torch.tensor([1.0])})["Y"].mean().backward()
    # d E[Y] / dW = x = 1.0 under do(X=1).
    assert weight.grad is not None
    torch.testing.assert_close(
        weight.grad.reshape(()), torch.tensor(1.0), atol=0.02, rtol=0,
    )


def test_intervening_on_the_target_returns_a_point_mass():
    """P(B | do(B=1)) is a delta; VE would otherwise fall through to uniform."""
    ve = _confounded_model("tensor_ve")
    lw = _confounded_model("likelihood_weighting")
    expected = torch.tensor([0.0, 1.0])
    torch.testing.assert_close(
        ve.query(["B"], do={"B": torch.tensor(1)}), expected, atol=1e-6, rtol=0,
    )
    torch.testing.assert_close(
        lw.query(["B"], do={"B": torch.tensor(1)}, n_samples=2048),
        expected, atol=1e-6, rtol=0,
    )
    batched = ve.query_batch(["B"], evidence={}, do={"B": torch.tensor([0, 1])})
    torch.testing.assert_close(
        batched, torch.tensor([[1.0, 0.0], [0.0, 1.0]]), atol=1e-6, rtol=0,
    )
