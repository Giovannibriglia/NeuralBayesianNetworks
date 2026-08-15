"""``NeuralBayesianNetwork.log_prob`` — complete-data likelihood, per row.

The method is deliberately strict and deliberately unreduced:

* **Strict** because the alternative is silently wrong.  A version that
  skipped nodes absent from ``data`` would return the likelihood of a
  different, smaller model — a plausible number that is the likelihood of
  nothing in particular.  Callers with latent variables must supply them.
* **Unreduced** because a caller weighting rows (an EM E-step multiplying by
  responsibilities) needs the per-row vector, and a scalar cannot be un-summed.
"""
from __future__ import annotations

import pytest
import torch

from nbn import NeuralBayesianNetwork as NBN
from nbn.mechanisms import LinearGaussianMechanism


def _discrete_chain(n=2000, seed=0):
    torch.manual_seed(seed)
    model = NBN(
        [("A", "B"), ("B", "C")],
        variables=dict.fromkeys("ABC", ("discrete", 2)),
        device="cpu",
    )
    model.auto_mechanisms()
    a = torch.bernoulli(torch.full((n,), 0.3))
    b = torch.bernoulli(torch.where(a > 0.5, 0.9, 0.1))
    c = torch.bernoulli(torch.where(b > 0.5, 0.8, 0.2))
    data = {"A": a, "B": b, "C": c}
    model.fit(data)
    return model, data


def test_returns_one_value_per_row():
    model, data = _discrete_chain()
    lp = model.log_prob(data)
    assert lp.shape == (2000,)
    assert torch.isfinite(lp).all()


def test_per_node_decomposes_the_total_exactly():
    model, data = _discrete_chain()
    total = model.log_prob(data)
    parts = model.log_prob(data, per_node=True)
    assert set(parts) == {"A", "B", "C"}
    assert all(v.shape == (2000,) for v in parts.values())
    torch.testing.assert_close(
        parts["A"] + parts["B"] + parts["C"], total, atol=1e-6, rtol=0,
    )


def test_missing_node_raises_and_names_it():
    """Never skip, never marginalise — see the module docstring."""
    model, data = _discrete_chain()
    del data["B"]
    with pytest.raises(ValueError, match=r"missing \['B'\]"):
        model.log_prob(data)


def test_unfitted_model_raises():
    model = NBN(
        [("A", "B")],
        variables=dict.fromkeys("AB", ("discrete", 2)),
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="No mechanism registered"):
        model.log_prob({"A": torch.zeros(4), "B": torch.zeros(4)})


def test_matches_hand_computed_mechanism_sum():
    """Independent of the implementation's loop: sum the mechanisms by hand."""
    from nbn.utils.batching import pack_parents

    model, data = _discrete_chain()
    expected = torch.zeros(2000)
    for node in model.dag.topological_order():
        pa = pack_parents(data, model.dag.parents(node))
        expected = expected + model.mechanisms[node].log_prob(data[node], pa)
    torch.testing.assert_close(model.log_prob(data), expected, atol=1e-6, rtol=0)


def test_continuous_model_agrees_with_the_analytic_gaussian():
    """A root LinearGaussian's row likelihood is a plain Normal log-density."""
    import math

    torch.manual_seed(0)
    model = NBN([], {"X": ("continuous", 1)}, device="cpu")
    model.set_mechanism("X", LinearGaussianMechanism())
    x = torch.randn(4000, 1) * 2.0 + 1.0
    model.fit({"X": x})

    mu = model.mechanisms["X"]._bias.detach().reshape(())
    sd = model.mechanisms["X"]._scale().detach().reshape(())
    analytic = (
        -0.5 * ((x.reshape(-1) - mu) / sd) ** 2
        - torch.log(sd)
        - 0.5 * math.log(2 * math.pi)
    )
    torch.testing.assert_close(model.log_prob({"X": x}), analytic, atol=1e-5, rtol=0)


def test_weighting_rows_is_the_caller_s_to_do():
    """The EM shape: responsibilities x per-row likelihood."""
    model, data = _discrete_chain()
    lp = model.log_prob(data)
    r = torch.rand(2000)
    weighted = (r * lp).sum() / r.sum()
    assert torch.isfinite(weighted)
