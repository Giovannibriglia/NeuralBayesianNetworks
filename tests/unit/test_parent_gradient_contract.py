"""Parent-gradient transparency is a contract, not an accident.

When a caller computes a parent value with its own ``nn.Module`` and hands it
to NBN, gradients must flow from the likelihood back through that tensor into
the caller's parameters.  This works today — but nothing tested it, and the
failure mode is not an exception.  A ``.detach()``, a ``torch.no_grad()``, or
a defensive ``clone().detach()`` added anywhere in the parent-marshalling path
would turn a caller's encoder into one that never trains: no error, no warning,
just a model that does not learn, which reads as a modelling problem and can
cost days to localise.

The paths that must stay differentiable:

    mechanism.log_prob(x, parents)      model.log_prob(data)
    model.sample(n)                     model.sample(n, do=...)

...and the paths that are deliberately NOT differentiable, pinned here so
nobody builds on a false assumption:

    model.query / query_batch  -- VE detaches at factor build; LW runs under
                                  torch.inference_mode() (a deliberate
                                  optimisation, see test_lw_query_inference_mode)
    model.intervene(...)       -- returns a deepcopy; the copy's parameters are
                                  fresh leaves

Every negative assertion below names ``sample(do=)`` as the differentiable
alternative, so the next person to hit this finds the answer in the failure
message rather than in the documentation.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from nbn import NeuralBayesianNetwork as NBN
from nbn.mechanisms import (
    CategoricalTableMechanism,
    LinearGaussianMechanism,
    MDNMechanism,
)

_USE_SAMPLE_DO = (
    "model.sample(n, do=...) is the differentiable interventional path"
)


class _Encoder(nn.Module):
    """Stands in for a downstream module that produces a parent value."""

    def __init__(self, w: float = 1.0) -> None:
        super().__init__()
        self.lin = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.lin.weight.fill_(w)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.lin(z)


def _fitted(mech_factory, n=600, epochs=30):
    torch.manual_seed(0)
    mech = mech_factory()
    pa = torch.randn(n, 1)
    x = 2.0 * pa + 0.1 * torch.randn(n, 1)
    mech.fit_local(x, pa, epochs=epochs)
    return mech, x, pa


# --------------------------------------------------------------------------
# mechanism.log_prob
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory", [LinearGaussianMechanism, MDNMechanism],
    ids=["linear_gaussian", "mdn"],
)
def test_mechanism_log_prob_is_parent_gradient_transparent(factory):
    """Closed-form and gradient-trained families marshal parents differently."""
    mech, x, _ = _fitted(factory)
    enc = _Encoder()
    z = torch.randn(64, 1)

    (-mech.log_prob(x[:64], enc(z)).mean()).backward()

    assert enc.lin.weight.grad is not None, (
        f"{factory.__name__}.log_prob severed the gradient to the caller's "
        f"parent tensor — check for a detach()/no_grad() in the parent path"
    )
    assert torch.isfinite(enc.lin.weight.grad).all()
    assert float(enc.lin.weight.grad.abs().sum()) > 0.0


def test_linear_gaussian_parent_gradient_matches_the_analytic_value():
    """Exact check against d/dw_enc of the Gaussian NLL, bias included.

    The bias term matters: omitting it disagrees with autograd by ~0.2%, which
    is close enough to look like a tolerance problem and is not one.
    """
    torch.manual_seed(0)
    mech = LinearGaussianMechanism()
    pa = torch.randn(2000, 1)
    mech.fit_local(3.0 * pa + 0.5 * torch.randn(2000, 1), pa)

    w = mech._weight.detach().reshape(())
    b = mech._bias.detach().reshape(())
    s = mech._scale().detach().reshape(())

    enc = _Encoder()
    z = torch.randn(8, 1)
    y = torch.randn(8, 1)
    (-mech.log_prob(y, enc(z)).mean()).backward()

    #   nll_i = 0.5*((y_i - (w*pa_i + b))/s)^2 + const,   pa_i = w_enc * z_i
    #   d nll_i / d w_enc = -(y_i - (w*z_i + b))/s^2 * w * z_i
    analytic = float((-(y - (w * z + b)) / s**2 * w * z).mean())
    torch.testing.assert_close(
        float(enc.lin.weight.grad), analytic, atol=1e-4, rtol=0,
    )


def test_gradient_survives_a_discrete_parent_cast():
    """LinearGaussian casts long parents to its own dtype (a91d8f9).

    The cast must not be the place a detach creeps in.
    """
    torch.manual_seed(0)
    mech = LinearGaussianMechanism()
    pa = torch.randn(400, 1)
    mech.fit_local(1.5 * pa + 0.1 * torch.randn(400, 1), pa)

    enc = _Encoder()
    z = torch.randn(16, 1)
    (-mech.log_prob(torch.randn(16, 1), enc(z)).mean()).backward()
    assert enc.lin.weight.grad is not None


# --------------------------------------------------------------------------
# model.log_prob
# --------------------------------------------------------------------------


def test_model_log_prob_is_gradient_transparent_to_caller_data():
    torch.manual_seed(0)
    model = NBN(
        [("X", "Y")],
        variables={"X": ("continuous", 1), "Y": ("continuous", 1)},
        device="cpu",
    )
    model.set_mechanism("X", LinearGaussianMechanism())
    model.set_mechanism("Y", LinearGaussianMechanism())
    x = torch.randn(500, 1)
    model.fit({"X": x, "Y": 2.0 * x + 0.1 * torch.randn(500, 1)})

    enc = _Encoder()
    z = torch.randn(32, 1)
    data = {"X": enc(z), "Y": torch.randn(32, 1)}

    (-model.log_prob(data).mean()).backward()
    assert enc.lin.weight.grad is not None, (
        "model.log_prob severed the gradient to caller-computed data"
    )
    assert torch.isfinite(enc.lin.weight.grad).all()


def test_model_log_prob_reaches_mechanism_parameters_too():
    torch.manual_seed(0)
    model = NBN([], {"X": ("continuous", 1)}, device="cpu")
    model.set_mechanism("X", LinearGaussianMechanism())
    x = torch.randn(400, 1)
    model.fit({"X": x})

    (-model.log_prob({"X": x}).mean()).backward()
    assert model.mechanisms["X"]._bias.grad is not None


def test_per_node_log_prob_is_independently_differentiable():
    """The decomposition must be usable for per-channel attribution."""
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

    enc = _Encoder()
    z = torch.randn(16, 1)
    parts = model.log_prob({"X": enc(z), "Y": torch.randn(16, 1)}, per_node=True)
    # Backward through the Y channel alone still reaches the encoder, because
    # X is Y's parent.
    (-parts["Y"].mean()).backward()
    assert enc.lin.weight.grad is not None


# --------------------------------------------------------------------------
# model.sample  (with and without do=)
# --------------------------------------------------------------------------


def _lg_chain(n=4000):
    torch.manual_seed(0)
    model = NBN(
        [("X", "Y")],
        variables={"X": ("continuous", 1), "Y": ("continuous", 1)},
        device="cpu",
    )
    model.set_mechanism("X", LinearGaussianMechanism())
    model.set_mechanism("Y", LinearGaussianMechanism())
    x = torch.randn(n, 1)
    model.fit({"X": x, "Y": 2.0 * x + 0.05 * torch.randn(n, 1)})
    return model


def test_sample_do_is_differentiable_wrt_model_parameters():
    """E[Y | do(X=1)] = w*1 + b, so dE[Y]/dw = 1."""
    model = _lg_chain()
    w = model.mechanisms["Y"]._weight
    torch.manual_seed(1)
    model.sample(50_000, do={"X": torch.tensor([1.0])})["Y"].mean().backward()
    assert w.grad is not None, _USE_SAMPLE_DO
    torch.testing.assert_close(
        float(w.grad.reshape(())), 1.0, atol=0.02, rtol=0,
    )


def test_sample_do_is_differentiable_wrt_the_intervention_value():
    """dE[Y | do(X=v)]/dv = w."""
    model = _lg_chain()
    w = float(model.mechanisms["Y"]._weight.detach().reshape(()))
    v = torch.tensor([1.0], requires_grad=True)
    torch.manual_seed(1)
    model.sample(50_000, do={"X": v})["Y"].mean().backward()
    assert v.grad is not None, (
        f"the do= value must stay differentiable; {_USE_SAMPLE_DO}"
    )
    torch.testing.assert_close(float(v.grad.reshape(())), w, atol=0.02, rtol=0)


def test_plain_sample_is_differentiable_wrt_model_parameters():
    model = _lg_chain()
    b = model.mechanisms["X"]._bias
    torch.manual_seed(2)
    model.sample(20_000)["X"].mean().backward()
    assert b.grad is not None
    torch.testing.assert_close(float(b.grad.reshape(())), 1.0, atol=0.05, rtol=0)


# --------------------------------------------------------------------------
# The deliberately non-differentiable paths
# --------------------------------------------------------------------------


def _discrete_model(engine):
    torch.manual_seed(0)
    model = NBN(
        [("A", "B")],
        variables=dict.fromkeys("AB", ("discrete", 2)),
        default_engine=engine,
        device="cpu",
    )
    model.auto_mechanisms()
    a = torch.bernoulli(torch.full((2000,), 0.3))
    model.fit({"A": a, "B": torch.bernoulli(torch.where(a > 0.5, 0.9, 0.1))})
    return model


@pytest.mark.parametrize("engine", ["tensor_ve", "likelihood_weighting"])
def test_query_is_not_differentiable_by_design(engine):
    """VE detaches at factor build; LW runs under torch.inference_mode().

    Both are deliberate performance decisions. Pinned so that a caller needing
    gradients is pointed at the path that has them, rather than discovering
    the absence as a silently non-training encoder.
    """
    model = _discrete_model(engine)
    out = model.query(["B"], evidence={"A": torch.tensor(1)}, n_samples=128)
    assert not out.requires_grad, (
        f"{engine} query became differentiable; that is not the contract — "
        f"{_USE_SAMPLE_DO}"
    )


def test_intervene_severs_the_gradient_to_the_original_model():
    """intervene() deep-copies; its parameters are fresh leaves."""
    model = _lg_chain()
    w = model.mechanisms["Y"]._weight
    cut = model.intervene({"X": torch.tensor([1.0])})
    cut.sample(2000)["Y"].mean().backward()
    assert w.grad is None, (
        f"intervene() is a deepcopy and must not backprop into the original; "
        f"{_USE_SAMPLE_DO}"
    )


def test_intervene_severs_the_gradient_to_the_callers_value():
    """The caller's tensor is copied into the new mechanism, not shared."""
    model = _lg_chain()
    v = torch.tensor([1.0], requires_grad=True)
    cut = model.intervene({"X": v})
    cut.sample(2000)["Y"].mean().backward()
    assert v.grad is None, (
        f"intervene() copies the value; for gradients w.r.t. the intervention "
        f"{_USE_SAMPLE_DO}"
    )


def test_the_two_do_paths_agree_numerically_even_though_one_is_differentiable():
    """Severed gradients must not mean different numbers."""
    torch.manual_seed(0)
    model = _discrete_model("tensor_ve")
    exact = model.query(["B"], do={"A": torch.tensor(1)})
    cut = model.intervene({"A": torch.tensor(1)})
    torch.testing.assert_close(exact, cut.query(["B"]), atol=1e-6, rtol=0)
    assert math.isfinite(float(exact.sum()))


def test_categorical_parents_keep_gradients_flowing_to_continuous_children():
    """discrete parent -> continuous child, the topology a91d8f9 unblocked."""
    torch.manual_seed(0)
    model = NBN(
        [("A", "R")],
        variables={"A": ("discrete", 3), "R": ("continuous", 1)},
        device="cpu",
    )
    model.set_mechanism("A", CategoricalTableMechanism())
    model.set_mechanism("R", LinearGaussianMechanism())
    a = torch.randint(0, 3, (800,))
    model.fit({"A": a, "R": a.float().unsqueeze(-1) * 1.5 + 0.1 * torch.randn(800, 1)})

    enc = _Encoder()
    z = torch.randn(32, 1)
    (-model.mechanisms["R"].log_prob(torch.randn(32, 1), enc(z)).mean()).backward()
    assert enc.lin.weight.grad is not None
