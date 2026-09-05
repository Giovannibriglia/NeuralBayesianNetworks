"""LW/AIS query paths run under ``torch.inference_mode`` (PR C).

Mechanism parameters still require grad after ``eval()``, so before this
change every particle's ``sample()``/``log_prob()`` in the LW ancestral
sweep built and retained an autograd graph across the whole topological
order — pure waste of query-time compute and memory (VE already detaches
at factor build).  ``LikelihoodWeightingEngine.query`` is now decorated
with ``@torch.inference_mode()``; ``query_batch`` delegates to it and
``AmortizedISEngine`` inherits both, so the whole hot path is covered.

Contract pinned here:
- query outputs carry no autograd state (``requires_grad`` False, no
  ``grad_fn``) for both the discrete-histogram and the continuous
  ``(weights, samples)`` return shapes;
- mechanism parameters still require grad afterwards and gradients still
  flow through ``log_prob`` — fit-ability is unaffected (proposal
  TRAINING in ``AmortizedISEngine.train_proposal`` is not routed through
  ``query`` and keeps its gradients).
"""
from __future__ import annotations

import torch

from nbn.bench.synthetic import make_synthetic_bn
from nbn.inference.amortized_is import AmortizedISEngine
from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine


def _discrete_model():
    return make_synthetic_bn(
        family="discrete", n_nodes=4, cardinality=3, max_in_degree=2,
        edge_density=0.50,
        n_train=50, n_test=10, n_reference=50,
        seed=0, device="cpu",
    ).true_model


def _mdn_model():
    """Neural-mechanism (MDN) model — the regime where the autograd-graph
    waste actually bites (per-particle graphs over MLP forward passes)."""
    return make_synthetic_bn(
        family="continuous_nongauss", n_nodes=4, max_in_degree=2,
        edge_density=0.50,
        n_train=50, n_test=10, n_reference=50,
        seed=0, device="cpu",
    ).true_model


def _assert_grad_free(t: torch.Tensor, what: str) -> None:
    assert not t.requires_grad, f"{what} must not require grad at query time"
    assert t.grad_fn is None, f"{what} must not carry an autograd graph"


def test_lw_discrete_query_outputs_carry_no_autograd_state() -> None:
    model = _discrete_model()
    topo = model.dag.topological_order()
    # Design check: mechanism params DO require grad after eval() — this is
    # exactly why the query path must opt out of autograd itself.
    assert any(
        p.requires_grad
        for mech in model.mechanisms.values() for p in mech.parameters()
    )
    eng = LikelihoodWeightingEngine(n_samples=64)
    probs = eng.query(model, [topo[-1]], {topo[0]: torch.tensor([0])})
    _assert_grad_free(probs, "LW discrete posterior")


def test_lw_continuous_query_outputs_carry_no_autograd_state() -> None:
    model = _mdn_model()
    topo = model.dag.topological_order()
    eng = LikelihoodWeightingEngine(n_samples=64)
    weights, samples = eng.query(
        model, [topo[-1]], {topo[0]: torch.tensor([0.1])},
    )
    _assert_grad_free(weights, "LW weights")
    _assert_grad_free(samples, "LW samples")


def test_lw_query_batch_and_ais_fallback_inherit_the_contract() -> None:
    """``query_batch`` delegates to ``query``; the untrained AIS engine
    falls back to the inherited LW path — both must be grad-free."""
    model = _discrete_model()
    topo = model.dag.topological_order()
    ev = {topo[0]: torch.tensor([0, 1, 2])}

    lw = LikelihoodWeightingEngine(n_samples=64)
    probs = lw.query_batch(model, [topo[-1]], ev)
    _assert_grad_free(probs, "LW query_batch posterior")

    ais = AmortizedISEngine(n_samples=64)   # recognition_net None → LW path
    probs = ais.query(model, [topo[-1]], {topo[0]: torch.tensor([0])})
    _assert_grad_free(probs, "AIS (LW-fallback) posterior")


def test_mechanism_params_still_require_grad_after_query() -> None:
    """Fit-ability is unaffected: after a query, mechanism parameters still
    require grad and gradients still flow through ``log_prob``."""
    model = _mdn_model()
    topo = model.dag.topological_order()
    eng = LikelihoodWeightingEngine(n_samples=64)
    eng.query(model, [topo[-1]], {topo[0]: torch.tensor([0.1])})

    root = topo[0]
    mech = model.mechanisms[root]
    params = [p for p in mech.parameters() if p.requires_grad]
    assert params, "mechanism params must still require grad after a query"

    lp = mech.log_prob(torch.randn(8, mech.output_dim), None)
    assert lp.grad_fn is not None, (
        "log_prob outside the query path must still build an autograd graph"
    )
    lp.sum().backward()
    assert any(p.grad is not None for p in params), (
        "gradients must still flow to mechanism params after a query"
    )
