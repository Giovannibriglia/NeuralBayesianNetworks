"""NBNAdapter.extract_learned_cpts + parameter-recovery verification (#109 PR 2).

Verifies the NBN side of the recovery metric end to end:
  * extract_learned_cpts returns canonical [n_parent_configs, K] CPTs for a
    fitted NBN (cat AND neuralcat), and the measurement's param_recovery_tv/kl
    match an independent recomputation — both against the measurement's own
    (deterministic) weights and against analytic parent-config weights;
  * the canonical row index for a multi-parent node matches forward() at a
    specific parent assignment (stride/order regression guard);
  * the gate: a discrete cell emits ok recovery rows, a continuous cell emits
    not_applicable while log_likelihood stays ok.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from benchmarking.adapters import NBNAdapter
from benchmarking.core.cpt_extraction import extract_discrete_cpts
from benchmarking.domains.base import BenchmarkProblem
from benchmarking.measurements import ParamLearningMeasurement
from benchmarking.metrics import param_recovery_kl, param_recovery_tv


# ---- fixed-CPT true-model construction --------------------------------------

def _cat_mech(probs: torch.Tensor, parent_cards: list[int]):
    """A CategoricalTableMechanism with CPT ``probs`` [n_parent_configs, K].

    Rows are indexed by the canonical mixed-radix convention (first parent
    slowest), matching ``_build_discrete_mechanism`` in the synthetic generator.
    """
    from nbn.mechanisms import CategoricalTableMechanism

    mech = CategoricalTableMechanism(alpha=0.0)
    k = int(probs.shape[-1])
    mech._logits = nn.Parameter(torch.log(probs.clamp_min(1e-12)))
    mech._n_classes = k
    mech._parent_cards = list(parent_cards)
    strides: list[int] = []
    stride = 1
    for c in reversed(parent_cards):
        strides.append(stride)
        stride *= c
    mech._parent_strides = list(reversed(strides))
    mech._class_values = torch.arange(k, dtype=torch.float)
    mech.output_dim = 1
    return mech


def _chain_true_model():
    """X0 -> X1 -> X2, binary, with FIXED hand-set CPTs. Returns (model, vars)."""
    from nbn import NeuralBayesianNetwork

    variables = {"X0": ("discrete", 2), "X1": ("discrete", 2), "X2": ("discrete", 2)}
    dag = [("X0", "X1"), ("X1", "X2")]
    model = NeuralBayesianNetwork(dag, variables=variables, device="cpu")
    model.set_mechanism("X0", _cat_mech(torch.tensor([[0.7, 0.3]]), []))
    model.set_mechanism("X1", _cat_mech(torch.tensor([[0.8, 0.2], [0.1, 0.9]]), [2]))
    model.set_mechanism("X2", _cat_mech(torch.tensor([[0.6, 0.4], [0.3, 0.7]]), [2]))
    return model, variables, dag


def _chain_problem(seed: int = 0, n_train: int = 4000, n_test: int = 500):
    model, variables, dag = _chain_true_model()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234 + seed)
        train = model.sample(n=n_train)
        test = model.sample(n=n_test)
    return BenchmarkProblem(
        name="chain", dag=dag, variables=variables,
        train_data=train, test_data=test, queries=[],
        true_model=model, family="discrete", problem_id="3", seed=seed,
    ), model, variables


# ---- TV / KL hand-recomputation against the measurement ---------------------

@pytest.mark.slow
@pytest.mark.parametrize("mechanism", ["cat", "neuralcat"])
def test_extract_and_recovery_match_recomputation(mechanism):
    problem, true_model, variables = _chain_problem(seed=0)

    adapter = NBNAdapter(mechanism=mechanism, engine=None, device="cpu")
    adapter.fit(problem, epochs=15)

    learned = adapter.extract_learned_cpts()
    true = extract_discrete_cpts(true_model, variables)

    # canonical layout: one entry per node, [n_parent_configs, K], rows sum to 1
    assert set(learned) == set(true) == {"X0", "X1", "X2"}
    assert tuple(learned["X0"].shape) == (1, 2)
    assert tuple(learned["X1"].shape) == (2, 2)
    assert tuple(learned["X2"].shape) == (2, 2)
    for cpt in learned.values():
        assert torch.allclose(cpt.sum(-1), torch.ones(cpt.shape[0]), atol=1e-5)

    m = ParamLearningMeasurement()
    rows = m.measure(problem, adapter, [], seed=problem.seed)
    by = {r.metric: r for r in rows}
    assert by["param_recovery_tv"].status == "ok"
    assert by["param_recovery_kl"].status == "ok"

    # (1) EXACT: measurement value == primitive on the measurement's own
    # (deterministic, same-seed) weights. Verifies the full assembly.
    nodes = list(true.keys())
    w = m._compute_weights(problem, true)            # same seed -> identical draw
    tv_exact = param_recovery_tv(
        [true[n] for n in nodes], [learned[n] for n in nodes], [w[n] for n in nodes]
    ).value
    kl_exact = param_recovery_kl(
        [true[n] for n in nodes], [learned[n] for n in nodes], [w[n] for n in nodes]
    ).value
    assert math.isclose(by["param_recovery_tv"].value, tv_exact, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(by["param_recovery_kl"].value, kl_exact, rel_tol=1e-9, abs_tol=1e-12)

    # (2) ANALYTIC by hand in numpy: parent-config weights propagated through the
    # chain, freq-weighted TV/KL over extracted true vs learned. Empirical (20k)
    # weights approximate these, so match to a sampling-error tolerance.
    t = {n: true[n].double().numpy() for n in nodes}
    le = {n: learned[n].double().numpy() for n in nodes}
    p_x0 = t["X0"][0]                                 # P(X0)
    p_x1 = p_x0 @ t["X1"]                             # P(X1) = sum_x0 P(x0) P(X1|x0)
    analytic_w = {"X0": np.array([1.0]), "X1": p_x0, "X2": p_x1}

    def _tv_rows(a, b):
        return 0.5 * np.abs(a - b).sum(-1)

    def _kl_rows(a, b):  # KL(true||learned), q unclamped
        mask = a > 0
        out = np.zeros(a.shape[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(mask, a * (np.log(np.clip(a, 1e-12, None)) - np.log(b)), 0.0)
        return term.sum(-1)

    tv_hand = np.mean([float((_tv_rows(t[n], le[n]) * analytic_w[n]).sum()) for n in nodes])
    kl_hand = np.mean([float((_kl_rows(t[n], le[n]) * analytic_w[n]).sum()) for n in nodes])
    assert math.isclose(by["param_recovery_tv"].value, tv_hand, abs_tol=0.02)
    assert math.isclose(by["param_recovery_kl"].value, kl_hand, abs_tol=0.05)


# ---- canonical row-index layout guard ---------------------------------------

@pytest.mark.slow
def test_canonical_row_index_multiparent():
    """For a node with two parents, the canonical row for (X0=a, X1=b) equals
    forward() at that assignment (catches stride/order surprises)."""
    from nbn import NeuralBayesianNetwork

    variables = {"X0": ("discrete", 2), "X1": ("discrete", 2), "X2": ("discrete", 3)}
    dag = [("X0", "X2"), ("X1", "X2")]
    model = NeuralBayesianNetwork(dag, variables=variables, device="cpu")
    model.set_mechanism("X0", _cat_mech(torch.tensor([[0.5, 0.5]]), []))
    model.set_mechanism("X1", _cat_mech(torch.tensor([[0.4, 0.6]]), []))
    # 4 parent configs (X0 in 0..1 slowest, X1 in 0..1 fastest), K=3.
    x2 = torch.tensor([
        [0.7, 0.2, 0.1],   # (X0=0, X1=0)
        [0.1, 0.8, 0.1],   # (X0=0, X1=1)
        [0.2, 0.2, 0.6],   # (X0=1, X1=0)
        [0.3, 0.3, 0.4],   # (X0=1, X1=1)
    ])
    model.set_mechanism("X2", _cat_mech(x2, [2, 2]))

    cpts = extract_discrete_cpts(model, variables)
    canonical = cpts["X2"]                            # [4, 3]
    assert tuple(canonical.shape) == (4, 3)

    mech_parents = list(model.dag.parents("X2"))      # mechanism's own column order
    for a in (0, 1):
        for b in (0, 1):
            vals = {"X0": a, "X1": b}
            pa = torch.tensor([[vals[p] for p in mech_parents]])
            expected = model.mechanisms["X2"].forward(pa).probs.reshape(-1)
            # canonical: parents sorted lex = [X0, X1]; first slowest -> a*2 + b
            row = canonical[a * 2 + b]
            assert torch.allclose(row, expected, atol=1e-6), (a, b)


# ---- gate: discrete ok / continuous not_applicable --------------------------

@pytest.mark.slow
def test_recovery_gate_discrete_ok_continuous_not_applicable():
    from benchmarking.synthetic import make_synthetic_bn

    def _problem(family):
        bn = make_synthetic_bn(
            n_nodes=4, family=family, cardinality=3, edge_density=0.5,
            max_in_degree=2, n_train=600, n_test=200, n_reference=200,
            seed=1, device="cpu",
        )
        return BenchmarkProblem(
            name=bn.name, dag=list(bn.dag.edges()), variables=bn.variable_specs,
            train_data=bn.train_data, test_data=bn.test_data, queries=[],
            true_model=bn.true_model, family=family, problem_id="4", seed=1,
        )

    m = ParamLearningMeasurement()

    # Discrete -> recovery ok.
    dp = _problem("discrete")
    da = NBNAdapter(mechanism="cat", engine=None, device="cpu")
    da.fit(dp, epochs=10)
    dby = {r.metric: r for r in m.measure(dp, da, [], seed=dp.seed)}
    assert dby["log_likelihood"].status == "ok"
    assert dby["param_recovery_tv"].status == "ok"
    assert dby["param_recovery_kl"].status == "ok"
    assert math.isfinite(dby["param_recovery_tv"].value)
    assert math.isfinite(dby["param_recovery_kl"].value)

    # Continuous -> recovery not_applicable, log_likelihood still ok.
    cp = _problem("continuous_lg")
    ca = NBNAdapter(mechanism="lg", engine=None, device="cpu")
    ca.fit(cp, epochs=10)
    cby = {r.metric: r for r in m.measure(cp, ca, [], seed=cp.seed)}
    assert cby["log_likelihood"].status == "ok"
    assert cby["param_recovery_tv"].status == "not_applicable"
    assert cby["param_recovery_kl"].status == "not_applicable"
