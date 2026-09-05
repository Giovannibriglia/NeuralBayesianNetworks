"""PgmpyAdapter.extract_learned_cpts + parameter-recovery verification (#109 PR 3).

Verifies the pgmpy side of the recovery metric:
  * cross-check A — the TabularCPD -> canonical reshape/permute matches pgmpy's
    OWN cpd.get_value at every (parent config, class), an oracle independent of
    the extraction's internal reshape;
  * cross-check B — representational correctness: held-out mean LL from the
    extracted CPTs equals the LL from an independent numpy MLE-counting
    reimplementation on the declared grid;
  * +inf KL: pgmpy-mle with a crafted config-specific hard zero diverges
    (KL=+inf, status="ok", TV finite in [0,1]); pgmpy-bayes stays finite;
  * the lg path: recovery not_applicable (LL not_supported — pgmpy has no
    score_data yet);
  * determinism: extract_learned_cpts is bit-identical across calls (no RNG in
    the re-estimation).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from nbn.bench.adapters import PgmpyAdapter
from nbn.bench.domains.base import BenchmarkProblem
from nbn.bench.measurements import ParamLearningMeasurement


# ---- fixtures ---------------------------------------------------------------

def _cat_mech(probs: torch.Tensor, parent_cards: list[int]):
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


def _bn_problem(seed: int = 1, n_nodes: int = 4):
    """A discrete synthetic problem with full state coverage at this n_train."""
    from nbn.bench.synthetic import make_synthetic_bn

    bn = make_synthetic_bn(
        n_nodes=n_nodes, family="discrete", cardinality=3, edge_density=0.5,
        max_in_degree=2, n_train=3000, n_test=400, n_reference=200,
        seed=seed, device="cpu",
    )
    return BenchmarkProblem(
        name=bn.name, dag=list(bn.dag.edges()), variables=bn.variable_specs,
        train_data=bn.train_data, test_data=bn.test_data, queries=[],
        true_model=bn.true_model, family="discrete", problem_id=str(n_nodes),
        seed=seed,
    )


def _reestimate_cpds(problem, param_method):
    """Re-estimate full-declared-grid TabularCPDs the way extract does, for use
    as the cross-check-A oracle (pgmpy's own get_value)."""
    import warnings

    import pandas as pd
    from pgmpy.models import DiscreteBayesianNetwork

    df = pd.DataFrame({
        k: v.cpu().long().reshape(-1).numpy()
        for k, v in problem.train_data.items()
    })
    bn = DiscreteBayesianNetwork(problem.dag)
    for node in problem.variables:
        if node not in bn.nodes():
            bn.add_node(node)
    state_names = {
        n: list(range(c)) for n, (k, c) in problem.variables.items()
        if k == "discrete"
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if param_method == "bayes":
            from pgmpy.estimators import BayesianEstimator
            cpds = BayesianEstimator(
                model=bn, data=df, state_names=state_names).get_parameters()
        else:
            from pgmpy.estimators import MaximumLikelihoodEstimator
            cpds = MaximumLikelihoodEstimator(
                model=bn, data=df, state_names=state_names).get_parameters()
    return {c.variable: c for c in cpds}


def _parents_of(node, dag, variables):
    return sorted(p for (p, c) in dag if c == node)


def _ll_from_cpts(cpts, data, variables, dag):
    """Mean held-out joint log-likelihood from canonical CPTs."""
    n = next(iter(data.values())).reshape(-1).shape[0]
    total = torch.zeros(n, dtype=torch.float64)
    for node, (kind, card) in variables.items():
        parents = _parents_of(node, dag, variables)
        cpt = cpts[node].double()
        x = data[node].long().reshape(-1)
        if not parents:
            total += torch.log(cpt[0, x])
        else:
            pcards = [variables[p][1] for p in parents]
            idx = torch.zeros(n, dtype=torch.long)
            stride = 1
            for d in reversed(range(len(parents))):
                idx = idx + data[parents[d]].long().reshape(-1) * stride
                stride *= pcards[d]
            total += torch.log(cpt[idx, x])
    return float(total.mean())


def _mle_counting_cpts(train, variables, dag):
    """Independent numpy MLE counting on the declared grid -> canonical CPTs."""
    out = {}
    n = next(iter(train.values())).reshape(-1).shape[0]
    for node, (kind, card) in variables.items():
        parents = _parents_of(node, dag, variables)
        pcards = [variables[p][1] for p in parents]
        n_cfg = int(np.prod(pcards)) if pcards else 1
        counts = np.zeros((n_cfg, card), dtype=np.float64)
        cfg_idx = np.zeros(n, dtype=np.int64)
        stride = 1
        for d in reversed(range(len(parents))):
            cfg_idx += train[parents[d]].reshape(-1).cpu().numpy().astype(np.int64) * stride
            stride *= pcards[d]
        xv = train[node].reshape(-1).cpu().numpy().astype(np.int64)
        for i in range(n):
            counts[cfg_idx[i], xv[i]] += 1.0
        row = counts.sum(1, keepdims=True)
        probs = np.divide(counts, row, out=np.full_like(counts, 1.0 / card),
                          where=row > 0)   # unseen config -> uniform (pgmpy convention)
        out[node] = torch.from_numpy(probs).float()
    return out


# ---- cross-check A: reshape/permute vs pgmpy get_value ----------------------

@pytest.mark.slow
@pytest.mark.parametrize("param_method", ["mle", "bayes"])
def test_canonical_matches_pgmpy_get_value(param_method):
    prob = _bn_problem(seed=2)
    adapter = PgmpyAdapter(param_method=param_method, inference_method="ve")
    adapter.fit(prob)
    extracted = adapter.extract_learned_cpts()
    cpd_by = _reestimate_cpds(prob, param_method)

    for node, (kind, card) in prob.variables.items():
        cpd = cpd_by[node]
        canon_parents = sorted(cpd.variables[1:])
        pcards = [prob.variables[p][1] for p in canon_parents]
        for cfg_idx, cfg in enumerate(itertools.product(*(range(c) for c in pcards))):
            assign = {p: cfg[i] for i, p in enumerate(canon_parents)}
            for cls in range(card):
                expected = float(cpd.get_value(**{node: cls, **assign}))
                got = float(extracted[node][cfg_idx, cls])
                assert abs(got - expected) < 1e-6, (node, cfg, cls, got, expected)


# ---- cross-check B: representational correctness via held-out LL -------------

@pytest.mark.slow
def test_extracted_cpts_match_independent_mle_counting():
    prob = _bn_problem(seed=3)
    adapter = PgmpyAdapter(param_method="mle", inference_method="ve")
    adapter.fit(prob)
    extracted = adapter.extract_learned_cpts()
    counted = _mle_counting_cpts(prob.train_data, prob.variables, prob.dag)

    ll_extract = _ll_from_cpts(extracted, prob.test_data, prob.variables, prob.dag)
    ll_counted = _ll_from_cpts(counted, prob.test_data, prob.variables, prob.dag)
    assert math.isfinite(ll_extract) and math.isfinite(ll_counted)
    assert math.isclose(ll_extract, ll_counted, rel_tol=1e-6, abs_tol=1e-6)


# ---- +inf KL on mle, finite on bayes ----------------------------------------

def _hard_zero_problem():
    from nbn import NeuralBayesianNetwork

    variables = {"X0": ("discrete", 2), "X1": ("discrete", 2)}
    dag = [("X0", "X1")]
    tm = NeuralBayesianNetwork(dag, variables=variables, device="cpu")
    tm.set_mechanism("X0", _cat_mech(torch.tensor([[0.5, 0.5]]), []))
    # true supports X1=1 | X0=0 (prob 0.3) — the class mle will zero out
    tm.set_mechanism("X1", _cat_mech(torch.tensor([[0.7, 0.3], [0.4, 0.6]]), [2]))
    # crafted train: X0=0 -> X1 ALWAYS 0 (zero count for class 1 | X0=0);
    # class 1 IS observed marginally at X0=1 so it stays in the grid.
    train = {"X0": torch.tensor([0, 0, 0, 0, 1, 1, 1]),
             "X1": torch.tensor([0, 0, 0, 0, 0, 1, 1])}
    test = {"X0": torch.tensor([0, 1, 0, 1]), "X1": torch.tensor([0, 1, 1, 0])}
    return BenchmarkProblem(
        name="tiny", dag=dag, variables=variables, train_data=train,
        test_data=test, queries=[], true_model=tm, family="discrete",
        problem_id="2", seed=0,
    )


def test_mle_hard_zero_yields_inf_kl_bayes_finite():
    prob = _hard_zero_problem()
    m = ParamLearningMeasurement()

    a_mle = PgmpyAdapter(param_method="mle", inference_method="ve")
    a_mle.fit(prob)
    # learned column for X0=0 must be a hard zero on class 1.
    assert float(a_mle.extract_learned_cpts()["X1"][0, 1]) == 0.0
    by = {r.metric: r for r in m.measure(prob, a_mle, [], seed=prob.seed)}
    assert by["param_recovery_kl"].status == "ok"
    assert math.isinf(by["param_recovery_kl"].value)
    assert by["param_recovery_tv"].status == "ok"
    assert 0.0 <= by["param_recovery_tv"].value <= 1.0
    assert math.isfinite(by["param_recovery_tv"].value)

    a_bayes = PgmpyAdapter(param_method="bayes", inference_method="ve")
    a_bayes.fit(prob)
    by_b = {r.metric: r for r in m.measure(prob, a_bayes, [], seed=prob.seed)}
    assert by_b["param_recovery_kl"].status == "ok"
    assert math.isfinite(by_b["param_recovery_kl"].value)   # BDeu prior, no zero


# ---- lg gate ----------------------------------------------------------------

@pytest.mark.slow
def test_lg_recovery_not_applicable():
    from nbn.bench.synthetic import make_synthetic_bn

    bn = make_synthetic_bn(
        n_nodes=4, family="continuous_lg", cardinality=3, edge_density=0.5,
        max_in_degree=2, n_train=400, n_test=200, n_reference=200, seed=1,
        device="cpu",
    )
    prob = BenchmarkProblem(
        name=bn.name, dag=list(bn.dag.edges()), variables=bn.variable_specs,
        train_data=bn.train_data, test_data=bn.test_data, queries=[],
        true_model=bn.true_model, family="continuous_lg", problem_id="4", seed=1,
    )
    a = PgmpyAdapter(param_method="lg", inference_method="predict")
    a.fit(prob)
    assert a.extract_learned_cpts() == {}          # continuous path -> no CPTs
    by = {r.metric: r for r in ParamLearningMeasurement().measure(prob, a, [], seed=1)}
    for mname in ("param_recovery_tv", "param_recovery_kl"):
        assert by[mname].status == "not_applicable"
    # pgmpy has no score_data yet -> log_likelihood is not_supported (PR 4+).
    assert by["log_likelihood"].status == "not_supported"


# ---- determinism ------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("param_method", ["mle", "bayes"])
def test_extract_is_deterministic(param_method):
    prob = _bn_problem(seed=5)
    adapter = PgmpyAdapter(param_method=param_method, inference_method="ve")
    adapter.fit(prob)
    a = adapter.extract_learned_cpts()
    b = adapter.extract_learned_cpts()
    assert set(a) == set(b)
    for node in a:
        assert torch.equal(a[node], b[node]), node


# ---- bad state name -> loud error -------------------------------------------

def test_non_integer_state_raises():
    from nbn.bench.adapters.pgmpy_adapter import _state_axis_index

    with pytest.raises(ValueError, match="not an integer in"):
        _state_axis_index("X", ["lo", "hi"], 2)
    with pytest.raises(ValueError, match=r"do not cover"):
        _state_axis_index("X", [0, 1], 3)   # missing state 2
