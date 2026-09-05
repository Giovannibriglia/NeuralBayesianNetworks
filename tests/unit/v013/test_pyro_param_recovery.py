"""PyroAdapter.extract_learned_cpts + parameter-recovery verification (#109 PR 5).

Verifies the pyro side of the recovery metric (the fifth and final adapter):
  * cross-check A — extracted CPTs equal an INDEPENDENT numpy LAPLACE counting
    (counts+1)/(sum+K) on the same training data with declared cardinalities.
    Must match alpha=1, NOT raw MLE — pins that extraction respects pyro's
    fit-time smoothing convention;
  * cross-check B — held-out mean LL from the extracted canonical CPTs equals
    the LL computed from the NATIVE flat self._cpts via dag-order indexing
    (orthogonal layout -> pins the permute);
  * finite-not-+inf — under the SAME hard-zero condition that makes pgmpy-mle
    and pomegranate diverge, pyro KL stays finite (Laplace smoothing); contrast
    pgmpy-mle +inf on identical data;
  * not_applicable gate — pyro is mixed-applicable, so a continuous_lg cell
    yields recovery not_applicable (LL not_supported). This is the case PR 4
    could not test (pomegranate was discrete-only);
  * determinism — extract bit-identical across calls;
  * multi-parent canonical layout (dag-edge order -> lex permute).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from nbn.bench.adapters import PgmpyAdapter, PyroAdapter
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


def _pyro(problem):
    a = PyroAdapter(mechanism="empirical", inference_method=None, device="cpu")
    a.fit(problem)
    return a


def _bn_problem(seed: int, family: str = "discrete", n_nodes: int = 4):
    from nbn.bench.synthetic import make_synthetic_bn

    bn = make_synthetic_bn(
        n_nodes=n_nodes, family=family, cardinality=3, edge_density=0.5,
        max_in_degree=2, n_train=3000, n_test=400, n_reference=200,
        seed=seed, device="cpu",
    )
    return BenchmarkProblem(
        name=bn.name, dag=list(bn.dag.edges()), variables=bn.variable_specs,
        train_data=bn.train_data, test_data=bn.test_data, queries=[],
        true_model=bn.true_model, family=family, problem_id=str(n_nodes),
        seed=seed,
    )


def _parents_lex(node, dag):
    return sorted(p for (p, c) in dag if c == node)


def _numpy_laplace_canonical(train, variables, dag):
    """Independent Laplace (alpha=1) counting -> canonical CPTs, declared grid."""
    out = {}
    n = next(iter(train.values())).reshape(-1).shape[0]
    for node, (kind, card) in variables.items():
        parents = _parents_lex(node, dag)
        pcards = [variables[p][1] for p in parents]
        n_cfg = int(np.prod(pcards)) if pcards else 1
        counts = np.zeros((n_cfg, card), dtype=np.float64)
        idx = np.zeros(n, dtype=np.int64)
        stride = 1
        for d in reversed(range(len(parents))):
            idx += train[parents[d]].reshape(-1).cpu().numpy().astype(np.int64) * stride
            stride *= pcards[d]
        xv = train[node].reshape(-1).cpu().numpy().astype(np.int64)
        for i in range(n):
            counts[idx[i], xv[i]] += 1.0
        probs = (counts + 1.0) / (counts.sum(1, keepdims=True) + card)   # Laplace
        out[node] = torch.from_numpy(probs).float()
    return out


def _canonical_ll(cpts, data, variables, dag):
    n = next(iter(data.values())).reshape(-1).shape[0]
    total = torch.zeros(n, dtype=torch.float64)
    for node, (kind, card) in variables.items():
        if kind != "discrete":
            continue
        parents = _parents_lex(node, dag)
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


def _native_ll(adapter, data):
    """Held-out LL from the NATIVE flat self._cpts (dag-order config index)."""
    n = next(iter(data.values())).reshape(-1).shape[0]
    total = torch.zeros(n, dtype=torch.float64)
    for node in adapter._topo:
        kind, _ = adapter.problem.variables[node]
        if kind != "discrete":
            continue
        cpt = adapter._cpts[node].detach().cpu().double()   # [n_pa, K] native
        cpt_parents = adapter._cpt_parents[node]
        x = data[node].long().reshape(-1)
        if not cpt_parents:
            total += torch.log(cpt[0, x])
        else:
            pcards = [adapter._cards[p] for p in cpt_parents]
            idx = torch.zeros(n, dtype=torch.long)
            stride = 1
            for d in reversed(range(len(cpt_parents))):
                idx = idx + data[cpt_parents[d]].long().reshape(-1) * stride
                stride *= pcards[d]
            total += torch.log(cpt[idx, x])
    return float(total.mean())


# ---- cross-check A: independent Laplace counting ----------------------------

@pytest.mark.slow
def test_extract_matches_independent_laplace_counting():
    prob = _bn_problem(seed=2)
    adapter = _pyro(prob)
    extracted = adapter.extract_learned_cpts()
    counted = _numpy_laplace_canonical(prob.train_data, prob.variables, prob.dag)

    assert set(extracted) == set(counted)
    for node in extracted:
        assert extracted[node].shape == counted[node].shape, node
        assert torch.allclose(extracted[node], counted[node], atol=1e-5), node


# ---- cross-check B: canonical LL == native flat LL --------------------------

@pytest.mark.slow
def test_canonical_ll_matches_native_flat_ll():
    prob = _bn_problem(seed=3)
    adapter = _pyro(prob)
    extracted = adapter.extract_learned_cpts()

    ll_canon = _canonical_ll(extracted, prob.test_data, prob.variables, prob.dag)
    ll_native = _native_ll(adapter, prob.test_data)
    assert math.isfinite(ll_canon) and math.isfinite(ll_native)
    assert math.isclose(ll_canon, ll_native, rel_tol=1e-5, abs_tol=1e-5)


# ---- finite (smoothed) vs pgmpy-mle +inf (unsmoothed) -----------------------

def _hard_zero_problem():
    from nbn import NeuralBayesianNetwork

    variables = {"X0": ("discrete", 2), "X1": ("discrete", 2)}
    dag = [("X0", "X1")]
    tm = NeuralBayesianNetwork(dag, variables=variables, device="cpu")
    tm.set_mechanism("X0", _cat_mech(torch.tensor([[0.5, 0.5]]), []))
    tm.set_mechanism("X1", _cat_mech(torch.tensor([[0.7, 0.3], [0.4, 0.6]]), [2]))
    train = {"X0": torch.tensor([0, 0, 0, 0, 1, 1, 1]).reshape(-1, 1),
             "X1": torch.tensor([0, 0, 0, 0, 0, 1, 1]).reshape(-1, 1)}
    test = {"X0": torch.tensor([0, 1, 0, 1]).reshape(-1, 1),
            "X1": torch.tensor([0, 1, 1, 0]).reshape(-1, 1)}
    return BenchmarkProblem(
        name="tiny", dag=dag, variables=variables, train_data=train,
        test_data=test, queries=[], true_model=tm, family="discrete",
        problem_id="2", seed=0,
    )


def test_pyro_finite_kl_contrasts_pgmpy_mle_inf():
    prob = _hard_zero_problem()
    m = ParamLearningMeasurement()

    pyro = _pyro(prob)
    # Laplace smoothing -> no hard zero even where the class has zero count.
    assert float(pyro.extract_learned_cpts()["X1"][0, 1]) > 0.0
    by = {r.metric: r for r in m.measure(prob, pyro, [], seed=0)}
    assert by["param_recovery_kl"].status == "ok"
    assert math.isfinite(by["param_recovery_kl"].value)            # smoothed -> finite
    assert 0.0 <= by["param_recovery_tv"].value <= 1.0

    mle = PgmpyAdapter(param_method="mle", inference_method="ve")
    mle.fit(prob)
    by_m = {r.metric: r for r in m.measure(prob, mle, [], seed=0)}
    assert math.isinf(by_m["param_recovery_kl"].value)             # unsmoothed -> +inf


# ---- not_applicable gate (pyro is mixed-applicable) -------------------------

@pytest.mark.slow
def test_continuous_cell_not_applicable():
    prob = _bn_problem(seed=1, family="continuous_lg")
    adapter = _pyro(prob)
    by = {r.metric: r for r in ParamLearningMeasurement().measure(prob, adapter, [], seed=1)}
    for mname in ("param_recovery_tv", "param_recovery_kl"):
        assert by[mname].status == "not_applicable"
    # pyro has no supports_scoring -> log_likelihood is not_supported.
    assert by["log_likelihood"].status == "not_supported"


# ---- determinism ------------------------------------------------------------

@pytest.mark.slow
def test_extract_is_deterministic():
    prob = _bn_problem(seed=5)
    a = _pyro(prob)
    first = a.extract_learned_cpts()
    second = a.extract_learned_cpts()
    assert set(first) == set(second)
    for node in first:
        assert torch.equal(first[node], second[node]), node


# ---- multi-parent canonical layout ------------------------------------------

@pytest.mark.slow
def test_canonical_row_index_multiparent():
    """Two parents stored in dag-edge order [X1, X0]; the canonical row for
    (X0=a, X1=b) must match the native flat CPT at the dag-order index."""
    variables = {"X0": ("discrete", 2), "X1": ("discrete", 2), "X2": ("discrete", 3)}
    dag = [("X1", "X2"), ("X0", "X2")]               # X2 parents (dag): [X1, X0]
    rng = torch.Generator().manual_seed(0)
    n = 4000
    x0 = torch.randint(0, 2, (n, 1), generator=rng)
    x1 = torch.randint(0, 2, (n, 1), generator=rng)
    x2 = ((x0 + 2 * x1) % 3)
    prob = BenchmarkProblem(
        name="mp", dag=dag, variables=variables,
        train_data={"X0": x0, "X1": x1, "X2": x2},
        test_data={"X0": x0, "X1": x1, "X2": x2}, queries=[],
        true_model=None, family="discrete", problem_id="3", seed=0,
    )
    a = _pyro(prob)
    canonical = a.extract_learned_cpts()["X2"]        # [4, 3], canonical (X0, X1)

    cpt_parents = a._cpt_parents["X2"]                # ['X1', 'X0']
    native = a._cpts["X2"].detach().cpu().float()     # [4, 3] flat, [X1,X0] order
    pcards = [a._cards[p] for p in cpt_parents]
    for a0 in (0, 1):
        for b1 in (0, 1):
            # native flat index over [X1, X0] (first slowest)
            assign = {"X0": a0, "X1": b1}
            flat = 0
            stride = 1
            for d in reversed(range(len(cpt_parents))):
                flat += assign[cpt_parents[d]] * stride
                stride *= pcards[d]
            # canonical row index over [X0, X1] (first slowest)
            row = canonical[a0 * 2 + b1]
            assert torch.allclose(row, native[flat], atol=1e-6), (a0, b1)
