"""PomegranateAdapter.extract_learned_cpts + parameter-recovery verification (#109 PR 4).

Verifies the pomegranate side of the recovery metric:
  * cross-check A — extracted CPTs equal an INDEPENDENT numpy MLE counting on the
    same training data with declared cardinalities + NaN->uniform fill (pins the
    extraction against the adapter's conceptual declared-grid counting, not its
    stored representation);
  * cross-check B — held-out mean LL from the extracted CPTs equals pomegranate's
    own model.log_probability (orthogonal oracle);
  * +inf KL: pomegranate (unsmoothed MLE) with a crafted hard zero diverges
    (KL=+inf, status="ok", TV finite in [0,1]); pgmpy-bayes (BDeu) on the SAME
    data stays finite — the unsmoothed-vs-smoothed split;
  * NaN unseen-config fill: an entirely-unseen parent config -> uniform 1/K,
    not NaN, metric finite;
  * determinism: extract_learned_cpts bit-identical across calls.

NOTE: pomegranate is discrete-only via is_applicable; the measurement's
not_applicable path is structurally UNREACHABLE for it (continuous cells never
reach fit — the applicability gate emits a not_supported sentinel first). So
there is intentionally NO not_applicable test here.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from benchmarking.adapters import PgmpyAdapter, PomegranateAdapter
from benchmarking.domains.base import BenchmarkProblem
from benchmarking.measurements import ParamLearningMeasurement


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
    from benchmarking.synthetic import make_synthetic_bn

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


def _parents_lex(node, dag):
    return sorted(p for (p, c) in dag if c == node)


def _numpy_mle_canonical(train, variables, dag):
    """Independent MLE counting -> canonical CPTs (declared grid, NaN->uniform)."""
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
        row = counts.sum(1, keepdims=True)
        probs = np.divide(counts, row, out=np.full_like(counts, 1.0 / card),
                          where=row > 0)   # unseen config -> uniform
        out[node] = torch.from_numpy(probs).float()
    return out


def _ll_from_cpts(cpts, data, variables, dag):
    n = next(iter(data.values())).reshape(-1).shape[0]
    total = torch.zeros(n, dtype=torch.float64)
    for node, (kind, card) in variables.items():
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


# ---- cross-check A: independent numpy MLE counting --------------------------

@pytest.mark.slow
def test_extract_matches_independent_numpy_counting():
    prob = _bn_problem(seed=2)
    adapter = PomegranateAdapter(device="cpu")
    adapter.fit(prob)
    extracted = adapter.extract_learned_cpts()
    counted = _numpy_mle_canonical(prob.train_data, prob.variables, prob.dag)

    assert set(extracted) == set(counted)
    for node in extracted:
        assert extracted[node].shape == counted[node].shape, node
        assert torch.allclose(extracted[node], counted[node], atol=1e-5), node


# ---- cross-check B: held-out LL via pomegranate's own log_probability --------

@pytest.mark.slow
def test_extracted_ll_matches_pomegranate_log_probability():
    prob = _bn_problem(seed=3)
    adapter = PomegranateAdapter(device="cpu")
    adapter.fit(prob)
    extracted = adapter.extract_learned_cpts()

    # LL from the extracted canonical CPTs.
    ll_extract = _ll_from_cpts(extracted, prob.test_data, prob.variables, prob.dag)

    # pomegranate's own joint log-prob; columns in the model's distribution
    # (topological) order. Model + data both on the adapter's (cpu) device.
    cols = torch.stack(
        [prob.test_data[n].reshape(-1).long() for n in adapter._topo], dim=1
    ).to(adapter.device)
    ll_pome = float(adapter.model.log_probability(cols).mean())

    assert math.isfinite(ll_extract) and math.isfinite(ll_pome)
    assert math.isclose(ll_extract, ll_pome, rel_tol=1e-4, abs_tol=1e-4)


# ---- +inf KL (unsmoothed) vs pgmpy-bayes finite (smoothed) ------------------

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


def test_pomegranate_inf_kl_contrasts_pgmpy_bayes_finite():
    prob = _hard_zero_problem()
    m = ParamLearningMeasurement()

    pome = PomegranateAdapter(device="cpu")
    pome.fit(prob)
    assert float(pome.extract_learned_cpts()["X1"][0, 1]) == 0.0   # hard zero
    by = {r.metric: r for r in m.measure(prob, pome, [], seed=0)}
    assert by["param_recovery_kl"].status == "ok"
    assert math.isinf(by["param_recovery_kl"].value)               # unsmoothed -> +inf
    assert by["param_recovery_tv"].status == "ok"
    assert 0.0 <= by["param_recovery_tv"].value <= 1.0
    assert math.isfinite(by["param_recovery_tv"].value)

    bayes = PgmpyAdapter(param_method="bayes", inference_method="ve")
    bayes.fit(prob)
    by_b = {r.metric: r for r in m.measure(prob, bayes, [], seed=0)}
    assert math.isfinite(by_b["param_recovery_kl"].value)          # smoothed -> finite


# ---- NaN unseen-config fill -------------------------------------------------

def test_unseen_config_filled_uniform_not_nan():
    variables = {"X0": ("discrete", 3), "X1": ("discrete", 2)}
    dag = [("X0", "X1")]
    # X0 declared card 3 but value 2 never observed -> config X0=2 unseen.
    train = {"X0": torch.tensor([0, 0, 1, 1]).reshape(-1, 1),
             "X1": torch.tensor([0, 1, 0, 1]).reshape(-1, 1)}
    prob = BenchmarkProblem(
        name="n", dag=dag, variables=variables, train_data=train,
        test_data=train, queries=[], true_model=None, family="discrete",
        problem_id="2", seed=0,
    )
    a = PomegranateAdapter(device="cpu")
    a.fit(prob)
    x1 = a.extract_learned_cpts()["X1"]                # [3, 2], row 2 = X0=2 unseen
    assert not torch.isnan(x1).any()
    assert torch.allclose(x1[2], torch.tensor([0.5, 0.5]), atol=1e-6)
    assert torch.allclose(x1.sum(-1), torch.ones(3), atol=1e-5)


# ---- determinism ------------------------------------------------------------

@pytest.mark.slow
def test_extract_is_deterministic():
    prob = _bn_problem(seed=5)
    a = PomegranateAdapter(device="cpu")
    a.fit(prob)
    first = a.extract_learned_cpts()
    second = a.extract_learned_cpts()
    assert set(first) == set(second)
    for node in first:
        assert torch.equal(first[node], second[node]), node


# ---- canonical multi-parent layout guard ------------------------------------

@pytest.mark.slow
def test_canonical_row_index_multiparent():
    """Two-parent node: the canonical row for (X0=a, X1=b) matches an explicit
    index into the stored dag-order CPT (catches permute/order surprises)."""
    variables = {"X0": ("discrete", 2), "X1": ("discrete", 2), "X2": ("discrete", 3)}
    # dag-edge order puts X1 before X0 deliberately (reverse of lex).
    dag = [("X1", "X2"), ("X0", "X2")]
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
    a = PomegranateAdapter(device="cpu")
    a.fit(prob)
    canonical = a.extract_learned_cpts()["X2"]          # [4, 3], canonical (X0,X1)

    dist = a.model.distributions[a._node_to_idx["X2"]]
    raw = dist.probs[0].detach().cpu().float()          # [*pa_cards(dag order), K]
    parents_dag = [p for p, c in dag if c == "X2"]      # ['X1','X0']
    for a0 in (0, 1):
        for b1 in (0, 1):
            # canonical: parents sorted = [X0, X1]; row index = a0*2 + b1
            row = canonical[a0 * 2 + b1]
            # stored: index raw by dag-order axes (X1 first, then X0)
            assign = {"X0": a0, "X1": b1}
            expected = raw[assign[parents_dag[0]], assign[parents_dag[1]]]
            assert torch.allclose(row, expected, atol=1e-6), (a0, b1)
