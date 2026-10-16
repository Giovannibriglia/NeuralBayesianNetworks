"""Certified sampling answers and barren-node pruning in likelihood weighting."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nbn.core.network import NeuralBayesianNetwork
from nbn.inference.certified import certified_query, evidence_weight_bound
from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
from nbn.inference.tensor_ve import TensorVariableElimination
from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism


def _random_bn(seed=0, n=9, k=3):
    g = torch.Generator().manual_seed(seed)
    names = [f"V{i}" for i in range(n)]
    edges = []
    for j in range(1, n):
        for i in range(max(0, j - 3), j):
            if torch.rand((), generator=g) < 0.6:
                edges.append((names[i], names[j]))
    m = NeuralBayesianNetwork(edges, variables={v: ("discrete", k) for v in names},
                              device="cpu")
    for v in names:
        pa = m.dag.parents(v)
        mech = CategoricalTableMechanism()
        mech._logits = nn.Parameter(torch.randn(k ** len(pa), k, generator=g))
        mech._n_classes = k
        mech._set_parent_layout([k] * len(pa))
        mech._class_values = torch.arange(k, dtype=torch.float)
        mech.output_dim = 1
        m.set_mechanism(v, mech)
    return m, names


def test_categorical_sampler_matches_cpt():
    m, names = _random_bn(1)
    v = next(x for x in names if m.dag.parents(x))
    mech = m.mechanisms[v]
    k = 3
    rows = mech._logits.shape[0]
    pa_vals = torch.tensor([[int(c) for c in _digits(r, k, len(m.dag.parents(v)))]
                            for r in range(rows)])
    torch.manual_seed(0)
    n = 40_000
    x = mech.sample(pa_vals.repeat_interleave(n, 0), 1).reshape(rows, n).long()
    freq = torch.stack([torch.bincount(x[r], minlength=k).float() / n for r in range(rows)])
    p = torch.softmax(mech._logits.detach(), -1)
    assert (freq - p).abs().max() < 5 * (0.25 / n) ** 0.5 * 2


def _digits(r, k, p):
    out = []
    for _ in range(p):
        out.append(r % k)
        r //= k
    return out[::-1]


def test_zero_probability_class_never_sampled():
    mech = CategoricalTableMechanism()
    mech._logits = nn.Parameter(torch.tensor([[0.0, -float("inf"), 1.0]]))
    mech._n_classes = 3
    mech._set_parent_layout([])
    mech._class_values = torch.arange(3, dtype=torch.float)
    mech._set_parent_layout([1])
    x = mech.sample(torch.zeros(100_000, 1), 1).reshape(-1)
    assert not bool((x == 1).any())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lw_pruning_same_answer_as_unpruned(seed):
    m, names = _random_bn(seed)
    tgt, ev = names[3], {names[6]: 1, names[1]: 0}
    exact = m.query([tgt], evidence=ev, engine=TensorVariableElimination())
    torch.manual_seed(seed)
    a = m.query([tgt], evidence=ev,
                engine=LikelihoodWeightingEngine(200_000, prune_barren=True))
    b = m.query([tgt], evidence=ev,
                engine=LikelihoodWeightingEngine(200_000, prune_barren=False))
    assert (a - exact).abs().max() < 0.02
    assert (b - exact).abs().max() < 0.02


def test_lw_pruning_skips_barren_nodes():
    m, names = _random_bn(0)
    eng = LikelihoodWeightingEngine(1000, prune_barren=True)
    calls = []
    for v in names:
        mech = m.mechanisms[v]
        orig = mech.sample

        def spy(*a, _v=v, _o=orig, **k):
            calls.append(_v)
            return _o(*a, **k)

        mech.sample = spy
    m.query([names[2]], evidence={names[1]: 0}, engine=eng)
    import networkx as nx
    anc = nx.ancestors(m.dag.networkx_graph, names[2]) | nx.ancestors(
        m.dag.networkx_graph, names[1]) | {names[2]}
    assert set(calls) <= anc


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_certified_query_covers_exact(seed):
    m, names = _random_bn(seed)
    tgt = names[4]
    ev = {names[7]: 2, names[2]: 1}
    exact = m.query([tgt], evidence=ev, engine=TensorVariableElimination()).double()
    torch.manual_seed(seed)
    a = certified_query(m, tgt, ev, n_samples=100_000, delta=0.05, chunk=30_000)
    assert a.n_samples == 100_000
    assert bool((a.lower <= exact + 1e-9).all()) and bool((exact <= a.upper + 1e-9).all())
    assert bool((a.lower <= a.estimate).all()) and bool((a.estimate <= a.upper).all())
    assert a.max_width < 0.5
    assert a.evidence_prob_lower <= a.evidence_prob_upper


def test_certified_query_no_evidence_and_bound():
    m, names = _random_bn(5)
    a = certified_query(m, names[-1], {}, n_samples=50_000)
    exact = m.query([names[-1]], engine=TensorVariableElimination()).double()
    assert bool((a.lower <= exact).all() and (exact <= a.upper).all())
    assert a.weight_bound == 1.0
    u = evidence_weight_bound(m, {names[3]: 0})
    assert 0.0 < u <= 1.0


def test_certified_query_rejects_continuous_evidence():
    m = NeuralBayesianNetwork([("A", "B")], variables={"A": ("continuous", 1),
                                                        "B": ("discrete", 2)})
    from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism
    m.set_mechanism("A", LinearGaussianMechanism())
    with pytest.raises(NotImplementedError):
        evidence_weight_bound(m, {"A": 0.3})


def test_interval_sets_identical_to_per_node():
    from nbn.uncertainty import interval_set, interval_sets
    m, names = _random_bn(3)
    torch.manual_seed(0)
    data = {k: v.reshape(-1).long() for k, v in m.sample(n=500).items()}
    for meth in ("kl", "hoeffding"):
        for sampling in ("iid", "trajectory"):
            b = interval_sets(m, data, names, delta=0.01, method=meth, sampling=sampling)
            for v in names:
                a = interval_set(m, data, v, delta=0.01, method=meth, sampling=sampling)
                assert torch.equal(a.lower, b[v].lower) and torch.equal(a.upper, b[v].upper)
                assert torch.equal(a.unseen, b[v].unseen) and a.level == b[v].level


def test_lw_particle_chunking_same_answer():
    m, names = _random_bn(4)
    tgt, ev = names[5], {names[8]: 1}
    exact = m.query([tgt], evidence=ev, engine=TensorVariableElimination())
    eng = LikelihoodWeightingEngine(60_000)
    eng.max_buffer_elements = 10 * 1024  # forces several passes
    assert eng._particle_chunk(1, len(names)) < 60_000
    torch.manual_seed(0)
    p, ess = m.query([tgt], evidence=ev, engine=eng, return_ess=True)
    assert p.shape == exact.shape and (p - exact).abs().max() < 0.02
    assert 0.0 < float(ess) <= 1.0
    # continuous-style return path (weights, samples) keeps all particles
    w, x = eng.query(m, [tgt, names[6]], {names[8]: torch.tensor([1])})
    assert w.shape == (1, 60_000) and x.shape == (1, 60_000, 2)
