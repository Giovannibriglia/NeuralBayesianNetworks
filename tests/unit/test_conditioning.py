"""Conditioning (exact VE under a memory cap) and bounded conditioning.

Checked against plain exact VE: answers equal to float precision at every
budget, intervals that contain the exact posterior and tighten as more
instantiations are enumerated, defaults unchanged.
"""
import pytest
import torch
import torch.nn as nn

from nbn import NeuralBayesianNetwork
from nbn.inference._memory import OutOfMemoryError
from nbn.inference.conditioning import ConditioningLimitError, conditioned_query
from nbn.inference.hybrid import HybridRouter
from nbn.inference.tensor_ve import TensorVariableElimination
from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism


def _dense_bn(seed=0, n=16, k=4, window=6, p=0.7):
    g = torch.Generator().manual_seed(seed)
    names = [f"V{i}" for i in range(n)]
    edges = [(names[i], names[j]) for j in range(1, n) for i in range(max(0, j - window), j)
             if torch.rand((), generator=g) < p]
    m = NeuralBayesianNetwork(edges, variables=dict.fromkeys(names, ("discrete", k)),
                              device="cpu")
    for v in names:
        pa = m.dag.parents(v)
        mech = CategoricalTableMechanism()
        mech._logits = nn.Parameter(torch.randn(k ** len(pa), k, generator=g) * 2)
        mech._n_classes = k
        mech._set_parent_layout([k] * len(pa))
        mech._class_values = torch.arange(k, dtype=torch.float)
        mech.output_dim = 1
        m.set_mechanism(v, mech)
    return m, names


def _query(m, names):
    return names[3], {names[-1]: 1, names[9]: 0}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("frac", [1.0, 0.3, 0.1, 0.03])
def test_conditioning_equals_exact_ve(seed, frac):
    m, names = _dense_bn(seed)
    tgt, ev = _query(m, names)
    exact = m.query([tgt], evidence=ev, engine=TensorVariableElimination())
    peak = TensorVariableElimination().estimate_peak_bytes(m, tgt, ev)
    a = conditioned_query(m, tgt, ev, memory_budget=peak * frac)
    assert a.exact
    assert torch.allclose(a.estimate.float(), exact, atol=1e-6)
    if frac <= 0.1:
        assert len(a.cutset) >= 1 and a.n_instantiations > 1


def test_conditioning_with_do_equals_exact_ve():
    m, names = _dense_bn(3)
    tgt, ev = _query(m, names)
    do = {names[2]: 1}
    exact = m.query([tgt], evidence=ev, do=do, engine=TensorVariableElimination())
    peak = TensorVariableElimination().estimate_peak_bytes(m, tgt, ev, do)
    a = conditioned_query(m, tgt, ev, do=do, memory_budget=peak * 0.05)
    assert a.cutset and torch.allclose(a.estimate.float(), exact, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1])
def test_bounded_conditioning_is_sound_and_tightens(seed):
    m, names = _dense_bn(seed, k=3)
    tgt, ev = _query(m, names)
    exact = m.query([tgt], evidence=ev, engine=TensorVariableElimination()).double()
    peak = TensorVariableElimination().estimate_peak_bytes(m, tgt, ev)
    prev = None
    n_total = None
    for limit in (1, 3, 9, 27, 81, 10 ** 9):
        b = conditioned_query(m, tgt, ev, memory_budget=peak * 0.05, enumerate_limit=limit)
        n_total = b.n_instantiations
        assert (b.lower <= exact + 1e-6).all() and (exact <= b.upper + 1e-6).all()
        if prev is not None:   # more instantiations: never looser
            assert (b.lower >= prev.lower - 1e-9).all() and (b.upper <= prev.upper + 1e-9).all()
        prev = b
    assert prev.exact and prev.n_enumerated == n_total
    assert torch.allclose(prev.estimate, exact, atol=1e-6)


def test_conditioning_refuses_beyond_max_instantiations():
    m, names = _dense_bn(0)
    tgt, ev = _query(m, names)
    peak = TensorVariableElimination().estimate_peak_bytes(m, tgt, ev)
    with pytest.raises(ConditioningLimitError):
        conditioned_query(m, tgt, ev, memory_budget=peak * 0.03, max_instantiations=3)
    assert issubclass(ConditioningLimitError, OutOfMemoryError)


def test_engine_default_still_raises_and_condition_answers_exactly():
    m, names = _dense_bn(0)
    tgt, ev = _query(m, names)
    exact = m.query([tgt], evidence=ev, engine=TensorVariableElimination())
    peak = TensorVariableElimination().estimate_peak_bytes(m, tgt, ev)
    with pytest.raises(OutOfMemoryError):
        TensorVariableElimination(memory_budget=peak * 0.05).query(m, [tgt], ev)
    eng = TensorVariableElimination(memory_budget=peak * 0.05, on_oom="condition")
    assert torch.allclose(eng.query(m, [tgt], ev), exact, atol=1e-6)
    assert eng.last_conditioning is not None and eng.last_conditioning.cutset
    evb = {names[-1]: torch.tensor([1, 0, 2]), names[9]: torch.tensor([0, 0, 1])}
    ref = TensorVariableElimination().query_batch(m, [tgt], evb)
    assert torch.allclose(eng.query_batch(m, [tgt], evb), ref, atol=1e-6)
    # a plan that fits is answered by plain VE (nothing conditioned)
    big = TensorVariableElimination(on_oom="condition")
    assert torch.allclose(big.query(m, [tgt], ev), exact, atol=1e-6)
    assert big.last_conditioning is None


def test_router_condition_fallback():
    m, names = _dense_bn(1)
    tgt, ev = _query(m, names)
    exact = m.query([tgt], evidence=ev, engine=TensorVariableElimination())
    peak = TensorVariableElimination().estimate_peak_bytes(m, tgt, ev)
    r = HybridRouter(memory_budget=peak * 0.05, fallback="condition")
    assert torch.allclose(m.query([tgt], evidence=ev, engine=r), exact, atol=1e-6)
    assert r.last_engine == "conditioning"
    capped = HybridRouter(memory_budget=peak * 0.05, fallback="condition",
                          condition_max_instantiations=2)
    m.query([tgt], evidence=ev, engine=capped)
    assert capped.last_engine == "likelihood_weighting"
    with pytest.raises(ValueError):
        HybridRouter(fallback="nope")


def _accelerators():
    out = []
    if torch.backends.mps.is_available():
        out.append("mps")
    if torch.cuda.is_available():
        out.append("cuda")
    return out


@pytest.mark.parametrize("device", _accelerators() or [pytest.param("none", marks=pytest.mark.skip(
    reason="no MPS or CUDA device"))])
def test_conditioning_on_accelerator_equals_cpu(device):
    m, names = _dense_bn(0, n=20, k=5, window=9)
    tgt, ev = _query(m, names)
    exact = m.query([tgt], evidence=ev, engine=TensorVariableElimination())
    md = m.to(device)
    peak = TensorVariableElimination().estimate_peak_bytes(md, tgt, ev)
    eng = TensorVariableElimination(memory_budget=peak * 0.2, on_oom="condition")
    p = eng.query(md, [tgt], ev)
    assert eng.last_conditioning is not None
    assert torch.allclose(p.cpu(), exact, atol=1e-5)
    b = conditioned_query(md, tgt, ev, memory_budget=peak * 0.2, enumerate_limit=2)
    assert (b.lower <= exact.double() + 1e-5).all() and (exact.double() <= b.upper + 1e-5).all()


def _grid_bn(rows, cols, seed=0, k=2):
    """Binary grid, parents up and left: in-degree <= 2, width ~ min(rows, cols)."""
    g = torch.Generator().manual_seed(seed)
    name = lambda i, j: f"G{i}_{j}"  # noqa: E731
    names = [name(i, j) for i in range(rows) for j in range(cols)]
    edges = [(name(i - 1, j), name(i, j)) for i in range(1, rows) for j in range(cols)]
    edges += [(name(i, j - 1), name(i, j)) for i in range(rows) for j in range(1, cols)]
    m = NeuralBayesianNetwork(edges, variables=dict.fromkeys(names, ("discrete", k)),
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
    return m, name


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_mps_rank_limit_is_conditioned_away():
    # width >= 15: a product tensor would need more than 16 dimensions on MPS,
    # although every CPT is small (in-degree 2)
    m, name = _grid_bn(17, 17)
    tgt, ev = name(8, 8), {name(16, 16): 1, name(16, 0): 0, name(0, 16): 1}
    eng = TensorVariableElimination()
    assert eng.plan_width(m, tgt, tuple(ev)) + 2 > 16
    exact = m.query([tgt], evidence=ev, engine=eng)
    md = m.to("mps")
    cond = TensorVariableElimination(on_oom="condition")
    p = cond.query(md, [tgt], ev)
    assert cond.last_conditioning is not None
    assert torch.allclose(p.cpu(), exact, atol=1e-5)
