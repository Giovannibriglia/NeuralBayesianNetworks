"""VE peak-memory model (#178): the estimate bounds the measured peak.

The guard used to price a query as "largest elimination factor x 3.0".  That
ignored factors that persist across steps: a query whose observed nodes are
all children of the target (naive Bayes, HMM smoothing) was priced at 0 bytes
or a sixth of its peak.  The estimate is now a tape of every allocation,
replayed through the device's allocator (``nbn.inference._ve_memory``).
These tests measure real peaks on CPU with ``torch.profiler`` (every
allocation and free of the CPU allocator, intra-op temporaries included) and
check the bound; the MPS heap and CUDA caching-allocator rules are checked
arithmetically.  ``scripts/ve_memory_validation.py`` runs the larger sweep.
"""
from __future__ import annotations

import gc
import json
import os
import tempfile
from unittest import mock

import pytest
import torch

from nbn import NeuralBayesianNetwork as NBN
from nbn import TensorVariableElimination
from nbn.inference import tensor_ve
from nbn.inference._ve_memory import (
    _PeakModel,
    _cuda_reserved_peak,
    _mps_charge,
    _plan_cost,
    _Scope,
)

MiB = 1 << 20


def _fit(edges, cards, seed=0, mechs=None):
    g = torch.Generator().manual_seed(seed)
    m = NBN(edges, variables={n: ("discrete", k) for n, k in cards.items()}, device="cpu")
    m.auto_mechanisms()
    for node, mech in (mechs or {}).items():
        m.set_mechanism(node, mech)
    m.fit({n: torch.randint(0, k, (200,), generator=g).float() for n, k in cards.items()})
    return m


def _naive_bayes(n=30, card_y=4, card_x=3):
    cards = {"Y": card_y, **{f"X{i}": card_x for i in range(n)}}
    return _fit([("Y", f"X{i}") for i in range(n)], cards)


def _hmm(n=30):
    hs, os_ = [f"H{i}" for i in range(n)], [f"O{i}" for i in range(n)]
    edges = [(hs[i], hs[i + 1]) for i in range(n - 1)] + list(zip(hs, os_))
    return _fit(edges, {**dict.fromkeys(hs, 4), **dict.fromkeys(os_, 5)})


def _wide(n=22, indeg=3, card=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    names = [f"V{i}" for i in range(n)]
    edges = [(names[p], names[j]) for j in range(1, n)
             for p in torch.randperm(j, generator=g)[: min(indeg, j)].tolist()]
    return _fit(edges, dict.fromkeys(names, card), seed)


def _cpu_peak(fn) -> int:
    """Peak bytes the CPU allocator holds during ``fn()`` beyond its start."""
    from torch.profiler import ProfilerActivity, profile

    gc.collect()
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        out = fn()
        del out
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as fh:
            events = json.load(fh)["traceEvents"]
    mem = sorted((e["ts"], e["args"].get("Ev Idx", 0), e["args"]["Bytes"])
                 for e in events
                 if e.get("name") == "[memory]" and e["args"].get("Device Type") == 0)
    cur = peak = 0
    for *_, b in mem:
        cur += b
        peak = max(peak, cur)
    return peak


def _evidence(model, nodes, B, seed=0, dtype=torch.long):
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randint(0, model.mechanisms[n].n_classes, (B,), generator=g).to(dtype)
            for n in nodes}


_CASES = {
    # name: (model, targets, evidence nodes, do nodes, B)
    "naive-bayes": ("naive-bayes", ["Y"], [f"X{i}" for i in range(30)], [], 4096),
    "hmm": ("hmm", ["H15"], [f"O{i}" for i in range(30)], [], 1024),
    "wide-tail-ev": ("wide", ["V8"], [f"V{i}" for i in range(17, 22)], [], 8),
    "wide-joint-clamped": ("wide", ["V8", "V9", "V20"], [f"V{i}" for i in range(17, 22)],
                           [], 8),
    "wide-do": ("wide", ["V8"], [f"V{i}" for i in range(18, 22)], ["V15", "V16"], 8),
}


@pytest.fixture(scope="module")
def models():
    return {"naive-bayes": _naive_bayes(), "hmm": _hmm(), "wide": _wide()}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_estimate_bounds_the_measured_cpu_peak(models, name):
    """Regression (#178): master priced naive-bayes at 0 bytes and the HMM at
    0.17x its measured peak.  Priced with the call's values, the estimate is
    within the 10 % margin plus 64 B alignment of the exact peak."""
    key, targets, ev_nodes, do_nodes, B = _CASES[name]
    model = models[key]
    ev, do = _evidence(model, ev_nodes, B), _evidence(model, do_nodes, B, seed=1)
    eng = TensorVariableElimination(memory_budget=float("inf"))

    def call():
        return eng.query_batch(model, targets, ev, do=do)

    call()  # tabulate the CPTs (a warm cache, as in any repeated use)
    target = targets if len(targets) > 1 else targets[0]
    estimate = eng.estimate_peak_bytes(model, target, ev, do, B=B)
    measured = _cpu_peak(call)
    assert measured > 0
    assert measured <= estimate <= 1.3 * measured, (name, estimate, measured)


def test_estimate_bounds_single_row_query_and_cold_cache(models):
    model = models["wide"]
    row = {f"V{i}": torch.tensor(1) for i in range(17, 22)}
    eng = TensorVariableElimination(memory_budget=float("inf"))
    cold = eng.estimate_peak_bytes(model, "V8", row)
    # Cold: this call also tabulates every CPT of the model.
    measured_cold = _cpu_peak(lambda: eng.query(model, ["V8"], row))
    warm = eng.estimate_peak_bytes(model, "V8", row)
    measured_warm = _cpu_peak(lambda: eng.query(model, ["V8"], row))
    assert cold >= measured_cold and warm >= measured_warm
    assert cold > warm  # the tables are priced only while they are not cached


# --------------------------------------------------------------------------- #
# Evidence pricing (review findings 3 and 4)
# --------------------------------------------------------------------------- #

def test_int64_evidence_on_device_is_not_priced_as_a_copy():
    """Regression (review 3): every evidence node was priced as an 8 B/row
    copy.  Target X0 of a naive Bayes with Y and 29 d-separated children
    observed, B=20000: the estimate was 12x the real peak, and a budget of 3x
    the real peak ran the query in 1-row chunks (4.7 s instead of 1.5 ms)."""
    model = _naive_bayes()
    B = 20000
    ev = _evidence(model, ["Y"] + [f"X{i}" for i in range(1, 30)], B)
    eng = TensorVariableElimination(memory_budget=float("inf"))
    eng.query_batch(model, ["X0"], ev)
    measured = _cpu_peak(lambda: eng.query_batch(model, ["X0"], ev))
    estimate = eng.estimate_peak_bytes(model, "X0", ev, B=B)
    assert measured <= estimate <= 1.5 * measured
    budgeted = TensorVariableElimination(memory_budget=3 * measured, memory_safety=1.0)
    budgeted.query_batch(model, ["X0"], ev)
    with mock.patch.object(budgeted, "_query_batch_chunked") as spy:
        budgeted.query_batch(model, ["X0"], ev)
    assert not spy.called


@pytest.mark.parametrize("kind", ["float-tensor", "python-list"])
def test_converted_evidence_is_priced(models, kind):
    """Float tensors are copied to int64 (8 B/row); Python lists are first
    materialised by ``torch.as_tensor`` (float32, 4 B/row) and then copied —
    regression (review 4): lists were priced at 8 B/row (ratio 0.947)."""
    model = models["naive-bayes"]
    _, targets, ev_nodes, _, B = _CASES["naive-bayes"]
    ev = _evidence(model, ev_nodes, B, dtype=torch.float32)
    if kind == "python-list":
        ev = {k: v.tolist() for k, v in ev.items()}
    eng = TensorVariableElimination(memory_budget=float("inf"))
    eng.query_batch(model, targets, ev)
    estimate = eng.estimate_peak_bytes(model, targets[0], ev, B=B)
    measured = _cpu_peak(lambda: eng.query_batch(model, targets, ev))
    assert measured <= estimate <= 1.3 * measured


def test_names_only_are_priced_as_the_costliest_input(models):
    """Without values (the router prices by names) every node is priced as a
    list of floats, an upper bound of any input."""
    model = models["naive-bayes"]
    _, targets, ev_nodes, _, B = _CASES["naive-bayes"]
    ev = {k: v.tolist() for k, v in _evidence(model, ev_nodes, B, dtype=torch.float32).items()}
    eng = TensorVariableElimination()
    eng._extract_factors(model)
    assert (eng.estimate_peak_bytes(model, "Y", tuple(ev), B=B)
            >= eng.estimate_peak_bytes(model, "Y", ev, B=B))


# --------------------------------------------------------------------------- #
# Chunking (review finding 2)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("frac", [0.15, 0.25, 0.4, 0.6])
def test_chunked_batch_stays_within_its_budget_or_is_refused(models, frac):
    """Regression (review 2): with float evidence the batch's int64 copies
    stay alive next to every chunk; when they alone exhaust the budget the
    guard degraded to 1-row chunks and ran at 2.2x / 1.3x the budget (0.15x
    / 0.25x).  Now the batch either fits its budget in chunks or is refused
    before allocating."""
    model = models["naive-bayes"]
    _, targets, ev_nodes, _, B = _CASES["naive-bayes"]
    ev = _evidence(model, ev_nodes, B, dtype=torch.float32)
    probe = TensorVariableElimination()
    probe._extract_factors(model)
    budget = int(frac * probe.estimate_peak_bytes(model, "Y", ev, B=B))
    eng = TensorVariableElimination(memory_budget=budget, memory_safety=1.0)
    eng._extract_factors(model)
    try:
        measured = _cpu_peak(lambda: eng.query_batch(model, targets, ev))
    except torch.cuda.OutOfMemoryError as exc:
        assert "pre-allocation guard" in str(exc)
        return
    assert measured <= budget, f"frac={frac}: peak {measured} > budget {budget}"


def test_chunks_are_large_when_the_budget_allows(models):
    """An index-heavy batch under 0.6x its estimate gets large chunks, not a
    degenerate run of tiny ones."""
    model = models["naive-bayes"]
    _, targets, ev_nodes, _, B = _CASES["naive-bayes"]
    ev = _evidence(model, ev_nodes, B)
    probe = TensorVariableElimination()
    probe._extract_factors(model)
    budget = int(0.6 * probe.estimate_peak_bytes(model, "Y", ev, B=B))
    eng = TensorVariableElimination(memory_budget=budget, memory_safety=1.0)
    eng._extract_factors(model)
    with mock.patch.object(eng, "_query_batch_chunked",
                           wraps=eng._query_batch_chunked) as spy:
        eng.query_batch(model, targets, ev)
    assert spy.called and spy.call_args.kwargs["chunk"] > B // 8


# --------------------------------------------------------------------------- #
# Cold-cache tabulation (review finding 5)
# --------------------------------------------------------------------------- #

def _seven_parent(mech):
    parents = [f"P{i}" for i in range(7)]
    cards = {**dict.fromkeys(parents, 4), "C": 4}
    return _fit([(p, "C") for p in parents], cards, mechs={"C": mech})


def test_cold_tabulation_working_memory_is_priced():
    """Regression (review 5): a neural-categorical node with 7 parents of 4
    states (16384 configurations through the MLP) measured 8.4 MiB on a cold
    cache against an estimate of 0.6 MiB."""
    from nbn.mechanisms import NeuralCategoricalMechanism

    model = _seven_parent(NeuralCategoricalMechanism(n_classes=4))
    ev = {"P0": torch.tensor([1, 2])}
    eng = TensorVariableElimination(memory_budget=float("inf"))
    estimate = eng.estimate_peak_bytes(model, "C", ev, B=2)
    measured = _cpu_peak(lambda: eng.query_batch(model, ["C"], ev))
    assert measured <= estimate, (estimate, measured)


@pytest.mark.parametrize("mech", ["neuralcat", "knn"])
def test_tabulate_working_bytes_bounds_tabulate(mech):
    """The hook bounds what ``tabulate`` holds besides the table it returns
    (kNN enumerates parent configurations against every training row)."""
    from nbn.mechanisms import NeuralCategoricalMechanism
    from nbn.mechanisms.non_parametric.knn_conditional import KNNConditionalMechanism

    m = (NeuralCategoricalMechanism(n_classes=4) if mech == "neuralcat"
         else KNNConditionalMechanism(discrete_child=True))
    g = torch.Generator().manual_seed(0)
    pa = torch.randint(0, 4, (500, 6), generator=g).float()
    x = torch.randint(0, 4, (500,), generator=g).float()
    m.fit_local(x, pa, parent_cards=[4] * 6, n_classes=4)
    cards = [4] * 6
    table = 4 ** 7 * 4
    measured = _cpu_peak(lambda: m.tabulate(cards).detach())
    assert measured - table <= m.tabulate_working_bytes(cards), (
        measured, m.tabulate_working_bytes(cards))


# --------------------------------------------------------------------------- #
# Allocators: MPS heaps (review 1) and CUDA's caching allocator
# --------------------------------------------------------------------------- #

def test_mps_buffers_are_charged_their_heap_share():
    """MPS packs 10 MiB..512 MiB buffers into 1 GiB heaps: a 342 MiB buffer
    fits twice, so twelve of them took 6 GiB (measured).  Regression (review
    1): the model charged them 12 x 342 MiB plus one heap."""
    import numpy as np

    charge = _mps_charge(np.array([342 * MiB] * 12, dtype=float))
    assert charge.sum() >= 6144 * MiB
    assert _mps_charge(np.array([600.0 * MiB]))[0] == 608 * MiB    # exact, 16 MiB
    assert _mps_charge(np.array([5.0 * MiB]))[0] == 32 * MiB / 6    # 32 MiB heaps


def test_mps_estimate_covers_the_reviewers_naive_bayes():
    """Y with 32 states, 16 binary observed children, B=2,801,664 (each
    conditioned factor 342 MiB) on MPS grew the driver by 8192 MiB while the
    previous model estimated 7836 MiB.  Priced arithmetically here."""
    scopes = {"Y": _Scope(["Y"], {"Y": 32})}
    for i in range(16):
        scopes[f"X{i}"] = _Scope(["Y", f"X{i}"], {"Y": 32, f"X{i}": 2})
    cost = _plan_cost([], scopes, evidence_keys=tuple(f"X{i}" for i in range(16)),
                      targets=("Y",), target_cards={"Y": 32}, joint=False)
    pm = _PeakModel(cost, device_type="mps")
    assert pm.bytes(2_801_664) >= 8192 * MiB


def test_cuda_simulation_holds_freed_segments_it_cannot_reuse():
    """Freed blocks stay cached and serve only requests they fit: 64 MiB
    blocks freed before a 256 MiB request leave the reserve at 64 + 256."""
    sizes = [64 * MiB, 64 * MiB, 256 * MiB]
    sign = [1, -1, 1]
    ref = [0, 0, 2]
    assert _cuda_reserved_peak(sizes, sign, ref) == 320 * MiB
    # A freed block that fits is reused (split): no growth.
    assert _cuda_reserved_peak([256 * MiB, 256 * MiB, 64 * MiB], [1, -1, 1],
                               [0, 0, 2]) == 256 * MiB
    # Small blocks share 2 MiB segments; blocks < 10 MiB share 20 MiB ones.
    assert _cuda_reserved_peak([100_000.0] * 10, [1] * 10, list(range(10))) == 2 * MiB
    assert _cuda_reserved_peak([3.0 * MiB] * 6, [1] * 6, list(range(6))) == 20 * MiB


def test_cuda_estimate_covers_the_measured_reserved_growth():
    """CUDA review run (RTX 4070, torch 2.11): ``wide24 tail-ev`` B=64 grew
    the caching allocator's reserve by 778,043,392 B — above the previous
    estimate (768,015,821 B): freed 64 MiB blocks could not serve the 256 MiB
    requests.  The simulated allocator must cover it with margin."""
    g = torch.Generator().manual_seed(3)
    names = [f"V{i}" for i in range(24)]
    edges = [(names[p], names[j]) for j in range(1, 24)
             for p in torch.randperm(j, generator=g)[: min(4, j)].tolist()]
    model = _fit(edges, dict.fromkeys(names, 4), 3)
    B = 64
    ev = _evidence(model, [f"V{i}" for i in range(19, 24)], B)
    eng = TensorVariableElimination()
    eng._extract_factors(model)
    pm = eng._peak_model(model, "V12", ev, B=B)
    pm.device_type = "cuda"
    assert pm.bytes(B) >= 1.05 * 778_043_392


def test_factor_dtype_follows_the_parameters(models):
    model = models["naive-bayes"]
    assert tensor_ve._factor_dtype_bytes(model) == 4
    m64 = torch.nn.Linear(2, 2).double()
    assert tensor_ve._factor_dtype_bytes(m64) == 8
