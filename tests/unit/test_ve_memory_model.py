"""VE peak-memory model (#178): the estimate bounds the measured peak.

The guard used to price a query as "largest elimination factor x 3.0".  That
ignored factors that persist across steps: a query whose observed nodes are
all children of the target (naive Bayes, HMM smoothing) was priced at 0 bytes
or a sixth of its peak, and chunk sizes assumed a peak proportional to B.
The estimate is now a walk of the live set (``_plan_cost``); these tests
measure real peaks on CPU with ``torch.profiler`` (every allocation and free
of the CPU allocator, intra-op temporaries included) and check the bound.
``scripts/ve_memory_validation.py`` runs the larger sweep on CPU / MPS / CUDA.
"""
from __future__ import annotations

import gc
import json
import os
import tempfile

import pytest
import torch

from nbn import NeuralBayesianNetwork as NBN
from nbn import TensorVariableElimination
from nbn.inference import tensor_ve
from nbn.inference.tensor_ve import (
    _PeakModel,
    _PlanCost,
    _heap_allowance,
    _max_chunk_rows,
)


def _fit(edges, cards, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = NBN(edges, variables={n: ("discrete", k) for n, k in cards.items()}, device="cpu")
    m.auto_mechanisms()
    m.fit({n: torch.randint(0, k, (200,), generator=g).float() for n, k in cards.items()})
    return m


def _naive_bayes(n=30):
    cards = {"Y": 4, **{f"X{i}": 3 for i in range(n)}}
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
    # name: (model builder, targets, evidence nodes, do nodes, B)
    "naive-bayes": (_naive_bayes, ["Y"], [f"X{i}" for i in range(30)], [], 4096),
    "hmm": (_hmm, ["H15"], [f"O{i}" for i in range(30)], [], 1024),
    "wide-tail-ev": (_wide, ["V8"], [f"V{i}" for i in range(17, 22)], [], 8),
    "wide-joint-clamped": (_wide, ["V8", "V9", "V20"], [f"V{i}" for i in range(17, 22)],
                           [], 8),
    "wide-do": (_wide, ["V8"], [f"V{i}" for i in range(18, 22)], ["V15", "V16"], 8),
}


@pytest.fixture(scope="module")
def models():
    return {"naive-bayes": _naive_bayes(), "hmm": _hmm(), "wide": _wide()}


def _model_for(models, name):
    return models["wide" if name.startswith("wide") else name]


@pytest.mark.parametrize("name", sorted(_CASES))
def test_estimate_bounds_the_measured_cpu_peak(models, name):
    """Regression (#178): master priced naive-bayes at 0 bytes and the HMM at
    0.17x its measured peak; the live-set walk bounds every case, tightly
    where factors dominate (index tensors are priced as copies)."""
    _, targets, ev_nodes, do_nodes, B = _CASES[name]
    model = _model_for(models, name)
    ev, do = _evidence(model, ev_nodes, B), _evidence(model, do_nodes, B, seed=1)
    eng = TensorVariableElimination(memory_budget=float("inf"))

    def call():
        return eng.query_batch(model, targets, ev, do=do)

    call()  # tabulate the CPTs (a warm cache, as in any repeated use)
    target = targets if len(targets) > 1 else targets[0]
    estimate = eng.estimate_peak_bytes(model, target, tuple(ev), tuple(do), B=B)
    measured = _cpu_peak(call)
    assert measured > 0
    assert estimate >= measured, f"{name}: estimate {estimate} < measured {measured}"
    assert estimate <= 1.8 * measured, f"{name}: estimate {estimate} vs {measured}"


def test_estimate_bounds_single_row_query_and_cold_cache(models):
    model = models["wide"]
    ev_nodes = [f"V{i}" for i in range(17, 22)]
    row = {n: torch.tensor(1) for n in ev_nodes}
    eng = TensorVariableElimination(memory_budget=float("inf"))
    cold = eng.estimate_peak_bytes(model, "V8", tuple(row))
    # Cold: this call also tabulates every CPT of the model.
    measured_cold = _cpu_peak(lambda: eng.query(model, ["V8"], row))
    warm = eng.estimate_peak_bytes(model, "V8", tuple(row))
    measured_warm = _cpu_peak(lambda: eng.query(model, ["V8"], row))
    assert cold >= measured_cold and warm >= measured_warm
    assert cold > warm  # the tables are priced only while they are not cached


def test_float_evidence_copies_are_priced(models):
    """Evidence that is not int64 is copied per row; the index term covers it."""
    model = models["naive-bayes"]
    _, targets, ev_nodes, _, B = _CASES["naive-bayes"]
    ev = _evidence(model, ev_nodes, B, dtype=torch.float32)
    eng = TensorVariableElimination(memory_budget=float("inf"))
    eng.query_batch(model, targets, ev)
    estimate = eng.estimate_peak_bytes(model, targets[0], tuple(ev), B=B)
    measured = _cpu_peak(lambda: eng.query_batch(model, targets, ev))
    assert measured <= estimate <= 1.3 * measured


@pytest.mark.parametrize("name", ["naive-bayes", "wide-tail-ev"])
def test_chunked_batch_stays_within_its_budget(models, name):
    """Regression (#178): a budget below the batch's peak must be respected by
    the chunks — master priced naive-bayes at 0 bytes and never chunked it."""
    _, targets, ev_nodes, _, B = _CASES[name]
    model = _model_for(models, name)
    ev = _evidence(model, ev_nodes, B)
    full = TensorVariableElimination().estimate_peak_bytes(model, targets[0], tuple(ev), B=B)
    budget = int(0.4 * full)
    eng = TensorVariableElimination(memory_budget=budget, memory_safety=1.0)
    eng.query_batch(model, targets, ev)  # warm
    measured = _cpu_peak(lambda: eng.query_batch(model, targets, ev))
    assert measured <= budget, f"{name}: chunked peak {measured} > budget {budget}"


# --------------------------------------------------------------------------- #
# Chunk sizing: the peak is convex in the row count, priced by its chord
# --------------------------------------------------------------------------- #

def test_max_chunk_rows_respects_the_batch_free_part_of_the_peak():
    """Regression (#178): with peak(c) = 40 + 6c (B=10 -> 100, B=1 -> 46) and
    a budget of 75, master's ``budget // (peak_at_B / B)`` gave 7 rows, whose
    peak is 82; the chord gives 5 (peak 70)."""
    rows = _max_chunk_rows(peak_at_B=100, peak_at_1=46, B=10, budget=75)
    assert 40 + 6 * rows <= 75
    assert rows == 5


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
def test_chunk_estimate_is_below_its_chord(models, device_type):
    """``bytes(c, heap_rows=B)`` is convex in c, so the chord through c=1 and
    c=B bounds every chunk size — including the heap step functions."""
    model = models["wide"]
    eng = TensorVariableElimination()
    pm = eng._peak_model(model, "V8", [f"V{i}" for i in range(17, 22)])
    pm = pm._replace(device_type=device_type)
    B = 4096
    p1, pB = pm.bytes(1, heap_rows=B), pm.bytes(B)
    for c in (1, 2, 3, 10, 100, 1000, 2047, 4095, B):
        chord = p1 + (c - 1) * (pB - p1) / (B - 1)
        assert pm.bytes(c) <= pm.bytes(c, heap_rows=B) <= chord + 1


# --------------------------------------------------------------------------- #
# Allocator reserves and dtype
# --------------------------------------------------------------------------- #

def test_heap_allowance_by_size_class():
    MiB = 1 << 20
    assert _heap_allowance("cpu", 10 ** 12) == 0
    assert _heap_allowance("mps", 100) == 8 * MiB
    assert _heap_allowance("mps", 5 * MiB) == 40 * MiB
    assert _heap_allowance("mps", 20 * MiB) == (8 + 32 + 1024) * MiB
    assert _heap_allowance("cuda", 20 * MiB) == (2 + 20 + 2) * MiB


def test_mps_estimate_includes_the_heap_reserve():
    """MPS carves 10 MiB..1 GiB buffers from 1 GiB heaps (measured): a plan
    with a 20 MiB tensor costs a 1 GiB heap, which the estimate must cover."""
    n = 5 << 20  # 20 MiB of float32
    cost = _PlanCost(tensor=(n, 0), steps=((2 * n, 0),), final=(n, 0), tensors=4)
    cpu = _PeakModel(cost, 4, 0, 0, cold=False, device_type="cpu").bytes(1)
    mps = _PeakModel(cost, 4, 0, 0, cold=False, device_type="mps").bytes(1)
    assert mps - cpu >= 1 << 30


def test_factor_dtype_follows_the_parameters(models):
    model = models["naive-bayes"]
    assert tensor_ve._factor_dtype_bytes(model) == 4
    m64 = torch.nn.Linear(2, 2).double()
    assert tensor_ve._factor_dtype_bytes(m64) == 8
