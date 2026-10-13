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
    _mps_heap_peak,
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
    work = m.tabulate_working_bytes(cards)
    total = work if isinstance(work, int) else sum(work)
    assert measured - table <= total, (measured, total)
    # Booked buffer by buffer (#301 review E): none of them is the whole sum.
    assert isinstance(work, list) and max(work) < total


# --------------------------------------------------------------------------- #
# Allocators: MPS heaps (review 1) and CUDA's caching allocator
# --------------------------------------------------------------------------- #

def _replay(sim, ops):
    """Run ``sim`` on ``[('+', name, MiB) | ('-', name)]``."""
    sizes, sign, ref, where = [], [], [], {}
    for op in ops:
        if op[0] == "+":
            where[op[1]] = len(sizes)
            sizes.append(op[2] * MiB)
            sign.append(1)
            ref.append(len(ref))
        else:
            j = where.pop(op[1])
            sizes.append(sizes[j])
            sign.append(-1)
            ref.append(j)
    return sim(sizes, sign, ref) / MiB


def test_mps_frees_are_not_credited_within_a_call():
    """#301 review A: MPS runs asynchronously; a buffer freed while queued
    GPU work still uses it is not reusable yet (barley q2, B=64: 73 MiB of
    driver growth unsynchronised, 33 MiB with a sync after every op).  The
    estimate therefore prices every allocation as new: two 600 MiB buffers
    allocated one after the other's free take two heaps."""
    ops = [("+", "a", 600), ("-", "a"), ("+", "b", 600)]
    assert _replay(_mps_heap_peak, ops) == 2 * 608
    assert _replay(lambda *a: _mps_heap_peak(*a, credit_frees=True), ops) == 608


def test_mps_heaps_follow_the_measured_allocator():
    """Metal heaps under PyTorch's MPS allocator, as measured with a sync
    after every op (#301 review A): freed buffers return their space to the
    heap and holes coalesce, but a request no single hole fits opens a new
    heap."""
    mps = lambda *a: _mps_heap_peak(*a, credit_frees=True)  # noqa: E731  (synced)
    assert _replay(mps, [("+", f"x{i}", 342) for i in range(12)]) == 6144
    # Reviewer's repro: the old per-buffer charge gave 0.931 of this.
    assert _replay(mps, [("+", "a", 330), ("+", "b", 12), ("+", "c", 400),
                         ("+", "d", 330), ("-", "a"), ("+", "e", 470)]) == 2048
    # 400 live, 400 freed, 224 free at the end: 500 needs a new heap ...
    assert _replay(mps, [("+", "a", 400), ("+", "b", 400), ("-", "a"),
                         ("+", "c", 500)]) == 2048
    # ... but the freed 400 serves a 200 and a 190 (split), with no new heap.
    assert _replay(mps, [("+", "f", 500), ("+", "c", 400), ("-", "c"),
                         ("+", "a", 200), ("+", "b", 190)]) == 1024
    assert _replay(mps, [("+", "a", 600)]) == 608          # own heap
    assert _replay(mps, [("+", f"x{i}", 9.5) for i in range(4)]) == 64


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
    assert pm.bytes(2_801_664) >= 1.05 * 8192 * MiB


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
    # Same tape on MPS: the heap model must also cover what CPU measures.
    pm.device_type, pm._memo = "mps", {}
    assert pm.bytes(B) >= 644 * MiB


def test_factor_dtype_follows_the_parameters(models):
    model = models["naive-bayes"]
    assert tensor_ve._factor_dtype_bytes(model) == 4
    m64 = torch.nn.Linear(2, 2).double()
    assert tensor_ve._factor_dtype_bytes(m64) == 8


# --------------------------------------------------------------------------- #
# Review round 2: cold cache + tight budget (B), per-call overhead (D), MPS
# tabulation buffers (E)
# --------------------------------------------------------------------------- #

def _naive_bayes_with_big_irrelevant_cpt():
    """Y with 30 observed children plus a node with six 8-state parents that
    no query about Y touches: its 8^7 CPT is tabulated on a cold cache."""
    cards = {"Y": 4, **{f"X{i}": 3 for i in range(30)},
             **{f"P{i}": 8 for i in range(6)}, "Z": 8}
    edges = [("Y", f"X{i}") for i in range(30)] + [(f"P{i}", "Z") for i in range(6)]
    return _fit(edges, cards)


@pytest.mark.parametrize("B", [4096, 65536])
@pytest.mark.parametrize("where", [0.25, 0.5, 0.9])
def test_cold_cache_tight_budget_chunks_without_recursion(B, where):
    """Regression (#301 review B): with a cold cache and a budget between the
    cold one-row and full-batch estimates, chunks were sized warm but
    re-entered the guard cold, re-chunked one row at a time 170-180 levels
    deep, and the batch was refused (HybridRouter then silently used LW).
    Now the CPTs are tabulated first and the chunks run in a loop."""
    model = _naive_bayes_with_big_irrelevant_cpt()
    ev = _evidence(model, [f"X{i}" for i in range(30)], B)
    probe = TensorVariableElimination()
    lo = probe.estimate_peak_bytes(model, "Y", {k: v[:1] for k, v in ev.items()}, B=1)
    hi = probe.estimate_peak_bytes(model, "Y", ev, B=B)
    budget = int(lo + where * (hi - lo))
    eng = TensorVariableElimination(memory_budget=budget, memory_safety=1.0)
    depth = [0, 0]
    real = eng.query_batch

    def counting(*a, **k):
        depth[0] += 1
        depth[1] = max(depth[1], depth[0])
        try:
            return real(*a, **k)
        finally:
            depth[0] -= 1

    with mock.patch.object(eng, "query_batch", side_effect=counting):
        measured = _cpu_peak(lambda: eng.query_batch(model, ["Y"], ev))
    assert depth[1] <= 2, f"recursion depth {depth[1]}"
    assert measured <= budget, (measured, budget)


def test_plan_replay_is_memoised_across_calls(models):
    """#301 review D: a repeated query shape replays only its evidence tapes;
    the plan's replay is cached on the plan cost."""
    from nbn.inference import _ve_memory

    model = models["wide"]
    row = {f"V{i}": torch.tensor(1) for i in range(17, 22)}
    eng = TensorVariableElimination()
    eng.query(model, ["V8"], row)
    calls = []
    real = _ve_memory._replay

    def spy(tape, *a, **k):
        calls.append(len(tape[0]))
        return real(tape, *a, **k)

    plan_len = len(eng._cached_plan_cost(model, "V8", tuple(sorted(row)), (),
                                         "min_fill").tape[0])
    with mock.patch.object(_ve_memory, "_replay", side_effect=spy):
        eng.query(model, ["V8"], row)
    assert plan_len not in calls


def test_mps_tabulation_buffers_are_priced_one_by_one():
    """#301 review E: ~20 MiB of neural-categorical activations booked as one
    buffer put it in the 1 GiB heap class (1.16 GiB estimated against 34 MiB
    of measured driver growth)."""
    from nbn.mechanisms import NeuralCategoricalMechanism

    model = _seven_parent(NeuralCategoricalMechanism(n_classes=4))
    eng = TensorVariableElimination()
    pm = eng._peak_model(model, "C", {"P0": torch.tensor([1, 2])}, B=2)
    pm.device_type = "mps"
    assert pm.bytes(2) < 256 * MiB


# --------------------------------------------------------------------------- #
# Review round 3
# --------------------------------------------------------------------------- #

def test_dtype_is_the_widest_cached_factor():
    """#301 review 3, finding 1: the warm-cache dtype came from the first
    cached factor.  In a float64 model with an intervened root, that factor is
    the DeterministicMechanism's float32 point mass while every product is
    float64: estimate 5.50 MiB vs 10.01 MiB measured (0.55)."""
    model = _wide().double().intervene({"V0": 1})
    ev = _evidence(model, [f"V{i}" for i in range(17, 22)], 8)
    eng = TensorVariableElimination(memory_budget=float("inf"))
    eng.query_batch(model, ["V8"], ev)  # warm
    first = next(iter(eng._cached_factors(model).values())).log_values
    assert first.dtype == torch.float32  # the trap: the first factor is narrow
    estimate = eng.estimate_peak_bytes(model, "V8", ev, B=8)
    measured = _cpu_peak(lambda: eng.query_batch(model, ["V8"], ev))
    assert measured <= estimate <= 1.3 * measured, (estimate, measured)


@pytest.mark.parametrize("api", ["estimate_peak_bytes", "fits_memory"])
def test_read_only_apis_leave_no_cache_token(models, api):
    """#301 review 3, finding 2: ``estimate_peak_bytes`` / ``fits_memory``
    left the guard's single-use cache token behind; an in-place parameter
    edit before the next extraction then got the stale factors."""
    model = _wide(seed=5)
    eng = TensorVariableElimination()
    eng._extract_factors(model)
    getattr(eng, api)(model, "V8", {"V20": torch.tensor([1])})
    mech = model.mechanisms["V8"]
    with torch.no_grad():
        mech._logits.add_(torch.randn_like(mech._logits))
    fresh = eng._extract_factors(model)["V8"].log_values
    expected = torch.log_softmax(mech.tabulate(None).detach(), dim=-1)
    torch.testing.assert_close(fresh, expected)


def test_chunk_sizer_does_not_subtract_resident_memory_from_live_headroom(models):
    """#301 review 3, finding 3: live headroom already reflects the evidence
    and CPTs this call holds, but the sizer subtracted them again, halving
    chunks (hmm60 at 2 GiB on MPS: 114688 rows instead of 262144)."""
    from nbn.inference import tensor_ve as tv
    from nbn.inference._ve_memory import _max_fitting_rows

    model = models["naive-bayes"]
    B = 4096
    ev = _evidence(model, [f"X{i}" for i in range(30)], B, dtype=torch.float32)
    probe = TensorVariableElimination()
    probe._extract_factors(model)
    pm = probe._peak_model(model, "Y", ev, B=B)
    budget = pm.bytes(B)
    head = 0.6 * budget / 0.9           # live room 0.6x the batch: chunk on it
    eng = TensorVariableElimination(memory_budget=budget, memory_safety=0.9)
    eng._extract_factors(model)
    with mock.patch.object(tv, "device_headroom_bytes", lambda d: head), \
         mock.patch.object(eng, "_query_batch_chunked",
                           wraps=eng._query_batch_chunked) as spy:
        eng.query_batch(model, ["Y"], ev)
    room = min(budget - pm.reserve_bytes(B), 0.9 * head - 2 * pm.output_bytes(B))
    assert spy.call_args.kwargs["chunk"] == _max_fitting_rows(pm.child().bytes, B - 1, room)


# --------------------------------------------------------------------------- #
# CUDA phase: PYTORCH_CUDA_ALLOC_CONF (#301 review C)
# --------------------------------------------------------------------------- #

def test_cuda_allocator_conf_is_parsed_like_torch():
    """``parseArgs``: known keys set their field, keys without effect on the
    reserve are ignored, another backend or an unknown key is ``known=False``."""
    from nbn.inference._ve_memory import _CudaConf, _parse_cuda_conf

    assert _parse_cuda_conf(None) == _CudaConf()
    assert _parse_cuda_conf("") == _CudaConf()
    c = _parse_cuda_conf("max_split_size_mb:128,max_non_split_rounding_mb:40,"
                         "garbage_collection_threshold:0.8,expandable_segments:True,"
                         "large_segment_size_mb:64")
    assert (c.max_split, c.non_split_rounding, c.expandable, c.large_segment, c.known) == (
        128 * MiB, 40 * MiB, True, 64 * MiB, True)
    assert _parse_cuda_conf("roundup_power2_divisions:4").divisions == (4,) * 17
    d = _parse_cuda_conf("roundup_power2_divisions:[64:8,256:4,>:1]").divisions
    assert d[:7] == (8,) * 7 and d[7] == 0 and d[8:] == (1,) * 9   # ">" fills from the last bucket on
    assert _parse_cuda_conf("backend:native,pinned_use_background_threads:True").known
    assert not _parse_cuda_conf("backend:cudaMallocAsync").known
    assert not _parse_cuda_conf("some_future_key:1").known
    assert not _parse_cuda_conf("max_split_size_mb:lots").known


def test_cuda_rounding_follows_power2_divisions():
    """``round_size``: 512 B by default; with divisions, up to the next
    multiple of (power of two below the size) / divisions."""
    from nbn.inference._ve_memory import _cuda_round

    assert _cuda_round(1) == 512
    assert _cuda_round(513) == 1024
    four = (4,) * 17
    assert _cuda_round(5 * MiB, four) == 5 * MiB           # 4 MiB floor, 1 MiB steps
    assert _cuda_round(4.5 * MiB, four) == 5 * MiB
    assert _cuda_round(2.1 * MiB, four) == 2.5 * MiB       # 2 MiB floor, 512 KiB steps
    assert _cuda_round(3 * MiB, four) == 3 * MiB
    assert _cuda_round(2 * MiB, four) == 2 * MiB            # powers of two are kept
    assert _cuda_round(1000, four) == 1024                   # size <= 512 * divisions: 512 B rounding
    assert _cuda_round(4.5 * MiB, (1,) * 17) == 4.5 * MiB   # divisions 1: plain rounding
    assert _cuda_round(3 * MiB + 1, (2,) * 17) == 4 * MiB   # 2 divisions: 1 MiB steps above 2 MiB


def test_cuda_max_split_size_blocks_oversize_reuse():
    """``get_free_block`` with ``max_split_size_mb``: an oversize cached block
    never serves a smaller request, an oversize request takes a block at most
    ``max_non_split_rounding_mb`` (20 MiB) larger, and oversize blocks are
    not split."""
    conf = "max_split_size_mb:128"
    seq = ([256 * MiB, 256 * MiB, 64 * MiB], [1, -1, 1], [0, 0, 2])
    assert _cuda_reserved_peak(*seq) == 256 * MiB                 # default: split and reuse
    assert _cuda_reserved_peak(*seq, conf=conf) == 320 * MiB      # oversize block refused
    seq = ([300 * MiB, 300 * MiB, 256 * MiB], [1, -1, 1], [0, 0, 2])
    assert _cuda_reserved_peak(*seq, conf=conf) == 556 * MiB      # 300 >= 256 + 20: refused
    seq = ([300 * MiB, 300 * MiB, 290 * MiB], [1, -1, 1], [0, 0, 2])
    assert _cuda_reserved_peak(*seq, conf=conf) == 300 * MiB      # within the rounding: taken
    # Taken whole: the 10 MiB rest is not split off, so a later 8 MiB request
    # opens a 20 MiB segment instead of reusing it.
    seq = ([300 * MiB, 300 * MiB, 290 * MiB, 8 * MiB], [1, -1, 1, 1], [0, 0, 2, 3])
    assert _cuda_reserved_peak(*seq, conf=conf) == 320 * MiB
    assert _cuda_reserved_peak(*seq) == 300 * MiB


def test_cuda_expandable_segments_grow_one_mapping_per_pool():
    """``expandable_segments:True``: the large pool maps 20 MiB units in place;
    the free block at the end is extended rather than left behind."""
    conf = "expandable_segments:True"
    seq = ([64 * MiB, 64 * MiB, 256 * MiB], [1, -1, 1], [0, 0, 2])
    assert _cuda_reserved_peak(*seq) == 320 * MiB
    assert _cuda_reserved_peak(*seq, conf=conf) == 260 * MiB      # 80 mapped, +180 for 256
    # A non-tail hole that fits is preferred over growing the mapping.
    seq = ([100 * MiB, 50 * MiB, 100 * MiB, 90 * MiB], [1, 1, -1, 1], [0, 1, 0, 3])
    assert _cuda_reserved_peak(*seq, conf=conf) == 160 * MiB
    # Small and large pools map separately, 2 MiB and 20 MiB at a time.
    seq = ([100_000.0, 3 * MiB], [1, 1], [0, 1])
    assert _cuda_reserved_peak(*seq, conf=conf) == 22 * MiB
    assert _cuda_reserved_peak(*seq, conf=conf + ",large_segment_size_mb:64") == 66 * MiB


def test_cuda_unknown_settings_credit_no_free():
    """Another backend or a setting the model does not know: every allocation
    is priced as new (an upper bound of any reuse)."""
    seq = ([64 * MiB, 64 * MiB, 64 * MiB], [1, -1, 1], [0, 0, 2])
    assert _cuda_reserved_peak(*seq) == 64 * MiB
    for conf in ("backend:cudaMallocAsync", "new_knob:1"):
        assert _cuda_reserved_peak(*seq, conf=conf) == 128 * MiB


def test_cuda_estimate_follows_the_process_allocator_conf(monkeypatch, models):
    """The estimate reads ``PYTORCH_CUDA_ALLOC_CONF`` at each call and memoises
    per configuration; measured on an RTX 4070 (torch 2.11) with
    ``roundup_power2_divisions:4``, ``wide34 tail-ev`` B=16 reserved
    1,012,924,416 B and ``wide36 scattered-ev`` B=16 339,738,624 B, both above
    the default-configuration estimate."""
    from nbn.inference import _ve_memory

    model = _wide(34, 4, 3, seed=2)
    B = 16
    ev = _evidence(model, [f"V{i}" for i in range(28, 34)], B)
    eng = TensorVariableElimination()
    eng._extract_factors(model)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    pm = eng._peak_model(model, "V10", ev, B=B)
    pm.device_type = "cuda"
    default = pm.bytes(B)
    assert default < 1_012_924_416
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "roundup_power2_divisions:4")
    pm._memo = {}
    assert pm.bytes(B) >= 1_012_924_416
    model36 = _wide(36, 3, 3, seed=6)
    ev = _evidence(model36, [f"V{i}" for i in range(0, 36, 3) if i != 18], B)
    eng = TensorVariableElimination()
    eng._extract_factors(model36)
    pm = eng._peak_model(model36, "V18", ev, B=B)
    pm.device_type = "cuda"
    assert pm.bytes(B) >= 339_738_624
    assert _ve_memory.cuda_allocator_conf().endswith("roundup_power2_divisions:4")

