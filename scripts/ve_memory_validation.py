"""Measured versus estimated peak memory of exact VE (#178).

``TensorVariableElimination`` refuses (or chunks) a query whose estimated
peak exceeds the device budget, so the estimate must be an *upper bound* of
what the query makes the device hold.  This script runs problems with
different treewidths, evidence patterns, evidence types, batch sizes and query
kinds, measures each call's peak, and prints it next to the estimate
(``estimate_peak_bytes`` with the call's values) and next to the rule it
replaced (largest elimination factor x 3.0, ``_LIVESET_MULTIPLIER`` of #177).

Measurement, per device (relative to just before the call, so the output
counts and pre-existing tensors do not):

* ``cpu``  -- exact: ``torch.profiler`` with ``profile_memory=True`` records
  every allocation and free of the CPU allocator, intra-op temporaries
  included; the peak is the maximum running sum.
* ``cuda`` -- ``measured`` = ``max_memory_allocated`` (exact, tensor level);
  ``driver`` = growth of ``max_memory_reserved`` (the caching allocator's
  segments, what the device actually gives up).  The estimate must bound
  both.
* ``mps``  -- no peak counter exists: ``measured`` =
  ``current_allocated_memory()`` read after every aten op (tensor level, a
  lower bound: blind to temporaries freed inside an op) and ``driver`` =
  growth of ``driver_allocated_memory()`` (heap level, what Metal gives up).
  The estimate must bound ``driver``.  ``--op-probe`` measures the
  temporaries of each op VE uses, at 1 GiB, where heaps are exact.

Every case runs in a fresh process on MPS / CUDA (``--isolate`` elsewhere):
blocks cached by earlier work absorb a case's allocations and would let an
under-estimate pass.  Before measuring, only the CPT cache is warmed
(``_extract_factors``), never a full query, for the same reason; ``cold``
cases skip even that, so the call also tabulates every CPT.

Usage::

    uv run --extra all python scripts/ve_memory_validation.py --device cpu
    uv run --extra all python scripts/ve_memory_validation.py --device mps
    uv run --extra all python scripts/ve_memory_validation.py --device cuda
    # --quick: peaks < ~1 GiB only;  --big-only;  --json out.json;
    # --op-probe (mps);  --mps-uncapped (lift the MPS element cap: large
    #   plans run unchunked, buffers > 512 MiB get exact heaps)

Exit status 1 if any estimate is below a bounded reading (cpu: measured;
cuda: measured and driver; mps: driver), or if a budget-forced case ran over
its budget instead of being chunked within it or refused.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nbn import NeuralBayesianNetwork as NBN  # noqa: E402
from nbn import TensorVariableElimination  # noqa: E402


# --------------------------------------------------------------------------- #
# Problems
# --------------------------------------------------------------------------- #

def _fit(edges, names, cards, seed):
    g = torch.Generator().manual_seed(seed)
    m = NBN(edges, variables={n: ("discrete", cards[n]) for n in names}, device="cpu")
    m.auto_mechanisms()
    m.fit({n: torch.randint(0, cards[n], (300,), generator=g).float() for n in names})
    return m


def wide_net(n: int, indeg: int, card: int, seed: int = 0):
    """Random DAG: node j takes ``min(indeg, j)`` random earlier parents."""
    g = torch.Generator().manual_seed(seed)
    names = [f"V{i}" for i in range(n)]
    edges = [(names[p], names[j]) for j in range(1, n)
             for p in torch.randperm(j, generator=g)[: min(indeg, j)].tolist()]
    return _fit(edges, names, dict.fromkeys(names, card), seed)


def naive_bayes(n_children: int, card_y: int = 4, card_x: int = 3):
    names = ["Y"] + [f"X{i}" for i in range(n_children)]
    cards = {n: (card_y if n == "Y" else card_x) for n in names}
    return _fit([("Y", n) for n in names[1:]], names, cards, 0)


def grid(rows: int, cols: int, card: int):
    """Edges right and down: treewidth ~ min(rows, cols)."""
    names = [f"G{r}_{c}" for r in range(rows) for c in range(cols)]
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((f"G{r}_{c}", f"G{r}_{c + 1}"))
            if r + 1 < rows:
                edges.append((f"G{r}_{c}", f"G{r + 1}_{c}"))
    return _fit(edges, names, dict.fromkeys(names, card), 0)


def hmm(n: int, card_h: int = 4, card_o: int = 5):
    """Hidden chain H0 -> H1 -> ... with one observed child O_i per step."""
    hs, os_ = [f"H{i}" for i in range(n)], [f"O{i}" for i in range(n)]
    edges = [(hs[i], hs[i + 1]) for i in range(n - 1)] + list(zip(hs, os_))
    cards = {**dict.fromkeys(hs, card_h), **dict.fromkeys(os_, card_o)}
    return _fit(edges, hs + os_, cards, 0)


def seven_parent_neural():
    """A neural-categorical child of 7 four-state parents: tabulating it on a
    cold cache runs 16384 configurations through the MLP (review #301)."""
    from nbn.mechanisms import NeuralCategoricalMechanism

    parents = [f"P{i}" for i in range(7)]
    names = parents + ["C"]
    g = torch.Generator().manual_seed(0)
    m = NBN([(p, "C") for p in parents], variables=dict.fromkeys(names, ("discrete", 4)),
            device="cpu")
    m.auto_mechanisms()
    m.set_mechanism("C", NeuralCategoricalMechanism(n_classes=4))
    m.fit({n: torch.randint(0, 4, (300,), generator=g).float() for n in names})
    return m


# Review #301 round 2: barley, UniformRandomSelector(n_evidence=12) seed 0,
# query 2 — mixed 10-512 MiB buffers that fragment MPS heaps.
BARLEY_Q2_TARGET = "dg25"
BARLEY_Q2_EVIDENCE = ["sort", "ngtilg", "frspdag", "tkvs", "nopt", "mod_nmin",
                      "dgv5980", "nmin", "markgrm", "ksort", "udb", "forfrugt"]
# Query 7 of the same selection: 187,600 elements per row at its largest, so
# the MPS element cap chunks a B=1024 batch into 715-row pieces.
BARLEY_Q7_TARGET = "potnmin"
BARLEY_Q7_EVIDENCE = ["nopt", "aks_vgt", "saamng", "pesticid", "saakern", "komm",
                      "nedbarea", "protein", "dgv1059", "antplnt", "sorttkv", "aar_mod"]


def barley():
    """bnlearn barley with tables fitted on 200 forward samples (seed 0).
    Needs the network once (cached in ~/.cache/nbn/bnlearn)."""
    from nbn.bench.core.config import BaselineSpec, build_adapter
    from nbn.bench.problems.bnlearn import BnlearnConfig, BnlearnProblemSource

    cfg = BnlearnConfig(networks=["barley"], seeds=[0], n_train=200, n_test=20,
                        n_reference=20)
    problem = next(BnlearnProblemSource().iter_problems(cfg))
    adapter = build_adapter(BaselineSpec(library="nbn", mechanism="cat",
                                         param_method="mle", inference_method="ve"))
    adapter.fit(problem)
    return adapter.model


@dataclass
class Case:
    name: str
    model_key: str
    targets: list
    evidence: list
    do: tuple = ()
    B: int = 1
    single_row: bool = False  # TensorVariableElimination.query instead of query_batch
    cold: bool = False        # fresh engine: the call also tabulates every CPT
    budget_frac: float = 0.0  # > 0: memory_budget = frac x estimate (forces chunking)
    big: bool = False         # peak >= ~1 GiB: skipped by --quick
    ev_kind: str = "long"     # "long" (int64 on the device), "float" tensors, "list"
    cpu_only: bool = False    # float64 models: MPS has no float64


def _tail(n: int, k: int) -> list[str]:
    return [f"V{i}" for i in range(n - k, n)]


def cases() -> tuple[dict[str, Callable], list[Case]]:
    models = {
        "wide30-d3-k4": lambda: wide_net(30, 3, 4, seed=1),
        "wide34-d4-k3": lambda: wide_net(34, 4, 3, seed=2),
        "wide24-d4-k4": lambda: wide_net(24, 4, 4, seed=3),
        "wide26-d3-k5": lambda: wide_net(26, 3, 5, seed=4),
        "wide36-d3-k3": lambda: wide_net(36, 3, 3, seed=6),
        "grid4x12-k4": lambda: grid(4, 12, 4),
        "grid6x10-k3": lambda: grid(6, 10, 3),
        "naive-bayes-40": lambda: naive_bayes(40),
        "naive-bayes-30": lambda: naive_bayes(30),
        "naive-bayes-y32-x16": lambda: naive_bayes(16, card_y=32, card_x=2),
        "hmm-60": lambda: hmm(60),
        "neuralcat-7-parents": seven_parent_neural,
        "barley": barley,
        # Review #301 round 3: float64 model whose first factor is the
        # intervened root's float32 point mass.
        "wide22-double-do-V0": lambda: wide_net(22, 3, 4, seed=1).double().intervene({"V0": 1}),
    }
    w30_ev = [f"V{i}" for i in range(24, 30)]
    scatter36 = [f"V{i}" for i in range(0, 36, 3) if i != 18]
    scatter24 = [f"V{i}" for i in range(0, 24, 3) if i != 12]
    out = []
    for B in (1, 4, 16, 64):
        out.append(Case("wide30 tail-ev", "wide30-d3-k4", ["V12"], w30_ev, B=B, big=B >= 64))
    for B in (1, 16):
        out.append(Case("wide34 tail-ev", "wide34-d4-k3", ["V10"], _tail(34, 6), B=B))
    for B in (1, 64):
        out.append(Case("wide24 tail-ev", "wide24-d4-k4", ["V12"], _tail(24, 5), B=B))
    for B in (1, 256):
        out.append(Case("wide24 scattered-ev", "wide24-d4-k4", ["V12"], scatter24, B=B))
    for B in (1, 4):
        out.append(Case("wide26 tail-ev", "wide26-d3-k5", ["V13"], _tail(26, 5), B=B, big=B >= 4))
    for B in (1, 16, 128):
        out.append(Case("wide36 scattered-ev", "wide36-d3-k3", ["V18"], scatter36, B=B,
                        big=B >= 128))
    for B in (1, 256, 2048):
        out.append(Case("grid4x12 bottom-ev", "grid4x12-k4", ["G0_0"],
                        [f"G3_{c}" for c in range(12)], B=B))
    for B in (1, 64, 256):
        out.append(Case("grid6x10 bottom-ev", "grid6x10-k3", ["G0_0"],
                        [f"G5_{c}" for c in range(10)], B=B))
    for B in (1024, 16384, 131072):
        out.append(Case("naive-bayes all-children", "naive-bayes-40", ["Y"],
                        [f"X{i}" for i in range(40)], B=B))
    for B in (256, 4096, 65536):
        out.append(Case("hmm-60 all-obs", "hmm-60", ["H30"], [f"O{i}" for i in range(60)], B=B))
    out.append(Case("naive-bayes float-ev", "naive-bayes-40", ["Y"],
                    [f"X{i}" for i in range(40)], B=131072, ev_kind="float"))
    # Review #301 counterexamples.
    nb30 = [f"X{i}" for i in range(30)]
    out.append(Case("naive-bayes list-ev", "naive-bayes-30", ["Y"], nb30, B=4096,
                    ev_kind="list"))
    out.append(Case("naive-bayes X0 d-sep int64", "naive-bayes-30", ["X0"],
                    ["Y"] + nb30[1:], B=20000))
    for frac in (0.15, 0.25):
        out.append(Case(f"naive-bayes float {frac}x budget", "naive-bayes-30", ["Y"], nb30,
                        B=4096, ev_kind="float", budget_frac=frac))
    out.append(Case("neuralcat 7 parents cold", "neuralcat-7-parents", ["C"], ["P0"], B=2,
                    cold=True))
    for B in (64, 256, 384, 512, 700, 704):
        out.append(Case("barley q2", "barley", [BARLEY_Q2_TARGET], BARLEY_Q2_EVIDENCE,
                        B=B, big=B >= 384))
    for B in (64, 256, 384, 512, 700, 715):
        out.append(Case("barley q7", "barley", [BARLEY_Q7_TARGET], BARLEY_Q7_EVIDENCE,
                        B=B, big=B >= 256))
    out.append(Case("wide22 float64 do(V0)", "wide22-double-do-V0", ["V8"],
                    [f"V{i}" for i in range(17, 22)], B=8, cpu_only=True))
    out.append(Case("naive-bayes Y32 x16 binary", "naive-bayes-y32-x16", ["Y"],
                    [f"X{i}" for i in range(16)], B=2_801_664, big=True))
    out.append(Case("wide30 no-ev", "wide30-d3-k4", ["V29"], [], B=64))
    out.append(Case("wide30 joint+clamped", "wide30-d3-k4", ["V12", "V13", "V25"], w30_ev, B=16))
    out.append(Case("wide30 do", "wide30-d3-k4", ["V12"], _tail(30, 4), do=("V20", "V21"), B=64))
    out.append(Case("wide30 query() row", "wide30-d3-k4", ["V12"], w30_ev, single_row=True))
    out.append(Case("wide30 cold", "wide30-d3-k4", ["V12"], w30_ev, B=16, cold=True))
    out.append(Case("wide30 chunked", "wide30-d3-k4", ["V12"], w30_ev, B=64, budget_frac=0.3))
    out.append(Case("naive-bayes chunked", "naive-bayes-40", ["Y"],
                    [f"X{i}" for i in range(40)], B=131072, budget_frac=0.6))
    return models, out


# --------------------------------------------------------------------------- #
# The rule #178 replaced: largest elimination factor x 3.0
# --------------------------------------------------------------------------- #

def old_estimate(eng, model, case: Case) -> int:
    ev = tuple(sorted(case.evidence))
    do = tuple(sorted(case.do))
    clamped = set(ev) | set(do)
    free = [t for t in case.targets if t not in clamped]
    target = free[0] if len(case.targets) == 1 else tuple(free)
    plan, relevant = eng._plan_with_relevant(model, target, ev, do_keys=do)
    ev_set = set(ev) | set(do)
    state = []
    for node in relevant:
        if node in do:
            continue
        scope = list(model.dag.parents(node)) + [node]
        cards = {n: model.mechanisms[n].n_classes for n in scope}
        state.append((set(scope) - ev_set, any(v in ev_set for v in scope), cards))
    batch_free = per_row = 0
    for var in plan:
        rel = [s for s in state if var in s[0]]
        if not rel:
            continue
        rest = [s for s in state if var not in s[0]]
        union = set().union(*(s[0] for s in rel))
        has_b = any(s[1] for s in rel)
        cards = {}
        for s in rel:
            cards.update(s[2])
        n = 1
        for v in union:
            n *= cards[v]
        if has_b:
            per_row = max(per_row, n)
        else:
            batch_free = max(batch_free, n)
        union.discard(var)
        state = rest + [(union, has_b, cards)]
    if len(case.targets) > 1:
        joint = 1
        for t in free:
            joint *= model.mechanisms[t].n_classes
        per_row = max(per_row, joint)
    return int(max(batch_free, case.B * per_row) * 4 * 3.0)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #

def _peak_cpu(fn) -> dict:
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
    mem = sorted(
        (e["ts"], e["args"].get("Ev Idx", 0), e["args"]["Bytes"])
        for e in events
        if e.get("name") == "[memory]" and e["args"].get("Device Type") == 0
    )
    cur = peak = 0
    for _, _, b in mem:
        cur += b
        peak = max(peak, cur)
    return {"measured": peak}


def _peak_cuda(fn, device) -> dict:
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    reserved0 = torch.cuda.memory_reserved(device)
    out = fn()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - base
    reserved = torch.cuda.max_memory_reserved(device) - reserved0
    del out
    # ``driver``: growth of the caching allocator's reserved segments.
    return {"measured": peak, "driver": reserved}


class _PollAllocated(torch.utils._python_dispatch.TorchDispatchMode):
    """Records ``torch.mps.current_allocated_memory()`` around every aten op."""

    def __init__(self) -> None:
        super().__init__()
        self.peak = torch.mps.current_allocated_memory()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.peak = max(self.peak, torch.mps.current_allocated_memory())
        return out


def _peak_mps(fn) -> dict:
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()
    c0 = torch.mps.current_allocated_memory()
    d0 = torch.mps.driver_allocated_memory()
    with _PollAllocated() as poll:
        out = fn()
        torch.mps.synchronize()
    driver = torch.mps.driver_allocated_memory() - d0
    del out
    return {"measured": poll.peak - c0, "driver": driver}


def measure(device: str, fn) -> dict:
    if device == "cpu":
        return _peak_cpu(fn)
    if device.startswith("cuda"):
        return _peak_cuda(fn, torch.device(device))
    if device == "mps":
        return _peak_mps(fn)
    raise ValueError(device)


# --------------------------------------------------------------------------- #
# MPS op probe: temporaries of each op VE uses, at 1 GiB (exact heaps)
# --------------------------------------------------------------------------- #

def mps_op_probe() -> list[dict]:
    G = 1 << 30
    B, X, Y = 1 << 10, 1 << 9, 1 << 9  # B*X*Y float32 = 1 GiB
    dev = "mps"
    probes = [
        ("broadcast add [B,X,1]+[B,1,Y]",
         lambda: (torch.randn(B, X, device=dev), torch.randn(B, Y, device=dev)),
         lambda a, b: a.unsqueeze(2) + b.unsqueeze(1)),
        ("add permuted view + contiguous",
         lambda: (torch.randn(X, Y, B, device=dev), torch.randn(B, X, Y, device=dev)),
         lambda a, b: a.permute(2, 0, 1) + b),
        ("logsumexp contiguous, middle dim",
         lambda: (torch.randn(B, X, Y, device=dev),), lambda a: torch.logsumexp(a, dim=1)),
        ("logsumexp permuted view",
         lambda: (torch.randn(X, Y, B, device=dev),),
         lambda a: torch.logsumexp(a.permute(2, 0, 1), dim=1)),
        ("softmax last dim",
         lambda: (torch.randn(B * X, Y, device=dev),), lambda a: torch.softmax(a, -1)),
        ("advanced index [i, j, :]",
         lambda: (torch.randn(4, 4, 1 << 18, device=dev),
                  torch.randint(0, 4, (B,), device=dev), torch.randint(0, 4, (B,), device=dev)),
         lambda t, i, j: t[i, j, :]),
    ]
    rows = []
    for name, setup, op in probes:
        gc.collect()
        torch.mps.empty_cache()
        ins = setup()
        torch.mps.synchronize()
        d0 = torch.mps.driver_allocated_memory()
        out = op(*ins)
        torch.mps.synchronize()
        grown = torch.mps.driver_allocated_memory() - d0
        out_b = out.numel() * out.element_size()
        del out, ins
        rows.append({"op": name, "driver_growth_GiB": round(grown / G, 3),
                     "output_GiB": round(out_b / G, 3),
                     "temporaries_GiB": round((grown - out_b) / G, 3)})
    return rows


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _values(model, nodes, B, kind, seed):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n in nodes:
        v = torch.randint(0, model.mechanisms[n].n_classes, (B,), generator=g)
        out[n] = v.float() if kind == "float" else v.float().tolist() if kind == "list" else v
    return out


def run_case(device: str, model, case: Case) -> dict:
    from nbn.inference._memory import OutOfMemoryError

    ev = _values(model, case.evidence, case.B, case.ev_kind, 0)
    do = _values(model, case.do, case.B, "long", 1)
    if case.ev_kind == "long":
        ev = {k: v.to(device) for k, v in ev.items()}
    do = {k: v.to(device) for k, v in do.items()}
    target = case.targets if len(case.targets) > 1 else case.targets[0]
    probe = TensorVariableElimination()
    if not case.cold:
        probe._extract_factors(model)  # the warm state the measured engine will be in
    estimate = probe.estimate_peak_bytes(model, target, ev, do, B=case.B)
    eng = TensorVariableElimination(memory_budget=float("inf"))
    row = {"case": case.name, "B": case.B, "estimate": estimate}
    if case.budget_frac:
        # The guard promises the call stays within its budget (chunking), or
        # refuses it before allocating.
        estimate = int(case.budget_frac * estimate)
        row["estimate"] = estimate
        row["budget"] = True
        eng = TensorVariableElimination(memory_budget=estimate, memory_safety=1.0)
    if not case.cold:
        eng._extract_factors(model)  # tabulate the CPTs only, never a full query

    if case.single_row:
        r1 = {k: (v[0] if isinstance(v, (torch.Tensor, list)) else v) for k, v in ev.items()}
        d1 = {k: v[0] for k, v in do.items()}

        def call():
            return eng.query(model, case.targets, r1, do=d1)
    else:
        def call():
            return eng.query_batch(model, case.targets, ev, do=do)

    row["old_3x"] = old_estimate(eng, model, case)
    try:
        m = measure(device, call)
    except OutOfMemoryError as exc:
        if not case.budget_frac or "pre-allocation guard" not in str(exc):
            raise
        row.update(refused=True, measured=0, ratio=float("nan"), old_ratio=float("nan"))
        return row
    row.update(m)
    row["ratio"] = estimate / m["measured"] if m["measured"] else float("nan")
    row["old_ratio"] = row["old_3x"] / m["measured"] if m["measured"] else float("nan")
    if "driver" in m:
        row["driver_ratio"] = estimate / m["driver"] if m["driver"] else float("nan")
    return row


def _fmt(b: float) -> str:
    for unit, k in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if b >= k:
            return f"{b / k:7.2f} {unit}"
    return f"{b:7.0f} B  "


def _print_row(r: dict) -> None:
    if r.get("refused"):
        print(f"{r['case']:30s} B={r['B']:<8d} REFUSED by the guard (budget "
              f"{_fmt(r['estimate'])})", flush=True)
        return
    extra = (f"  driver {_fmt(r['driver'])}  est/driver {r['driver_ratio']:5.2f}"
             if "driver" in r else "")
    print(f"{r['case']:30s} B={r['B']:<8d} measured {_fmt(r['measured'])}  "
          f"{'budget  ' if r.get('budget') else 'estimate'} {_fmt(r['estimate'])}  "
          f"est/meas {r['ratio']:6.2f}  old {_fmt(r['old_3x'])}  "
          f"old/meas {r['old_ratio']:6.2f}{extra}", flush=True)


def _failures(device: str, rows: list[dict]) -> list[str]:
    bad = []
    for r in rows:
        if r.get("refused"):
            continue
        readings = (("driver",) if device == "mps"
                    else ("measured", "driver") if device.startswith("cuda") else ("measured",))
        for key in readings:
            if key in r and r["estimate"] < r[key]:
                what = "over budget" if r.get("budget") else "UNDER-ESTIMATE"
                bad.append(f"{what}: {r['case']} B={r['B']}: "
                           f"{'budget' if r.get('budget') else 'estimate'} "
                           f"{r['estimate']} < {key} {r[key]}")
    return bad


def _run_one(device: str, index: int, args) -> dict:
    """One case in this process."""
    if args.mps_uncapped:
        from nbn.inference import tensor_ve
        tensor_ve._MPS_MAX_ELEMENTS = 1 << 62
    builders, todo = cases()
    case = todo[index]
    model = builders[case.model_key]().to(device)
    return run_case(device, model, case)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--big-only", action="store_true",
                    help="only the cases marked big (peak ~1 GiB or more)")
    ap.add_argument("--json")
    ap.add_argument("--op-probe", action="store_true")
    ap.add_argument("--mps-uncapped", action="store_true",
                    help="lift the MPS element cap (measurement only: MPS reductions "
                         "past it return wrong values, but allocate as usual)")
    ap.add_argument("--isolate", action="store_true",
                    help="run every case in a fresh process (default on mps / cuda)")
    ap.add_argument("--case", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    device = args.device

    if args.case is not None:  # child process: one case, one JSON line
        print("ROW " + json.dumps(_run_one(device, args.case, args)), flush=True)
        return 0

    if args.op_probe and device == "mps":
        for r in mps_op_probe():
            print(r)

    _, todo = cases()
    picked = [i for i, c in enumerate(todo)
              if not (args.quick and c.big) and not (args.big_only and not c.big)
              and not (c.cpu_only and device != "cpu")]
    isolate = args.isolate or device != "cpu"
    rows = []
    models: dict = {}
    for i in picked:
        if isolate:
            cmd = [sys.executable, os.path.abspath(__file__), "--device", device,
                   "--case", str(i)] + (["--mps-uncapped"] if args.mps_uncapped else [])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("ROW ")]
            if proc.returncode or not lines:
                print(proc.stdout[-2000:], proc.stderr[-4000:], file=sys.stderr)
                raise SystemExit(f"case {todo[i].name} B={todo[i].B} failed")
            r = json.loads(lines[-1][4:])
        else:
            builders, _ = cases()
            case = todo[i]
            if case.model_key not in models:
                models[case.model_key] = builders[case.model_key]().to(device)
            r = run_case(device, models[case.model_key], case)
        rows.append(r)
        _print_row(r)

    est = [r for r in rows if not r.get("budget") and not r.get("refused")]
    for label, sel in (("all", est), ("measured >= 1 MiB",
                                       [r for r in est if r["measured"] >= (1 << 20)])):
        if not sel:
            continue
        for key in ("ratio", "old_ratio") + (("driver_ratio",) if "driver_ratio" in sel[0] else ()):
            vals = [r[key] for r in sel if r[key] == r[key]]
            if vals:
                print(f"[{device}] {label:18s} {key:12s} min {min(vals):6.2f}  "
                      f"median {statistics.median(vals):6.2f}  max {max(vals):6.2f}  "
                      f"(n={len(vals)})")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"device": device, "torch": torch.__version__, "rows": rows}, fh,
                      indent=1)
    bad = _failures(device, rows)
    for line in bad:
        print(line, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
