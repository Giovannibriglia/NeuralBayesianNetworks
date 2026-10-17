"""Round 4: does a larger budget buy a better answer with the new options?

E1  exact VE vs ``on_oom="condition"``: queries answered exactly, by memory
    budget, time and device (denser ER graphs, 100 nodes, K = 3, 10 queries).
E2  ``certified_query`` anytime mode: certified interval width vs time budget,
    plain likelihood weighting vs collapsed (cutset) sampling.
E3  bounded conditioning: width of the sound bounds vs time budget, on the
    generator's CPTs and on sharper ("skewed") CPTs.

One configuration per process (threads and device are fixed at start-up);
records are appended to ``--out`` as JSON lines.  Problems are rebuilt with
``gen.build`` (same seeds and queries as rounds 1-3).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _setup(threads: int):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    import torch
    torch.set_num_threads(threads)
    sys.path.insert(0, str(HERE.parent.parent))
    sys.path.insert(0, str(HERE))
    return torch


def _sync(torch, dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def _problem(deg, n=100, topology="er", skew=1.0):
    import torch
    from cell import discrete_queries
    from gen import build
    pr = build("discrete", topology, n, deg, k=3, seed=0)
    if skew != 1.0:
        with torch.no_grad():
            for v in pr["nodes"]:
                mech = pr["true_model"].mechanisms[v]
                mech._logits.mul_(skew)
    return pr, discrete_queries(pr, n_q=10, n_ev=5)


def e1(a, torch, dev, base):
    from nbn.inference.conditioning import (ConditioningLimitError, conditioned_query,
                                            plan_conditioning)
    from nbn.inference._ve_memory import _factor_dtype_bytes
    from nbn.inference._memory import OutOfMemoryError
    from nbn.inference.tensor_ve import MPSTensorLimitError, TensorVariableElimination
    mem = a.mem_gb * 2 ** 30
    for deg in a.degs:
        pr, qs = _problem(deg)
        model = pr["true_model"].to(dev)
        for qi, (tgt, ev) in enumerate(qs):
            rec = dict(base, exp="e1", deg=deg, q=qi, mem_gb=a.mem_gb)
            plain = TensorVariableElimination(memory_budget=mem)
            t = time.perf_counter()
            try:
                p = plain.query(model, [tgt], ev)
                _sync(torch, dev)
                rec.update(plain_status="ok", plain_s=time.perf_counter() - t,
                           plain_answer=[float(x) for x in p.cpu()])
            except (OutOfMemoryError, MPSTensorLimitError, RuntimeError) as e:
                rec.update(plain_status=type(e).__name__)
            # the cutset the plan needs (planning only, no cap)
            _, rel = plain._plan_with_relevant(model, tgt, tuple(sorted(ev)))
            try:
                pl = plan_conditioning(model, {n for n in rel}, targets=(tgt,),
                                       clamped=set(ev), budget=mem, device=dev,
                                       dtype_bytes=_factor_dtype_bytes(model),
                                       max_instantiations=float("inf"))
                rec.update(cutset=len(pl.cutset),
                           n_inst=math.prod(int(model.mechanisms[c].n_classes)
                                            for c in pl.cutset))
            except ConditioningLimitError as e:
                rec.update(cutset=None, n_inst=None, plan_error=str(e)[:120])
            if rec.get("plain_status") == "ok":
                rec.update(cond_status="ok", cond_s=rec["plain_s"], conditioned=False,
                           cond_answer=rec["plain_answer"])
            elif rec.get("n_inst") is not None and rec["n_inst"] <= a.max_inst:
                # what on_oom="condition" runs, with a give-up time so that a
                # query needing hours is measured, not waited for
                t = time.perf_counter()
                try:
                    ans = conditioned_query(model, tgt, ev, memory_budget=mem,
                                            max_instantiations=a.max_inst,
                                            max_seconds=a.query_limit_s, engine=plain)
                    _sync(torch, dev)
                    el = time.perf_counter() - t
                    rec.update(conditioned=True, cond_done=ans.n_enumerated)
                    if ans.exact:
                        rec.update(cond_status="ok", cond_s=el,
                                   cond_answer=[float(x) for x in ans.estimate])
                    else:
                        rec.update(cond_status="gave_up", cond_elapsed_s=el,
                                   cond_projected_s=el * ans.n_instantiations
                                   / max(1, ans.n_enumerated))
                except (OutOfMemoryError, MPSTensorLimitError, RuntimeError) as e:
                    rec.update(cond_status=type(e).__name__, cond_error=str(e)[:160])
            else:
                rec.update(cond_status="over_cap")
            if rec.get("plain_answer") and rec.get("cond_answer"):
                rec["max_abs_diff"] = max(abs(x - y) for x, y in
                                          zip(rec["plain_answer"], rec["cond_answer"]))
            print(json.dumps({k: v for k, v in rec.items() if "answer" not in k}), flush=True)
            yield rec


def e2(a, torch, dev, base):
    from nbn.inference.certified import certified_query
    from nbn.inference.tensor_ve import TensorVariableElimination
    for topo, n, deg in a.e2_problems:
        pr, qs = _problem(deg, n=n, topology=topo)
        model = pr["true_model"].to(dev)
        exact = {}
        ve = TensorVariableElimination(memory_budget=a.mem_gb * 2 ** 30)
        for qi, (tgt, ev) in enumerate(qs[: a.n_q]):
            try:
                exact[qi] = [float(x) for x in ve.query(model, [tgt], ev).cpu()]
            except Exception:  # noqa: BLE001  no exact reference for this query
                exact[qi] = None
        for mode in a.modes:
            for tb in a.times:
                for qi, (tgt, ev) in enumerate(qs[: a.n_q]):
                    torch.manual_seed(1000 + qi)
                    kw = dict(time_budget_s=tb)
                    if mode == "collapsed":
                        kw.update(collapse="auto", memory_budget=a.mem_gb * 2 ** 30)
                    rec = dict(base, exp="e2", topology=topo, n=n, deg=deg, q=qi, mode=mode,
                               time_budget_s=tb)
                    try:
                        c = certified_query(model, tgt, ev, **kw)
                        rec.update(status="ok", width=c.max_width, n_samples=c.n_samples,
                                   checkpoints=c.n_checkpoints, seconds=c.seconds,
                                   collapsed=list(c.collapsed),
                                   estimate=[float(x) for x in c.estimate],
                                   lower=[float(x) for x in c.lower],
                                   upper=[float(x) for x in c.upper])
                        ex = exact[qi]
                        if ex is not None:
                            rec["exact"] = ex
                            rec["covers"] = all(lo - 1e-9 <= e <= hi + 1e-9 for lo, e, hi
                                                in zip(rec["lower"], ex, rec["upper"]))
                            rec["err"] = max(abs(x - e) for x, e in zip(rec["estimate"], ex))
                    except Exception as e:  # noqa: BLE001
                        rec.update(status=type(e).__name__, error=str(e)[:160])
                    print(json.dumps({k: v for k, v in rec.items()
                                      if k not in ("estimate", "lower", "upper", "exact")}),
                          flush=True)
                    yield rec


def e3(a, torch, dev, base):
    from nbn.inference.conditioning import ConditioningLimitError, conditioned_query
    for skew in a.skews:
        pr, qs = _problem(a.e3_deg, skew=skew)
        model = pr["true_model"].to(dev)
        for qi, (tgt, ev) in enumerate(qs[: a.n_q]):
            for tb in a.times:
                rec = dict(base, exp="e3", deg=a.e3_deg, skew=skew, q=qi, time_budget_s=tb,
                           mem_gb=a.mem_gb)
                try:
                    b = conditioned_query(model, tgt, ev, memory_budget=a.mem_gb * 2 ** 30,
                                          time_budget_s=tb, max_instantiations=10 ** 12)
                    rec.update(status="ok", width=b.max_width, exact=b.exact,
                               n_enumerated=b.n_enumerated, n_inst=b.n_instantiations,
                               mass=b.prior_mass_enumerated, seconds=b.seconds,
                               cutset=len(b.cutset))
                except ConditioningLimitError as e:
                    rec.update(status="refused", error=str(e)[:160])
                print(json.dumps(rec), flush=True)
                yield rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["e1", "e2", "e3"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--mem-gb", type=float, default=4.0)
    ap.add_argument("--degs", type=float, nargs="+", default=[3, 4, 4.5, 5, 5.5, 6])
    ap.add_argument("--max-inst", type=int, default=3 ** 11)
    ap.add_argument("--query-limit-s", type=float, default=20.0)
    ap.add_argument("--times", type=float, nargs="+", default=[0.25, 1, 4, 16, 64])
    ap.add_argument("--modes", nargs="+", default=["plain", "collapsed"])
    ap.add_argument("--n-q", type=int, default=10)
    ap.add_argument("--e2-problems", default="local:1000:2,er:100:3,er:100:4.5")
    ap.add_argument("--e3-deg", type=float, default=4.5)
    ap.add_argument("--skews", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--host", default=os.uname().nodename)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.e2_problems = [(t, int(n), float(d)) for t, n, d in
                     (x.split(":") for x in a.e2_problems.split(","))]
    torch = _setup(a.threads)
    dev = torch.device(a.device)
    base = dict(host=a.host, device=dev.type, threads=a.threads, torch=torch.__version__)
    gen = {"e1": e1, "e2": e2, "e3": e3}[a.exp](a, torch, dev, base)
    with open(a.out, "a") as fh:
        for rec in gen:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
