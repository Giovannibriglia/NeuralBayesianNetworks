"""Orchestrate scaling sweeps: one subprocess per cell, budget enforced.

Each cell runs ``cell.py`` in a fresh interpreter with
* torch / BLAS threads pinned (``--threads``),
* an RSS watchdog that kills the process above ``--mem-gb`` (status ``oom``),
* a hard wall-clock kill at budget + load slack (status ``timeout``).

Sweeps stop escalating a method once it fails at some size (failures are
monotone in size for our generators), and record the failure.

Usage: python run.py <sweep-name> [--out results.jsonl] [--mem-gb 4] ...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PY = sys.executable
sys.path.insert(0, str(HERE))
import gen  # noqa: E402

PROBLEMS = HERE / "problems"


def ensure_problem(family, topology, n, deg, k=3, window=5, seed=0):
    if family == "discadd":
        name = f"discadd_fixed_n{n}_deg{deg}_k{k}_s{seed}"
    else:
        name = gen.problem_name(family, topology, n, deg, k, seed, window)
    p = PROBLEMS / f"{name}.pt"
    if not p.exists():
        if family == "discadd":
            pr = gen.build_additive(n, int(deg), k=k, seed=seed)
        else:
            pr = gen.build(family, topology, n, deg, k=k, window=window, seed=seed)
        gen.save(pr, PROBLEMS)
    return p


def run_cell(problem, task, method, *, device="cpu", threads=1, budget_s=30.0,
             mem_gb=4.0, opt=None, slack_s=60.0, code_root=None):
    cmd = [PY, str(HERE / "cell.py"), str(problem), task, method,
           "--device", device, "--threads", str(threads),
           "--budget-s", str(budget_s)]
    if opt:
        cmd += ["--opt", *[f"{k}={v}" for k, v in opt.items()]]
    env = dict(os.environ, PYTHONPATH=str(code_root or ROOT))
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
    ps = psutil.Process(proc.pid)
    peak = 0.0
    killed = None
    while proc.poll() is None:
        try:
            rss = ps.memory_info().rss / 2**20
            peak = max(peak, rss)
            if rss > mem_gb * 1024:
                proc.kill()
                killed = "oom"
                break
        except psutil.NoSuchProcess:
            break
        if time.time() - t0 > budget_s + slack_s:
            proc.kill()
            killed = "timeout"
            break
        time.sleep(0.02)
    out, err = proc.communicate()
    rec = None
    for line in out.splitlines():
        if line.startswith("RESULT "):
            rec = json.loads(line[7:])
    if rec is None:
        rec = dict(problem=Path(problem).stem, task=task, method=method, device=device,
                   status=killed or "crash", error=err[-800:])
        # recover graph stats from the name
    if killed:
        rec["status"] = killed
    rec["watch_peak_rss_mb"] = peak
    rec["mem_cap_gb"] = mem_gb
    return rec


def meta_from_problem(p):
    import torch
    pr = torch.load(p, weights_only=False)
    return dict(n=pr["n"], deg=pr["deg"], n_edges=pr["n_edges"], max_in=pr["max_in"],
                topology=pr["topology"], family=pr["family"], window=pr["window"])


# --------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------- #

NODES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
DEGS = [1, 1.5, 2, 3, 4, 6, 8]


def sweep_specs(name):
    """Yield (axis_key, family, topology, sizes, deg_or_n, task, methods, kwargs)."""
    specs = []
    if name == "discrete_fit":
        for topo in ("er",):
            specs.append(("n", "discrete", topo, NODES, 2, "fit", ["nbn", "pgmpy"], {}))
            specs.append(("deg", "discrete", topo, DEGS, 200, "fit", ["nbn", "pgmpy"], {}))
    elif name == "discrete_infer":
        for topo in ("er", "local"):
            specs.append(("n", "discrete", topo, NODES, 2, "infer",
                          ["nbn_ve", "pgmpy_ve", "nbn_lw"], {}))
        specs.append(("deg", "discrete", "er", DEGS, 100, "infer",
                      ["nbn_ve", "pgmpy_ve", "nbn_lw"], {}))
    elif name == "discrete_infer_cert":
        for topo in ("er", "local"):
            specs.append(("n", "discrete", topo, NODES, 2, "infer", ["nbn_cert"], {}))
        specs.append(("deg", "discrete", "er", DEGS, 100, "infer", ["nbn_cert"], {}))
    elif name == "discrete_fit_indeg":
        specs.append(("deg", "discadd", "fixed", [2, 4, 6, 8, 10, 12, 13, 14, 16, 18, 20],
                      30, "fit", ["nbn", "nbn_sparse", "pgmpy"], {}))
    elif name == "r2_fit_ncat":
        specs.append(("n", "discrete", "er", NODES[:7], 2, "fitnn",
                      ["ncat_local", "ncat_stacked", "pyro_ncat"], {}))
    elif name == "r2_fit_flex":
        specs.append(("n", "ngs", "er", NODES[:7], 2, "fitnn",
                      ["flex_local", "flex_stacked"], {}))
    elif name == "r2_fit_flow":
        specs.append(("n", "ngs", "er", NODES[:7], 2, "fitnn",
                      ["flow_local", "flow_stacked"], {}))
    elif name == "r2_knn_query":
        specs.append(("n", "ngs", "er", NODES[:7], 2, "mechq", ["knn_lw"], {}))
    elif name == "r2_flex_query":
        specs.append(("n", "ngs", "er", NODES[:7], 2, "mechq", ["flex_lw"], {}))
    elif name == "r2_ais":
        specs.append(("n", "discrete", "local", NODES[:7], 2, "ais", ["ais"], {}))
    elif name == "r2_avi":
        specs.append(("n", "discrete", "er", NODES[:7], 2, "avi", ["avi"], {}))
    elif name == "discrete_sets":
        specs.append(("n", "discrete", "er", NODES, 2, "sets",
                      ["nbn_interval", "nbn_intervals_batched", "nbn_l1"], {}))
        specs.append(("deg", "discrete", "er", DEGS, 200, "sets",
                      ["nbn_interval", "nbn_intervals_batched", "nbn_l1"], {}))
    elif name == "cont_fit":
        specs.append(("n", "ngs", "er", NODES, 2, "fit",
                      ["nbn_local", "nbn_stacked", "pyro_svi"], {}))
        specs.append(("deg", "ngs", "er", DEGS, 100, "fit",
                      ["nbn_local", "nbn_stacked", "pyro_svi"], {}))
    elif name == "cont_infer":
        specs.append(("n", "ngs", "er", NODES, 2, "infer",
                      ["nbn_lw", "pyro_vec", "pyro_is"], {}))
        specs.append(("deg", "ngs", "er", DEGS, 100, "infer",
                      ["nbn_lw", "pyro_vec", "pyro_is"], {}))
    else:
        raise ValueError(name)
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mem-gb", type=float, default=4.0)
    ap.add_argument("--budget-s", type=float, default=30.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--opt", nargs="*", default=[])
    ap.add_argument("--code", default="branch",
                    help="'branch' (this worktree) or a path to another NBN checkout")
    ap.add_argument("--sizes", type=float, nargs="*", default=None,
                    help="override the sweep's x-axis values")
    ap.add_argument("--axes", nargs="*", default=None, help="restrict to 'n' and/or 'deg'")
    ap.add_argument("--slack", type=float, default=60.0,
                    help="extra wall-clock seconds for untimed set-up (fits, proposal training)")
    ap.add_argument("--tag", default=None, help="label appended to the method names")
    ap.add_argument("--topologies", nargs="*", default=None)
    a = ap.parse_args()
    code_root = None if a.code == "branch" else Path(a.code)
    tag = ("" if a.code == "branch" else "@master") if a.tag is None else a.tag
    out = Path(a.out or HERE / "results" / f"{a.sweep}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    extra_opt = dict(kv.split("=", 1) for kv in a.opt)
    for axis, fam, topo, sizes, other, task, methods, kw in sweep_specs(a.sweep):
        if a.methods:
            methods = [m for m in methods if m in a.methods]
        if a.axes and axis not in a.axes:
            continue
        if a.topologies and topo not in a.topologies:
            continue
        if a.sizes:
            sizes = [int(x) if axis == "n" else x for x in a.sizes]
        for seed in a.seeds:
            dead = set()
            for s in sizes:
                if axis == "n":
                    n, deg = s, other
                else:
                    n, deg = other, s
                if a.max_n and n > a.max_n:
                    continue
                live = [m for m in methods if m not in dead]
                if not live:
                    break
                p = ensure_problem(fam, topo, n, deg, seed=seed)
                meta = meta_from_problem(p)
                for m in live:
                    rec = run_cell(p, task, m, device=a.device, threads=a.threads,
                                   budget_s=a.budget_s, mem_gb=a.mem_gb,
                                   opt={**kw, **extra_opt}, code_root=code_root,
                                   slack_s=a.slack)
                    rec["method"] = m + tag
                    rec["code"] = a.code
                    rec.update({k: v for k, v in meta.items() if k not in rec})
                    rec["axis"] = axis
                    rec["sweep"] = a.sweep
                    rec["seed"] = seed
                    with open(out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    msg = (f"[{a.sweep}] {axis}={s:<6} {topo:5} {m + tag:14} {rec['status']:8} "
                           f"t={rec.get('time_s', float('nan')):.3f}s "
                           f"rss={rec.get('watch_peak_rss_mb', 0):.0f}MB "
                           f"E={meta['n_edges']}")
                    if rec["status"] not in ("ok",):
                        msg += f"  {rec.get('error', '')[:160]}"
                        dead.add(m)
                    print(msg, flush=True)


if __name__ == "__main__":
    main()
