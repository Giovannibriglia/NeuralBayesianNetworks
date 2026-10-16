"""Figures and tables for the scaling study (reads results/*.jsonl).

Style: NBN's shared example style (``examples/_figures.py``, Okabe-Ito).
Colour = method family (fixed per entity), line style = code version
(solid: this branch, dashed: master).  A cross marks the first size at which a
method failed its budget, labelled with the reason.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "examples"))
from _figures import (  # noqa: E402
    AQUA, BLUE, DOUBLE, GREY, INK, MAGENTA, ORANGE, SKY, VERMILLION, use_style,
)

import os  # noqa: E402

RES = Path(os.environ.get("SCALING_RES", HERE / "results"))
FIG = Path(os.environ.get("SCALING_FIG", HERE / "figures"))

COLOR = {
    "nbn": BLUE, "nbn_ve": BLUE, "nbn_local": SKY, "nbn_stacked": BLUE,
    "pgmpy": ORANGE, "pgmpy_ve": ORANGE,
    "nbn_lw": AQUA, "nbn_cert": MAGENTA,
    "pyro_svi": VERMILLION, "pyro_vec": VERMILLION, "pyro_is": INK,
    "nbn_interval": SKY, "nbn_intervals_batched": BLUE, "nbn_l1": AQUA,
}
LABEL = {
    "nbn": "NBN MLE", "pgmpy": "pgmpy MLE",
    "nbn_ve": "NBN exact VE", "pgmpy_ve": "pgmpy exact VE",
    "nbn_lw": "NBN LW (approx.)", "nbn_cert": "NBN certified LW",
    "nbn_local": "NBN local fit", "nbn_stacked": "NBN stacked fit",
    "pyro_svi": "Pyro SVI (MLE)", "pyro_vec": "Pyro IS (vectorised)",
    "pyro_is": "Pyro Importance", "nbn_interval": "NBN interval sets",
    "nbn_l1": "NBN L1 sets",
    "nbn_intervals_batched": "NBN interval sets (batched)",
}


def load(*names):
    rows = []
    for nm in names:
        p = RES / f"{nm}.jsonl"
        if p.exists():
            rows += [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    return rows


def _base(m):
    return m.split("@")[0]


def _x(r, axis):
    return r["n"] if axis == "n" else r["n_edges"]


def agg(rows, axis, topo=None, methods=None):
    """{method: [cell, ...]} with one cell per x value, aggregated over seeds.

    A cell has x (mean nodes or mean edges), t (median time over seeds that
    answered every query), ok (all seeds answered all), answered/asked
    totals, and fail (first failure status, if any seed failed).
    """
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("axis") != axis or (topo and r.get("topology") != topo):
            continue
        m = r["method"] + ("" if r.get("device", "cpu") == "cpu" else f"[{r['device']}]")
        if methods and m not in methods:
            continue
        key = r["n"] if axis == "n" else r["deg"]
        groups[m][key].append(r)
    out = {}
    for m, by in groups.items():
        cells = []
        for key in sorted(by):
            rs = by[key]
            full = [r for r in rs if r["status"] == "ok"
                    and r.get("n_answered", r.get("n_q")) == r.get("n_q")]
            part = [r for r in rs if r["status"] == "ok" and r not in full]
            bad = [r for r in rs if r["status"] != "ok"]
            ts = [r["time_s"] for r in full + part]
            cells.append(dict(
                key=key, x=float(np.mean([_x(r, axis) for r in rs])),
                t=float(np.median(ts)) if ts else None,
                ok=len(full) == len(rs), n_runs=len(rs),
                answered=sum(r.get("n_answered", r.get("n_q", 0)) or 0 for r in full + part),
                asked=sum(r.get("n_q", 0) or 0 for r in rs),
                fail=(bad[0]["status"] if bad else None), n_fail=len(bad),
                rows=rs))
        out[m] = cells
    return out


def plot_time(ax, rows, axis, topo=None, budget=30.0, methods=None, ylabel=True):
    for mi, (m, cells) in enumerate(sorted(agg(rows, axis, topo, methods).items())):
        c = COLOR.get(_base(m.split("[")[0]), GREY)
        ls = "--" if "@" in m else "-"
        lab = LABEL.get(_base(m.split("[")[0]), m) + (" [master]" if "@" in m else "")
        good = [cl for cl in cells if cl["ok"]]
        if good:
            ax.plot([cl["x"] for cl in good], [cl["t"] for cl in good], ls, color=c,
                    marker="o", ms=3.5, lw=1.6, label=lab)
        partial = [cl for cl in cells if not cl["ok"] and cl["t"] is not None]
        if partial:
            ax.plot([cl["x"] for cl in partial], [cl["t"] for cl in partial], linestyle="none",
                    marker="o", mfc="none", color=c, ms=5, label=None if good else lab)
        dead = [cl for cl in cells if cl["t"] is None]
        if dead:
            cl = dead[0]
            xd = cl["x"] * (1.0 + 0.12 * mi)   # keep coincident failure marks apart
            ax.plot([xd], [budget], "x", color=c, ms=7, mew=2,
                    label=None if (good or partial) else lab)
            ax.annotate({"timeout": "time", "oom": "mem"}.get(cl["fail"], cl["fail"]),
                        (xd, budget), textcoords="offset points", xytext=(4, 4 + 10 * (mi % 2)),
                        fontsize=7, color=INK)
    ax.axhline(budget, color=GREY, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=budget * 4)   # room for the failure labels under the title
    ax.set_xlabel("nodes" if axis == "n" else "edges (mean over seeds)")
    if ylabel:
        ax.set_ylabel("time (s)")


def fig_discrete_fit():
    rows = load("discrete_fit")
    if not rows:
        return
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    plot_time(axs[0], rows, "n")
    plot_time(axs[1], rows, "deg", ylabel=False)
    axs[0].set_title("Discrete MLE vs nodes (ER, deg 2)")
    axs[1].set_title("Discrete MLE vs edges (ER, n=200)")
    axs[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "discrete_fit.png", dpi=200)
    fig.savefig(FIG / "discrete_fit.pdf")


def fig_discrete_infer():
    rows = load("discrete_infer", "discrete_infer_master", "discrete_infer_cert")
    if not rows:
        return
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 5.4))
    plot_time(axs[0, 0], rows, "n", "local")
    plot_time(axs[0, 1], rows, "n", "er", ylabel=False)
    plot_time(axs[1, 0], rows, "deg", "er")
    axs[0, 0].set_title("10 queries vs nodes, bounded tree-width (window 5)")
    axs[0, 1].set_title("10 queries vs nodes, Erdős–Rényi (deg 2)")
    axs[1, 0].set_title("10 queries vs edges, Erdős–Rényi (n = 100)")
    ax = axs[1, 1]
    A = agg(rows, "n", "er", ["nbn_ve", "nbn_ve@master", "pgmpy_ve"])
    for m, cells in sorted(A.items()):
        c = COLOR.get(_base(m), GREY)
        ax.plot([cl["x"] for cl in cells],
                [100.0 * cl["answered"] / max(1, cl["asked"]) for cl in cells],
                "--" if "@" in m else "-", color=c, marker="o", ms=3.5,
                label=LABEL[_base(m)] + (" [master]" if "@" in m else ""))
    ax.set_xscale("log")
    ax.set_ylim(-3, 103)
    ax.set_xlabel("nodes")
    ax.set_ylabel("queries answered exactly (%)")
    ax.set_title("Exact answers within budget, Erdős–Rényi (deg 2)\nhollow markers in other panels = partial")
    h, labs = [], []
    for a in axs.flat[:3]:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in labs:
                h.append(hh)
                labs.append(ll)
    fig.legend(h, labs, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "discrete_infer.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "discrete_infer.pdf", bbox_inches="tight")


def lw_errors(rows):
    """max |LW - exact| per problem, exact from NBN VE or pgmpy VE (agreeing)."""
    exact = {}
    for r in rows:
        if r["method"] in ("nbn_ve", "pgmpy_ve") and r["status"] == "ok":
            for qi, a in enumerate(r.get("answers") or []):
                if a is not None:
                    exact.setdefault((r["problem"], qi), a)
    out = []
    for r in rows:
        if _base(r["method"]).startswith(("nbn_lw", "nbn_cert")) and r["status"] == "ok":
            for qi, a in enumerate(r.get("answers") or []):
                e = exact.get((r["problem"], qi))
                if e is not None and a is not None:
                    out.append(dict(problem=r["problem"], method=r["method"], n=r["n"],
                                    topology=r["topology"], axis=r["axis"],
                                    err=max(abs(x - y) for x, y in zip(a, e)),
                                    ess=(r.get("ess_frac") or [None] * 99)[qi],
                                    covered=(None if not r.get("lower") else all(
                                        lo - 1e-9 <= x <= hi + 1e-9 for lo, x, hi in
                                        zip(r["lower"][qi], e, r["upper"][qi]))),
                                    width=(r.get("max_width") or [None] * 99)[qi]))
    return out


def width_bound_stats(rows, k=3, budget_bytes=0.75 * 4 * 2**30, elem=4):
    """How often the tree-width lower bound *proves* a refused query infeasible."""
    need = math.log(budget_bytes / elem, k) - 1  # lb above this => factor > budget
    tot = refused = proven = 0
    gaps = []
    for r in rows:
        if r["method"] != "nbn_ve" or r["status"] != "ok" or not r.get("width_bounds"):
            continue
        for a, (lb, ub) in zip(r["answers"], r["width_bounds"]):
            tot += 1
            gaps.append(ub - lb)
            if a is None:
                refused += 1
                proven += lb > need
    if not tot:
        return ""
    return (f"Width bounds on {tot} exact-VE queries: median gap (min-fill width − lower bound) = "
            f"{np.median(gaps):.0f}; {refused} refused by the memory guard, of which {proven} are "
            f"*proven* infeasible by the lower bound (needs lb > {need:.1f}).\n")


def ve_agreement(rows):
    by = defaultdict(dict)
    for r in rows:
        if r["method"] in ("nbn_ve", "pgmpy_ve") and r["status"] == "ok":
            by[r["problem"]][r["method"]] = r["answers"]
    diffs = []
    for p, d in by.items():
        if len(d) == 2:
            for a, b in zip(d["nbn_ve"], d["pgmpy_ve"]):
                if a is not None and b is not None:
                    diffs.append(max(abs(x - y) for x, y in zip(a, b)))
    return diffs


def fig_sets():
    rows = load("discrete_sets")
    if not rows:
        return
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    plot_time(axs[0], rows, "n")
    plot_time(axs[1], rows, "deg", ylabel=False)
    axs[0].set_title("S4 per-factor sets vs nodes (ER, deg 2)")
    axs[1].set_title("S4 per-factor sets vs edges (ER, n=200)")
    axs[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "discrete_sets.png", dpi=200)
    fig.savefig(FIG / "discrete_sets.pdf")


def fig_cont_fit():
    rows = load("cont_fit", "cont_fit_mps")
    if not rows:
        return
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE * 1.45, 2.8))
    plot_time(axs[0], [r for r in rows if r.get("device", "cpu") == "cpu"], "n")
    plot_time(axs[1], [r for r in rows if r.get("device", "cpu") == "cpu"], "deg", ylabel=False)
    axs[0].set_title("Non-Gaussian fit vs nodes (ER, deg 2)\n1 CPU thread")
    axs[1].set_title("Non-Gaussian fit vs edges (ER, n=100)\n1 CPU thread")
    ax = axs[2]
    for m in ("nbn_local", "nbn_stacked", "pyro_svi"):
        rs = sorted([r for r in rows if r["method"] == m and r["status"] == "ok"
                     and r.get("axis") == "n" and r.get("device", "cpu") == "cpu"],
                    key=lambda r: r["n"])
        if rs:
            ax.plot([r["n"] for r in rs],
                    [r["true_test_ll_median"] - r["test_ll_median"] for r in rs
                     ] if "test_ll_median" in rs[0] else [r["true_test_ll"] - r["test_ll"] for r in rs],
                    "-o", ms=3.5, color=COLOR[m], label=LABEL[m])
    ax.set_xscale("log")
    ax.set_xlabel("nodes")
    ax.set_ylabel("LL gap to truth\n(median row, nats / node)")
    ax.set_ylim(bottom=0)
    ax.set_title("Fit quality (same budget)")
    h, labs = axs[0].get_legend_handles_labels()
    fig.legend(h, labs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "cont_fit.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "cont_fit.pdf", bbox_inches="tight")


def fig_cont_infer():
    rows = load("cont_infer", "cont_infer_master")
    if not rows:
        return
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.8))
    plot_time(axs[0], rows, "n")
    plot_time(axs[1], rows, "deg", ylabel=False)
    axs[0].set_title("Non-Gaussian queries vs nodes\n(10 queries, 10k particles)")
    axs[1].set_title("Non-Gaussian queries vs edges (n=100)")
    h, labs = axs[0].get_legend_handles_labels()
    fig.legend(h, labs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "cont_infer.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "cont_infer.pdf", bbox_inches="tight")


def cont_errors():
    ref = {r["problem"]: r for r in load("cont_infer_ref")}
    out = []
    for r in load("cont_infer", "cont_infer_master"):
        if r["status"] != "ok" or r["problem"] not in ref:
            continue
        for qi, a in enumerate(r.get("answers") or []):
            out.append(dict(method=r["method"], n=r["n"], axis=r["axis"],
                            err=abs(a - ref[r["problem"]]["answers"][qi]),
                            ess=(r.get("ess_frac") or [None] * 99)[qi]))
    return out


def fig_approx():
    rows = load("discrete_infer", "discrete_infer_cert")
    errs = lw_errors(rows)
    if not errs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.7))
    for ax, topo in zip(axs, ("er", "local")):
        for m, c, lab in (("nbn_lw", AQUA, "LW |error| vs exact"),
                          ("nbn_cert", MAGENTA, "certified LW |error|")):
            es = [e for e in errs if e["method"] == m and e["topology"] == topo and e["axis"] == "n"]
            if es:
                ax.scatter([e["n"] * (1.0 + 0.04 * (m == "nbn_cert")) for e in es],
                           [max(e["err"], 1e-5) for e in es], s=6, color=c, alpha=0.6, label=lab)
        ws = [e for e in errs if e["method"] == "nbn_cert" and e["topology"] == topo
              and e["axis"] == "n" and e.get("width") is not None]
        if ws:
            ax.scatter([e["n"] * 1.04 for e in ws], [e["width"] for e in ws], s=10,
                       marker="_", color=INK, label="certified width (max over classes)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("nodes")
        ax.set_title({"er": "Erdős–Rényi", "local": "bounded tree-width"}[topo]
                     + ": approximate arms where exact is known")
    axs[0].set_ylabel("absolute error / width")
    axs[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "discrete_approx.png", dpi=200)
    fig.savefig(FIG / "discrete_approx.pdf")


def fig_indeg():
    rows = load("discrete_fit_indeg")
    if not rows:
        return
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE * 1.45, 2.6))
    for r in rows:
        r["_d"] = r["deg"]
    for ax, field, lab in [(axs[0], "time_s", "time (s)"),
                           (axs[1], "watch_peak_rss_mb", "peak RSS (MB)"),
                           (axs[2], "acc_tv", "TV to true CPD\n(mean over rows)")]:
        for m in ("nbn", "nbn_sparse", "pgmpy"):
            rs = sorted([r for r in rows if r["method"] == m], key=lambda r: r["_d"])
            ok = [r for r in rs if r["status"] == "ok"]
            bad = [r for r in rs if r["status"] != "ok"]
            c = {"nbn": SKY, "nbn_sparse": BLUE, "pgmpy": ORANGE}[m]
            lbl = {"nbn": "NBN dense table", "nbn_sparse": "NBN sparse table",
                   "pgmpy": "pgmpy MLE"}[m]
            ax.plot([r["_d"] for r in ok], [r[field] for r in ok], "-o", ms=3.5, color=c,
                    label=lbl)
            if bad and field != "acc_tv":
                r = bad[0]
                y = 30.0 if field == "time_s" else r["watch_peak_rss_mb"]
                ax.plot([r["_d"]], [y], "x", color=c, ms=7, mew=2)
                ax.annotate({"timeout": "time", "oom": "mem"}.get(r["status"], r["status"]),
                            (r["_d"], y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("in-degree (parents per node)")
        ax.set_ylabel(lab)
        if field != "acc_tv":
            ax.set_yscale("log")
    axs[0].axhline(30, color=GREY, lw=0.8, ls=":")
    axs[1].axhline(4096, color=GREY, lw=0.8, ls=":")
    axs[0].set_title("Discrete MLE vs in-degree\n(30 nodes, K=3, 10k rows)")
    axs[1].set_title("Memory")
    axs[2].set_title("Accuracy (identical where all run):\nfragmentation, not compute")
    axs[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "discrete_fit_indeg.png", dpi=200)
    fig.savefig(FIG / "discrete_fit_indeg.pdf")


def fig_mps():
    cpu = [r for r in load("cont_fit") if r.get("axis") == "n"]
    gpu = [r for r in load("cont_fit_mps") if r.get("axis") == "n"]
    full = [r for r in load("cont_fit_full") if r.get("axis") == "n"]
    if not gpu:
        return
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, 3.0))
    for rows, ls, dev in [(cpu, "-", "1 thread"), (full, ":", "10 threads"),
                          (gpu, "--", "MPS GPU")]:
        for m in ("nbn_local", "nbn_stacked", "pyro_svi"):
            rs = sorted([r for r in rows if r["method"] == m], key=lambda r: r["n"])
            ok = [r for r in rs if r["status"] == "ok"]
            if ok:
                ax.plot([r["n"] for r in ok], [r["time_s"] for r in ok], ls, marker="o", ms=3,
                        color=COLOR[m], label=f"{LABEL[m]}, {dev}")
    ax.axhline(30, color=GREY, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nodes")
    ax.set_ylabel("fit time (s)")
    ax.set_title("Non-Gaussian fit: 1 thread vs 10 threads vs GPU\n(same model, same Adam steps)")
    ax.legend(fontsize=5.5, loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(FIG / "cont_fit_mps.png", dpi=200)
    fig.savefig(FIG / "cont_fit_mps.pdf")


def frontier(rows, topo=None, axis="n"):
    """Largest x at which each method succeeded (all queries answered), per seed."""
    best = defaultdict(dict)
    for r in rows:
        if r.get("axis") != axis or (topo and r.get("topology") != topo):
            continue
        ok = r["status"] == "ok" and r.get("n_answered", r.get("n_q")) == r.get("n_q")
        key = (r["method"] + ("" if r.get("device", "cpu") == "cpu" else f"[{r['device']}]"))
        x = r["n"] if axis == "n" else r["n_edges"]
        if ok:
            best[key][r.get("seed", 0)] = max(best[key].get(r.get("seed", 0), 0), x)
        else:
            best[key].setdefault(r.get("seed", 0), 0)
    return {m: sorted(v.values()) for m, v in best.items()}


def table(rows, axis="n", topo=None, field="time_s"):
    """Markdown table: rows = x values, columns = methods (aggregated over seeds)."""
    A = agg(rows, axis, topo)
    meths = sorted(A)
    keys = sorted({cl["key"] for cells in A.values() for cl in cells})
    xs = {}
    for cells in A.values():
        for cl in cells:
            xs.setdefault(cl["key"], []).append(cl["x"])
    hdr = ("| nodes | " if axis == "n" else "| mean in-degree (edges) | ") + " | ".join(meths) + " |"
    lines = [hdr, "|" + "---|" * (len(meths) + 1)]
    for k in keys:
        lab = f"{k}" if axis == "n" else f"{k:g} ({np.mean(xs[k]):.0f})"
        row = []
        for m in meths:
            cl = next((c for c in A[m] if c["key"] == k), None)
            if cl is None:
                row.append("")
            elif cl["t"] is None:
                row.append(f"{cl['fail']}" + (f" ({cl['n_fail']}/{cl['n_runs']})" if cl["n_runs"] > 1 else ""))
            else:
                if field == "time_s":
                    v = cl["t"]
                else:
                    vals = [r.get(field) for r in cl["rows"] if r["status"] == "ok" and r.get(field) is not None]
                    v = float(np.median(vals)) if vals else float("nan")
                txt = f"{v:.3g}"
                if not cl["ok"]:
                    txt += f" ({cl['answered']}/{cl['asked']})" if cl["n_fail"] == 0 else f" ({cl['fail']} in {cl['n_fail']}/{cl['n_runs']})"
                row.append(txt)
        lines.append(f"| {lab} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    FIG.mkdir(exist_ok=True)
    use_style()
    plt.rcParams.update({"legend.fontsize": 7, "axes.titlesize": 8})
    fig_discrete_fit()
    fig_discrete_infer()
    fig_sets()
    fig_cont_fit()
    fig_cont_infer()
    fig_indeg()
    fig_mps()
    fig_approx()
    out = []
    out.append("## Frontier: largest size solved within the budget (per seed)\n")
    for nm, tp in [("discrete_infer", "local"), ("discrete_infer", "er"), ("discrete_fit", None),
                   ("discrete_sets", None), ("cont_fit", None), ("cont_infer", None)]:
        rows = load(nm, nm + "_master", nm + "_mps", nm + "_cert")
        if nm == "discrete_infer":
            rows += load("discrete_infer_cert")
        if rows:
            fr = frontier(rows, tp)
            out.append(f"- **{nm}{' (' + tp + ')' if tp else ''}**: "
                       + "; ".join(f"{m}: {v}" for m, v in sorted(fr.items())) + "\n")
    for nm, axes in [("discrete_fit", [("n", None), ("deg", None)]),
                     ("discrete_sets", [("n", None), ("deg", None)]),
                     ("cont_fit", [("n", None), ("deg", None)]),
                     ("cont_infer", [("n", None), ("deg", None)])]:
        rows = load(nm, nm + "_master", nm + "_mps")
        for ax, tp in axes:
            out.append(f"### {nm}, x = {ax}\n\n" + table(rows, ax, tp) + "\n")
    rows = load("discrete_infer", "discrete_infer_master", "discrete_infer_cert")
    rows = [r for r in rows if r.get("seed", 0) == 0] + [r for r in rows if r.get("seed", 0) != 0]
    for ax, tp in [("n", "local"), ("n", "er"), ("deg", "er")]:
        out.append(f"### discrete_infer ({tp}), x = {ax}\n\n" + table(rows, ax, tp) + "\n")
    wb = width_bound_stats(rows)
    if wb:
        out.append(wb)
    d = ve_agreement(rows)
    if d:
        out.append(f"NBN VE vs pgmpy VE: {len(d)} queries, max |diff| = {max(d):.2e}\n")
    errs = lw_errors(rows)
    if errs:
        for m in sorted({e["method"] for e in errs}):
            es = [e["err"] for e in errs if e["method"] == m]
            out.append(f"{m}: {len(es)} queries with an exact reference; "
                       f"median |err| = {np.median(es):.4f}, max = {max(es):.4f}\n")
            cov = [e["covered"] for e in errs if e["method"] == m and e["covered"] is not None]
            if cov:
                ws = [e["width"] for e in errs if e["method"] == m and e.get("width") is not None]
                out.append(f"{m}: certified interval covers exact answer in "
                           f"{sum(cov)}/{len(cov)} queries; median max-width "
                           f"{np.median(ws):.3f}\n")
    klp = RES / "prelim" / "discrete_infer_cert_klonly.jsonl"
    if klp.exists():
        kl_rows = [json.loads(ln) for ln in klp.read_text().splitlines() if ln.strip()]
        base = [r for r in load("discrete_infer") if r["method"] in ("nbn_ve", "pgmpy_ve")]
        for r in kl_rows:
            r["method"] = "nbn_cert_klonly"
        ek = [e for e in lw_errors(base + kl_rows) if e["method"] == "nbn_cert_klonly"]
        if ek:
            ws = [e["width"] for e in ek if e.get("width") is not None]
            cov = [e["covered"] for e in ek if e["covered"] is not None]
            out.append(f"nbn_cert with bound='kl' only (earlier run): coverage {sum(cov)}/{len(cov)}; "
                       f"median max-width {np.median(ws):.3f}\n")
    ce = cont_errors()
    for m in sorted({e["method"] for e in ce}):
        es = [e["err"] for e in ce if e["method"] == m]
        out.append(f"{m} (continuous, vs 4M-particle reference): {len(es)} queries; "
                   f"median |err| = {np.median(es):.4f}, 90% = {np.quantile(es, 0.9):.4f}\n")
    (RES / "tables.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
