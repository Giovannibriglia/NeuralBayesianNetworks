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

RES = HERE / "results"
FIG = HERE / "figures"

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
            rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def _base(m):
    return m.split("@")[0]


def _x(r, axis):
    return r["n"] if axis == "n" else r["n_edges"]


def plot_time(ax, rows, axis, topo=None, budget=30.0, methods=None, ylabel=True):
    groups = defaultdict(list)
    for r in rows:
        if r.get("axis") != axis or (topo and r.get("topology") != topo):
            continue
        if methods and r["method"] not in methods:
            continue
        groups[r["method"]].append(r)
    for m, rs in sorted(groups.items()):
        rs.sort(key=lambda r: _x(r, axis))
        ok = [r for r in rs if r["status"] == "ok" and r.get("n_answered", 1) == r.get("n_q", 1)]
        bad = [r for r in rs if r not in ok]
        c = COLOR.get(_base(m), GREY)
        ls = "--" if "@" in m else "-"
        lab = LABEL.get(_base(m), m) + (" [master]" if "@" in m else "")
        if ok:
            ax.plot([_x(r, axis) for r in ok], [r["time_s"] for r in ok], ls, color=c,
                    marker="o", ms=3.5, lw=1.6, label=lab)
        if bad:
            r = bad[0]
            y = budget if r["status"] == "timeout" else (ok[-1]["time_s"] if ok else budget)
            why = {"timeout": "time", "oom": "mem"}.get(r["status"], r["status"])
            if r["status"] == "ok":
                why = f"{r.get('n_answered')}/{r.get('n_q')}"
            ax.plot([_x(r, axis)], [y], "x", color=c, ms=7, mew=2,
                    label=None if ok else lab)
            ax.annotate(why, (_x(r, axis), y), textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=INK)
    ax.axhline(budget, color=GREY, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nodes" if axis == "n" else "edges")
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
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE * 1.45, 2.8))
    plot_time(axs[0], rows, "n", "local")
    plot_time(axs[1], rows, "n", "er", ylabel=False)
    plot_time(axs[2], rows, "deg", "er", ylabel=False)
    axs[0].set_title("10 queries vs nodes\nbounded treewidth (window 5)")
    axs[1].set_title("10 queries vs nodes\nErdős–Rényi (deg 2)")
    axs[2].set_title("10 queries vs edges\nErdős–Rényi (n=100)")
    h, l = axs[0].get_legend_handles_labels()
    h2, l2 = axs[1].get_legend_handles_labels()
    for hh, ll in zip(h2, l2):
        if ll not in l:
            h.append(hh)
            l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
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
        if _base(r["method"]) in ("nbn_lw", "nbn_cert") and r["status"] == "ok":
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
    ax.set_title("Fit quality (same budget)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
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
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "cont_infer.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "cont_infer.pdf", bbox_inches="tight")


def table(rows, axis="n", topo=None, field="time_s"):
    """Markdown table: rows = x values, columns = methods."""
    cells = defaultdict(dict)
    meths = set()
    for r in rows:
        if r.get("axis") != axis or (topo and r.get("topology") != topo):
            continue
        x = _x(r, axis) if axis == "deg" else r["n"]
        m = r["method"] + ("" if r.get("device", "cpu") == "cpu" else f"[{r['device']}]")
        meths.add(m)
        if r["status"] == "ok":
            v = r.get(field)
            s = f"{v:.3g}" if isinstance(v, (int, float)) else "?"
            if r.get("n_answered", r.get("n_q")) != r.get("n_q"):
                s += f" ({r['n_answered']}/{r['n_q']})"
        else:
            s = r["status"]
        cells[x][m] = s
    meths = sorted(meths)
    hdr = "| " + ("nodes" if axis == "n" else "edges") + " | " + " | ".join(meths) + " |"
    sep = "|" + "---|" * (len(meths) + 1)
    lines = [hdr, sep]
    for x in sorted(cells):
        lines.append(f"| {x} | " + " | ".join(cells[x].get(m, "") for m in meths) + " |")
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
    out = []
    for nm, axes in [("discrete_fit", [("n", None), ("deg", None)]),
                     ("discrete_sets", [("n", None), ("deg", None)]),
                     ("cont_fit", [("n", None), ("deg", None)]),
                     ("cont_infer", [("n", None), ("deg", None)])]:
        rows = load(nm, nm + "_master", nm + "_mps")
        for ax, tp in axes:
            out.append(f"### {nm}, x = {ax}\n\n" + table(rows, ax, tp) + "\n")
    rows = load("discrete_infer", "discrete_infer_master", "discrete_infer_cert")
    for ax, tp in [("n", "local"), ("n", "er"), ("deg", "er")]:
        out.append(f"### discrete_infer ({tp}), x = {ax}\n\n" + table(rows, ax, tp) + "\n")
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
                out.append(f"{m}: certified interval covers exact answer in "
                           f"{sum(cov)}/{len(cov)} queries\n")
    (HERE / "results" / "tables.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
