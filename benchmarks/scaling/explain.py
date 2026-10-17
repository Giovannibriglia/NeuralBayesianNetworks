"""Explanatory plots: what was changed and why it works (figures/explained/).

Reads the saved results (results/*.jsonl), the problem files and
``results/ordering_split.jsonl`` (from ``measure_ordering.py``).  One figure
per question; see figures/explained/README.md.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "examples"))
import analyze as A  # noqa: E402
import cell  # noqa: E402
from _figures import (  # noqa: E402
    AQUA, BLUE, DOUBLE, GREY, INK, MAGENTA, ORANGE, SKY, VERMILLION, use_style,
)

OUT = HERE / "figures" / "explained"
RES = HERE / "results"
PROB = HERE / "problems"
NEW, OLD, BASE = BLUE, SKY, ORANGE   # this branch / NBN before / pgmpy-Pyro


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def med_time(rows, method, axis, key, topo=None):
    cells = A.agg(rows, axis, topo, [method]).get(method, [])
    for c in cells:
        if c["key"] == key:
            return c["t"]
    return None


def frontier_median(rows, method, topo=None, axis="n"):
    fr = A.frontier(rows, topo, axis).get(method)
    return float(np.median(fr)) if fr else None


# --------------------------------------------------------------------- #
def p01_overview():
    di = A.load("discrete_infer", "discrete_infer_master", "discrete_infer_cert")
    df = A.load("discrete_fit")
    ds = A.load("discrete_sets")
    cf = A.load("cont_fit", "cont_fit_mps")
    ci = A.load("cont_infer", "cont_infer_master")
    rows = [  # (label, rows, topo, {role: method})
        ("Discrete fit", df, None, {"new": "nbn", "base": "pgmpy"}),
        ("Exact query, loose graph", di, "local",
         {"new": "nbn_ve", "old": "nbn_ve@master", "base": "pgmpy_ve"}),
        ("Exact query, dense graph", di, "er",
         {"new": "nbn_ve", "old": "nbn_ve@master", "base": "pgmpy_ve"}),
        ("Sampling query, dense graph", di, "er", {"new": "nbn_lw", "old": "nbn_lw@master"}),
        ("Certified query, dense graph", di, "er", {"new": "nbn_cert"}),
        ("S4 sets", ds, None, {"new": "nbn_intervals_batched", "old": "nbn_interval"}),
        ("Non-Gaussian fit, 1 CPU thread", cf, None,
         {"new": "nbn_stacked", "old": "nbn_local", "base": "pyro_svi"}),
        ("Non-Gaussian fit, GPU", cf, None,
         {"new": "nbn_stacked[mps]", "old": "nbn_local[mps]", "base": "pyro_svi[mps]"}),
        ("Non-Gaussian query", ci, None,
         {"new": "nbn_lw", "old": "nbn_lw@master", "base": "pyro_vec"}),
    ]
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.6))
    style = {"new": (NEW, "o", "NBN, this branch"),
             "old": (OLD, "s", "NBN before (master / sequential)"),
             "base": (BASE, "D", "pgmpy (discrete) / Pyro (continuous)")}
    seen = set()
    for i, (lab, rs, topo, roles) in enumerate(rows):
        y = len(rows) - 1 - i
        for j, (role, m) in enumerate(roles.items()):
            v = frontier_median(rs, m, topo)
            c, mk, name = style[role]
            off = {"new": 0.0, "old": 0.27, "base": -0.27}[role]
            if not v:
                ax.plot([7], [y + off], mk, color=c, mfc="none", ms=6)
                continue
            ax.plot([v], [y + off], mk, color=c, ms=6, zorder=3 if role == "new" else 2,
                    label=None if role in seen else name)
            seen.add(role)
        ax.text(6.2, y, lab, ha="right", va="center", fontsize=7)
    ax.axvline(5000, color=GREY, lw=0.8, ls="--")
    ax.set_ylim(-0.6, len(rows) - 0.05)
    ax.text(4600, len(rows) - 0.35, "largest size tested", fontsize=6, ha="right", color=GREY)
    ax.set_xscale("log")
    ax.set_xlim(6, 9000)
    ax.set_yticks([])
    ax.set_xlabel("largest graph (nodes) handled within 1 thread / 4 GB / 30 s\n"
                  "(median over seeds; hollow = failed at the smallest size)")
    ax.set_title("01  How far does each method get within the budget?")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.42), ncol=3, frameon=False, fontsize=7)
    save(fig, "01_overview_frontier")


# --------------------------------------------------------------------- #
def p02_speedups():
    di = A.load("discrete_infer", "discrete_infer_master")
    ci = A.load("cont_infer", "cont_infer_master")
    cf = A.load("cont_fit")
    ds = A.load("discrete_sets")
    ind = {(r["deg"], r["method"]): r for r in A.load("discrete_fit_indeg")}
    items = []

    def ratio(a, b):
        return None if (a is None or b is None) else a / b

    items.append(("Exact queries: incremental min-fill\n(loose graph, 1 000 nodes)", "time",
                  ratio(med_time(di, "nbn_ve@master", "n", 1000, "local"),
                        med_time(di, "nbn_ve", "n", 1000, "local"))))
    items.append(("Sampling queries: barren pruning + fast sampling\n(dense graph, 200 nodes)", "time",
                  ratio(med_time(di, "nbn_lw@master", "n", 200, "er"),
                        med_time(di, "nbn_lw", "n", 200, "er"))))
    items.append(("Non-Gaussian queries: barren pruning\n(2 000 nodes)", "time",
                  ratio(med_time(ci, "nbn_lw@master", "n", 2000), med_time(ci, "nbn_lw", "n", 2000))))
    items.append(("Non-Gaussian fit: stacked training\n(200 nodes, 1 thread)", "time",
                  ratio(med_time(cf, "nbn_local", "n", 200), med_time(cf, "nbn_stacked", "n", 200))))
    items.append(("S4 sets: one vectorised pass\n(5 000 nodes)", "time",
                  ratio(med_time(ds, "nbn_interval", "n", 5000),
                        med_time(ds, "nbn_intervals_batched", "n", 5000))))
    d, s = ind.get((14, "nbn")), ind.get((14, "nbn_sparse"))
    items.append(("Discrete fit: sparse table\n(memory, 14 parents per node)", "memory",
                  ratio(d["watch_peak_rss_mb"], s["watch_peak_rss_mb"]) if d and s else None))
    klw = _cert_widths(RES / "prelim" / "discrete_infer_cert_klonly.jsonl")
    nw = _cert_widths(RES / "discrete_infer_cert.jsonl")
    items.append(("Certified queries: + empirical Bernstein\n(interval width)", "width",
                  ratio(np.median(klw), np.median(nw)) if klw and nw else None))
    items = [it for it in items if it[2]]
    col = {"time": BLUE, "memory": AQUA, "width": MAGENTA}
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.85, 3.4))
    for i, (lab, kind, v) in enumerate(items[::-1]):
        ax.barh(i, v, color=col[kind], height=0.6)
        ax.text(v * 1.08, i, f"{v:.1f}×", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([it[0] for it in items[::-1]], fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlim(1, max(it[2] for it in items) * 3)
    ax.set_xlabel("improvement factor (before / after; same answers)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label={"time": "faster", "memory": "less memory",
                                             "width": "narrower guaranteed interval"}[k])
                       for k, c in col.items()], loc="lower right", fontsize=7, frameon=False)
    ax.set_title("02  What each change bought")
    save(fig, "02_what_each_change_bought")


def _cert_widths(path):
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            if r["status"] == "ok":
                out += r.get("max_width") or []
    return out


# --------------------------------------------------------------------- #
def p03_ordering_split():
    p = RES / "ordering_split.jsonl"
    if not p.exists():
        return
    recs = [x for ln in p.read_text().splitlines() if ln.strip() for x in json.loads(ln)]
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    ns = sorted({r["n"] for r in recs})
    w = 0.38
    for k, code in enumerate(("master", "branch")):
        rs = {r["n"]: r for r in recs if r["code"] == code}
        xs = np.arange(len(ns)) + (k - 0.5) * w
        c = OLD if code == "master" else NEW
        lab = "NBN before" if code == "master" else "NBN, this branch"
        axs[0].bar(xs, [rs[n]["total_s"] for n in ns], w, color=c, label=lab)
        axs[1].bar(xs, [100 * rs[n]["ordering_s"] / rs[n]["total_s"] for n in ns], w, color=c)
        for x, n in zip(xs, ns):
            axs[0].text(x, rs[n]["total_s"] * 1.15, f"{rs[n]['total_s']:.2f} s", ha="center",
                        fontsize=6.5)
            axs[1].text(x, 100 * rs[n]["ordering_s"] / rs[n]["total_s"] + 2,
                        f"{100 * rs[n]['ordering_s'] / rs[n]['total_s']:.0f}%", ha="center",
                        fontsize=6.5)
    for ax in axs:
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f"{n} nodes" for n in ns])
    axs[0].set_yscale("log")
    axs[0].set_ylabel("10 exact queries (s)")
    axs[0].set_title("Total time")
    axs[1].set_ylabel("share spent choosing the\nelimination order (%)")
    axs[1].set_ylim(0, 110)
    axs[1].set_title("Where the time went")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("03  Exact queries were slow because of bookkeeping, not arithmetic "
                 "(loose graphs, 1 thread)", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "03_where_exact_query_time_went")


# --------------------------------------------------------------------- #
def p04_treewidth_wall():
    di = [r for r in A.load("discrete_infer") if r["method"] == "nbn_ve" and r["axis"] == "n"
          and r["status"] == "ok" and r.get("width_bounds")]
    need = math.log(0.75 * 4 * 2**30 / 4, 3) - 1
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.9), sharey=True)
    rng = np.random.default_rng(0)
    for ax, topo, title in ((axs[0], "local", "Loose graph (each node's parents among the 5 before it)"),
                            (axs[1], "er", "Dense random graph (Erdős–Rényi)")):
        for ok, c, lab in ((True, NEW, "answered exactly"), (False, VERMILLION, "refused (too big)")):
            xs, ys = [], []
            for r in di:
                if r["topology"] != topo:
                    continue
                for a, (lb, ub) in zip(r["answers"], r["width_bounds"]):
                    if (a is not None) == ok:
                        xs.append(r["n"] * math.exp(rng.normal(0, 0.08)))
                        ys.append(ub + rng.normal(0, 0.12))
            if xs:
                ax.scatter(xs, ys, s=8, color=c, alpha=0.6, label=lab)
        ax.axhline(need, color=INK, lw=0.8, ls="--")
        ax.text(11, need + 0.6, "largest width whose biggest table fits in 3 GB (K = 3)",
                fontsize=6)
        ax.set_xscale("log")
        ax.set_xlabel("nodes")
        ax.set_title(title, fontsize=7.5)
    axs[0].set_ylabel("induced width of the query\n(table of K^(width+1) numbers)")
    axs[1].legend(fontsize=7, loc="upper left", frameon=False)
    fig.suptitle("04  The tree-width wall: exact cost is exponential in the width; "
                 "on dense graphs the width grows with size", fontsize=8)
    fig.tight_layout()
    save(fig, "04_treewidth_wall")


# --------------------------------------------------------------------- #
def p05_certified():
    prob, seed = "discrete_er_n100_deg2_k3_s1", 1
    exact = lw = cert = klo = None
    for r in A.load("discrete_infer"):
        if r["problem"] == prob and r["status"] == "ok":
            if r["method"] == "nbn_ve":
                exact = r["answers"]
            if r["method"] == "nbn_lw":
                lw = r["answers"]
    for r in A.load("discrete_infer_cert"):
        if r["problem"] == prob and r["status"] == "ok":
            cert = r
    kp = RES / "prelim" / "discrete_infer_cert_klonly.jsonl"
    if kp.exists():
        for ln in kp.read_text().splitlines():
            r = json.loads(ln)
            if r["problem"] == prob and r["status"] == "ok":
                klo = r
    if not (exact and cert):
        return
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.75, 3.3))
    nq = len(exact)
    for q in range(nq):
        y = nq - 1 - q
        c = 0
        if klo:
            ax.plot([klo["lower"][q][c], klo["upper"][q][c]], [y + 0.18] * 2, color=GREY, lw=4,
                    solid_capstyle="butt", label="guaranteed interval, KL bound" if q == 0 else None)
        ax.plot([cert["lower"][q][c], cert["upper"][q][c]], [y] * 2, color=MAGENTA, lw=4,
                solid_capstyle="butt",
                label="guaranteed interval, KL ∩ empirical Bernstein" if q == 0 else None)
        if lw:
            ax.plot(lw[q][c], y - 0.2, "^", color=AQUA, ms=5,
                    label="plain LW estimate (no guarantee)" if q == 0 else None)
        ax.plot(exact[q][c], y, "o", color=INK, ms=4, label="exact answer (VE)" if q == 0 else None)
    ax.set_yticks(range(nq))
    ax.set_yticklabels([f"query {nq - i}" for i in range(nq)], fontsize=6.5)
    ax.set_xlabel("P(target = first class | 5 observed nodes)")
    ax.set_title("05  Certified answers: an interval that holds with probability ≥ 95%\n"
                 "(100-node dense graph; 200 000 samples per query)", fontsize=8)
    ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False)
    save(fig, "05_certified_answers")


# --------------------------------------------------------------------- #
def p06_stacked():
    cf = A.load("cont_fit", "cont_fit_mps")
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.7, 2.8))
    for m, dev, c, ls, lab in (("nbn_local", "cpu", OLD, "-", "NBN before: one node at a time (CPU)"),
                               ("pyro_svi", "cpu", BASE, "-", "Pyro SVI (CPU)"),
                               ("nbn_stacked", "cpu", NEW, "-", "NBN stacked: all nodes at once (CPU)"),
                               ("nbn_local", "mps", OLD, "--", "NBN one at a time (GPU)"),
                               ("nbn_stacked", "mps", NEW, "--", "NBN stacked (GPU)")):
        rs = sorted([r for r in cf if r["method"] == m and r.get("device", "cpu") == dev
                     and r["axis"] == "n" and r["status"] == "ok"], key=lambda r: r["n"])
        if rs:
            ax.plot([r["n"] for r in rs], [1000 * r["time_s"] / r["n"] for r in rs], ls,
                    marker="o", ms=3, color=c, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nodes")
    ax.set_ylabel("fit time per node (ms)")
    ax.set_title("06  Stacked fitting: same per-node objective, one batched network\n"
                 "(20 epochs, 10 000 rows; lower = cheaper per node)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    save(fig, "06_stacked_fit_cost_per_node")


# --------------------------------------------------------------------- #
def _indeg_stats():
    stats = []
    for p in sorted(PROB.glob("discadd_fixed_n30_deg*_k3_s0.pt")):
        pr = torch.load(p, weights_only=False)
        d = pr["deg"]
        rows = []
        for j, v in enumerate(pr["nodes"]):
            pa = [pr["nodes"][i] for i in pr["add_parents"][j]]
            if len(pa) != d:
                continue
            key = torch.zeros(pr["train"][v].shape[0], dtype=torch.long)
            for a in pa:
                key = key * 3 + pr["train"][a]
            rows.append(int(torch.unique(key).numel()))
        n_train = pr["train"][pr["nodes"][0]].shape[0]
        stats.append(dict(d=d, observed=float(np.mean(rows)), n=n_train))
    return sorted(stats, key=lambda s: s["d"])


def p07_sparse_memory(stats):
    if not stats:
        return
    ds = [s["d"] for s in stats]
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.6, 2.8))
    ax.plot(ds, [3 ** (d + 1) for d in ds], "-o", ms=3, color=OLD,
            label="dense table: K^(parents+1) numbers")
    ax.plot(ds, [3 * (s["observed"] + 1) for s in stats], "-o", ms=3, color=NEW,
            label="sparse table: rows actually seen (+1 uniform row)")
    ax.axhline(3 * stats[0]["n"], color=GREY, lw=0.8, ls=":")
    ax.text(ds[0], 3 * stats[0]["n"] * 1.5, "cap: K × 10 000 training rows", fontsize=6, color=GREY)
    ax.set_yscale("log")
    ax.set_xlabel("parents per node")
    ax.set_ylabel("numbers stored per node")
    ax.set_title("07  Sparse table: memory follows the data, not the parent space\n"
                 "(same answers; K = 3 values per variable)", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    save(fig, "07_sparse_table_memory")


def p08_fragmentation(stats):
    ind = [r for r in A.load("discrete_fit_indeg") if r["method"] == "nbn_sparse" and r["status"] == "ok"]
    if not stats or not ind:
        return
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    axs[0].plot([s["d"] for s in stats], [s["n"] / s["observed"] for s in stats], "-o", ms=3,
                color=NEW)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("parents per node")
    axs[0].set_ylabel("training rows per\nobserved parent combination")
    axs[0].set_title("Data per table row shrinks…", fontsize=8)
    ind.sort(key=lambda r: r["deg"])
    axs[1].plot([r["deg"] for r in ind], [r["acc_tv"] for r in ind], "-o", ms=3, color=VERMILLION)
    axs[1].set_ylim(bottom=0)
    axs[1].set_xlabel("parents per node")
    axs[1].set_ylabel("error of the fitted table\n(total variation to the truth)")
    axs[1].set_title("…so the error grows, however fast we compute", fontsize=8)
    fig.suptitle("08  Many parents: a statistical limit, not a computational one "
                 "(10 000 rows, K = 3)", fontsize=8)
    fig.tight_layout()
    save(fig, "08_data_fragmentation")


# --------------------------------------------------------------------- #
def p09_pruning():
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, 2.8))
    for fam, topo, c, mk, lab, qfun in (
            ("discrete", "local", OLD, "s", "discrete, loose graph", "d"),
            ("discrete", "er", NEW, "o", "discrete, dense graph", "d"),
            ("ngs", "er", AQUA, "^", "non-Gaussian, dense graph", "c")):
        xs, ys = [], []
        for n in A.NODE_SIZES if hasattr(A, "NODE_SIZES") else (10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
            name = (f"{fam}_{topo}_w5_n{n}_deg2_k3_s0" if topo == "local"
                    else f"{fam}_{topo}_n{n}_deg2_k3_s0")
            p = PROB / f"{name}.pt"
            if not p.exists():
                continue
            pr = torch.load(p, weights_only=False)
            g = nx.DiGraph(pr["edges"])
            g.add_nodes_from(pr["nodes"])
            qs = cell.discrete_queries(pr) if qfun == "d" else [(t, e) for t, e, _ in
                                                                 cell.continuous_queries(pr)]
            fr = []
            for t, e in qs:
                keep = {t, *e}
                for v in list(keep):
                    keep |= nx.ancestors(g, v)
                fr.append(len(keep) / pr["n"])
            xs.append(n)
            ys.append(100 * float(np.median(fr)))
        if xs:
            ax.plot(xs, ys, "-" + mk, ms=3.5, color=c, label=lab)
    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.set_xlabel("nodes")
    ax.set_ylabel("nodes a query must sample (%)\n(median over 10 queries)")
    ax.set_title("09  Barren-node pruning: sampling only what can affect the answer\n"
                 "(before: 100% of the nodes for every query)", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False)
    save(fig, "09_barren_pruning")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    use_style()
    plt.rcParams.update({"axes.titlesize": 8, "legend.fontsize": 7})
    p01_overview()
    p02_speedups()
    p03_ordering_split()
    p04_treewidth_wall()
    p05_certified()
    p06_stacked()
    stats = _indeg_stats()
    p07_sparse_memory(stats)
    p08_fragmentation(stats)
    p09_pruning()
    print(sorted(x.name for x in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
