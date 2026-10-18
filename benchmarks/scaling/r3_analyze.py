"""Round 3: does NBN turn a larger budget into more scale?

Reads results/r3_threads.jsonl (CPU threads 1-8, this branch and master),
results/r3_mps.jsonl (GPU), results/r3_big.jsonl (8 threads, 16 GB, 120 s) and
the reference-budget sweeps of rounds 1-2 (1 thread, 4 GB, 30 s).  Writes
figures/round3/*.{png,pdf}, figures/round3/README.md, report/tables_r3/*.tex.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "report"))
sys.path.insert(0, str(HERE.parent.parent / "examples"))
from _figures import AQUA, BLUE, DOUBLE, GREY, INK, MAGENTA, ORANGE, SKY, VERMILLION, use_style  # noqa: E402
from quant_tables import Table  # noqa: E402

RES = HERE / "results"
OUT = HERE / "figures" / "round3"
TAB = HERE / "report" / "tables_r3"
THREADS = (1, 2, 4, 8)

# (sweep, method) -> (label, group)
CELLS = {
    ("discrete_infer", "nbn_ve"): ("exact VE (discrete)", "exact"),
    ("discrete_infer", "pgmpy_ve"): ("pgmpy VE", "exact"),
    ("discrete_sets", "nbn_intervals_batched"): ("S4 interval sets", "exact"),
    ("discrete_fit", "nbn"): ("MLE fit (discrete)", "exact"),
    ("discrete_fit", "pgmpy"): ("pgmpy MLE fit", "exact"),
    ("discrete_infer", "nbn_lw"): ("LW queries (discrete)", "sampling"),
    ("discrete_infer_cert", "nbn_cert"): ("certified queries", "sampling"),
    ("cont_infer", "nbn_lw"): ("LW queries (non-Gaussian)", "sampling"),
    ("cont_infer", "pyro_vec"): ("Pyro vectorised LW", "sampling"),
    ("r2_ais", "ais"): ("amortized IS", "sampling"),
    ("cont_fit", "nbn_stacked"): ("MDN fit (stacked)", "neural"),
    ("cont_fit", "nbn_local"): ("MDN fit (sequential)", "neural"),
    ("cont_fit", "pyro_svi"): ("Pyro SVI (MDN)", "neural"),
    ("r2_fit_ncat", "ncat_stacked"): ("neural categorical fit", "neural"),
    ("r2_fit_ncat", "ncat_local"): ("neural categorical fit", "neural"),
    ("r2_fit_ncat", "pyro_ncat"): ("Pyro SVI (categorical)", "neural"),
    ("r2_fit_flex", "flex_stacked"): ("FlexCode fit", "neural"),
    ("r2_fit_flex", "flex_local"): ("FlexCode fit", "neural"),
    ("r2_fit_flow", "flow_stacked"): ("flow fit", "neural"),
    ("r2_fit_flow", "flow_local"): ("flow fit", "neural"),
    ("r2_avi", "avi"): ("amortized VI training", "neural"),
    ("r2_knn_query", "knn_lw"): ("kNN LW queries", "nonpar"),
    ("r2_flex_query", "flex_lw"): ("FlexCode LW queries", "nonpar"),
}
PYRO_PGMPY = ("pgmpy", "pyro")


def load(*names):
    out = []
    for name in names:
        p = RES / f"{name}.jsonl"
        if p.exists():
            out += [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    return out


def base(m):
    return m.split("@")[0]


def is_master(r):
    return "@master" in r["method"]


def label(r):
    lab = CELLS.get((r["sweep"], base(r["method"])), (base(r["method"]), "?"))[0]
    return lab


def size_of(r):
    return r["n"] if r.get("axis") == "n" else r["deg"]


def complete(r):
    """ok and, for exact VE, every query answered."""
    if r.get("status") != "ok":
        return False
    if base(r["method"]) == "nbn_ve" and r.get("n_answered") is not None:
        return r["n_answered"] == 10
    return True


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def plain(axis, ticks, fmt="{:g}"):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))
    axis.set_minor_formatter(NullFormatter())


def fmt_t(x):
    if isinstance(x, (int, float)):
        return f"{x:.3g}"
    return x or "—"


# ------------------------------------------------------------------ threads
def thread_table(rows):
    """{(sweep, method, master?, size): {threads: (time, rec)}}"""
    d = {}
    for r in rows:
        key = (r["sweep"], base(r["method"]), is_master(r), size_of(r))
        d.setdefault(key, {})[r["threads"]] = r
    return d


def fig_threads(tables):
    rows = [r for r in load("r3_threads") if r.get("device", "cpu") == "cpu"]
    if not rows:
        return
    d = thread_table(rows)
    T = Table("r3threads", "R3-1. CPU threads: time (s) at 1, 2, 4, 8 threads and speed-up "
              "1→8 (budget 120 s, 16 GB; same size per row)",
              ["computation", "version", "size", "1", "2", "4", "8", "speed-up 1→4", "speed-up 1→8"])
    keys = sorted(d, key=lambda k: (CELLS.get((k[0], k[1]), ("", "z"))[1], k[0], k[1], k[2]))
    dots = []
    for k in keys:
        sw, m, mst, size = k
        ts = {t: (d[k][t]["time_s"] if complete(d[k][t]) else d[k][t]["status"])
              for t in THREADS if t in d[k]}
        if base(m) == "nbn_ve":  # compare equal work: same answered count
            ts = {t: (d[k][t]["time_s"] if d[k][t].get("status") == "ok" else d[k][t]["status"])
                  for t in THREADS if t in d[k]}
        lab = CELLS.get((sw, m), (m, "?"))[0]
        ver = "master" if mst else ("baseline" if m.startswith(PYRO_PGMPY) else "this branch")
        t1 = ts.get(1)

        def sp(t):
            v = ts.get(t)
            return t1 / v if isinstance(t1, float) and isinstance(v, float) else None
        T.add(lab, ver, f"{size:g}", *[fmt_t(ts.get(t)) for t in THREADS],
              f"{sp(4):.2f}×" if sp(4) else "—", f"{sp(8):.2f}×" if sp(8) else "—")
        if sp(8):
            best = max((t for t in THREADS if isinstance(ts.get(t), float)),
                       key=lambda t: t1 / ts[t])
            dots.append((lab, ver, sp(4), sp(8), t1 / ts[best], best,
                         CELLS.get((sw, m), (m, "?"))[1]))
    tables.append(T)
    # dot plot: speed-up 1->4 and 1->8 per computation
    order = {"exact": 0, "sampling": 1, "neural": 2, "nonpar": 3}
    dots.sort(key=lambda x: (order.get(x[6], 9), x[1] != "this branch", x[0]))
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.8, 0.24 * len(dots) + 1.0))
    ys = np.arange(len(dots))[::-1]
    col = {"this branch": BLUE, "master": SKY, "baseline": ORANGE}
    for y, (lab, ver, s4, s8, sb, tb, grp) in zip(ys, dots):
        ax.plot([1, s8], [y, y], color=GREY, lw=0.6, zorder=1)
        ax.scatter([s4], [y], marker="o", s=18, facecolor="white", edgecolor=col[ver], zorder=3)
        ax.scatter([s8], [y], marker="o", s=18, color=col[ver], zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{lab} ({ver})" for lab, ver, *_ in dots], fontsize=6.5)
    ax.axvline(1, color=INK, lw=0.6)
    ax.set_xscale("log", base=2)
    plain(ax.xaxis, [0.5, 1, 2, 4, 8], "{:g}×")
    ax.set_xlim(0.4, 9)
    ax.set_xlabel("speed-up over 1 thread  (○ 4 threads, ● 8 threads)")
    ax.set_title("More CPU threads: who uses them? (Apple M5: 4 performance + 6 efficiency cores)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=v) for v, c in col.items()],
              loc="lower right", frameon=False, fontsize=6.5)
    save(fig, "r3_1_threads_speedup")

    # absolute time vs threads, branch vs master, for the cells that have both
    pairs = []
    for k in keys:
        sw, m, mst, size = k
        if mst:
            continue
        mk = [kk for kk in keys if kk[0] == sw and kk[2] and
              (kk[1] == m or kk[1] == m.replace("stacked", "local"))]
        if mk:
            pairs.append((k, mk[0]))
    if not pairs:
        return
    ncol = 4
    nrow = (len(pairs) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(DOUBLE * 1.15, 1.9 * nrow + 0.4), squeeze=False)
    for ax, (kb, km) in zip(axs.flat, pairs):
        for kk, c, ls, lab in ((kb, BLUE, "-", "this branch"), (km, SKY, ":", "master")):
            pts = [(t, d[kk][t]["time_s"]) for t in THREADS
                   if t in d[kk] and d[kk][t].get("status") == "ok"]
            if pts:
                ax.plot(*zip(*pts), ls, marker="o", ms=3, color=c,
                        label=f"{lab} ({kk[3]:g})")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        plain(ax.xaxis, list(THREADS))
        ax.set_title(CELLS.get((kb[0], kb[1]), (kb[1],))[0], fontsize=7)
        ax.set_xlabel("threads", fontsize=6.5)
        ax.set_ylabel("time (s)", fontsize=6.5)
        ax.legend(fontsize=5.5, frameon=False, title="version (size)", title_fontsize=5.5)
    for ax in list(axs.flat)[len(pairs):]:
        ax.axis("off")
    fig.tight_layout()
    save(fig, "r3_2_threads_branch_vs_master")


# ---------------------------------------------------------------------- GPU
def fig_gpu(tables):
    g = load("r3_mps")
    c = [r for r in load("r3_threads") if not is_master(r)]
    if not g:
        return
    T = Table("r3gpu", "R3-2. GPU (Apple MPS) vs CPU: time (s); speed-up = CPU time / GPU time. * exact VE: the guard answers a different number of queries per device, so times are not comparable",
              ["computation", "size", "CPU 1 thread", "CPU best (threads)", "GPU", "GPU vs 1 thread",
               "GPU vs best CPU"])
    bars = []
    for r in sorted(g, key=lambda r: (CELLS.get((r["sweep"], base(r["method"])), ("", "z"))[1],
                                      label(r))):
        sw, m, size = r["sweep"], base(r["method"]), size_of(r)
        cpu = {x["threads"]: x for x in c if x["sweep"] == sw and base(x["method"]) == m
               and size_of(x) == size and x.get("device", "cpu") == "cpu"}
        okf = (lambda x: x.get("status") == "ok") if m == "nbn_ve" else complete
        t1 = cpu[1]["time_s"] if 1 in cpu and okf(cpu[1]) else None
        okc = {t: x["time_s"] for t, x in cpu.items() if okf(x)}
        tb = min(okc, key=okc.get) if okc else None
        tg = r["time_s"] if okf(r) else None
        if m == "nbn_ve":  # answered queries differ: the guard's budget differs by device
            T.add(label(r) + " *", f"{size:g}", f"{fmt_t(t1)} ({cpu[1]['n_answered']}/10)",
                  f"{okc[tb]:.3g} ({tb})", f"{fmt_t(tg)} ({r['n_answered']}/10)", "—", "—")
            continue
        T.add(label(r), f"{size:g}", fmt_t(t1),
              f"{okc[tb]:.3g} ({tb})" if tb else "—",
              fmt_t(tg if tg else r["status"]),
              f"{t1 / tg:.2f}×" if t1 and tg else "—",
              f"{okc[tb] / tg:.2f}×" if tb and tg else "—")
        if t1 and tg:
            bars.append((label(r), t1 / tg, okc[tb] / tg))
    tables.append(T)
    if not bars:
        return
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.8, 0.26 * len(bars) + 1.0))
    ys = np.arange(len(bars))[::-1]
    ax.barh(ys + 0.18, [b[1] for b in bars], height=0.34, color=VERMILLION, label="vs CPU, 1 thread")
    ax.barh(ys - 0.18, [b[2] for b in bars], height=0.34, color=SKY, label="vs best CPU (1–8 threads)")
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars], fontsize=6.5)
    ax.axvline(1, color=INK, lw=0.6)
    ax.set_xscale("log")
    plain(ax.xaxis, [0.1, 0.25, 0.5, 1, 2, 4, 8, 16], "{:g}×")
    ax.set_xlabel("GPU speed-up (>1: GPU faster)")
    ax.set_title("GPU (Apple MPS, unified memory) vs CPU, same size as the thread study")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    save(fig, "r3_3_gpu")


# ----------------------------------------------------------------- frontier
FRONTIER = [  # (label, sweep, method, axis, topology, reference files)
    ("exact VE, loose ER graphs", "discrete_infer", "nbn_ve", "n", "er", ("discrete_infer",)),
    ("pgmpy VE, loose ER graphs", "discrete_infer", "pgmpy_ve", "n", "er", ("discrete_infer",)),
    ("LW queries, ER", "discrete_infer", "nbn_lw", "n", "er", ("discrete_infer",)),
    ("LW queries, local graphs", "discrete_infer", "nbn_lw", "n", "local", ("discrete_infer",)),
    ("certified queries, local", "discrete_infer_cert", "nbn_cert", "n", "local",
     ("discrete_infer_cert",)),
    ("MDN fit (stacked)", "cont_fit", "nbn_stacked", "n", "er", ("cont_fit",)),
    ("Pyro SVI (MDN)", "cont_fit", "pyro_svi", "n", "er", ("cont_fit",)),
    ("neural categorical fit", "r2_fit_ncat", "ncat_stacked", "n", "er", ("r2_fit_ncat",)),
    ("Pyro SVI (categorical)", "r2_fit_ncat", "pyro_ncat", "n", "er", ("r2_fit_ncat",)),
    ("FlexCode fit", "r2_fit_flex", "flex_stacked", "n", "er", ("r2_fit_flex",)),
    ("flow fit", "r2_fit_flow", "flow_stacked", "n", "er", ("r2_fit_flow",)),
    ("kNN LW queries", "r2_knn_query", "knn_lw", "n", "er", ("r2_knn_query",)),
    ("FlexCode LW queries", "r2_flex_query", "flex_lw", "n", "er", ("r2_flex_query",)),
    ("amortized IS", "r2_ais", "ais", "n", "local", ("r2_ais",)),
    ("amortized VI (10 epochs)", "r2_avi", "avi", "n", "er", ("r2_avi",)),
]


def frontier(rows, sweep, method, axis, topo):
    rs = [r for r in rows if r.get("sweep") == sweep and r["method"] == method
          and r.get("axis") == axis and r.get("topology") == topo and r.get("seed", 0) == 0]
    ok = [size_of(r) for r in rs if complete(r)]
    tried = [size_of(r) for r in rs]
    fail = sorted((size_of(r), r["status"]) for r in rs if not complete(r))
    return (max(ok) if ok else None), (max(tried) if tried else None), (fail[0] if fail else None)


def fig_frontier(tables):
    big = load("r3_big")
    if not big:
        return
    T = Table("r3frontier", "R3-3. Largest graph (nodes) finished: reference budget "
              "(1 thread, 4 GB, 30 s) vs large budget (8 threads, 16 GB, 120 s)",
              ["computation", "reference: largest", "stopped by", "large: largest", "stopped by",
               "gain in nodes"])
    items = []
    for lab, sw, m, ax_, topo, ref in FRONTIER:
        r_ok, _, r_fail = frontier(load(*ref), sw, m, ax_, topo)
        b_ok, b_tried, b_fail = frontier(big, sw, m, ax_, topo)
        if b_tried is None:
            continue
        gain = (b_ok / r_ok) if (b_ok and r_ok) else None
        T.add(lab, fmt_t(r_ok), f"{r_fail[1]} at {r_fail[0]:g}" if r_fail else "none (max tested)",
              fmt_t(b_ok), f"{b_fail[1]} at {b_fail[0]:g}" if b_fail else "none (max tested)",
              f"{gain:.0f}×" if gain else "—")
        items.append((lab, r_ok, b_ok, r_fail, b_fail))
    tables.append(T)
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.8, 0.26 * len(items) + 1.0))
    ys = np.arange(len(items))[::-1]
    for y, (lab, r_ok, b_ok, r_fail, b_fail) in zip(ys, items):
        if r_ok and b_ok:
            ax.annotate("", xy=(b_ok, y), xytext=(r_ok, y),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.8, shrinkA=3, shrinkB=3))
        if r_ok:
            ax.scatter([r_ok], [y], s=22, facecolor="white", edgecolor=BLUE, zorder=3)
        if b_ok:
            ax.scatter([b_ok], [y], s=22, color=BLUE, zorder=3)
            if not b_fail:
                ax.text(b_ok * 1.25, y, "no limit hit", fontsize=5.5, va="center", color=GREY)
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items], fontsize=6.5)
    ax.set_xscale("log")
    plain(ax.xaxis, [10, 100, 1000, 10000], "{:,.0f}")
    ax.set_xlim(7, 40000)
    ax.set_xlabel("largest graph finished (nodes)  ○ 1 thread, 4 GB, 30 s   ● 8 threads, 16 GB, 120 s")
    ax.set_title("A larger budget moves the frontier")
    save(fig, "r3_4_frontier")


def fig_ve_width(tables):
    """Exact VE on denser graphs (100 nodes): queries answered vs degree, both budgets."""
    big = [r for r in load("r3_big") if r.get("axis") == "deg"]
    ref = [r for r in load("discrete_infer", "r3_small_deg") if r.get("axis") == "deg"
           and r.get("seed", 0) == 0 and "@" not in r["method"]]
    if not big:
        return
    T = Table("r3vewidth", "R3-4. Exact inference on denser graphs (100 nodes, 10 queries): "
              "queries answered and the largest elimination width answered",
              ["avg degree", "budget", "NBN answered", "NBN max width answered", "NBN time (s)",
               "pgmpy"])
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    for rows, name, mk, fill in ((ref, "1 thread, 4 GB, 30 s", "o", "white"),
                                 (big, "8 threads, 16 GB, 120 s", "o", BLUE)):
        pts, wpts = [], []
        for deg in sorted({r["deg"] for r in rows}):
            nb = next((r for r in rows if r["method"] == "nbn_ve" and r["deg"] == deg), None)
            pg = next((r for r in rows if r["method"] == "pgmpy_ve" and r["deg"] == deg), None)
            if nb is None:
                continue
            ans = nb.get("n_answered") if nb["status"] == "ok" else 0
            widths = []
            if nb["status"] == "ok" and nb.get("width_bounds"):
                # width of NBN's elimination order, for the queries it answered
                answered = [i for i, a in enumerate(nb.get("answers", [])) if a is not None]
                widths = [nb["width_bounds"][i][1] for i in answered]
            pts.append((deg, ans or 0))
            if widths:
                wpts.append((deg, max(widths)))
            T.add(f"{deg:g}", name, f"{ans or 0}/10", str(max(widths)) if widths else "—",
                  fmt_t(nb.get("time_s") if nb["status"] == "ok" else nb["status"]),
                  "—" if pg is None else (f"10/10 ({pg['time_s']:.3g} s)" if pg["status"] == "ok"
                                          else pg["status"]))
        if pts:
            axs[0].plot(*zip(*pts), "-", marker=mk, ms=4, color=BLUE, mfc=fill, label=name)
        if wpts:
            axs[1].plot(*zip(*wpts), "-", marker=mk, ms=4, color=BLUE, mfc=fill, label=name)
    axs[0].set_xlabel("average degree (100 nodes)")
    axs[0].set_ylabel("queries answered exactly (of 10)")
    axs[0].set_ylim(-0.5, 10.5)
    axs[0].set_title("Exact VE: answered queries")
    axs[0].legend(frameon=False, fontsize=6.5)
    axs[1].set_xlabel("average degree (100 nodes)")
    axs[1].set_ylabel("largest width answered")
    axs[1].set_title("Width reached: memory buys about $+\\log_K c$")
    fig.tight_layout()
    save(fig, "r3_5_ve_width")
    tables.append(T)


def fig_particles(tables):
    """Sampling with many particles: do threads pay off when the batch is large?"""
    rows = load("r3_particles")
    if not rows:
        return
    ve = next((r for r in rows if r["method"] == "nbn_ve" and r["status"] == "ok"), None)
    exact = ve["answers"] if ve else None

    def err(r):
        if r["sweep"] != "discrete_infer" or not exact or r["status"] != "ok":
            return None
        e = [np.abs(np.array(a) - np.array(b)).max() for a, b in zip(r["answers"], exact)
             if a is not None and b is not None]
        return float(np.mean(e)) if e else None

    T = Table("r3particles", "R3-5. Likelihood weighting with more particles: time (s) for 10 "
              "queries at 1 and 8 threads, and mean max-abs error vs exact VE (discrete only)",
              ["network", "particles", "1 thread", "8 threads", "speed-up", "error vs exact",
               "ESS/N"])
    series = {}
    for sw, lab in (("discrete_infer", "discrete, local, 1,000 nodes"),
                    ("cont_infer", "non-Gaussian MDN, ER, 5,000 nodes")):
        rs = [r for r in rows if r["sweep"] == sw and r["method"].startswith("nbn_lw")]
        for S in sorted({int(r["opt"]["n_samples"]) for r in rs}):
            by = {r["threads"]: r for r in rs if int(r["opt"]["n_samples"]) == S}
            t1 = by[1]["time_s"] if 1 in by and by[1]["status"] == "ok" else None
            t8 = by[8]["time_s"] if 8 in by and by[8]["status"] == "ok" else None
            e = err(by.get(8) or by.get(1))
            ess = (by.get(8) or by.get(1)).get("ess_frac")
            T.add(lab, f"{S:,}", fmt_t(t1 if t1 else by.get(1, {}).get("status")),
                  fmt_t(t8 if t8 else by.get(8, {}).get("status")),
                  f"{t1 / t8:.2f}×" if t1 and t8 else "—",
                  f"{e:.4f}" if e is not None else "—",
                  f"{np.median(ess):.3f}" if ess else "—")
            series.setdefault(lab, []).append((S, t1, t8, e))
    tables.append(T)
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    for (lab, pts), c in zip(series.items(), (BLUE, VERMILLION)):
        pts = [p for p in pts if p[1] and p[2]]
        if pts:
            axs[0].plot([p[0] for p in pts], [p[1] / p[2] for p in pts], "-o", ms=4, color=c,
                        label=lab)
    axs[0].set_xscale("log")
    axs[0].axhline(1, color=INK, lw=0.6)
    axs[0].set_xlabel("particles per query")
    axs[0].set_ylabel("speed-up, 8 threads vs 1")
    axs[0].set_title("Threads pay off when batches are large")
    axs[0].legend(frameon=False, fontsize=6)
    d = [p for p in series.get("discrete, local, 1,000 nodes", []) if p[3] is not None]
    if d:
        S = np.array([p[0] for p in d], float)
        axs[1].plot(S, [p[3] for p in d], "-o", ms=4, color=BLUE, label="measured")
        axs[1].plot(S, d[0][3] * np.sqrt(S[0] / S), ":", color=GREY, label=r"$\propto 1/\sqrt{M}$")
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        axs[1].set_xlabel("particles per query $M$")
        axs[1].set_ylabel("mean max-abs error vs exact")
        axs[1].set_title("More particles: error falls like $1/\\sqrt{M}$")
        axs[1].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "r3_6_particles")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    use_style()
    plt.rcParams.update({"legend.fontsize": 7, "axes.titlesize": 8})
    tables = []
    fig_threads(tables)
    fig_gpu(tables)
    fig_frontier(tables)
    fig_ve_width(tables)
    fig_particles(tables)
    md = ["# Round 3: does NBN use a larger budget? — figures and numbers", ""]
    for t in tables:
        (TAB / f"{t.key}.tex").write_text(t.tex())
        md.append(t.md())
    (OUT / "README.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
