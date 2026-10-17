"""Round 4 figures and tables: a larger budget buys a better answer.

Reads results/r4_e1.jsonl, r4_e2.jsonl, r4_e3.jsonl (any hosts / devices);
writes figures/round4/*.{png,pdf}, figures/round4/README.md and
report/tables_r4/*.tex.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "report"))
sys.path.insert(0, str(HERE.parent.parent / "examples"))
from _figures import BLUE, DOUBLE, GREY, INK, ORANGE, SKY, VERMILLION, use_style  # noqa: E402
from quant_tables import Table  # noqa: E402

RES = HERE / "results"
OUT = HERE / "figures" / "round4"
TAB = HERE / "report" / "tables_r4"


def load(name):
    p = RES / f"{name}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def cfg(r):
    dev = r["device"]
    if dev == "cpu":
        return f"CPU, {r['threads']} thread" + ("s" if r["threads"] > 1 else "")
    return {"cuda": "CUDA (TITAN X)", "mps": "Apple MPS"}.get(dev, dev)


CFG_COLOR = {"CPU, 1 thread": SKY, "CPU, 20 threads": BLUE, "CUDA (TITAN X)": VERMILLION,
             "Apple MPS": ORANGE}
CFG_ORDER = list(CFG_COLOR)


def plain(axis, ticks, fmt="{:g}"):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))
    axis.set_minor_formatter(NullFormatter())


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- E1
def e1(tables):
    rows = load("r4_e1")
    if not rows:
        return
    by = defaultdict(list)
    for r in rows:
        by[(cfg(r), r["mem_gb"], r["deg"])].append(r)
    T = Table("r4e1", "R4-1. Exact queries answered (of 10) on denser graphs (ER, 100 nodes, "
              "K = 3): plain VE vs conditioning within 20 s per query; the time a query "
              "would need (projected from the measured rate) for those that did not finish",
              ["config", "memory", "avg degree", "plain VE", "conditioning <= 20 s",
               "median s (answered)", "max cutset", "projected s (others, median / max)"])
    for (c, mem, deg), rs in sorted(by.items(), key=lambda kv: (CFG_ORDER.index(kv[0][0])
                                                                if kv[0][0] in CFG_ORDER else 9,
                                                                kv[0][1], kv[0][2])):
        pl = sum(r["plain_status"] == "ok" for r in rs)
        co = [r for r in rs if r.get("cond_status") == "ok"]
        ts = [r["cond_s"] for r in co]
        diffs = [r["max_abs_diff"] for r in rs if "max_abs_diff" in r]
        cuts = [r["cutset"] for r in co if r.get("cutset") is not None]
        proj = [r["cond_projected_s"] for r in rs if r.get("cond_status") == "gave_up"]
        cuts = [r["cutset"] for r in rs if r.get("cutset") is not None]
        T.add(c, f"{mem:g} GB", f"{deg:g}", f"{pl}/{len(rs)}", f"{len(co)}/{len(rs)}",
              f"{np.median(ts):.3g}" if ts else "—", str(max(cuts)) if cuts else "—",
              f"{np.median(proj):.3g} / {max(proj):.3g}" if proj else "—")
    tables.append(T)

    # Figure 1: answered vs degree, per memory budget (best CPU config + CUDA)
    mems = sorted({r["mem_gb"] for r in rows})
    fig, axs = plt.subplots(1, len(mems), figsize=(DOUBLE, 2.4), sharey=True, squeeze=False)
    for ax, mem in zip(axs[0], mems):
        for c in CFG_ORDER:
            sub = {deg: rs for (cc, m, deg), rs in by.items() if cc == c and m == mem}
            if not sub:
                continue
            degs = sorted(sub)
            ax.plot(degs, [sum(r.get("cond_status") == "ok" for r in sub[d]) for d in degs],
                    "-o", ms=3, color=CFG_COLOR[c], label=f"conditioning, {c}")
        sub = {deg: rs for (cc, m, deg), rs in by.items() if m == mem}
        degs = sorted(sub)
        ax.plot(degs, [max(sum(r["plain_status"] == "ok" for r in rs2)
                           for (cc, m, d2), rs2 in by.items() if m == mem and d2 == d)
                       for d in degs], ":s", ms=3, color=INK, label="plain VE (any config)")
        ax.set_title(f"memory budget {mem:g} GB")
        ax.set_xlabel("average degree")
        ax.set_ylim(-0.5, 10.5)
    axs[0][0].set_ylabel("queries answered exactly (of 10)")
    axs[0][-1].legend(frameon=False, fontsize=5.5, loc="lower left")
    fig.tight_layout()
    save(fig, "r4_1_exact_answered")

    # Figure 2: time of a conditioned answer vs instantiations, per config
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.55, 2.6))
    for c in CFG_ORDER:
        pts = [(r["n_inst"], r["cond_s"]) for r in rows if cfg(r) == c
               and r.get("cond_status") == "ok" and r.get("conditioned")]
        if pts:
            x, y = zip(*pts)
            ax.scatter(x, y, s=8, color=CFG_COLOR[c], label=c, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("instantiations of the cutset $K^{|C|}$")
    ax.set_ylabel("time of the exact answer (s)")
    ax.set_title("Conditioning: time grows with $K^{|C|}$; workers divide it")
    ax.legend(frameon=False, fontsize=6)
    save(fig, "r4_2_time_vs_instantiations")

    # Figure 3: answered within a time limit T (memory 1 GB): time buys answers
    mem = min(mems)
    degs_all = sorted({r["deg"] for r in rows})
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.4))
    Ts = np.logspace(-1, 3.3, 60)
    for c in CFG_ORDER:
        rs = [r for r in rows if cfg(r) == c and r["mem_gb"] == mem]
        if not rs:
            continue
        ts = [r["cond_s"] if r.get("cond_status") == "ok" else
              r.get("cond_projected_s", np.inf) for r in rs]
        axs[0].plot(Ts, [sum(t <= T for t in ts) for T in Ts], color=CFG_COLOR[c], label=c)
        pl = [r.get("plain_s", np.inf) if r["plain_status"] == "ok" else np.inf for r in rs]
        axs[0].plot(Ts, [sum(t <= T for t in pl) for T in Ts], ":", color=CFG_COLOR[c])
    axs[0].set_xscale("log")
    axs[0].set_xlabel("time limit per query (s)")
    axs[0].set_ylabel(f"queries answered exactly (of {10 * len(degs_all)})")
    axs[0].set_title(f"Memory {mem:g} GB: time now buys exact answers\n"
                     "(solid: conditioning, dotted: plain VE; beyond 20 s projected)")
    axs[0].legend(frameon=False, fontsize=6)
    for c in CFG_ORDER:
        rs = [r for r in rows if cfg(r) == c]
        if not rs:
            continue
        pts = sorted({r["mem_gb"] for r in rs})
        axs[1].plot(pts, [sum(r.get("cond_status") == "ok" for r in rs if r["mem_gb"] == m)
                          for m in pts], "-o", ms=3, color=CFG_COLOR[c], label=f"conditioning, {c}")
        axs[1].plot(pts, [sum(r["plain_status"] == "ok" for r in rs if r["mem_gb"] == m)
                          for m in pts], ":s", ms=3, color=CFG_COLOR[c])
    axs[1].set_xscale("log", base=2)
    plain(axs[1].xaxis, sorted({r["mem_gb"] for r in rows}), "{:g} GB")
    axs[1].set_xlabel("memory budget")
    axs[1].set_ylabel("queries answered exactly")
    axs[1].set_title("Memory: plain VE gains slowly;\nconditioning answers (almost) all")
    fig.tight_layout()
    save(fig, "r4_3_budget_buys_answers")


# ---------------------------------------------------------------------- E2
def e2(tables):
    rows = [r for r in load("r4_e2") if r.get("status") == "ok"]
    if not rows:
        return
    probs = sorted({(r["topology"], r["n"], r["deg"]) for r in rows})
    T = Table("r4e2", "R4-2. Certified sampling, anytime mode: median certified width (95%, "
              "all classes) by time budget; coverage of the exact answer where it is known",
              ["problem", "config", "mode", *[f"{t:g} s" for t in
                                              sorted({r['time_budget_s'] for r in rows})],
               "covers exact"])
    fig, axs = plt.subplots(1, len(probs), figsize=(DOUBLE, 2.5), squeeze=False)
    for ax, (topo, n, deg) in zip(axs[0], probs):
        for c in CFG_ORDER:
            for mode, ls in (("plain", "-"), ("collapsed", "--")):
                rs = [r for r in rows if (r["topology"], r["n"], r["deg"]) == (topo, n, deg)
                      and cfg(r) == c and r["mode"] == mode]
                if not rs:
                    continue
                tbs = sorted({r["time_budget_s"] for r in rs})
                med = [float(np.median([r["width"] for r in rs if r["time_budget_s"] == t]))
                       for t in tbs]
                ax.plot(tbs, med, ls, marker="o", ms=3, color=CFG_COLOR[c],
                        label=f"{mode}, {c}")
                cov = [r["covers"] for r in rs if "covers" in r]
                T.add(f"{topo} n={n} deg {deg:g}", c, mode, *[f"{m:.3g}" for m in med],
                      f"{sum(cov)}/{len(cov)}" if cov else "—")
        tt = np.array([0.25, 16])
        ax.plot(tt, 0.2 * (tt / 0.25) ** -0.5, ":", color=GREY, lw=0.8, label=r"$\propto t^{-1/2}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{topo}, {n} nodes, degree {deg:g}")
        ax.set_xlabel("time budget per query (s)")
    axs[0][0].set_ylabel("certified width (median)")
    axs[0][-1].legend(frameon=False, fontsize=5, loc="lower left")
    fig.tight_layout()
    save(fig, "r4_4_certified_anytime")
    tables.append(T)


# ---------------------------------------------------------------------- E3
def e3(tables):
    rows = [r for r in load("r4_e3") if r.get("status") == "ok"]
    if not rows:
        return
    T = Table("r4e3", "R4-3. Bounded conditioning (sound bounds, no probability): median "
              "width by time budget; skew = factor on the CPT logits",
              ["config", "skew", *[f"{t:g} s" for t in sorted({r['time_budget_s'] for r in rows})],
               "exact at largest", "median prior mass covered"])
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.55, 2.5))
    for c in CFG_ORDER:
        for skew, ls in ((1.0, "-"), (3.0, "--")):
            rs = [r for r in rows if cfg(r) == c and r["skew"] == skew]
            if not rs:
                continue
            tbs = sorted({r["time_budget_s"] for r in rs})
            med = [float(np.median([r["width"] for r in rs if r["time_budget_s"] == t]))
                   for t in tbs]
            ax.plot(tbs, med, ls, marker="o", ms=3, color=CFG_COLOR[c],
                    label=f"skew {skew:g}, {c}")
            last = [r for r in rs if r["time_budget_s"] == tbs[-1]]
            T.add(c, f"{skew:g}", *[f"{m:.3g}" for m in med],
                  f"{sum(r['exact'] for r in last)}/{len(last)}",
                  f"{np.median([r['mass'] or 0 for r in last]):.3g}")
    ax.set_xscale("log")
    ax.set_xlabel("time budget per query (s)")
    ax.set_ylabel("width of the bounds (median)")
    ax.set_title("Bounded conditioning: sound bounds tighten with time")
    ax.legend(frameon=False, fontsize=5.5)
    save(fig, "r4_5_bounded_conditioning")
    tables.append(T)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    use_style()
    plt.rcParams.update({"legend.fontsize": 7, "axes.titlesize": 8})
    tables = []
    e1(tables)
    e2(tables)
    e3(tables)
    md = ["# Round 4: a larger budget buys a better answer — figures and numbers", ""]
    for t in tables:
        (TAB / f"{t.key}.tex").write_text(t.tex())
        md.append(t.md())
    (OUT / "README.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
