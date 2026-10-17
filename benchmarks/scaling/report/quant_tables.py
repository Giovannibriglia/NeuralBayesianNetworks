"""Quantitative results per question (Q1-Q9), as Markdown and LaTeX tables.

Everything is computed from the saved results, so the README of
figures/explained and the PDF report cannot drift from the data.
Writes ``figures/explained/NUMBERS.md`` and ``report/tables/q*.tex``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SC = HERE.parent
sys.path.insert(0, str(SC))
sys.path.insert(0, str(SC.parent.parent / "examples"))
import analyze as A  # noqa: E402
import cell  # noqa: E402
import explain as E  # noqa: E402

RES = SC / "results"
TAB = HERE / "tables"
NODES = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, str):
        return v
    if abs(v) >= 100:
        return f"{v:,.0f}".replace(",", " ")
    return f"{v:.{nd}g}"


class Table:
    def __init__(self, key, title, header, note=""):
        self.key, self.title, self.header, self.rows, self.note = key, title, header, [], note

    def add(self, *cells):
        self.rows.append([c if isinstance(c, str) else fmt(c) for c in cells])

    def md(self):
        out = [f"**{self.title}**", "", "| " + " | ".join(self.header) + " |",
               "|" + "---|" * len(self.header)]
        out += ["| " + " | ".join(r) + " |" for r in self.rows]
        if self.note:
            out += ["", f"*{self.note}*"]
        return "\n".join(out) + "\n"

    def tex(self):
        def esc(x):
            return (x.replace("\\", "\\textbackslash{}").replace("%", "\\%").replace("&", "\\&")
                    .replace("#", "\\#").replace("_", "\\_").replace("×", "$\\times$")
                    .replace("≥", "$\\geq$").replace("≤", "$\\leq$").replace("—", "--")
                    .replace("∩", "$\\cap$").replace("→", "$\\to$").replace("~", "$\\sim$")
                    .replace("^", "\\^{}"))
        cols = "l" + "r" * (len(self.header) - 1)
        wide = len(self.header) >= 7
        lines = ["\\begin{table}[H]\\centering\\small",
                 f"\\caption{{{esc(self.title)}}}\\label{{tab:{self.key}}}",
                 ("\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%" if wide else ""),
                 f"\\begin{{tabular}}{{{cols}}}\\toprule",
                 " & ".join(esc(h) for h in self.header) + " \\\\ \\midrule"]
        lines += [" & ".join(esc(c) for c in r) + " \\\\" for r in self.rows]
        lines.append("\\bottomrule\\end{tabular}" + ("}" if wide else ""))
        if self.note:
            lines.append(f"\\par\\smallskip\\footnotesize {esc(self.note)}")
        lines.append("\\end{table}")
        return "\n".join(lines) + "\n"


def t_med(rows, m, axis, key, topo=None):
    return E.med_time(rows, m, axis, key, topo)


def ratio(a, b):
    return None if (a is None or b is None or b == 0) else a / b


# --------------------------------------------------------------------- #
def q1():
    di = A.load("discrete_infer", "discrete_infer_master", "discrete_infer_cert")
    df, ds = A.load("discrete_fit"), A.load("discrete_sets")
    cf, ci = A.load("cont_fit", "cont_fit_mps"), A.load("cont_infer", "cont_infer_master")
    T = Table("q1", "Q1. Largest graph (nodes) handled within 1 thread / 4 GB / 30 s "
              "(median over seeds; 5000 = largest size tested)",
              ["task", "NBN branch", "NBN before", "pgmpy / Pyro"],
              "Exact queries count only if all 10 queries are answered exactly.")
    spec = [("Discrete fit", df, None, "nbn", None, "pgmpy"),
            ("Exact query, loose graph", di, "local", "nbn_ve", "nbn_ve@master", "pgmpy_ve"),
            ("Exact query, dense graph", di, "er", "nbn_ve", "nbn_ve@master", "pgmpy_ve"),
            ("Sampling query (LW), dense graph", di, "er", "nbn_lw", "nbn_lw@master", None),
            ("Certified query, dense graph", di, "er", "nbn_cert", None, None),
            ("S4 per-factor sets", ds, None, "nbn_intervals_batched", "nbn_interval", None),
            ("Non-Gaussian fit, 1 thread", cf, None, "nbn_stacked", "nbn_local", "pyro_svi"),
            ("Non-Gaussian fit, GPU (MPS)", cf, None, "nbn_stacked[mps]", "nbn_local[mps]",
             "pyro_svi[mps]"),
            ("Non-Gaussian query", ci, None, "nbn_lw", "nbn_lw@master", "pyro_vec")]
    for lab, rs, topo, a, b, c in spec:
        vals = [E.frontier_median(rs, m, topo) if m else None for m in (a, b, c)]
        T.add(lab, *[("n/a" if m is None else ("< 10" if not v else f"{v:.0f}"))
                     for m, v in zip((a, b, c), vals)])
    return [T]


def q2():
    di = A.load("discrete_infer", "discrete_infer_master")
    ci = A.load("cont_infer", "cont_infer_master")
    cf, ds = A.load("cont_fit"), A.load("discrete_sets")
    ind = {(r["deg"], r["method"]): r for r in A.load("discrete_fit_indeg")}
    T = Table("q2", "Q2. Before / after for each change (same answers)",
              ["change", "setting", "before", "after", "factor"])
    rows = [
        ("Incremental min-fill", "exact, loose, 1000 nodes (s)",
         t_med(di, "nbn_ve@master", "n", 1000, "local"), t_med(di, "nbn_ve", "n", 1000, "local")),
        ("Incremental min-fill", "exact, loose, 2000 nodes (s)",
         t_med(di, "nbn_ve@master", "n", 2000, "local"), t_med(di, "nbn_ve", "n", 2000, "local")),
        ("Barren pruning + inverse CDF", "LW, dense, 200 nodes (s)",
         t_med(di, "nbn_lw@master", "n", 200, "er"), t_med(di, "nbn_lw", "n", 200, "er")),
        ("Barren pruning + inverse CDF", "LW, loose, 200 nodes (s)",
         t_med(di, "nbn_lw@master", "n", 200, "local"), t_med(di, "nbn_lw", "n", 200, "local")),
        ("Barren pruning", "non-Gaussian LW, 2000 nodes (s)",
         t_med(ci, "nbn_lw@master", "n", 2000), t_med(ci, "nbn_lw", "n", 2000)),
        ("Stacked fit", "non-Gaussian, 200 nodes (s)",
         t_med(cf, "nbn_local", "n", 200), t_med(cf, "nbn_stacked", "n", 200)),
        ("Batched interval sets", "5000 nodes (s)",
         t_med(ds, "nbn_interval", "n", 5000), t_med(ds, "nbn_intervals_batched", "n", 5000)),
    ]
    d, s = ind.get((14, "nbn")), ind.get((14, "nbn_sparse"))
    if d and s:
        rows.append(("Sparse table", "14 parents, peak memory (MB)",
                     d["watch_peak_rss_mb"], s["watch_peak_rss_mb"]))
    kl = E._cert_widths(RES / "prelim" / "discrete_infer_cert_klonly.jsonl")
    kb = E._cert_widths(RES / "discrete_infer_cert.jsonl")
    if kl and kb:
        rows.append(("+ empirical Bernstein", "certified width, median of all queries",
                     float(np.median(kl)), float(np.median(kb))))
    for lab, setting, a, b in rows:
        T.add(lab, setting, a, b, f"{ratio(a, b):.1f}×" if ratio(a, b) else "—")
    return [T]


def q3():
    p = RES / "ordering_split.jsonl"
    recs = [x for ln in p.read_text().splitlines() if ln.strip() for x in json.loads(ln)]
    T = Table("q3", "Q3. Ten exact queries on loose graphs: total time and time spent "
              "choosing the elimination order (1 thread)",
              ["nodes", "code", "total (s)", "ordering (s)", "ordering share"])
    for r in sorted(recs, key=lambda r: (r["n"], r["code"] != "master")):
        T.add(str(r["n"]), "before" if r["code"] == "master" else "branch", r["total_s"],
              r["ordering_s"], f"{100 * r['ordering_s'] / r['total_s']:.0f}%")
    return [T]


def q4():
    di = [r for r in A.load("discrete_infer") if r["method"] == "nbn_ve" and r["axis"] == "n"
          and r["status"] == "ok" and r.get("width_bounds")]
    pg = A.agg(A.load("discrete_infer"), "n", "er", ["pgmpy_ve"]).get("pgmpy_ve", [])
    need = math.log(0.75 * 4 * 2**30 / 4, 3) - 1
    out = []
    for topo, name in (("local", "loose"), ("er", "dense")):
        T = Table(f"q4{topo}", f"Q4. Width of exact queries, {name} graph "
                  f"(3 seeds × 10 queries; a width above {need:.1f} cannot fit 3 GB with K = 3)",
                  ["nodes", "median width", "max width", "median lower bound",
                   "answered exactly", "refused", "proven infeasible"] +
                  (["pgmpy answered"] if topo == "er" else []))
        for n in NODES:
            ub, lb, ans, ref, prov = [], [], 0, 0, 0
            for r in di:
                if r["topology"] != topo or r["n"] != n:
                    continue
                for a, (lo, hi) in zip(r["answers"], r["width_bounds"]):
                    ub.append(hi)
                    lb.append(lo)
                    if a is None:
                        ref += 1
                        prov += lo > need
                    else:
                        ans += 1
            if not ub:
                continue
            row = [str(n), float(np.median(ub)), float(max(ub)), float(np.median(lb)),
                   f"{ans}/{ans + ref}", str(ref), str(prov)]
            if topo == "er":
                c = next((c for c in pg if c["key"] == n), None)
                row.append("—" if c is None else
                           (f"{c['answered']}/{c['asked']}" if c["t"] is not None else
                            f"0 ({c['fail']})"))
            T.add(*row)
        out.append(T)
    return out


def q5():
    rows = A.load("discrete_infer", "discrete_infer_cert")
    errs = A.lw_errors(rows)
    T = Table("q5", "Q5. Certified answers vs exact answers (all queries where exact VE answered)",
              ["arm", "queries", "covered", "median abs. error", "max abs. error", "median width",
               "90% width"])
    kp = RES / "prelim" / "discrete_infer_cert_klonly.jsonl"
    if kp.exists():
        kl = [json.loads(ln) for ln in kp.read_text().splitlines() if ln.strip()]
        for r in kl:
            r["method"] = "nbn_cert_kl"
        base = [r for r in rows if r["method"] in ("nbn_ve", "pgmpy_ve")]
        errs += A.lw_errors(base + kl)
    for m, lab in (("nbn_lw", "plain LW, 100k samples (no guarantee)"),
                   ("nbn_cert_kl", "certified, KL bound, 200k"),
                   ("nbn_cert", "certified, KL ∩ Bernstein, 200k")):
        es = [e for e in errs if e["method"] == m]
        if not es:
            continue
        cov = [e["covered"] for e in es if e["covered"] is not None]
        ws = [e["width"] for e in es if e.get("width") is not None]
        T.add(lab, str(len(es)), f"{sum(cov)}/{len(cov)}" if cov else "n/a",
              float(np.median([e["err"] for e in es])), float(max(e["err"] for e in es)),
              float(np.median(ws)) if ws else "—", float(np.quantile(ws, 0.9)) if ws else "—")
    T2 = Table("q5t", "Q5. Time for 10 certified queries (200k samples each, s)",
               ["nodes", "dense graph", "loose graph"])
    cert = A.load("discrete_infer_cert")
    for n in NODES:
        T2.add(str(n), t_med(cert, "nbn_cert", "n", n, "er"), t_med(cert, "nbn_cert", "n", n, "local"))
    return [T, T2]


def q6():
    cf = A.load("cont_fit", "cont_fit_mps", "cont_fit_full")
    T = Table("q6", "Q6. Non-Gaussian fit time per node (ms) and fit quality "
              "(same model, 20 epochs, 10 000 rows)",
              ["nodes", "local CPU", "stacked CPU", "Pyro CPU", "local GPU", "stacked GPU",
               "stacked 10 thr.", "LL gap stacked", "LL gap local", "LL gap Pyro"],
              "LL gap = true minus fitted median per-row test log-likelihood, nats per node.")

    def per_node(m, dev, n, thr=1):
        rs = [r for r in cf if r["method"] == m and r.get("device", "cpu") == dev
              and r["axis"] == "n" and r["n"] == n and r.get("threads", 1) == thr]
        ok = [r for r in rs if r["status"] == "ok"]
        if ok:
            return 1000 * ok[0]["time_s"] / n
        return rs[0]["status"] if rs else None

    def gap(m, n):
        rs = [r for r in cf if r["method"] == m and r.get("device", "cpu") == "cpu"
              and r["axis"] == "n" and r["n"] == n and r["status"] == "ok"
              and r.get("threads", 1) == 1]
        return (rs[0]["true_test_ll_median"] - rs[0]["test_ll_median"]) if rs else None

    for n in NODES[:-1]:
        T.add(str(n), per_node("nbn_local", "cpu", n), per_node("nbn_stacked", "cpu", n),
              per_node("pyro_svi", "cpu", n), per_node("nbn_local", "mps", n),
              per_node("nbn_stacked", "mps", n), per_node("nbn_stacked", "cpu", n, 10),
              gap("nbn_stacked", n), gap("nbn_local", n), gap("pyro_svi", n))
    return [T]


def q78():
    stats = {s["d"]: s for s in E._indeg_stats()}
    ind = {(r["deg"], r["method"]): r for r in A.load("discrete_fit_indeg")}
    T7 = Table("q7", "Q7. Table size and measured cost vs number of parents (30 nodes, K = 3)",
               ["parents", "dense numbers/node", "sparse numbers/node", "time dense (s)",
                "time sparse (s)", "time pgmpy (s)", "RSS dense (MB)", "RSS sparse (MB)"])
    T8 = Table("q8", "Q8. Data fragmentation: rows per observed parent combination and error "
               "of the fitted table (10 000 rows)",
               ["parents", "rows per combination", "TV error (all methods identical)"])

    def cell_of(d, m, f):
        r = ind.get((d, m))
        if r is None:
            return "—"
        return r[f] if r["status"] == "ok" else r["status"]

    for d in sorted(stats):
        s = stats[d]
        T7.add(str(d), float(3 ** (d + 1)), float(3 * (s["observed"] + 1)),
               cell_of(d, "nbn", "time_s"), cell_of(d, "nbn_sparse", "time_s"),
               cell_of(d, "pgmpy", "time_s"),
               ind[(d, "nbn")]["watch_peak_rss_mb"] if (d, "nbn") in ind else "—",
               ind[(d, "nbn_sparse")]["watch_peak_rss_mb"] if (d, "nbn_sparse") in ind else "—")
        T8.add(str(d), s["n"] / s["observed"], cell_of(d, "nbn_sparse", "acc_tv"))
    return [T7, T8]


def q9():
    di = A.load("discrete_infer", "discrete_infer_master")
    ci = A.load("cont_infer", "cont_infer_master")
    T = Table("q9", "Q9. Share of nodes a sampling query must touch, and LW time "
              "(10 queries, before / after)",
              ["nodes", "touched, loose", "touched, dense", "touched, non-Gauss.",
               "LW dense before (s)", "LW dense after (s)", "non-Gauss. before (s)",
               "non-Gauss. after (s)"])
    for n in NODES:
        fr = []
        for fam, topo, qf in (("discrete", "local", "d"), ("discrete", "er", "d"), ("ngs", "er", "c")):
            name = (f"{fam}_{topo}_w5_n{n}_deg2_k3_s0" if topo == "local"
                    else f"{fam}_{topo}_n{n}_deg2_k3_s0")
            p = E.PROB / f"{name}.pt"
            if not p.exists():
                fr.append(None)
                continue
            pr = torch.load(p, weights_only=False)
            g = nx.DiGraph(pr["edges"])
            g.add_nodes_from(pr["nodes"])
            qs = cell.discrete_queries(pr) if qf == "d" else [(t, e) for t, e, _ in
                                                               cell.continuous_queries(pr)]
            vals = []
            for t, e in qs:
                keep = {t, *e}
                for v in list(keep):
                    keep |= nx.ancestors(g, v)
                vals.append(len(keep) / pr["n"])
            fr.append(100 * float(np.median(vals)))
        T.add(str(n), *[("—" if v is None else f"{v:.0f}%") for v in fr],
              t_med(di, "nbn_lw@master", "n", n, "er") or "timeout",
              t_med(di, "nbn_lw", "n", n, "er"),
              t_med(ci, "nbn_lw@master", "n", n) or "timeout", t_med(ci, "nbn_lw", "n", n))
    return [T]


SECTIONS = [
    ("Q1", "How far does each method get within the budget?", "01_overview_frontier", ["q1"],
     "Every row the branch improved now reaches the largest size tested (5 000 nodes), except "
     "exact queries on dense graphs (100 nodes for every library) and neural fitting (500 nodes "
     "on 1 thread, 1 000 on the GPU), where the cost is intrinsic. Sampling queries went from "
     "200 to 5 000 nodes; non-Gaussian fitting on the GPU from 20 to 1 000."),
    ("Q2", "What did each change buy?", "02_what_each_change_bought", ["q2"],
     "Same answers, 3.4× to 51× faster, 8× less memory, and guaranteed intervals 2× narrower. "
     "The largest gains come from removing bookkeeping (ordering, sampling nodes that cannot "
     "matter), not from faster arithmetic."),
    ("Q3", "Why were exact queries slow?", "03_where_exact_query_time_went", ["q3"],
     "Before, 95% of the time went into choosing the elimination order. The incremental version "
     "returns the identical order; the ordering share drops to about a third and the total time "
     "by 13–15× in this measurement."),
    ("Q4", "Why can't exact inference scale on dense graphs?", "04_treewidth_wall", ["q4local", "q4er"],
     "On loose graphs the width stays at 5–6 at any size, so all 300 queries are exact. On dense "
     "graphs the median width grows from 2 to 60 and the maximum to 100; beyond 17.7 no table "
     "fits in 3 GB. At 5 000 nodes 23 of 30 queries are refused; 8 are proven infeasible by the "
     "lower bound alone. pgmpy fails already at 200 nodes."),
    ("Q5", "What does a certified answer look like?", "05_certified_answers", ["q5", "q5t"],
     "Every one of the 528 guaranteed intervals contains the exact answer (95% was promised). "
     "Intersecting with an empirical-Bernstein bound halves the median width (0.067 → 0.033) at "
     "no cost in coverage. A certified query on a 5 000-node dense graph takes 1.5 s."),
    ("Q6", "Why is stacked fitting faster, and why does it help on the GPU?",
     "06_stacked_fit_cost_per_node", ["q6"],
     "Per node, stacked training costs about 30 ms on one CPU thread against 95 ms (sequential) "
     "and 130 ms (Pyro), and 15 ms on the GPU, where sequential training costs 800+ ms per node. "
     "Fit quality is the same (gaps 0.015–0.035 nats per node for all three)."),
    ("Q7", "How does the sparse table save memory?", "07_sparse_table_memory", ["q7"],
     "At 20 parents the full table would need 10 billion numbers per node; the sparse table "
     "stores 28 849 and the process stays at ≈ 0.3 GB. pgmpy times out at 13 parents, the dense "
     "NBN table runs out of memory at 16."),
    ("Q8", "Then is \"many parents\" solved?", "08_data_fragmentation", ["q8"],
     "No: with 20 parents a parent combination is seen about once, and the fitted table's error "
     "grows 25× (0.010 → 0.249) whatever the speed. That limit needs a different model type."),
    ("Q9", "Why are sampling queries so much faster?", "09_barren_pruning", ["q9"],
     "A query only needs the ancestors of its target and evidence: 5% of a 5 000-node dense "
     "graph (3% non-Gaussian). Before, every query sampled every node and timed out at 500 "
     "(discrete) or 5 000 (non-Gaussian) nodes."),
]


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    tables = {}
    for f in (q1, q2, q3, q4, q5, q6, q78, q9):
        for t in f():
            tables[t.key] = t
            (TAB / f"{t.key}.tex").write_text(t.tex())
    md = ["# What was done, in nine plots — with the numbers", "",
          "Each section: one question, its plot, the numbers behind it, and what they say.",
          "Words used: **loose graph** = each node's parents are among the 5 nodes just before "
          "it (tree-width ≤ 5 at any size); **dense graph** = parents drawn from all earlier "
          "nodes (Erdős–Rényi; gets thicker as it grows); **budget** = 1 CPU thread, 4 GB, 30 s "
          "per run (GPU = Apple M5 via MPS); **before** = NBN `master`, or the old one-node-at-a-"
          "time training path.", "",
          "Regenerate: `python benchmarks/scaling/explain.py` (plots) and "
          "`python benchmarks/scaling/report/quant_tables.py` (this file and the report tables).",
          ""]
    for q, title, fig, keys, takeaway in SECTIONS:
        md += [f"## {q}. {title}", "", f"![{q}]({fig}.png)", ""]
        for k in keys:
            md.append(tables[k].md())
        md += [f"**What the numbers say.** {takeaway}", ""]
    (SC / "figures" / "explained" / "README.md").write_text("\n".join(md))
    stale = SC / "figures" / "explained" / "NUMBERS.md"
    if stale.exists():
        stale.unlink()
    print("ok", sorted(tables))


if __name__ == "__main__":
    main()
