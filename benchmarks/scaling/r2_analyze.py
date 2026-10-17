"""Round-2 figures and tables: other mechanisms and engines.

Versions: this branch (``feat/scaling-mechanisms``), the first scaling branch
(``@v1``, ``feat/scaling-study``) and NBN ``master`` (``@master``); baselines
Pyro (neural categorical fit) where an equivalent exists.  Writes
``figures/round2/*.{png,pdf}``, ``figures/round2/README.md`` and
``report/tables_r2/*.tex``.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "report"))
sys.path.insert(0, str(HERE.parent.parent / "examples"))
import analyze as A  # noqa: E402
from _figures import AQUA, BLUE, DOUBLE, GREY, INK, MAGENTA, ORANGE, SKY, VERMILLION, use_style  # noqa: E402
from quant_tables import Table  # noqa: E402

OUT = HERE / "figures" / "round2"


def _plain_ticks(axis, ticks, fmt="{:g}"):
    """Readable log-axis ticks: fixed positions, plain labels, no minor labels."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))
    axis.set_minor_formatter(NullFormatter())
TAB = HERE / "report" / "tables_r2"
RES = HERE / "results"
VERS = {"": ("this branch", BLUE, "-", "o"), "@v1": ("first scaling branch", AQUA, "--", "s"),
        "@master": ("NBN master", SKY, ":", "D")}
NODES = (10, 20, 50, 100, 200, 500, 1000)


def load(name):
    p = RES / f"{name}.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def series(rows, method, field="time_s"):
    """(n, value) for ok cells, first failure (n, status)."""
    rs = sorted([r for r in rows if r["method"] == method], key=lambda r: r["n"])
    ok = [(r["n"], r[field]) for r in rs if r["status"] == "ok" and r.get(field) is not None]
    bad = next(((r["n"], r["status"]) for r in rs if r["status"] != "ok"), None)
    return ok, bad


def val(rows, method, n, field="time_s"):
    for r in rows:
        if r["method"] == method and r["n"] == n and r["status"] == "ok":
            return r.get(field)
    for r in rows:
        if r["method"] == method and r["n"] == n:
            return r["status"]
    return None


def draw(ax, rows, specs, budget=30.0, ylabel="time (s)"):
    for mi, (method, label, c, ls, mk) in enumerate(specs):
        ok, bad = series(rows, method)
        if ok:
            ax.plot([n for n, _ in ok], [v for _, v in ok], ls, marker=mk, ms=3.5, color=c,
                    label=label)
        if bad:
            x = bad[0] * (1 + 0.12 * mi)
            ax.plot([x], [budget], "x", color=c, ms=7, mew=2, label=None if ok else label)
            ax.annotate({"timeout": "time", "oom": "mem"}.get(bad[1], bad[1]), (x, budget),
                        textcoords="offset points", xytext=(3, 3 + 9 * (mi % 2)), fontsize=6)
    ax.axhline(budget, color=GREY, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=budget * 4)
    ax.set_xlabel("nodes")
    ax.set_ylabel(ylabel)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def ratio(a, b):
    return a / b if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b else None


# --------------------------------------------------------------------- #
def fig_fits(tables):
    sw = [("r2_fit_ncat", "ncat", "Neural categorical (discrete)"),
          ("r2_fit_flex", "flex", "FlexCode (non-Gaussian)"),
          ("r2_fit_flow", "flow", "Normalising flow (non-Gaussian)")]
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE * 1.45, 2.8))
    T = Table("r2fit", "R2-1. Neural-mechanism fitting time (s) and fit quality "
              "(10 epochs, 10 000 rows, 1 thread)",
              ["mechanism", "nodes", "NBN before (sequential)", "NBN stacked", "Pyro SVI",
               "speed-up vs before", "LL before", "LL stacked", "LL Pyro"],
              "LL = mean test log-likelihood per node and row (higher is better). "
              "'Before' is the sequential path, identical on master and the first branch.")
    for ax, (name, kind, title) in zip(axs, sw):
        rows = load(name)
        specs = [(f"{kind}_local@master", "NBN before (sequential)", SKY, ":", "D"),
                 (f"{kind}_stacked", "NBN stacked (this branch)", BLUE, "-", "o")]
        if kind == "ncat":
            specs.append(("pyro_ncat", "Pyro SVI (same model)", VERMILLION, "-", "^"))
        draw(ax, rows, specs)
        ax.set_title(title)
        for n in (50, 200, 500):
            b = val(rows, f"{kind}_local@master", n)
            s = val(rows, f"{kind}_stacked", n)
            py = val(rows, "pyro_ncat", n) if kind == "ncat" else "n/a"
            T.add(kind, str(n), b, s, py,
                  f"{ratio(b, s):.1f}×" if ratio(b, s) else "—",
                  val(rows, f"{kind}_local@master", n, "test_ll"),
                  val(rows, f"{kind}_stacked", n, "test_ll"),
                  val(rows, "pyro_ncat", n, "test_ll") if kind == "ncat" else "n/a")
    h, lab = [], []
    for a in axs:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "r2_1_neural_fits")
    tables.append(T)


def fig_queries(tables):
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.8))
    T = Table("r2q", "R2-2. Ten likelihood-weighting queries (10k samples) on networks of kNN "
              "or FlexCode mechanisms (s)",
              ["mechanism", "nodes", "master", "first branch", "this branch",
               "vs master", "vs first branch"])
    for ax, (name, kind, title) in zip(axs, (("r2_knn_query", "knn", "kNN mechanisms"),
                                             ("r2_flex_query", "flex", "FlexCode mechanisms"))):
        rows = load(name)
        specs = [(f"{kind}_lw{t}", lab, c, ls, mk) for t, (lab, c, ls, mk) in VERS.items()]
        draw(ax, rows, specs[::-1])
        ax.set_title(f"{title}: 10 LW queries")
        sizes = (10, 20, 50, 100, 200) if kind == "knn" else (10, 50, 100, 500, 1000)
        for n in sizes:
            m_, v1, b = (val(rows, f"{kind}_lw{t}", n) for t in ("@master", "@v1", ""))
            T.add(kind, str(n), m_, v1, b,
                  f"{ratio(m_, b):.1f}×" if ratio(m_, b) else "—",
                  f"{ratio(v1, b):.1f}×" if ratio(v1, b) else "—")
    axs[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "r2_2_knn_flex_queries")
    tables.append(T)


def fig_ais(tables):
    rows = load("r2_ais")
    if not rows:
        return
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE * 1.45, 2.7))
    specs = [(f"ais{t}", lab, c, ls, mk) for t, (lab, c, ls, mk) in VERS.items()]
    draw(axs[0], rows, specs[::-1])
    axs[0].set_title("Amortized IS: 10 queries (10k samples)")
    for t, (lab, c, ls, mk) in VERS.items():
        rs = sorted([r for r in rows if r["method"] == f"ais{t}" and r["status"] == "ok"],
                    key=lambda r: r["n"])
        if not rs:
            continue
        axs[1].plot([r["n"] for r in rs], [100 * float(np.median(r["ess_frac"])) for r in rs],
                    ls, marker=mk, ms=3.5, color=c, label=lab)
        errs = [(r["n"], [e for e in r["abs_err"] if e is not None]) for r in rs]
        axs[2].plot([n for n, e in errs if e], [float(np.median(e)) for n, e in errs if e],
                    ls, marker=mk, ms=3.5, color=c, label=lab)
    for ax, yl, tt in ((axs[1], "effective sample size (%)", "Effective sample size (median)"),
                       (axs[2], "abs. error vs exact", "Error vs exact VE (median)")):
        ax.set_xscale("log")
        ax.set_xlabel("nodes")
        ax.set_ylabel(yl)
        ax.set_title(tt)
    axs[2].set_yscale("log")
    axs[1].set_ylim(0, 105)
    axs[1].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "r2_3_amortized_is")
    T = Table("r2ais", "R2-3. Amortized IS on loose graphs: time for 10 queries (s), median "
              "effective sample size, median error vs exact",
              ["nodes", "time master", "time this branch", "speed-up", "ESS/N master",
               "ESS/N this branch", "error master", "error this branch"])
    for n in (50, 200, 500, 1000):
        def med(t, f):
            for r in rows:
                if r["method"] == f"ais{t}" and r["n"] == n and r["status"] == "ok":
                    xs = [x for x in r[f] if x is not None]
                    return float(np.median(xs)) if xs else None
            return None
        a, b = val(rows, "ais@master", n), val(rows, "ais", n)
        T.add(str(n), a, b, f"{ratio(a, b):.1f}×" if ratio(a, b) else "—",
              med("@master", "ess_frac"), med("", "ess_frac"),
              med("@master", "abs_err"), med("", "abs_err"))
    tables.append(T)
    # same estimator: error and ESS across seeds (seed 0 from the sweep + seeds 1-5)
    allr = [r for r in rows + load("r2_ais_seeds") if r["status"] == "ok"]
    if any(r["seed"] != 0 for r in allr):
        S = Table("r2aisseeds", "R2-3b. Amortized IS, same estimator: mean absolute error vs "
                  "exact VE and median ESS/N across seeds (10 queries per seed)",
                  ["nodes", "version", "seeds", "mean error", "sd across seeds",
                   "median ESS/N", "mean time (s)"])
        for n in (200, 1000):
            for t, lab in (("@master", "master"), ("", "this branch")):
                rs = [r for r in allr if r["method"] == f"ais{t}" and r["n"] == n]
                if not rs:
                    continue
                per = [float(np.mean([e for e in r["abs_err"] if e is not None])) for r in rs]
                ess = [x for r in rs for x in r["ess_frac"] if x is not None]
                S.add(str(n), lab, str(len(rs)), float(np.mean(per)),
                      float(np.std(per, ddof=1)) if len(per) > 1 else "—",
                      float(np.median(ess)), float(np.mean([r["time_s"] for r in rs])))
        tables.append(S)


def fig_avi(tables):
    rows = load("r2_avi")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.55, 2.7))
    for t, (lab, c, ls, mk) in VERS.items():
        ok, _ = series(rows, f"avi{t}", "time_per_step_s")
        if ok:
            ax.plot([n for n, _ in ok], [1000 * v for _, v in ok], ls, marker=mk, ms=3.5,
                    color=c, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nodes")
    ax.set_ylabel("ms per ELBO step")
    _plain_ticks(ax.yaxis, [10, 20, 50, 100, 200, 500, 1000])
    ax.set_title("Amortized VI training step\n(discrete, 1 024 rows per step)")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "r2_4_amortized_vi")
    T = Table("r2avi", "R2-4. Amortized VI: time per ELBO training step (ms)",
              ["nodes", "master", "first branch", "this branch", "speed-up vs master"])
    for n in (50, 200, 500, 1000):
        a = val(rows, "avi@master", n, "time_per_step_s")
        v1 = val(rows, "avi@v1", n, "time_per_step_s")
        b = val(rows, "avi", n, "time_per_step_s")
        T.add(str(n), *[1000 * x if isinstance(x, float) else x for x in (a, v1, b)],
              f"{ratio(a, b):.1f}×" if ratio(a, b) else "—")
    tables.append(T)


def fig_micro(tables):
    rows = load("r2_micro")
    knn = [r for r in rows if r["kind"] == "knn_search"]
    if not knn:
        return
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.7))
    for d, c in zip((1, 2, 4, 8), (BLUE, AQUA, ORANGE, MAGENTA)):
        rs = sorted([r for r in knn if r["d_pa"] == d], key=lambda r: r["n_train"])
        axs[0].plot([r["n_train"] for r in rs], [r["old_any"] / r["exact_kdtree"] for r in rs],
                    "-o", ms=3.5, color=c, label=f"{d} parent dim(s)")
    axs[0].axhline(1, color=GREY, lw=0.8)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("training rows")
    _plain_ticks(axs[0].xaxis, [2000, 10000, 50000], "{:,.0f}")
    _plain_ticks(axs[0].yaxis, [0.5, 1, 2, 4, 8])
    axs[0].set_ylabel("speed-up of the k-d tree\nover the old search")
    axs[0].set_title("kNN search, 10 000 queries")
    axs[0].legend(fontsize=6.5, frameon=False)
    rs = sorted(knn, key=lambda r: (r["d_pa"], r["n_train"]))
    xs = np.arange(len(rs))
    axs[1].bar(xs, [r["disc_old_any"] / r["disc_new"] for r in rs], color=BLUE)
    axs[1].set_xticks(xs)
    axs[1].set_xticklabels([f"{r['n_train'] // 1000}k/d{r['d_pa']}" for r in rs], rotation=90,
                           fontsize=6)
    axs[1].set_yscale("log")
    axs[1].set_ylabel("speed-up (old / new)")
    axs[1].set_title("Discrete parents: repeated rows searched once")
    fig.tight_layout()
    save(fig, "r2_5_knn_search_micro")
    T = Table("r2micro", "R2-5. kNN neighbour search, 10 000 queries (s, 1 thread)",
              ["training rows", "parent dims", "old (float32, ties unspecified)",
               "new brute (exact)", "new k-d tree (exact)", "discrete parents old", "discrete new"])
    for r in rs:
        T.add(str(r["n_train"]), str(r["d_pa"]), r["old_any"], r["exact_brute"],
              r["exact_kdtree"], r["disc_old_any"], r["disc_new"])
    tables.append(T)
    fx = [r for r in rows if r["kind"] == "flex_grid"]
    if fx:
        T2 = Table("r2flex", "R2-6. FlexCode grid density (normaliser), seconds",
                   ["rows", "before (broadcast)", "after (matmul)", "speed-up"])
        for r in fx:
            T2.add(str(r["rows"]), r["old"], r["new"], f"{r['old'] / r['new']:.0f}×")
        tables.append(T2)


def _largest_common(rows, m_old, m_new, field="time_s"):
    """Largest n where both methods finished; returns (n, ratio old/new)."""
    best = None
    for n in NODES:
        a, b = val(rows, m_old, n, field), val(rows, m_new, n, field)
        r = ratio(a, b)
        if r:
            best = (n, r)
    return best


def fig_summary(tables):
    items = []
    for name, kind, label in (("r2_fit_ncat", "ncat", "Neural categorical fit"),
                              ("r2_fit_flex", "flex", "FlexCode fit"),
                              ("r2_fit_flow", "flow", "Flow fit")):
        rows = load(name)
        c = _largest_common(rows, f"{kind}_local@master", f"{kind}_stacked")
        if c:
            items.append((f"{label} ({c[0]} nodes)", "master", c[1]))
        if kind == "ncat":
            c = _largest_common(rows, "pyro_ncat", "ncat_stacked")
            if c:
                items.append((f"{label} ({c[0]} nodes)", "Pyro", c[1]))
    for name, kind, label in (("r2_knn_query", "knn", "kNN LW queries"),
                              ("r2_flex_query", "flex", "FlexCode LW queries")):
        rows = load(name)
        for t, who in (("@master", "master"), ("@v1", "first branch")):
            c = _largest_common(rows, f"{kind}_lw{t}", f"{kind}_lw")
            if c:
                items.append((f"{label} ({c[0]} nodes)", who, c[1]))
    rows = load("r2_ais")
    for t, who in (("@master", "master"), ("@v1", "first branch")):
        c = _largest_common(rows, f"ais{t}", "ais")
        if c:
            items.append((f"Amortized IS queries ({c[0]} nodes)", who, c[1]))
    rows = load("r2_avi")
    c = _largest_common(rows, "avi@master", "avi", "time_per_step_s")
    if c:
        items.append((f"Amortized VI step ({c[0]} nodes)", "master", c[1]))
    if not items:
        return
    col = {"master": SKY, "first branch": AQUA, "Pyro": VERMILLION}
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.85, 0.32 * len(items) + 0.9))
    for i, (lab, who, v) in enumerate(items[::-1]):
        ax.barh(i, v, color=col[who], height=0.6)
        ax.text(v * 1.08, i, f"{v:.1f}×", va="center", fontsize=7)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f"{lab}  vs {who}" for lab, who, _ in items[::-1]], fontsize=6.5)
    ax.set_xscale("log")
    ax.axvline(1, color=GREY, lw=0.8)
    ax.set_xlim(0.8, max(v for *_, v in items) * 3)
    ax.set_xlabel("speed-up of this branch at the largest size both finished\n"
                  "(same answers / same estimator)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f"vs {w}") for w, c in col.items()],
              fontsize=6.5, frameon=False, loc="lower right")
    ax.set_title("Round 2: speed-ups")
    save(fig, "r2_0_summary")
    T = Table("r2sum", "R2-0. Speed-ups of this branch at the largest size both versions finished",
              ["what", "compared with", "speed-up"],
              "Where the older version timed out at every larger size, the gap is larger than shown.")
    for lab, who, v in items:
        T.add(lab, who, f"{v:.1f}×")
    tables.insert(0, T)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    use_style()
    plt.rcParams.update({"legend.fontsize": 7, "axes.titlesize": 8})
    tables = []
    fig_fits(tables)
    fig_queries(tables)
    fig_ais(tables)
    fig_avi(tables)
    fig_micro(tables)
    fig_summary(tables)
    md = ["# Round 2: other mechanisms and engines — figures and numbers", ""]
    for t in tables:
        (TAB / f"{t.key}.tex").write_text(t.tex())
        md.append(t.md())
    (OUT / "README.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
