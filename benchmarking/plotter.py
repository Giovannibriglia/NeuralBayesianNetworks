"""Publication-ready plots from a benchmark parquet."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarking.style import NBN_PALETTE, apply_style, savefig_multi


def _ensure(df) -> Any:
    """Pandas import guard with a friendly error."""
    try:
        import pandas as pd
        return pd
    except ImportError as e:
        raise ImportError("Plotting requires pandas: pip install pandas") from e


def plot_speed_pareto(df, out_stem: str = "results/speed_pareto") -> list[str]:
    """Accuracy-vs-time scatter, one point per (baseline, problem, device)."""
    apply_style()
    import matplotlib.pyplot as plt
    pd = _ensure(df)

    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)
    if "tv" not in df.columns or "time_s" not in df.columns:
        # No accuracy info — fall back to time-only
        plot_throughput_bar(df, out_stem)
        return []
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for b in df["baseline"].dropna().unique():
        sub = df[(df["baseline"] == b) & df["tv"].notna() & df["time_s"].notna()]
        if sub.empty:
            continue
        ax.scatter(
            sub["time_s"], sub["tv"],
            label=b, color=NBN_PALETTE.get(b, "#888888"), s=24, alpha=0.8,
        )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Time per query (s, log)")
    ax.set_ylabel("TV distance to ground truth (log)")
    ax.set_title("Accuracy vs. speed Pareto")
    ax.legend(frameon=False)
    return savefig_multi(fig, out_stem)


def plot_throughput_bar(df, out_stem: str = "results/throughput") -> list[str]:
    """Median latency per (baseline, device) as a grouped bar chart."""
    apply_style()
    import matplotlib.pyplot as plt
    pd = _ensure(df)

    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)
    g = df[df["ok"]].groupby(["baseline", "device"])["time_s"].median().reset_index()
    if g.empty:
        return []
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    labels = [f"{r.baseline}\n({r.device})" for r in g.itertuples()]
    times_ms = (g["time_s"] * 1000).tolist()
    bars = ax.bar(
        labels, times_ms,
        color=[NBN_PALETTE.get(b, "#888888") for b in g["baseline"]],
    )
    ax.set_ylabel("Median latency (ms)")
    ax.set_yscale("log")
    ax.set_title("Median single-query latency")
    for bar, v in zip(bars, times_ms, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    return savefig_multi(fig, out_stem)


def _color(b: str) -> str:
    base = b.split("_")[0] if isinstance(b, str) else "ground"
    return NBN_PALETTE.get(base, "#888888")


def plot_scaling_ablation(
    df,
    *,
    axis: str = "n_nodes",
    metric: str = "ms_per_query",
    problem_kind: str = "all",
    out_stem: str = "results/scaling_ablation",
) -> list[str]:
    """Line plot of ``metric`` vs ``axis`` (n_nodes / density / cardinality).

    Parses each scaling-grid problem name (``hybrid_n{N}_d{DDD}_k{K}_r{R}``)
    to extract the axis value, then groups by (baseline, device) and draws
    one line per group with mean ± std error band over replicates.
    """
    apply_style()
    import re
    import matplotlib.pyplot as plt
    pd = _ensure(df)

    if not hasattr(df, "columns"):
        df = pd.DataFrame(df)
    if "problem" not in df.columns:
        return []
    re_n = re.compile(r"hybrid_n(\d+)_d(\d+)_k(\d+)_r(\d+)")
    parsed = []
    for _, row in df.iterrows():
        m = re_n.match(str(row.get("problem", "")))
        if not m:
            continue
        n, d, k, r = int(m.group(1)), int(m.group(2)) / 100.0, int(m.group(3)), int(m.group(4))
        parsed.append({**row.to_dict(), "n_nodes": n, "density": d, "cardinality": k, "rep": r})
    if not parsed:
        return []
    pdf = pd.DataFrame(parsed)
    if metric not in pdf.columns or axis not in pdf.columns:
        return []

    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
    for (baseline, device), grp in pdf.groupby(["baseline", "device"]):
        agg = grp.groupby(axis)[metric].agg(["mean", "std"]).reset_index()
        ax.plot(agg[axis], agg["mean"], marker="o", lw=1.5, ms=5,
                color=_color(baseline), label=f"{baseline} ({device})")
        if agg["std"].notna().any():
            ax.fill_between(agg[axis], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            color=_color(baseline), alpha=0.15)
    if axis == "n_nodes":
        ax.set_xscale("log")
    if metric in {"ms_per_query", "time_s", "throughput_qps"}:
        ax.set_yscale("log")
    ax.set_xlabel({"n_nodes": "n_nodes (log)", "density": "edge density",
                    "cardinality": "discrete cardinality"}.get(axis, axis))
    ax.set_ylabel({"ms_per_query": "ms / query (log)", "time_s": "time (s, log)",
                    "tv": "TV", "throughput_qps": "queries / s (log)"}.get(metric, metric))
    ax.set_title(f"Scaling ablation: {metric} vs {axis}")
    ax.legend(frameon=False, fontsize=8)
    return savefig_multi(fig, out_stem)


def plot_accuracy_bars(df, out_stem: str = "results/accuracy") -> list[str]:
    """Mean TV / KL per baseline as a bar chart."""
    apply_style()
    import matplotlib.pyplot as plt
    _ensure(df)

    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)
    sub = df[df["kind"] == "marginal_metric"]
    if sub.empty:
        return []
    g = sub.groupby("baseline")[["kl", "tv", "mae"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=False)
    for ax, m in zip(axes, ["kl", "tv", "mae"], strict=False):
        ax.bar(g["baseline"], g[m],
               color=[NBN_PALETTE.get(b, "#888888") for b in g["baseline"]])
        ax.set_title(m.upper())
        ax.tick_params(axis="x", rotation=20)
    return savefig_multi(fig, out_stem)
