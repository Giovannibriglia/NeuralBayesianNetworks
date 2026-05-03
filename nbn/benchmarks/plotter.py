"""Publication-ready plots from a benchmark parquet."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nbn.benchmarks.style import NBN_PALETTE, apply_style, savefig_multi


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
