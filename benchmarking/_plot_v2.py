"""v0.6c-C-3 — plotter v2 (legend filtering + b&w-safe markers).

Renders 2 figures per crash-test run (accuracy and total_time_s)
with a 2×2 panel grid (one panel per family).  Each panel's legend
filters to baselines that are *applicable* to that family per the
C-1a registry — so e.g. discrete panels don't list
``pgmpy-lg-predict``.

Replaces the v0.5b ``plot_metric_vs_n_nodes`` (which was designed for
the pre-C-1a flat 5-baseline list).  The runner calls
:func:`render_figures` directly.

Marker scheme is **b&w-safe**: each baseline gets a unique shape
(circle / square / triangle / diamond / plus / cross / star / etc.)
so reviewers reading printouts can still distinguish baselines.
Colors group by library (pgmpy → blue tones, nbn → red/orange,
gpytorch → green, pomegranate → purple, pyro → brown), with
mechanism × engine variants as different shades within the hue.

Mean ± std bands: per-family line plot of mean accuracy/time across
seeds, with a semi-transparent ±1σ band.  NaN values (cells with no
ok rows) break the line naturally.

Error/timeout cells are tallied per-family and listed in the figure
footer so reviewers see what's missing without hunting through the
parquet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from benchmarking._baseline_registry import is_applicable


# ---------------------------------------------------------------------- #
# Metric metadata
# ---------------------------------------------------------------------- #


# Metrics that are non-negative by definition (W₁ ≥ 0, TV ≥ 0, time ≥ 0).
# Std bands around the mean are clipped at 0 for these so the rendered
# region stays geometrically valid (#42).
NON_NEGATIVE_METRICS = frozenset({
    "accuracy", "tv_per_node", "jsd_per_node", "w1_per_node",
    "cpd_accuracy", "total_time_s",
})


# Parameter-learning uses different accuracy metrics per family
# (different units; can't share a y-axis, but each panel has its own).
# Inference uses "accuracy" uniformly across families and is unaffected.
PARAM_LEARN_FAMILY_METRIC: Dict[str, str] = {
    "discrete":            "tv_per_node",
    "continuous_lg":       "w1_per_node",
    "continuous_nongauss": "w1_per_node",
    "hybrid":              "w1_per_node",
}


# ---------------------------------------------------------------------- #
# Stable per-baseline color and marker schemes
# ---------------------------------------------------------------------- #


# (library prefix, hue base) — colormap base hue per library.  Mechanism
# × engine variants spread across a small hue range using these bases.
_LIBRARY_HUES: Dict[str, str] = {
    "pgmpy":       "tab:blue",
    "nbn":         "tab:red",
    "gpytorch":    "tab:green",
    "pomegranate": "tab:purple",
    "pyro":        "tab:brown",
}

# B&w-safe marker shapes — distinguishable in monochrome printout.
_MARKERS: Sequence[str] = (
    "o", "s", "^", "v", "D", "P", "X", "*", ">", "<", "p", "h", "H", "+",
)


def _stable_baseline_style(baselines: Sequence[str]) -> Dict[str, dict]:
    """Assign each baseline a stable (color, marker, linestyle) triple.

    Color hue groups by library (matplotlib named colors with shaded
    variants).  Marker shape is unique per baseline (cycling
    deterministically through `_MARKERS`).  Linestyle distinguishes
    visually-similar baselines from the same library.

    Sorting by baseline name within each library ensures the same
    baseline always gets the same style across runs.
    """
    import matplotlib.colors as mcolors

    # Group baselines by library prefix, sort within group.
    by_lib: Dict[str, List[str]] = {}
    for b in baselines:
        prefix = b.split("-")[0]
        by_lib.setdefault(prefix, []).append(b)
    for lib in by_lib:
        by_lib[lib].sort()

    style: Dict[str, dict] = {}
    marker_idx = 0
    for lib, names in by_lib.items():
        base_color = _LIBRARY_HUES.get(lib, "tab:gray")
        # Generate shades of the base color: light → dark within the
        # group.  We convert the base color to RGB and interpolate to
        # white/dark per index.
        try:
            base_rgb = mcolors.to_rgb(base_color)
        except ValueError:
            base_rgb = mcolors.to_rgb("gray")
        n_in_lib = max(1, len(names))
        for i, name in enumerate(names):
            t = 0.35 + 0.55 * (i / max(1, n_in_lib - 1)) if n_in_lib > 1 else 0.6
            shade = tuple(min(1.0, c * t + (1 - t) * 0.5) for c in base_rgb)
            style[name] = {
                "color": shade,
                "marker": _MARKERS[marker_idx % len(_MARKERS)],
                "linestyle": ("-", "--", ":", "-.")[i % 4],
            }
            marker_idx += 1
    return style


# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #


def render_figures(
    parquet_path: str | Path,
    output_dir: str | Path,
    output_prefix: str,
    formats: Sequence[str] = ("png", "pdf", "svg"),
    log_y_for_time: bool = True,
    highlight_pareto: bool = False,
) -> Dict[str, List[str]]:
    """Render accuracy and total_time figures from a metrics parquet.

    Per-panel applicability filter: each panel's legend only lists
    baselines applicable to that panel's family per the registry.
    Mean ± std bands across seeds; b&w-safe markers; error/timeout
    cells indicated in the figure footer.

    Returns ``{'accuracy': [paths], 'total_time_s': [paths]}``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_parquet(parquet_path)
    out_dir = Path(output_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick the metric pair to render.  Inference uses
    # (accuracy, total_time_s); param-learning uses
    # (tv_per_node|w1_per_node, total_time_s) — fall through both.
    # v0.8 Pass-9: jsd_per_node is also stored in the parquet but
    # isn't rendered as an auto-figure to keep the figure naming
    # backward-compatible (view_name "accuracy_vs_size" produces one
    # file path).  Downstream consumers can read jsd_per_node rows
    # directly; a dedicated JSD figure-render is deferred.
    accuracy_metric = None
    for cand in ("accuracy", "tv_per_node", "w1_per_node", "cpd_accuracy"):
        if cand in df["metric"].unique():
            accuracy_metric = cand
            break
    metrics_to_render: List[str] = []
    if accuracy_metric is not None:
        metrics_to_render.append(accuracy_metric)
    if "total_time_s" in df["metric"].unique():
        metrics_to_render.append("total_time_s")

    out_paths: Dict[str, List[str]] = {}
    families = ["discrete", "continuous_lg", "continuous_nongauss", "hybrid"]
    baselines = sorted(df["baseline"].unique())
    style = _stable_baseline_style(baselines)
    n_nodes_list = sorted(df["n_nodes"].unique())

    for metric in metrics_to_render:
        figure_paths = _render_single_metric(
            df, metric=metric, families=families,
            baselines=baselines, style=style, n_nodes_list=n_nodes_list,
            out_dir=out_dir, output_prefix=output_prefix,
            formats=formats,
            log_y=(log_y_for_time and metric == "total_time_s"),
            highlight_pareto=highlight_pareto,
        )
        view_name = "accuracy_vs_size" if metric != "total_time_s" else "total_time_vs_size"
        out_paths[metric] = figure_paths
        plt.close("all")
        # Stash the view-name'd paths under the friendly key too for
        # downstream consumers.
        out_paths[view_name] = figure_paths

    return out_paths


def _render_single_metric(
    df: pd.DataFrame, *, metric: str, families: Sequence[str],
    baselines: Sequence[str], style: Dict[str, dict],
    n_nodes_list: Sequence[int], out_dir: Path, output_prefix: str,
    formats: Sequence[str], log_y: bool, highlight_pareto: bool,
) -> List[str]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False)
    flat_axes = axes.flatten()
    error_footnotes: List[str] = []

    for ax, family in zip(flat_axes, families):
        # Per-family metric resolution.  Parameter-learning uses
        # tv_per_node for discrete and w1_per_node for the others
        # (different units; can't share a y-axis, but each panel has
        # its own).  Inference and total-time are uniform across
        # families, so the else branch is the pre-fix path (#44).
        if metric in ("tv_per_node", "w1_per_node"):
            family_metric = PARAM_LEARN_FAMILY_METRIC.get(family, metric)
        else:
            family_metric = metric

        ax.set_title(f"{family}", fontsize=11, fontweight="bold")
        ax.set_xlabel("n_nodes")
        ylabel = (
            "Wasserstein-1 (W₁)" if family_metric in ("accuracy", "w1_per_node")
            else "Total variation (TV)" if family_metric == "tv_per_node"
            else "Jensen-Shannon (JSD/log 2)" if family_metric == "jsd_per_node"
            else "Wall-clock seconds" if family_metric == "total_time_s"
            else family_metric
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if log_y:
            ax.set_yscale("log")

        any_drawn = False
        for baseline in baselines:
            if not is_applicable(baseline, family):
                continue
            sub = df[
                (df["family"] == family)
                & (df["baseline"] == baseline)
                & (df["metric"] == family_metric)
            ]
            if sub.empty:
                continue
            # Aggregate across seeds → mean ± std per n_nodes.
            agg = (
                sub[sub["status"] == "ok"]
                .groupby("n_nodes")["value"]
                .agg(["mean", "std", "count"])
                .reindex(n_nodes_list)
            )
            agg["std"] = agg["std"].fillna(0.0)
            xs = np.array(agg.index, dtype=float)
            ys = agg["mean"].to_numpy(dtype=float)
            stds = agg["std"].to_numpy(dtype=float)

            sty = style[baseline]
            ax.plot(
                xs, ys,
                label=baseline,
                color=sty["color"], marker=sty["marker"],
                linestyle=sty["linestyle"], markersize=7, linewidth=1.5,
            )
            # Std band; clip lower bound at 0 for non-negative metrics
            # so the rendered region stays geometrically valid (#42).
            lower = ys - stds
            if family_metric in NON_NEGATIVE_METRICS:
                lower = np.maximum(lower, 0.0)
            ax.fill_between(
                xs, lower, ys + stds,
                color=sty["color"], alpha=0.18, linewidth=0,
            )
            any_drawn = True

        # Tally error cells for the footer.
        err_sub = df[
            (df["family"] == family)
            & (df["status"].isin(["error", "timeout", "oom"]))
        ]
        if not err_sub.empty:
            err_grouped = err_sub.groupby(
                ["baseline", "n_nodes", "status"],
            ).size()
            for (baseline, n, status), _cnt in err_grouped.items():
                error_footnotes.append(
                    f"  {family} n={n} {baseline}: {status}"
                )

        if any_drawn:
            ax.legend(loc="best", fontsize=8, framealpha=0.85)
        else:
            ax.text(
                0.5, 0.5, "(no applicable baselines)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, alpha=0.5,
            )

    # Figure-level title.
    metric_label = (
        "accuracy" if metric in ("accuracy", "tv_per_node", "jsd_per_node",
                                  "w1_per_node", "cpd_accuracy")
        else "total query time"
    )
    fig.suptitle(
        f"{output_prefix} — {metric_label} vs network size",
        fontsize=13, fontweight="bold",
    )

    view_name = "accuracy_vs_size" if metric != "total_time_s" else "total_time_vs_size"

    # DNF sidecar + corner annotation (#49): the multi-line footer that
    # used to render via ``fig.text(0.02, 0.005, ...)`` overlapped axes
    # on dense-DNF panels (continuous_nongauss especially).  The full DNF
    # list now goes to ``<prefix>_<view>_dnf.txt`` next to the figure;
    # the figure gains only a single-line corner annotation pointing at
    # the sidecar.
    if error_footnotes:
        sidecar_name = f"{output_prefix}_{view_name}_dnf.txt"
        sidecar_path = out_dir / sidecar_name
        sidecar_path.write_text(
            f"{output_prefix}_{view_name} — DNF cells\n\n"
            "Error/timeout/oom cells:\n"
            + "\n".join(error_footnotes)
            + "\n",
            encoding="utf-8",
        )
        # Annotation text uses ``*_dnf.txt`` (rather than the verbose
        # sidecar_name) so the rendered string is short enough to dodge
        # the centered suptitle on canonical figures (longest suptitle
        # right-edge ≈ 0.76 figure-fraction; verbose text would extend
        # left to 0.68 and collide).  The sidecar file on disk keeps
        # its full deterministic name so it's discoverable per-figure.
        fig.text(
            0.98, 0.98,
            f"DNF: {len(error_footnotes)} cells (see *_dnf.txt)",
            fontsize=7, alpha=0.6, ha="right", va="top",
        )

    fig.tight_layout(rect=[0, 0.01, 1, 0.96])

    written: List[str] = []
    for ext in formats:
        out = out_dir / f"{output_prefix}_{view_name}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        written.append(str(out))
    plt.close(fig)
    return written


__all__ = ["render_figures", "_stable_baseline_style"]
