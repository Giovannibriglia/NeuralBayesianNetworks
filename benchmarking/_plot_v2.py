"""v0.13 plotter — one figure per (family, metric).

Produces one figure per (family, metric) combination present in the
parquet.  Each figure shows only baselines applicable to that family,
scaled to that family's data range.

Layout:
  - x-axis: problem_id values (numeric for synthetic, string for bnlearn).
  - y-axis: mean metric value across seeds; ±1σ band.
  - Color encodes **baseline library** (stable hue per library, shaded
    variants within the library group — the v0.12 b&w-safe scheme).
  - Marker + linestyle also encode baseline (unique per baseline).
  - Legend inside the axes (≤6 baselines per family figure; fits cleanly).

File naming::

    {output_prefix}_{family}_{metric}_vs_problem_id.{ext}

Example::

    inference_complete_discrete_tv_per_node_vs_problem_id.pdf

Return dict key: ``"{family}_{metric}"`` (e.g., ``"discrete_tv_per_node"``).

DNF sidecar (``*_dnf.txt``) lists error/timeout/oom cells for the
specific (family, metric) figure.

Backward compatibility: v0.12 parquets that have ``n_nodes`` (int)
instead of ``problem_id`` (str) are transparently up-cast at load time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from benchmarking.core.applicability import is_applicable


# ---------------------------------------------------------------------------
# Metric display metadata
# ---------------------------------------------------------------------------

# y-axis label per metric name.
METRIC_YLABELS: Dict[str, str] = {
    "accuracy":       "Accuracy (W₁ or TV, lower is better)",
    "tv_per_node":    "TV per node (lower is better)",
    "jsd_per_node":   "JSD / log 2 per node (lower is better)",
    "w1_per_node":    "W₁ per node (lower is better)",
    "total_time_s":   "Total query time (s)",
    "fit_time_s":     "Fit time (s)",
    "query_time_s":   "Query time per query (s)",
    "metrics_time_s": "Metrics scoring time (s)",
    "cpd_accuracy":   "CPD accuracy",
}

# Metrics where lower bound of std band is clipped at 0 (non-negative).
NON_NEGATIVE_METRICS = frozenset({
    "accuracy", "tv_per_node", "jsd_per_node", "w1_per_node",
    "cpd_accuracy", "total_time_s", "fit_time_s", "query_time_s", "metrics_time_s",
})

# Metrics rendered with a log y-axis by default.
LOG_Y_METRICS = frozenset({"total_time_s", "fit_time_s", "query_time_s"})


# ---------------------------------------------------------------------------
# Stable per-baseline color + marker + linestyle (v0.12 library-hue scheme)
# ---------------------------------------------------------------------------

# Base hue per library prefix.  Colors group by library; shaded variants
# within the group distinguish mechanism × engine combinations.
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
    deterministically through ``_MARKERS``).  Linestyle distinguishes
    visually-similar baselines from the same library.

    Sorting by baseline name within each library group ensures the same
    baseline always gets the same style across runs.
    """
    import matplotlib.colors as mcolors

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_figures(
    parquet_path: str | Path,
    output_dir: str | Path,
    output_prefix: str,
    formats: Sequence[str] = ("png", "pdf", "svg"),
    log_y_for_time: bool = True,
    metrics: Sequence[str] | None = None,
) -> Dict[str, List[str]]:
    """Render one figure per (family, metric) pair from a metrics parquet.

    Each figure shows only the baselines applicable to that family,
    scaled to that family's data range — solving the y-axis scale
    incompatibility between families (e.g., discrete TV ≈ 0.05 vs
    continuous W₁ ≈ 0.2–1.4 cannot share an axis).

    Parameters
    ----------
    parquet_path:
        Raw metrics parquet (v3 schema with ``problem_id``; or v0.12
        schema with ``n_nodes`` — transparently up-cast).
    output_dir:
        Root output directory.  Figures land in ``{output_dir}/figures/``.
    output_prefix:
        Run label prefix (e.g. ``"inference_complete"``).
    formats:
        File formats to produce per figure.  Default: png + pdf + svg.
    log_y_for_time:
        Apply log scale to timing metrics.
    metrics:
        Subset of metric names to render.  ``None`` renders all metrics
        present in the parquet except the ``"status"`` sentinel.

    Returns
    -------
    dict[str, list[str]]
        Written file paths keyed by ``"{family}_{metric}"``
        (e.g., ``"discrete_tv_per_node"``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_parquet(parquet_path)

    # Backward-compat: v0.12 parquets use n_nodes (int); promote to problem_id (str).
    if "n_nodes" in df.columns and "problem_id" not in df.columns:
        df = df.rename(columns={"n_nodes": "problem_id"})
        df["problem_id"] = df["problem_id"].astype(str)

    out_dir = Path(output_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = [m for m in df["metric"].unique() if m != "status"]
    metrics_to_render = list(metrics) if metrics is not None else all_metrics

    problem_ids = sorted(
        df["problem_id"].unique(),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, x),
    )
    baselines = sorted(df["baseline"].unique())
    families  = sorted(df["family"].unique())
    style = _stable_baseline_style(baselines)

    out_paths: Dict[str, List[str]] = {}
    for family in families:
        for metric in metrics_to_render:
            if metric not in df["metric"].unique():
                continue
            sub = df[(df["family"] == family) & (df["metric"] == metric)]
            if sub.empty:
                continue
            log_y = log_y_for_time and metric in LOG_Y_METRICS
            key = f"{family}_{metric}"
            figure_paths = _render_family_metric(
                df=df,
                family=family,
                metric=metric,
                baselines=baselines,
                style=style,
                problem_ids=problem_ids,
                out_dir=out_dir,
                output_prefix=output_prefix,
                formats=formats,
                log_y=log_y,
            )
            out_paths[key] = figure_paths
            plt.close("all")

    return out_paths


def _render_family_metric(
    df: pd.DataFrame,
    *,
    family: str,
    metric: str,
    baselines: Sequence[str],
    style: Dict[str, dict],
    problem_ids: Sequence[str],
    out_dir: Path,
    output_prefix: str,
    formats: Sequence[str],
    log_y: bool,
) -> List[str]:
    """Render one figure for a single (family, metric) pair.

    Only baselines applicable to ``family`` are plotted.  Color, marker,
    and linestyle come from the baseline (library-hue scheme) rather than
    from the family, so each baseline is visually distinct within the
    figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # x-axis: numeric when all problem_ids are digit strings.
    if all(str(p).isdigit() for p in problem_ids):
        xs = [float(p) for p in problem_ids]
        ax.set_xlabel("n_nodes (problem_id)")
    else:
        xs = list(range(len(problem_ids)))
        ax.set_xticks(xs)
        ax.set_xticklabels([str(p) for p in problem_ids], rotation=15, ha="right")
        ax.set_xlabel("problem_id")

    xs_map = dict(zip(problem_ids, xs))

    ylabel = METRIC_YLABELS.get(metric, metric)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{output_prefix} — {family} — {metric}",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.set_yscale("log")

    any_drawn = False
    error_footnotes: List[str] = []

    for baseline in baselines:
        if not is_applicable(baseline, family):
            continue

        sub = df[
            (df["family"] == family)
            & (df["baseline"] == baseline)
            & (df["metric"] == metric)
        ]
        if sub.empty:
            continue

        agg = (
            sub[sub["status"] == "ok"]
            .groupby("problem_id")["value"]
            .agg(["mean", "std"])
            .reindex(problem_ids)
        )
        agg["std"] = agg["std"].fillna(0.0)

        ys   = agg["mean"].to_numpy(dtype=float)
        stds = agg["std"].to_numpy(dtype=float)
        xs_plot = [xs_map[p] for p in problem_ids]

        sty = style.get(baseline, {"color": "tab:gray", "marker": "o", "linestyle": "-"})
        ax.plot(
            xs_plot, ys,
            color=sty["color"],
            marker=sty["marker"],
            linestyle=sty["linestyle"],
            markersize=6,
            linewidth=1.5,
            label=baseline,
        )

        lower = ys - stds
        if metric in NON_NEGATIVE_METRICS:
            lower = np.maximum(lower, 0.0)
        ax.fill_between(
            xs_plot, lower, ys + stds,
            color=sty["color"], alpha=0.18, linewidth=0,
        )
        any_drawn = True

    # DNF cells for this specific (family, metric).
    err_sub = df[
        (df["family"] == family)
        & (df["status"].isin(["error", "timeout", "oom"]))
    ]
    if not err_sub.empty:
        err_grouped = err_sub.groupby(["baseline", "problem_id", "status"]).size()
        for (bl, pid, st), _ in err_grouped.items():
            error_footnotes.append(f"  n={pid} {bl}: {st}")

    if not any_drawn:
        ax.text(
            0.5, 0.5, "(no applicable baselines for this family)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, alpha=0.5,
        )
    else:
        ax.legend(loc="best", fontsize=8, framealpha=0.85)

    # DNF sidecar + corner annotation.
    view_name = f"{family}_{metric}_vs_problem_id"
    if error_footnotes:
        sidecar_name = f"{output_prefix}_{view_name}_dnf.txt"
        sidecar_path = out_dir / sidecar_name
        sidecar_path.write_text(
            f"Error/timeout/oom cells for family={family}, metric={metric}:\n"
            + "\n".join(error_footnotes)
            + "\n",
            encoding="utf-8",
        )
        ax.text(
            0.98, 0.98,
            f"DNF: {len(error_footnotes)} cells (see *_dnf.txt)",
            fontsize=7, alpha=0.6, ha="right", va="top",
            transform=ax.transAxes,
        )

    fig.tight_layout()

    written: List[str] = []
    for ext in formats:
        fname = f"{output_prefix}_{view_name}.{ext}"
        fig.savefig(out_dir / fname, dpi=120, bbox_inches="tight")
        written.append(str(out_dir / fname))
    plt.close(fig)
    return written


__all__ = ["render_figures", "_stable_baseline_style"]
