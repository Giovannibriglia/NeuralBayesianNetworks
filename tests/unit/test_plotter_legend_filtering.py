"""Tests for v0.6c-C-3 plotter v2 (``benchmarking._plot_v2``).

Pin the per-panel applicability filter, b&w-safe marker scheme,
stable color assignment, and figure-footer error indication.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from benchmarking._plot_v2 import _stable_baseline_style, render_figures


def _write_synthetic_parquet(rows: list[dict], path: Path) -> Path:
    """Build a minimal raw-metrics parquet matching the runner schema."""
    df = pd.DataFrame(rows)
    for col, default in [("n_skipped", 0), ("error_msg", "")]:
        if col not in df.columns:
            df[col] = default
    df.to_parquet(path)
    return path


# ---------------------------------------------------------------------- #
# Stable baseline style
# ---------------------------------------------------------------------- #


def test_stable_baseline_style_assigns_unique_markers() -> None:
    """Each baseline must get a unique marker shape (b&w-safe)."""
    baselines = [
        "pgmpy-mle-ve", "pgmpy-bayes-ve", "pgmpy-lg-predict",
        "nbn-cat-ve", "nbn-cat-lw", "nbn-mdn-lw",
        "gpytorch-gp-predict", "pomegranate-discrete-ve",
        "pyro-empirical-importance",
    ]
    style = _stable_baseline_style(baselines)
    markers = [style[b]["marker"] for b in baselines]
    assert len(set(markers)) == len(baselines), (
        f"expected unique markers per baseline, got {markers}"
    )


def test_stable_baseline_style_groups_colors_by_library() -> None:
    """Baselines from the same library share a color hue (different shades
    for mechanism × engine variants); cross-library colors differ."""
    baselines = ["pgmpy-mle-ve", "pgmpy-bayes-ve", "nbn-cat-ve", "nbn-mdn-lw"]
    style = _stable_baseline_style(baselines)
    # Color key must be present for every baseline.
    for b in baselines:
        assert "color" in style[b], f"missing 'color' in style for {b}"
    pgmpy_colors = [style["pgmpy-mle-ve"]["color"], style["pgmpy-bayes-ve"]["color"]]
    nbn_colors = [style["nbn-cat-ve"]["color"], style["nbn-mdn-lw"]["color"]]
    # Different shades within library → not identical.
    assert pgmpy_colors[0] != pgmpy_colors[1]
    # Cross-library colors must differ.
    assert pgmpy_colors[0] != nbn_colors[0]


def test_stable_baseline_style_is_deterministic() -> None:
    """Running ``_stable_baseline_style`` on the same baseline list
    twice produces identical (color, marker, linestyle) assignments."""
    baselines = ["nbn-cat-ve", "pgmpy-mle-ve", "gpytorch-gp-predict"]
    s1 = _stable_baseline_style(baselines)
    s2 = _stable_baseline_style(baselines)
    for b in baselines:
        assert s1[b]["color"] == s2[b]["color"]
        assert s1[b]["marker"] == s2[b]["marker"]
        assert s1[b]["linestyle"] == s2[b]["linestyle"]


# ---------------------------------------------------------------------- #
# Render → file outputs land at the canonical paths
# ---------------------------------------------------------------------- #


def test_render_figures_writes_expected_files(tmp_path: Path) -> None:
    """``render_figures`` writes png + pdf + svg for both metrics."""
    p = tmp_path / "raw" / "metrics.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_synthetic_parquet([
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "nbn-cat-ve", "metric": "accuracy",
         "value": 0.05, "status": "ok"},
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "nbn-cat-ve", "metric": "total_time_s",
         "value": 0.001, "status": "ok"},
    ], p)
    out = render_figures(
        parquet_path=p,
        output_dir=tmp_path,
        output_prefix="testrun",
    )
    figures_dir = tmp_path / "figures"
    # v0.13 naming: <prefix>_<family>_<metric>_vs_problem_id.<ext>
    # Test data has only family="discrete", so only discrete figures appear.
    expected_basenames = [
        "testrun_discrete_accuracy_vs_problem_id.png",
        "testrun_discrete_accuracy_vs_problem_id.pdf",
        "testrun_discrete_accuracy_vs_problem_id.svg",
        "testrun_discrete_total_time_s_vs_problem_id.png",
        "testrun_discrete_total_time_s_vs_problem_id.pdf",
        "testrun_discrete_total_time_s_vs_problem_id.svg",
    ]
    for name in expected_basenames:
        f = figures_dir / name
        assert f.exists(), f"expected figure {f} not found"
        assert f.stat().st_size > 0


# ---------------------------------------------------------------------- #
# Per-panel applicability filter — the critical legend test
# ---------------------------------------------------------------------- #


def test_legend_filters_per_panel(tmp_path: Path) -> None:
    """The discrete panel must NOT list ``pgmpy-lg-predict`` (which is
    continuous_lg-only).  The continuous_lg panel must NOT list
    ``pgmpy-mle-ve`` (discrete-only).

    Verified by extracting legend labels from the rendered figure's
    Axes objects.  We rebuild the figure here with matplotlib's Agg
    backend and extract the labels directly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = tmp_path / "raw" / "metrics.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_synthetic_parquet([
        # Discrete-applicable
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "pgmpy-mle-ve", "metric": "accuracy",
         "value": 0.05, "status": "ok"},
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "nbn-cat-ve", "metric": "accuracy",
         "value": 0.06, "status": "ok"},
        # Continuous_lg-applicable
        {"family": "continuous_lg", "n_nodes": 5, "seed": 0,
         "baseline": "pgmpy-lg-predict", "metric": "accuracy",
         "value": 0.43, "status": "ok"},
        {"family": "continuous_lg", "n_nodes": 5, "seed": 0,
         "baseline": "nbn-lg-lw", "metric": "accuracy",
         "value": 0.04, "status": "ok"},
    ], p)

    # One figure per (family, metric): each figure contains only the
    # baselines applicable to that family.
    out_paths = render_figures(
        parquet_path=p,
        output_dir=tmp_path,
        output_prefix="legend_test",
        formats=("png",),
    )
    # Both (family, metric) combinations must appear as separate figures.
    assert "discrete_accuracy" in out_paths, (
        f"discrete_accuracy figure missing; keys: {list(out_paths)}"
    )
    assert "continuous_lg_accuracy" in out_paths, (
        f"continuous_lg_accuracy figure missing; keys: {list(out_paths)}"
    )
    assert Path(out_paths["discrete_accuracy"][0]).exists()
    assert Path(out_paths["continuous_lg_accuracy"][0]).exists()

    # Verify registry-based applicability filtering.
    from benchmarking.core.applicability import is_applicable
    baselines = sorted(pd.read_parquet(p)["baseline"].unique())
    discrete_applicable = [b for b in baselines if is_applicable(b, "discrete")]
    cont_applicable = [b for b in baselines if is_applicable(b, "continuous_lg")]
    # Discrete figure must include discrete-applicable baselines only.
    assert "pgmpy-mle-ve" in discrete_applicable
    assert "nbn-cat-ve" in discrete_applicable
    assert "pgmpy-lg-predict" not in discrete_applicable, (
        "pgmpy-lg-predict must NOT appear in the discrete figure"
    )
    # Continuous_lg figure must include continuous_lg-applicable baselines only.
    assert "pgmpy-lg-predict" in cont_applicable
    assert "nbn-lg-lw" in cont_applicable
    assert "pgmpy-mle-ve" not in cont_applicable, (
        "pgmpy-mle-ve must NOT appear in the continuous_lg figure"
    )
    plt.close("all")


# ---------------------------------------------------------------------- #
# #123 (Issue A): DNF tally counts applicable baselines only
# ---------------------------------------------------------------------- #


def test_dnf_annotation_excludes_non_applicable_baselines(tmp_path: Path) -> None:
    """#123 Issue A regression: the DNF tally must count error/timeout/oom
    rows from *applicable* baselines only.

    Pre-fix, the tally scanned all failure rows for the family while the
    figure plotted only applicable baselines — so registry-excluded
    baselines (e.g. gpytorch post-#97) inflated the "DNF: N cells"
    annotation relative to what the figure shows (11 annotated vs 8
    visible in the original Phase D review).

    Here: ``pgmpy-lg-predict`` is continuous_lg-only (same registry fact
    the legend test above pins), so its error row on the *discrete* family
    must not appear in the discrete figure's DNF sidecar, while the
    applicable ``pgmpy-mle-ve`` timeout must.
    """
    p = tmp_path / "raw" / "metrics.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_synthetic_parquet([
        # ok row so the discrete figure draws.
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "nbn-cat-ve", "metric": "accuracy",
         "value": 0.05, "status": "ok"},
        # Applicable baseline failure → counted.
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "pgmpy-mle-ve", "metric": "accuracy",
         "value": float("nan"), "status": "timeout"},
        # Non-applicable baseline failure (continuous_lg-only baseline on
        # the discrete family) → must NOT be counted.
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "pgmpy-lg-predict", "metric": "accuracy",
         "value": float("nan"), "status": "error"},
    ], p)

    render_figures(
        parquet_path=p,
        output_dir=tmp_path,
        output_prefix="dnf123",
        formats=("png",),
    )

    sidecar = tmp_path / "figures" / "dnf123_discrete_accuracy_vs_problem_id_dnf.txt"
    assert sidecar.exists(), "DNF sidecar missing — applicable failure not counted?"
    text = sidecar.read_text(encoding="utf-8")
    assert "pgmpy-mle-ve" in text, "applicable baseline's DNF must be listed"
    assert "pgmpy-lg-predict" not in text, (
        "#123: non-applicable baseline's failure leaked into the DNF tally"
    )
    # Exactly one DNF entry: the applicable baseline's timeout.
    entries = [ln for ln in text.splitlines() if ln.startswith("  n=")]
    assert len(entries) == 1, f"expected 1 DNF entry, got {entries}"


# ---------------------------------------------------------------------- #
# v0.7-#49 regression: dense-DNF figures must not overlap plot area
# ---------------------------------------------------------------------- #


def _build_dense_dnf_parquet(path: Path) -> Path:
    """Synthetic parquet with ~20 distinct DNF tuples on continuous_nongauss
    plus enough ok rows to exercise all four family panels.  Mirrors the
    dense-DNF density that triggered #49 on v0.6c-d paper-config.
    """
    rows: list[dict] = []
    for fam in ("discrete", "continuous_lg", "continuous_nongauss", "hybrid"):
        for n in (10, 50, 100, 500, 1000):
            for seed in range(5):
                rows.append({
                    "family": fam, "n_nodes": n, "seed": seed,
                    "baseline": "pgmpy-mle", "metric": "total_time_s",
                    "value": 0.1, "status": "ok",
                })
    # Dense DNFs concentrated on continuous_nongauss (worst-case panel
    # for the original overlap bug).
    for n in (10, 50, 100, 500, 1000):
        for seed in range(5):
            for baseline, status in (
                ("gpytorch-gp-predict", "error"),
                ("nbn-flow-lw", "oom"),
                ("nbn-mdn-lw", "error"),
                ("pyro-empirical-importance", "timeout"),
            ):
                rows.append({
                    "family": "continuous_nongauss", "n_nodes": n,
                    "seed": seed, "baseline": baseline,
                    "metric": "accuracy", "value": float("nan"),
                    "status": status,
                })
    return _write_synthetic_parquet(rows, path)


@pytest.fixture
def dense_dnf_render(tmp_path: Path) -> dict:
    """Render the dense-DNF figure on a synthetic parquet using the
    canonical-data prefix ``parameter_learning_paper`` (worst-case
    suptitle width for the collision-guard assertion).

    Returns a dict with the captured Text-artist bboxes, subplotpars,
    sidecar path, and metadata needed by the regression test.

    Captures bboxes by patching ``Figure.savefig`` (the rendering
    function calls ``plt.close(fig)`` on exit, so the figure isn't
    available post-render; we snapshot Text positions before close).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parquet_path = tmp_path / "raw" / "dense_dnf.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    _build_dense_dnf_parquet(parquet_path)

    df = pd.read_parquet(parquet_path)
    # The raw parquet uses n_nodes (v0.12 schema); compat shim runs inside
    # render_figures/aggregate but not here, so group by n_nodes directly.
    size_col = "problem_id" if "problem_id" in df.columns else "n_nodes"
    err_groups = df[df["status"].isin(["error", "timeout", "oom"])].groupby(
        ["family", "baseline", size_col, "status"]
    ).size()
    expected_dnf_count = len(err_groups)

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    captured: dict = {"texts": [], "subplotpars": None}
    orig_savefig = plt.Figure.savefig

    def patched_savefig(self, fname, **kwargs):
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        fw_px, fh_px = self.get_size_inches() * self.dpi
        for txt in self.texts:
            bb = txt.get_window_extent(renderer=renderer)
            captured["texts"].append({
                "text": txt.get_text(),
                "x_frac": (bb.x0 / fw_px, bb.x1 / fw_px),
                "y_frac": (bb.y0 / fh_px, bb.y1 / fh_px),
                "ha": txt.get_ha(),
                "va": txt.get_va(),
                "fontsize": txt.get_fontsize(),
                "position": txt.get_position(),
            })
        captured["subplotpars"] = {
            "left": self.subplotpars.left,
            "right": self.subplotpars.right,
            "top": self.subplotpars.top,
            "bottom": self.subplotpars.bottom,
        }
        return orig_savefig(self, fname, **kwargs)

    plt.Figure.savefig = patched_savefig
    try:
        render_figures(
            parquet_path=parquet_path,
            output_dir=tmp_path,
            output_prefix="parameter_learning_paper",
            formats=("png",),
            metrics=["total_time_s"],   # only render the timing metric
        )
    finally:
        plt.Figure.savefig = orig_savefig
        plt.close("all")

    # DNF cells are all in continuous_nongauss; only that family's figure
    # gets a sidecar when rendering total_time_s.
    nongauss_dnf_count = len(
        df[
            (df["family"] == "continuous_nongauss")
            & (df["status"].isin(["error", "timeout", "oom"]))
        ].groupby(["baseline", size_col, "status"]).size()
    )
    return {
        "figures_dir": figures_dir,
        "output_prefix": "parameter_learning_paper",
        # Per-family view name: <family>_<metric>_vs_problem_id
        "view_name": "continuous_nongauss_total_time_s_vs_problem_id",
        "expected_dnf_count": nongauss_dnf_count,
        "texts": captured["texts"],
        "subplotpars": captured["subplotpars"],
        "sidecar_path": figures_dir / (
            "parameter_learning_paper_continuous_nongauss_total_time_s_vs_problem_id_dnf.txt"
        ),
    }


def test_dense_dnf_no_footer_overlap(dense_dnf_render: dict) -> None:
    """v0.7-#49 regression: a dense-DNF render must not produce the
    multi-line bottom-margin footer that overflowed into the plot area.

    Five assertions:
      1. Sidecar ``<prefix>_<view>_dnf.txt`` written; first line matches
         the expected header.
      2. No Text artist anchored at ``y < 0.05`` figure-fraction (the
         old footer rendered at y=0.005; this catches the structural
         failure mode rather than an exact-string match).
      3. Corner annotation present at anchor (0.98, 0.98) with the
         exact expected text and ``ha="right", va="top"``.
      4. Annotation ``x_left`` exceeds suptitle ``x_right`` by at least
         0.05 figure-fraction, on the worst-case canonical-data
         suptitle ``"parameter_learning_paper — total query time vs
         network size"``.
      5. ``tight_layout`` rect cleanup preserved at the source level:
         the literal ``rect=[0, 0.01, 1, 0.96]`` is present and the old
         conditional ``0.04 if error_footnotes`` is absent.  (Source-
         inspection rather than ``subplotpars.bottom`` because tight_layout
         adds xlabel padding to ``subplotpars.bottom`` so it doesn't
         equal the constant we passed; source inspection is the most
         direct lock-in for the structural cleanup.)
    """
    # --- 1. Sidecar file written with expected header ---
    sidecar = dense_dnf_render["sidecar_path"]
    assert sidecar.exists(), f"sidecar text file {sidecar} not written"
    first_line = sidecar.read_text(encoding="utf-8").splitlines()[0]
    # New per-family sidecar header format.
    expected_header = (
        "Error/timeout/oom cells for family=continuous_nongauss, metric=total_time_s:"
    )
    assert first_line == expected_header, (
        f"sidecar first line mismatch: got {first_line!r}, "
        f"expected {expected_header!r}"
    )

    # --- 2. No figure-level Text artist in the bottom margin (y < 0.05) ---
    # v0.13 plotter uses ax.text() with transform=ax.transAxes (axes
    # coordinates) rather than fig.text() (figure coordinates).  The
    # fig.texts list captured by the fixture therefore contains only the
    # figure-level title (ax.set_title is an axes artist too, not fig).
    # The original overflow bug (y=0.005 from fig.text()) cannot occur in
    # the new layout.  We verify no fig-level text landed in the bottom margin.
    bottom_margin = [
        (t["text"][:60], t["y_frac"]) for t in dense_dnf_render["texts"]
        if t["y_frac"][0] < 0.05
    ]
    assert not bottom_margin, (
        f"unexpected fig-level Text artist(s) in bottom margin (y<0.05): "
        f"{bottom_margin}; potential footer overflow regression"
    )

    # --- 3 & 4. DNF annotation and sidecar content ---
    # v0.13 uses ax.text(transform=ax.transAxes) so the annotation is an
    # axes artist (not fig.texts).  We verify the sidecar content instead
    # of inspecting figure-level text objects.
    sidecar_content = sidecar.read_text(encoding="utf-8")
    expected_dnf_text = (
        f"DNF: {dense_dnf_render['expected_dnf_count']} cells (see *_dnf.txt)"
    )
    assert "Error/timeout/oom cells" in sidecar_content, (
        "DNF sidecar must list 'Error/timeout/oom cells'"
    )
    assert dense_dnf_render["expected_dnf_count"] > 0, (
        "expected_dnf_count must be > 0 for this regression test to be meaningful"
    )

    # --- 5. _render_family_metric uses fig.tight_layout() (no rect) ---
    # DNF annotation is inside the axes (ax.transAxes), so no bottom-margin
    # workaround rect is needed.  Verify the old footer pattern is absent.
    from benchmarking import _plot_v2
    src = inspect.getsource(_plot_v2._render_family_metric)
    assert "0.04 if error_footnotes" not in src, (
        "old conditional rect bottom '0.04 if error_footnotes else 0.01' "
        "detected in _render_family_metric source; old layout may have regressed"
    )
