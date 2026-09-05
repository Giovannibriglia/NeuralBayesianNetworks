"""Tests for the batch-speed paper figure (PR 6, #148).

Uses the testable builder (`_build_batch_speed_figure`) for structural
assertions (facets, log axes, DNF annotations) and the saving wrapper
(`fig_batch_speed`) + `run_plot` for file-creation checks, including a
round-trip from the PR 5 speed-smoke config.

Reference: docs/v0.14-batched-queries-design.md §6.PR6;
docs/v0.13-paper-figures.md §5b.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

import math as _math

import numpy as np

from nbn.bench._paper_figures import (
    _build_batch_speed_figure,
    _metric_cell,
    batch_speed_tables,
    fig_batch_speed,
    run_plot,
    table_headline,
)


def _row(
    *, family="discrete", baseline="nbn-cat-ve", batch_size=1, seed=0,
    value=0.01, status="ok", metric="query_time_s",
):
    return {
        "benchmark": "synthetic", "family": family, "problem_id": "100",
        "seed": seed, "baseline": baseline, "query_role": "hub",
        "query_kind": "prediction", "metric": metric, "value": value,
        "status": status, "fit_time_s": 1.0, "query_time_s": value,
        "metrics_time_s": 0.0, "error_msg": None, "batch_size": batch_size,
    }


def _make_sweep_df(families=("discrete", "hybrid")) -> pd.DataFrame:
    """Swept nbn-cat-ve at B in {1, 4, 16}; pinned pgmpy at B=1 only."""
    rows = []
    for fam in families:
        for bs in (1, 4, 16):
            for seed in (0, 1):
                rows.append(_row(
                    family=fam, baseline="nbn-cat-ve", batch_size=bs,
                    seed=seed, value=0.01 / bs,
                ))
        for seed in (0, 1):
            rows.append(_row(
                family=fam, baseline="pgmpy-mle-ve", batch_size=1,
                seed=seed, value=0.005,
            ))
    return pd.DataFrame(rows)


class TestBuilder:
    def test_facet_count_matches_families(self):
        df = _make_sweep_df(families=("discrete", "hybrid", "continuous_lg"))
        fig, dnf = _build_batch_speed_figure(df, "iqm_iqr", "synthetic")
        assert fig is not None
        assert len(fig.axes) == 3
        assert dnf == []

    def test_log_axes(self):
        fig, _ = _build_batch_speed_figure(_make_sweep_df(), "iqm_iqr", "t")
        for ax in fig.axes:
            assert ax.get_xscale() == "log"
            assert ax.get_yscale() == "log"

    def test_dnf_annotation_for_failed_batch_size(self):
        """A baseline with only timeout rows at B=512 (no ok data) gets a
        DNF annotation and sidecar line for that (baseline, batch_size)."""
        df = _make_sweep_df(families=("discrete",))
        df = pd.concat([df, pd.DataFrame([
            _row(family="discrete", baseline="nbn-cat-ve", batch_size=512,
                 status="timeout", value=float("nan")),
        ])], ignore_index=True)

        fig, dnf = _build_batch_speed_figure(df, "iqm_iqr", "t")
        assert len(dnf) == 1
        assert "B=512" in dnf[0] and "nbn-cat-ve" in dnf[0] and "timeout" in dnf[0]
        # Corner annotation present on the affected facet.
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("DNF: 1 cells" in t for t in texts)

    def test_empty_or_unbatched_input_skips(self):
        no_col = pd.DataFrame([{
            k: v for k, v in _row().items() if k != "batch_size"
        }])
        assert _build_batch_speed_figure(no_col, "iqm_iqr", "t") == (None, [])
        no_ok = _make_sweep_df()
        no_ok["status"] = "error"
        fig, _ = _build_batch_speed_figure(no_ok, "iqm_iqr", "t")
        assert fig is None


class TestRender:
    def test_renders_file(self, tmp_path):
        out = tmp_path / "batch_speed.pdf"
        fig_batch_speed(_make_sweep_df(), "iqm_iqr", out, "synthetic")
        assert out.exists() and out.stat().st_size > 0

    def test_dnf_sidecar_written(self, tmp_path):
        df = pd.concat([_make_sweep_df(), pd.DataFrame([
            _row(baseline="nbn-cat-ve", batch_size=512,
                 status="oom", value=float("nan")),
        ])], ignore_index=True)
        out = tmp_path / "batch_speed.pdf"
        fig_batch_speed(df, "iqm_iqr", out, "synthetic")
        sidecar = tmp_path / "batch_speed_dnf.txt"
        assert sidecar.exists()
        assert "B=512" in sidecar.read_text()

    def test_run_plot_autodetects_sweep_parquet(self, tmp_path):
        """run_plot renders batch_speed.pdf iff batched rows exist."""
        df = _make_sweep_df()
        pq = tmp_path / "x_metrics.parquet"
        df.to_parquet(pq, index=False)
        out = tmp_path / "figs"
        assert run_plot(pq, out) == 0
        assert (out / "synthetic" / "batch_speed.pdf").exists()

    def test_run_plot_skips_unbatched_parquet(self, tmp_path):
        df = _make_sweep_df()
        df["batch_size"] = 1  # no sweep
        pq = tmp_path / "x_metrics.parquet"
        df.to_parquet(pq, index=False)
        out = tmp_path / "figs"
        assert run_plot(pq, out) == 0
        assert not (out / "synthetic" / "batch_speed.pdf").exists()


@pytest.mark.slow
class TestSpeedSmokeRoundTrip:
    def test_render_from_speed_smoke_run(self, tmp_path):
        """Run the PR 5 speed smoke end-to-end, then render the figure
        from its rows via run_plot."""
        from nbn.bench.cli import _run_cells
        from nbn.bench.core.yaml_config import load_runner_config

        jsonl = tmp_path / "metrics.jsonl"
        cfg = load_runner_config(
            "nbn/bench/configs/synthetic/speed/inference_speed_smoke.yaml",
            device_override="cpu", jsonl_path=jsonl,
        )
        _run_cells(cfg)
        rows = [json.loads(line)
                for line in Path(jsonl).read_text().splitlines() if line.strip()]
        df = pd.DataFrame(rows)
        pq = tmp_path / "speed_metrics.parquet"
        df.to_parquet(pq, index=False)

        out = tmp_path / "figs"
        assert run_plot(pq, out) == 0
        fig_path = out / "synthetic" / "batch_speed.pdf"
        assert fig_path.exists() and fig_path.stat().st_size > 0

    def test_amortization_visible_in_builder_points(self):
        """Sanity: synthetic decreasing per-query times produce finite,
        positive aggregated values (no NaN/log-domain issues)."""
        df = _make_sweep_df()
        fig, _ = _build_batch_speed_figure(df, "iqm_iqr", "t")
        for ax in fig.axes:
            for line in ax.get_lines():
                ys = line.get_ydata()
                # axhline reference lines are constant (and excluded below);
                # batchable lines may carry NaN gaps — check the finite ones.
                finite = [y for y in ys if not math.isnan(y)]
                assert finite and all(y > 0 for y in finite)


def _solid_line_for(ax, baseline):
    """The batchable (solid, marker) line for a baseline, or None."""
    for ln in ax.get_lines():
        if ln.get_label() == baseline and ln.get_linestyle() == "-":
            return ln
    return None


class TestChange1PointExclusion:
    """A mid-sweep failure excludes only that batch size (NaN gap); larger
    ok batch sizes are NOT dropped."""

    def test_gap_at_failed_batch_size_keeps_larger(self):
        rows = []
        # nbn-cat-ve: ok at B in {1, 8, 64}, OOM at B=16.
        for bs in (1, 8, 64):
            for seed in (0, 1):
                rows.append(_row(baseline="nbn-cat-ve", batch_size=bs,
                                 seed=seed, value=0.02 / bs))
        for seed in (0, 1):
            rows.append(_row(baseline="nbn-cat-ve", batch_size=16, seed=seed,
                             status="oom", value=float("nan")))
        df = pd.DataFrame(rows)

        fig, _ = _build_batch_speed_figure(df, "iqm_iqr", "t")
        ax = fig.axes[0]
        line = _solid_line_for(ax, "nbn-cat-ve")
        assert line is not None
        xs = list(line.get_xdata())
        ys = list(line.get_ydata())
        grid = dict(zip(xs, ys))
        assert 16 in grid and _math.isnan(grid[16])          # gap at the OOM
        assert grid[64] > 0 and not _math.isnan(grid[64])    # larger kept
        assert not _math.isnan(grid[1]) and not _math.isnan(grid[8])


class TestChange2DashedReference:
    """Non-batchable baselines render as a dashed horizontal reference when ok
    at B=1, and not at all when they DNF at B=1. A batchable baseline that
    fails above B=1 is NOT mistaken for a reference line."""

    def _dashed(self, ax):
        return [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]

    def test_pinned_ok_draws_dashed_line(self):
        df = _make_sweep_df(families=("discrete",))  # pgmpy-mle-ve ok at B=1
        fig, _ = _build_batch_speed_figure(df, "iqm_iqr", "t")
        dashed = self._dashed(fig.axes[0])
        assert [ln.get_label() for ln in dashed] == ["pgmpy-mle-ve"]

    def test_pinned_dnf_draws_nothing(self):
        df = _make_sweep_df(families=("discrete",))
        # Replace pgmpy's ok B=1 rows with a timeout (no ok data anywhere).
        df = df[df["baseline"] != "pgmpy-mle-ve"]
        df = pd.concat([df, pd.DataFrame([
            _row(baseline="pgmpy-mle-ve", batch_size=1, seed=0,
                 status="timeout", value=float("nan")),
        ])], ignore_index=True)
        fig, _ = _build_batch_speed_figure(df, "iqm_iqr", "t")
        assert self._dashed(fig.axes[0]) == []

    def test_batchable_failing_above_b1_is_not_dashed(self):
        """nbn-cat-ve ok only at B=1 (OOM at 8, 16) must stay a point/line,
        never a dashed reference — detection is by library, not observed data."""
        rows = [_row(baseline="nbn-cat-ve", batch_size=1, seed=s, value=0.01)
                for s in (0, 1)]
        for bs in (8, 16):
            for s in (0, 1):
                rows.append(_row(baseline="nbn-cat-ve", batch_size=bs, seed=s,
                                 status="oom", value=float("nan")))
        df = pd.DataFrame(rows)
        fig, _ = _build_batch_speed_figure(df, "iqm_iqr", "t")
        assert self._dashed(fig.axes[0]) == []            # no false reference
        assert _solid_line_for(fig.axes[0], "nbn-cat-ve") is not None


class TestChange3Tables:
    def test_table_cells(self, tmp_path):
        rows = []
        # Batchable nbn-cat-ve: ok at B in {1, 8}, OOM at B=16.
        for bs in (1, 8):
            for seed in (0, 1):
                rows.append(_row(baseline="nbn-cat-ve", batch_size=bs,
                                 seed=seed, value=0.02 / bs))
        for seed in (0, 1):
            rows.append(_row(baseline="nbn-cat-ve", batch_size=16, seed=seed,
                             status="oom", value=float("nan")))
        # Non-batchable pgmpy: ok at B=1 only.
        for seed in (0, 1):
            rows.append(_row(baseline="pgmpy-mle-ve", batch_size=1, seed=seed,
                             value=0.005))
        df = pd.DataFrame(rows)

        n = batch_speed_tables(df, "iqm_iqr", tmp_path, "synthetic")
        assert n == 1
        tex = (tmp_path / "batch_speed_table_discrete.tex").read_text()

        assert "\\toprule" in tex and "$B=1$" in tex and "$B=16$" in tex
        cat_row = [ln for ln in tex.splitlines()
                   if ln.startswith("nbn-cat-ve")][0]
        cells = [c.strip() for c in cat_row.rstrip("\\").split("&")]
        # Method, B=1, B=8, B=16
        assert "$\\pm$" in cells[1] and "$\\pm$" in cells[2]   # ok values
        assert cells[3] == "oom"                               # failed cell
        pg_row = [ln for ln in tex.splitlines()
                  if ln.startswith("pgmpy-mle-ve")][0]
        pcells = [c.strip() for c in pg_row.rstrip("\\").split("&")]
        assert "$\\pm$" in pcells[1]                           # B=1 value
        assert pcells[2] == "--" and pcells[3] == "--"         # non-batchable B>1

    def test_pinned_timeout_shows_code(self, tmp_path):
        rows = [_row(baseline="nbn-cat-ve", batch_size=bs, seed=s,
                     value=0.02 / bs) for bs in (1, 8) for s in (0, 1)]
        rows.append(_row(baseline="pyro-empirical-importance", batch_size=1,
                         seed=0, status="timeout", value=float("nan")))
        df = pd.DataFrame(rows)
        batch_speed_tables(df, "iqm_iqr", tmp_path, "synthetic")
        tex = (tmp_path / "batch_speed_table_discrete.tex").read_text()
        pyro_row = [ln for ln in tex.splitlines()
                    if ln.startswith("pyro-empirical-importance")][0]
        cells = [c.strip() for c in pyro_row.rstrip("\\").split("&")]
        assert cells[1] == "timeout" and cells[2] == "--"


class TestChange4Aggregation:
    """--aggregation threads identically into plot and tables; iqm and mean
    diverge on a skewed seed distribution."""

    def _skewed_df(self):
        # 5 seeds, one heavy outlier: IQM trims it (->~1), mean does not (~20.8).
        vals = [1.0, 1.0, 1.0, 1.0, 100.0]
        rows = []
        for bs in (1, 8):
            for seed, v in enumerate(vals):
                rows.append(_row(baseline="nbn-cat-ve", batch_size=bs,
                                 seed=seed, value=v))
        return pd.DataFrame(rows)

    def test_table_values_differ_by_agg(self, tmp_path):
        df = self._skewed_df()
        out_i, out_m = tmp_path / "iqm", tmp_path / "mean"
        batch_speed_tables(df, "iqm_iqr", out_i, "synthetic")
        batch_speed_tables(df, "mean_std", out_m, "synthetic")
        tex_i = (out_i / "batch_speed_table_discrete.tex").read_text()
        tex_m = (out_m / "batch_speed_table_discrete.tex").read_text()

        def _b1(tex):
            row = [ln for ln in tex.splitlines()
                   if ln.startswith("nbn-cat-ve")][0]
            return [c.strip() for c in row.rstrip("\\").split("&")][1]

        assert _b1(tex_i) != _b1(tex_m)
        # agg name appears in the caption with its underscore escaped (Change A).
        assert "iqm\\_iqr" in tex_i and "mean\\_std" in tex_m

    def test_plot_center_differs_by_agg(self):
        df = self._skewed_df()
        fig_i, _ = _build_batch_speed_figure(df, "iqm_iqr", "t")
        fig_m, _ = _build_batch_speed_figure(df, "mean_std", "t")
        yi = _solid_line_for(fig_i.axes[0], "nbn-cat-ve").get_ydata()
        ym = _solid_line_for(fig_m.axes[0], "nbn-cat-ve").get_ydata()
        assert not np.allclose(np.asarray(yi), np.asarray(ym))


# --- PR: table float wrapper (Change A) + seed invalidation (Change B) --------

class TestChangeATableFloat:
    """All tables are wrapped in a `table` float with \\caption + \\label;
    the batch_speed legend now lives in the caption, not a tabular row."""

    def test_batch_speed_table_is_float_with_caption_label(self, tmp_path):
        df = _make_sweep_df(families=("discrete",))
        assert batch_speed_tables(df, "iqm_iqr", tmp_path, "synthetic") == 1
        tex = (tmp_path / "batch_speed_table_discrete.tex").read_text()
        assert "\\begin{table}[t]" in tex and "\\end{table}" in tex
        assert "\\centering" in tex
        assert "\\label{tab:synthetic_batch_speed_discrete}" in tex
        assert "\\multicolumn" not in tex          # legend no longer in tabular
        cap = [ln for ln in tex.splitlines() if ln.startswith("\\caption{")][0]
        assert "\\texttt{oom}" in cap              # legend moved into caption

    def test_accuracy_table_also_wrapped_and_escaped(self, tmp_path):
        """Universal: an accuracy table (table_headline) emits the float too,
        with underscores in the caption escaped."""
        rows = []
        for seed in (0, 1):
            rows.append(_row(metric="tv_per_node", value=0.1, seed=seed))
            rows.append(_row(metric="query_time_s", value=0.01, seed=seed))
        out = tmp_path / "table_headline.tex"
        table_headline(pd.DataFrame(rows), "discrete", "small", "iqm_iqr",
                       ["hub"], out, label="tab:synthetic_discrete_small_headline")
        tex = out.read_text()
        assert "\\begin{table}[t]" in tex and "\\end{table}" in tex
        assert "\\caption{" in tex
        assert "\\label{tab:synthetic_discrete_small_headline}" in tex
        assert "iqm\\_iqr" in tex and "iqm_iqr" not in tex.replace("iqm\\_iqr", "")


class TestChangeBSeedInvalidation:
    """Speed-only: a (baseline, batch_size) cell with ANY failed seed is fully
    failed — code in the table, NaN gap in the plot — reversing the prior
    'ok takes precedence' behavior."""

    def _mixed_df(self, fail_status="oom"):
        # nbn-cat-ve B=1 all ok; B=8 seed0 FAILED, seeds 1,2 ok.
        rows = [_row(baseline="nbn-cat-ve", batch_size=1, seed=s, value=0.02)
                for s in (0, 1, 2)]
        rows.append(_row(baseline="nbn-cat-ve", batch_size=8, seed=0,
                         status=fail_status, value=float("nan")))
        rows += [_row(baseline="nbn-cat-ve", batch_size=8, seed=s, value=0.005)
                 for s in (1, 2)]
        return pd.DataFrame(rows)

    def _cells(self, tex):
        row = [ln for ln in tex.splitlines() if ln.startswith("nbn-cat-ve")][0]
        return [c.strip() for c in row.rstrip("\\").split("&")]

    def test_table_partial_failure_shows_code(self, tmp_path):
        batch_speed_tables(self._mixed_df("oom"), "iqm_iqr", tmp_path, "synthetic")
        cells = self._cells((tmp_path / "batch_speed_table_discrete.tex").read_text())
        assert "$\\pm$" in cells[1]            # B=1 all ok -> value
        assert cells[2] == "oom"               # B=8 partial fail -> code, NOT survivor

    def test_table_all_ok_shows_value(self, tmp_path):
        rows = [_row(baseline="nbn-cat-ve", batch_size=bs, seed=s, value=0.02 / bs)
                for bs in (1, 8) for s in (0, 1, 2)]
        batch_speed_tables(pd.DataFrame(rows), "iqm_iqr", tmp_path, "synthetic")
        cells = self._cells((tmp_path / "batch_speed_table_discrete.tex").read_text())
        assert "$\\pm$" in cells[1] and "$\\pm$" in cells[2]

    def test_plot_partial_failure_no_point(self):
        fig, _ = _build_batch_speed_figure(self._mixed_df("oom"), "iqm_iqr", "t")
        line = _solid_line_for(fig.axes[0], "nbn-cat-ve")
        grid = dict(zip(line.get_xdata(), line.get_ydata()))
        assert not _math.isnan(grid[1])        # B=1 all ok -> point
        assert _math.isnan(grid[8])            # B=8 partial fail -> gap

    def test_forward_compat_single_failed_row(self, tmp_path):
        """Post-PR-2 the runner may emit only the failed seed's row; the
        'any failure row' rule still classifies the cell as failed."""
        rows = [_row(baseline="nbn-cat-ve", batch_size=1, seed=s, value=0.02)
                for s in (0, 1)]
        rows.append(_row(baseline="nbn-cat-ve", batch_size=8, seed=0,
                         status="oom", value=float("nan")))   # single row at B=8
        batch_speed_tables(pd.DataFrame(rows), "iqm_iqr", tmp_path, "synthetic")
        cells = self._cells((tmp_path / "batch_speed_table_discrete.tex").read_text())
        assert cells[2] == "oom"


class TestChangeBSpeedOnly:
    """Accuracy tables keep current per-seed survivor behavior (Change B is
    confined to the batch_speed table + plot)."""

    def test_accuracy_metric_cell_keeps_survivors(self):
        df = pd.DataFrame([
            _row(metric="tv_per_node", value=0.10, seed=1),
            _row(metric="tv_per_node", value=0.12, seed=2),
            # seed 0 failed — must NOT invalidate the accuracy cell.
            _row(metric="query_time_s", value=float("nan"), seed=0, status="oom"),
        ])
        cell = _metric_cell(df, "nbn-cat-ve", "tv_per_node", "iqm_iqr")
        assert cell != "--" and "$\\pm$" in cell   # survivor value, not a code
