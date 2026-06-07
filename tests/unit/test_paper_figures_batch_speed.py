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

from benchmarking._paper_figures import (
    _build_batch_speed_figure,
    fig_batch_speed,
    run_plot,
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
        from benchmarking.cli import _run_cells
        from benchmarking.core.yaml_config import load_runner_config

        jsonl = tmp_path / "metrics.jsonl"
        cfg = load_runner_config(
            "benchmarking/configs/synthetic/speed/inference_speed_smoke.yaml",
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
                assert all(y > 0 and not math.isnan(y) for y in ys)
