"""Tests for the migrated _aggregate.py and _tables.py (v0.13 schema).

Uses synthetic in-memory DataFrames written to parquet as fixtures
(no real fitting or runner invocation).

Covers:
  TestAggregate      — aggregate() output shape, column names, numeric sort
  TestBackwardCompat — v0.12 n_nodes parquet transparently up-cast
  TestTables         — write_all() produces expected files
  TestPlotterSmoke   — render_figures() produces figure files (@pytest.mark.slow)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarking._aggregate import aggregate, _NA_NOT_APPLICABLE, _NA_CELL_ERRORED
from benchmarking._tables import write_all


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_v3_parquet(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    """Write a minimal v3-schema parquet and return its path."""
    if rows is None:
        rows = _default_rows()
    df = pd.DataFrame(rows)
    # Enforce v3 dtypes.
    df["seed"] = df["seed"].astype("int64")
    for col in ("value", "fit_time_s", "query_time_s", "metrics_time_s"):
        if col in df.columns:
            df[col] = df[col].astype("float64")
    df["problem_id"] = df["problem_id"].astype(str)
    path = tmp_path / "test_metrics.parquet"
    df.to_parquet(path, index=False)
    return path


def _default_rows() -> list[dict]:
    """8 rows: 2 families × 2 baselines × 2 metrics, seed=0."""
    base = dict(seed=0, query_role="random", fit_time_s=0.5,
                query_time_s=0.01, metrics_time_s=0.05, error_msg=None,
                benchmark="synthetic")
    return [
        {**base, "family": "discrete", "problem_id": "4", "baseline": "nbn-cat-lw",
         "metric": "tv_per_node",  "value": 0.10, "status": "ok"},
        {**base, "family": "discrete", "problem_id": "4", "baseline": "nbn-cat-lw",
         "metric": "query_time_s", "value": 0.01, "status": "ok"},
        {**base, "family": "discrete", "problem_id": "4", "baseline": "pgmpy-mle-ve",
         "metric": "tv_per_node",  "value": 0.12, "status": "ok"},
        {**base, "family": "discrete", "problem_id": "4", "baseline": "pgmpy-mle-ve",
         "metric": "query_time_s", "value": 0.02, "status": "ok"},
        {**base, "family": "continuous_lg", "problem_id": "4", "baseline": "nbn-lg-lw",
         "metric": "w1_per_node",  "value": 0.20, "status": "ok"},
        {**base, "family": "continuous_lg", "problem_id": "4", "baseline": "nbn-lg-lw",
         "metric": "query_time_s", "value": 0.015, "status": "ok"},
        # pgmpy-mle-ve on continuous_lg is not_supported (applicability gate)
        {**base, "family": "continuous_lg", "problem_id": "4", "baseline": "pgmpy-mle-ve",
         "metric": "status",       "value": float("nan"), "status": "not_supported"},
        # A timeout row
        {**base, "family": "discrete", "problem_id": "10", "baseline": "nbn-cat-lw",
         "metric": "tv_per_node",  "value": float("nan"), "status": "timeout",
         "query_time_s": float("nan"), "metrics_time_s": float("nan")},
    ]


def _multi_size_rows() -> list[dict]:
    """Rows across 3 problem_id values to test numeric sort."""
    base = dict(seed=0, query_role="random", fit_time_s=0.1,
                query_time_s=0.005, metrics_time_s=0.01, error_msg=None,
                benchmark="synthetic", family="discrete",
                baseline="nbn-cat-lw", metric="tv_per_node", status="ok")
    return [
        {**base, "problem_id": "5",  "value": 0.20},
        {**base, "problem_id": "10", "value": 0.15},
        {**base, "problem_id": "100","value": 0.08},
    ]


# ---------------------------------------------------------------------------
# TestAggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_returns_four_keys(self, tmp_path):
        path = _make_v3_parquet(tmp_path)
        result = aggregate(path)
        assert set(result.keys()) == {"wide", "long", "status", "pareto"}

    def test_long_uses_problem_id_column(self, tmp_path):
        path = _make_v3_parquet(tmp_path)
        result = aggregate(path)
        assert "problem_id" in result["long"].columns
        assert "n_nodes" not in result["long"].columns

    def test_wide_uses_problem_id_index_level(self, tmp_path):
        path = _make_v3_parquet(tmp_path)
        result = aggregate(path)
        assert "problem_id" in result["wide"].index.names
        assert "n_nodes" not in result["wide"].index.names

    def test_status_uses_problem_id_column(self, tmp_path):
        path = _make_v3_parquet(tmp_path)
        result = aggregate(path)
        assert "problem_id" in result["status"].columns

    def test_numeric_sort_of_problem_ids(self, tmp_path):
        """["5", "10", "100"] must sort numerically, not lexicographically."""
        rows = _multi_size_rows()
        path = _make_v3_parquet(tmp_path, rows)
        result = aggregate(path)
        # Extract the problem_id level from the wide MultiIndex.
        pids = list(dict.fromkeys(
            result["wide"].index.get_level_values("problem_id")
        ))
        assert pids == ["5", "10", "100"], f"Unexpected order: {pids}"

    def test_not_applicable_cell_formatted(self, tmp_path):
        """pgmpy-mle-ve on continuous_lg → n/a (not applicable) in wide."""
        path = _make_v3_parquet(tmp_path)
        result = aggregate(path)
        wide = result["wide"]
        # Look for the continuous_lg × w1_per_node row for pgmpy-mle-ve.
        idx = ("continuous_lg", "w1_per_node", "4")
        if idx in wide.index and "pgmpy-mle-ve" in wide.columns:
            cell = wide.loc[idx, "pgmpy-mle-ve"]
            assert cell == _NA_NOT_APPLICABLE

    def test_long_ok_rows_have_numeric_mean(self, tmp_path):
        path = _make_v3_parquet(tmp_path)
        result = aggregate(path)
        long = result["long"]
        ok = long[long["n_ok"] > 0]
        assert ok["mean"].notna().all()

    def test_status_sentinel_rows_excluded_from_metrics(self, tmp_path):
        """'status' metric rows must not appear in the metrics loop."""
        path = _make_v3_parquet(tmp_path)
        result = aggregate(path)
        long = result["long"]
        assert "status" not in long["metric"].unique()

    def test_empty_parquet_returns_empty_frames(self, tmp_path):
        df = pd.DataFrame(columns=[
            "benchmark", "family", "problem_id", "seed", "baseline",
            "query_role", "metric", "value", "status",
            "fit_time_s", "query_time_s", "metrics_time_s", "error_msg",
        ])
        path = tmp_path / "empty.parquet"
        df.to_parquet(path, index=False)
        result = aggregate(path)
        assert result["long"].empty
        assert result["wide"].empty


# ---------------------------------------------------------------------------
# TestBackwardCompat
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_v012_n_nodes_parquet_accepted(self, tmp_path):
        """v0.12 parquet with n_nodes (int) column is transparently up-cast."""
        df = pd.DataFrame([{
            "family": "discrete", "n_nodes": 4, "seed": 0,
            "baseline": "nbn-cat-lw", "metric": "tv_per_node",
            "value": 0.15, "status": "ok", "n_skipped": 0, "error_msg": "",
        }])
        df["n_nodes"] = df["n_nodes"].astype("int64")
        path = tmp_path / "v012.parquet"
        df.to_parquet(path, index=False)
        result = aggregate(path)
        assert "problem_id" in result["long"].columns
        assert result["long"].iloc[0]["problem_id"] == "4"


# ---------------------------------------------------------------------------
# TestTables
# ---------------------------------------------------------------------------

class TestTables:
    def test_write_all_produces_four_files(self, tmp_path):
        parquet = _make_v3_parquet(tmp_path)
        result = aggregate(parquet)
        paths = write_all(result, output_dir=tmp_path, output_prefix="test_run")
        for key in ("csv", "md", "tex", "parquet"):
            assert key in paths
            assert paths[key].exists(), f"{key} file missing"

    def test_markdown_contains_problem_id_header(self, tmp_path):
        parquet = _make_v3_parquet(tmp_path)
        result = aggregate(parquet)
        paths = write_all(result, output_dir=tmp_path, output_prefix="test_run")
        md = paths["md"].read_text()
        assert "problem_id" in md

    def test_latex_contains_booktabs(self, tmp_path):
        parquet = _make_v3_parquet(tmp_path)
        result = aggregate(parquet)
        paths = write_all(result, output_dir=tmp_path, output_prefix="test_run")
        tex = paths["tex"].read_text()
        assert "\\toprule" in tex
        assert "\\bottomrule" in tex


# ---------------------------------------------------------------------------
# TestPlotterSmoke (@pytest.mark.slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPlotterSmoke:
    def test_render_figures_produces_files(self, tmp_path):
        """Smoke test: render_figures runs end-to-end and creates figure files."""
        from benchmarking._plot_v2 import render_figures

        parquet = _make_v3_parquet(tmp_path, _multi_size_rows())
        out_paths = render_figures(
            parquet_path=parquet,
            output_dir=tmp_path,
            output_prefix="test_smoke",
            formats=("png",),   # fastest format for smoke test
        )
        # At least one metric figure must be produced.
        assert len(out_paths) > 0
        for metric, file_list in out_paths.items():
            assert len(file_list) > 0, f"No files produced for metric {metric}"
            for p in file_list:
                path = Path(p)
                assert path.exists(), f"Missing figure file: {p}"
                assert path.stat().st_size > 0, f"Empty figure file: {p}"

    def test_render_figures_file_naming_convention(self, tmp_path):
        """Produced files follow the <prefix>_<metric>_vs_problem_id.<ext> convention."""
        from benchmarking._plot_v2 import render_figures

        parquet = _make_v3_parquet(tmp_path, _multi_size_rows())
        out_paths = render_figures(
            parquet_path=parquet,
            output_dir=tmp_path,
            output_prefix="smoke",
            formats=("png",),
        )
        for metric, file_list in out_paths.items():
            for p in file_list:
                fname = Path(p).name
                assert f"_{metric}_vs_problem_id." in fname, (
                    f"File name {fname!r} doesn't follow convention for metric {metric}"
                )

    def test_render_figures_backward_compat_v012(self, tmp_path):
        """render_figures accepts v0.12 n_nodes parquets without error."""
        from benchmarking._plot_v2 import render_figures

        df = pd.DataFrame([{
            "family": "discrete", "n_nodes": 4, "seed": 0,
            "baseline": "nbn-cat-lw", "metric": "tv_per_node",
            "value": 0.15, "status": "ok", "n_skipped": 0, "error_msg": "",
        }])
        df["n_nodes"] = df["n_nodes"].astype("int64")
        path = tmp_path / "v012.parquet"
        df.to_parquet(path, index=False)
        # Should not raise.
        out_paths = render_figures(
            parquet_path=path,
            output_dir=tmp_path,
            output_prefix="v012_compat",
            formats=("png",),
        )
        assert isinstance(out_paths, dict)
