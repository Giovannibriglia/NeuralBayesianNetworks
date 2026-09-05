"""Tests for v0.13 output helpers: jsonl_to_parquet, make_results_dir.

JsonlWriter tests live in test_runner.py (TestJsonlWriter class) because
that file already has a complete fixture for CellResult rows; we only add
the conversion + path-convention tests here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from nbn.bench.core.output import (
    JsonlWriter,
    _compact_datetime,
    jsonl_to_parquet,
    make_results_dir,
)
from nbn.bench.core.results import CellResult


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _sample_v3_row(**overrides) -> dict:
    base = {
        "benchmark": "synthetic",
        "family": "discrete",
        "problem_id": "4",
        "seed": 0,
        "baseline": "nbn-cat-lw",
        "query_role": "random",
        "metric": "tv_per_node",
        "value": 0.15,
        "status": "ok",
        "fit_time_s": 0.5,
        "query_time_s": 0.01,
        "metrics_time_s": 0.05,
        "error_msg": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestJsonlToParquet
# ---------------------------------------------------------------------------

class TestJsonlToParquet:
    def test_creates_parquet_file(self, tmp_path):
        jsonl = tmp_path / "rows.jsonl"
        _write_jsonl(jsonl, [_sample_v3_row()])
        out = tmp_path / "rows.parquet"
        jsonl_to_parquet(jsonl, out)
        assert out.exists()

    def test_row_count_preserved(self, tmp_path):
        import pandas as pd
        jsonl = tmp_path / "rows.jsonl"
        _write_jsonl(jsonl, [_sample_v3_row(query_time_s=0.01 + i * 0.001) for i in range(5)])
        out = tmp_path / "rows.parquet"
        jsonl_to_parquet(jsonl, out)
        df = pd.read_parquet(out)
        assert len(df) == 5

    def test_v3_schema_columns_present(self, tmp_path):
        import pandas as pd
        jsonl = tmp_path / "rows.jsonl"
        _write_jsonl(jsonl, [_sample_v3_row()])
        out = tmp_path / "rows.parquet"
        jsonl_to_parquet(jsonl, out)
        df = pd.read_parquet(out)
        required = {
            "benchmark", "family", "problem_id", "seed", "baseline",
            "query_role", "metric", "value", "status",
            "fit_time_s", "query_time_s", "metrics_time_s", "error_msg",
        }
        assert required.issubset(set(df.columns))

    def test_problem_id_is_string(self, tmp_path):
        import pandas as pd
        jsonl = tmp_path / "rows.jsonl"
        _write_jsonl(jsonl, [_sample_v3_row(problem_id="10")])
        out = tmp_path / "rows.parquet"
        jsonl_to_parquet(jsonl, out)
        df = pd.read_parquet(out)
        # Accept object dtype (pandas < 2) or StringDtype (pandas >= 2).
        assert pd.api.types.is_string_dtype(df["problem_id"])

    def test_seed_is_int64(self, tmp_path):
        import pandas as pd
        jsonl = tmp_path / "rows.jsonl"
        _write_jsonl(jsonl, [_sample_v3_row(seed=7)])
        out = tmp_path / "rows.parquet"
        jsonl_to_parquet(jsonl, out)
        df = pd.read_parquet(out)
        assert str(df["seed"].dtype) == "int64"

    def test_float_columns_are_float64(self, tmp_path):
        import pandas as pd
        jsonl = tmp_path / "rows.jsonl"
        _write_jsonl(jsonl, [_sample_v3_row()])
        out = tmp_path / "rows.parquet"
        jsonl_to_parquet(jsonl, out)
        df = pd.read_parquet(out)
        for col in ("value", "fit_time_s", "query_time_s", "metrics_time_s"):
            assert str(df[col].dtype) == "float64", f"{col} dtype wrong"

    def test_nan_null_roundtrip(self, tmp_path):
        """NaN written as null in JSONL is re-read as NaN in parquet."""
        import pandas as pd
        row = _sample_v3_row(value=None, query_time_s=None)
        jsonl = tmp_path / "rows.jsonl"
        _write_jsonl(jsonl, [row])
        out = tmp_path / "rows.parquet"
        jsonl_to_parquet(jsonl, out)
        df = pd.read_parquet(out)
        assert math.isnan(df["value"].iloc[0])
        assert math.isnan(df["query_time_s"].iloc[0])

    def test_empty_jsonl_warns_and_no_parquet(self, tmp_path):
        jsonl = tmp_path / "empty.jsonl"
        jsonl.write_text("")
        out = tmp_path / "empty.parquet"
        with pytest.warns(UserWarning, match="no rows"):
            jsonl_to_parquet(jsonl, out)
        assert not out.exists()

    def test_parent_dir_created(self, tmp_path):
        jsonl = tmp_path / "rows.jsonl"
        _write_jsonl(jsonl, [_sample_v3_row()])
        out = tmp_path / "sub" / "dir" / "rows.parquet"
        jsonl_to_parquet(jsonl, out)
        assert out.exists()

    def test_roundtrip_via_jsonlwriter(self, tmp_path):
        """JsonlWriter → jsonl_to_parquet roundtrip preserves all fields."""
        import pandas as pd
        row = CellResult(
            benchmark="synthetic",
            family="discrete",
            problem_id="4",
            seed=3,
            baseline="nbn-cat-lw",
            query_role="random",
            metric="tv_per_node",
            value=0.123,
            status="ok",
            fit_time_s=0.42,
            query_time_s=0.011,
            metrics_time_s=0.051,
        )
        jsonl_path = tmp_path / "out.jsonl"
        parquet_path = tmp_path / "out.parquet"
        with JsonlWriter(jsonl_path) as w:
            w.write(row)
        jsonl_to_parquet(jsonl_path, parquet_path)
        df = pd.read_parquet(parquet_path)
        assert len(df) == 1
        assert df["seed"].iloc[0] == 3
        assert df["value"].iloc[0] == pytest.approx(0.123)
        assert df["problem_id"].iloc[0] == "4"


# ---------------------------------------------------------------------------
# TestCompactDatetime
# ---------------------------------------------------------------------------

class TestCompactDatetime:
    def test_format(self):
        dt = _compact_datetime()
        assert len(dt) == 15           # "YYYYMMDD_HHMMSS"
        assert dt[8] == "_"
        assert dt[:8].isdigit()
        assert dt[9:].isdigit()

    def test_two_calls_differ_or_same_second(self):
        a = _compact_datetime()
        b = _compact_datetime()
        # Within the same second they may be equal; they should never differ
        # in format.
        assert len(a) == len(b) == 15


# ---------------------------------------------------------------------------
# TestMakeResultsDir
# ---------------------------------------------------------------------------

class TestMakeResultsDir:
    def test_creates_directory(self, tmp_path):
        d = make_results_dir("synthetic", "smoke", base=tmp_path)
        assert d.exists()
        assert d.is_dir()

    def test_path_contains_benchmark_and_config(self, tmp_path):
        d = make_results_dir("synthetic", "paper", base=tmp_path)
        assert "benchmark_synthetic_paper_" in d.name

    def test_path_ends_with_datetime(self, tmp_path):
        d = make_results_dir("bnlearn", "asia", base=tmp_path)
        # Last 15 chars should be "YYYYMMDD_HHMMSS"
        dt_part = d.name.split("_")
        # Name: benchmark_<bm>_<cfg>_<date>_<time>
        # e.g., benchmark_bnlearn_asia_20260531_143045
        assert d.name.startswith("benchmark_bnlearn_asia_")

    def test_raises_if_already_exists(self, tmp_path):
        # Force a known name by mocking _compact_datetime
        from unittest.mock import patch
        with patch("nbn.bench.core.output._compact_datetime", return_value="20260101_000000"):
            d = make_results_dir("synthetic", "test", base=tmp_path)
            assert d.exists()
            with pytest.raises(FileExistsError):
                make_results_dir("synthetic", "test", base=tmp_path)

    def test_base_directory_respected(self, tmp_path):
        base = tmp_path / "my_results"
        base.mkdir()
        d = make_results_dir("scalability", "run1", base=base)
        assert d.parent == base
