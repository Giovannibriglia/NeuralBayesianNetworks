"""Crash-recovery acceptance tests for the JSONL sidecar (#80).

Tests verify that:
- JSONLSidecarWriter appends valid JSON lines, flushes per row, and closes cleanly.
- NaN values are serialised as JSON null and round-trip back to float NaN.
- jsonl_to_parquet() reconstructs a parquet that is byte-identical to the
  parquet produced by write_parquet() on the same CellResult list.
- Column order matches: 8 core fields first, then extra keys in first-occurrence order.
- Per-column dtypes match.
- An empty JSONL produces a warning and no parquet file.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import pandas as pd
import pytest

from benchmarking._crash_test_utils import (
    CellResult,
    JSONLSidecarWriter,
    jsonl_to_parquet,
    write_parquet,
)


# ------------------------------------------------------------------ helpers --


def _make_rows() -> list[CellResult]:
    return [
        CellResult(
            family="discrete",
            n_nodes=5,
            seed=0,
            baseline="pgmpy-mle-ve",
            metric="total_time_s",
            value=0.001,
            status="ok",
            n_skipped=0,
        ),
        CellResult(
            family="discrete",
            n_nodes=5,
            seed=0,
            baseline="pgmpy-mle-ve",
            metric="accuracy",
            value=0.75,
            status="ok",
            n_skipped=0,
        ),
        # NaN value — e.g. accuracy for a not_supported baseline
        CellResult(
            family="continuous_lg",
            n_nodes=10,
            seed=1,
            baseline="gpytorch-gp-predict",
            metric="accuracy",
            value=float("nan"),
            status="not_supported",
            n_skipped=0,
        ),
        # Row with extra keys
        CellResult(
            family="discrete",
            n_nodes=10,
            seed=2,
            baseline="nbn-cat-ve",
            metric="total_time_s",
            value=0.042,
            status="ok",
            n_skipped=0,
            extra={"error_msg": ""},
        ),
    ]


# ------------------------------------------------------------------ tests ---


def test_writer_produces_valid_jsonl(tmp_path: Path) -> None:
    import json

    rows = _make_rows()
    path = tmp_path / "test.jsonl"
    writer = JSONLSidecarWriter(path)
    try:
        for row in rows:
            writer.append(row)
    finally:
        writer.close()

    lines = path.read_text().splitlines()
    assert len(lines) == len(rows)
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["baseline"] == "pgmpy-mle-ve"
    assert parsed[0]["value"] == pytest.approx(0.001)


def test_nan_serialised_as_null(tmp_path: Path) -> None:
    import json

    rows = _make_rows()
    path = tmp_path / "test.jsonl"
    writer = JSONLSidecarWriter(path)
    try:
        for row in rows:
            writer.append(row)
    finally:
        writer.close()

    lines = path.read_text().splitlines()
    nan_line = json.loads(lines[2])  # the not_supported row
    assert nan_line["value"] is None, "NaN must be serialised as JSON null"


def test_round_trip_parity(tmp_path: Path) -> None:
    """JSONL→parquet must be byte-identical to write_parquet on the same rows."""
    rows = _make_rows()

    # Write via the official parquet writer
    parquet_direct = tmp_path / "direct.parquet"
    write_parquet(rows, parquet_direct)

    # Write via JSONL sidecar, then recover
    jsonl_file = tmp_path / "test.jsonl"
    writer = JSONLSidecarWriter(jsonl_file)
    try:
        for row in rows:
            writer.append(row)
    finally:
        writer.close()

    parquet_recovered = tmp_path / "recovered.parquet"
    jsonl_to_parquet(jsonl_file, parquet_recovered)

    orig = pd.read_parquet(parquet_direct)
    recov = pd.read_parquet(parquet_recovered)

    assert orig.shape == recov.shape, f"shape mismatch: {orig.shape} vs {recov.shape}"
    assert list(orig.columns) == list(recov.columns), (
        f"column order mismatch:\n  direct:    {list(orig.columns)}\n"
        f"  recovered: {list(recov.columns)}"
    )
    assert (orig.dtypes == recov.dtypes).all(), (
        f"dtype mismatch:\n{orig.dtypes}\nvs\n{recov.dtypes}"
    )
    assert orig.equals(recov), "DataFrames differ — round-trip is not lossless"


def test_column_order(tmp_path: Path) -> None:
    """8 core fields must appear first; extra keys follow in first-occurrence order."""
    core = ["family", "n_nodes", "seed", "baseline", "metric", "value", "status", "n_skipped"]

    jsonl_file = tmp_path / "test.jsonl"
    writer = JSONLSidecarWriter(jsonl_file)
    try:
        for row in _make_rows():
            writer.append(row)
    finally:
        writer.close()

    parquet_file = tmp_path / "recovered.parquet"
    jsonl_to_parquet(jsonl_file, parquet_file)
    df = pd.read_parquet(parquet_file)

    assert list(df.columns[: len(core)]) == core, (
        f"Core columns out of order: {list(df.columns)}"
    )


def test_nan_roundtrip_dtype(tmp_path: Path) -> None:
    """NaN values must come back as float64 NaN, not None or object."""
    rows = [
        CellResult(
            family="continuous_lg",
            n_nodes=5,
            seed=0,
            baseline="gpytorch-gp-predict",
            metric="accuracy",
            value=float("nan"),
            status="not_supported",
            n_skipped=0,
        )
    ]
    jsonl_file = tmp_path / "test.jsonl"
    writer = JSONLSidecarWriter(jsonl_file)
    try:
        for row in rows:
            writer.append(row)
    finally:
        writer.close()

    parquet_file = tmp_path / "recovered.parquet"
    jsonl_to_parquet(jsonl_file, parquet_file)
    df = pd.read_parquet(parquet_file)

    assert df["value"].dtype == "float64"
    assert math.isnan(df["value"].iloc[0])


def test_empty_jsonl_warns_no_parquet(tmp_path: Path) -> None:
    """An empty JSONL must emit a warning and not create a parquet file."""
    jsonl_file = tmp_path / "empty.jsonl"
    jsonl_file.write_text("")
    parquet_file = tmp_path / "should_not_exist.parquet"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jsonl_to_parquet(jsonl_file, parquet_file)

    assert not parquet_file.exists(), "parquet must not be created for empty JSONL"
    assert any("no rows" in str(w.message).lower() for w in caught), (
        "Expected a 'no rows' warning"
    )


def test_partial_jsonl_recovery(tmp_path: Path) -> None:
    """A truncated JSONL (simulating a mid-run crash) recovers the rows written so far."""
    rows = _make_rows()
    jsonl_file = tmp_path / "partial.jsonl"
    writer = JSONLSidecarWriter(jsonl_file)
    # Write only the first two rows, then "crash" without calling close()
    writer.append(rows[0])
    writer.append(rows[1])
    writer._fh.flush()
    # Don't call close() — simulate crash

    parquet_file = tmp_path / "partial_recovered.parquet"
    jsonl_to_parquet(jsonl_file, parquet_file)
    df = pd.read_parquet(parquet_file)

    assert len(df) == 2
    assert df.iloc[0]["baseline"] == "pgmpy-mle-ve"
    assert df.iloc[0]["metric"] == "total_time_s"
