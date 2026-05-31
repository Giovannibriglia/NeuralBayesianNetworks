"""Tests for v0.6c-C-3 aggregator (``benchmarking._aggregate``).

Pin the aggregation contract: 4 DataFrames out (wide, long, status,
pareto), explicit ``"n/a (...)"`` strings instead of NaN/0 for missing
cells, and Pareto-frontier correctness.

Synthetic parquets are constructed in-memory to avoid depending on a
live smoke run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from benchmarking._aggregate import (
    _NA_CELL_ERRORED,
    _NA_METRIC_MISSING,
    _NA_NOT_APPLICABLE,
    aggregate,
)


def _write_synthetic_parquet(rows: list[dict], tmp_path: Path) -> Path:
    """Build a minimal raw-metrics parquet matching the runner schema."""
    df = pd.DataFrame(rows)
    # Fill default columns the runner emits.
    for col, default in [
        ("n_skipped", 0), ("error_msg", ""),
    ]:
        if col not in df.columns:
            df[col] = default
    out = tmp_path / "metrics.parquet"
    df.to_parquet(out)
    return out


def test_aggregate_returns_four_dataframes(tmp_path: Path) -> None:
    """Aggregator output has the four expected keys."""
    p = _write_synthetic_parquet([
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "pgmpy-mle-ve", "metric": "accuracy",
         "value": 0.05, "status": "ok"},
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "pgmpy-mle-ve", "metric": "total_time_s",
         "value": 0.001, "status": "ok"},
    ], tmp_path)
    out = aggregate(p)
    assert set(out.keys()) == {"wide", "long", "status", "pareto"}
    assert isinstance(out["wide"], pd.DataFrame)
    assert isinstance(out["long"], pd.DataFrame)
    assert isinstance(out["status"], pd.DataFrame)
    assert isinstance(out["pareto"], pd.DataFrame)


def test_aggregate_not_applicable_cell(tmp_path: Path) -> None:
    """``pgmpy-mle-ve`` on continuous_lg is not applicable per the
    registry; the aggregator must emit ``'n/a (not applicable)'``,
    NEVER NaN or 0."""
    p = _write_synthetic_parquet([
        # pgmpy-mle-ve is registry-applicable to discrete only.
        {"family": "continuous_lg", "n_nodes": 5, "seed": 0,
         "baseline": "pgmpy-mle-ve", "metric": "accuracy",
         "value": 0.123, "status": "ok"},
        # Add a discrete-applicable baseline so the wide DataFrame
        # has the column.
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "pgmpy-mle-ve", "metric": "accuracy",
         "value": 0.05, "status": "ok"},
    ], tmp_path)
    out = aggregate(p)
    cell = out["wide"].loc[("continuous_lg", "accuracy", "5"), "pgmpy-mle-ve"]
    assert cell == _NA_NOT_APPLICABLE, (
        f"expected {_NA_NOT_APPLICABLE!r}, got {cell!r}"
    )


def test_aggregate_cell_errored(tmp_path: Path) -> None:
    """Cell where every seed produced ``status='error'`` must report
    ``'n/a (cell errored)'`` — not zero, not NaN."""
    p = _write_synthetic_parquet([
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "pgmpy-mle-ve", "metric": "accuracy",
         "value": float("nan"), "status": "error"},
        {"family": "discrete", "n_nodes": 5, "seed": 1,
         "baseline": "pgmpy-mle-ve", "metric": "accuracy",
         "value": float("nan"), "status": "error"},
    ], tmp_path)
    out = aggregate(p)
    cell = out["wide"].loc[("discrete", "accuracy", "5"), "pgmpy-mle-ve"]
    assert cell == _NA_CELL_ERRORED


def test_aggregate_metric_missing(tmp_path: Path) -> None:
    """Issue #35 workaround: when a cell exists for one metric but not
    the other, the missing metric's cell reports
    ``'n/a (metric missing)'``.

    The parquet must include at least one row for every metric so the
    metric appears in the aggregator's iteration; the *cell of
    interest* still has the metric missing for that specific baseline.
    """
    p = _write_synthetic_parquet([
        # nbn-cat-ve emits accuracy only — issue #35 case.
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "nbn-cat-ve", "metric": "accuracy",
         "value": 0.07, "status": "ok"},
        # Another baseline emits total_time_s so the metric is present
        # in the parquet's metric column overall.
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "pgmpy-mle-ve", "metric": "accuracy",
         "value": 0.05, "status": "ok"},
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "pgmpy-mle-ve", "metric": "total_time_s",
         "value": 0.001, "status": "ok"},
    ], tmp_path)
    out = aggregate(p)
    cell = out["wide"].loc[("discrete", "total_time_s", "5"), "nbn-cat-ve"]
    assert cell == _NA_METRIC_MISSING
    # And accuracy is the formatted numeric value.
    acc = out["wide"].loc[("discrete", "accuracy", "5"), "nbn-cat-ve"]
    assert acc == "0.0700"


def test_aggregate_ok_rows_with_nan_value_treated_as_missing(tmp_path: Path) -> None:
    """Issue #35 specific: runner emits ``status='ok'`` rows where the
    ``value`` itself is NaN (metric path didn't populate the timing).
    The aggregator must treat these as metric-missing — never let a
    literal ``"nan"`` reach the formatted output."""
    p = _write_synthetic_parquet([
        # NBN-style: status=ok but value=NaN.
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "nbn-cat-ve", "metric": "total_time_s",
         "value": float("nan"), "status": "ok"},
        {"family": "discrete", "n_nodes": 5, "seed": 1,
         "baseline": "nbn-cat-ve", "metric": "total_time_s",
         "value": float("nan"), "status": "ok"},
        # Another baseline emits a real value so the metric is present.
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "pgmpy-mle-ve", "metric": "total_time_s",
         "value": 0.001, "status": "ok"},
    ], tmp_path)
    out = aggregate(p)
    cell = out["wide"].loc[("discrete", "total_time_s", "5"), "nbn-cat-ve"]
    assert cell == _NA_METRIC_MISSING, (
        f"expected {_NA_METRIC_MISSING!r}, got {cell!r}"
    )
    # And no literal "nan" anywhere in the wide table.
    flat = out["wide"].to_numpy().flatten().tolist()
    assert not any(isinstance(v, str) and "nan" in v.lower() for v in flat), (
        f"literal 'nan' leaked into wide table: {flat}"
    )


def test_aggregate_multi_seed_mean_std(tmp_path: Path) -> None:
    """Multiple ok seeds → ``mean ± std`` formatting with sample stddev."""
    p = _write_synthetic_parquet([
        {"family": "discrete", "n_nodes": 5, "seed": s,
         "baseline": "nbn-cat-ve", "metric": "accuracy",
         "value": v, "status": "ok"}
        for s, v in enumerate([0.04, 0.05, 0.06])
    ], tmp_path)
    out = aggregate(p)
    cell = out["wide"].loc[("discrete", "accuracy", "5"), "nbn-cat-ve"]
    # mean=0.05, sample-std (ddof=1) of {0.04, 0.05, 0.06} = 0.01
    assert cell.startswith("0.0500"), f"unexpected mean format: {cell!r}"
    assert "± 0.0100" in cell, f"expected ± 0.0100, got {cell!r}"


def test_aggregate_single_seed_no_std_suffix(tmp_path: Path) -> None:
    """Single seed → format is ``mean`` only, no ``± std`` (since it'd
    always be zero and adds clutter to the table)."""
    p = _write_synthetic_parquet([
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "nbn-cat-ve", "metric": "accuracy",
         "value": 0.0429, "status": "ok"},
    ], tmp_path)
    out = aggregate(p)
    cell = out["wide"].loc[("discrete", "accuracy", "5"), "nbn-cat-ve"]
    assert cell == "0.0429"
    assert "±" not in cell


def test_aggregate_pareto_frontier(tmp_path: Path) -> None:
    """A baseline that's worse on both axes must be marked
    non-Pareto; the dominator(s) must be Pareto-optimal.

    Construct a 2-baseline scenario where baseline_dominant is
    Pareto and baseline_dominated is dominated.
    """
    rows = []
    for baseline, t, a in [
        ("nbn-cat-ve", 0.001, 0.04),  # fast and accurate → Pareto
        ("nbn-cat-lw", 0.020, 0.10),  # slow and inaccurate → dominated
    ]:
        rows.append({"family": "discrete", "n_nodes": 5, "seed": 0,
                     "baseline": baseline, "metric": "accuracy",
                     "value": a, "status": "ok"})
        rows.append({"family": "discrete", "n_nodes": 5, "seed": 0,
                     "baseline": baseline, "metric": "total_time_s",
                     "value": t, "status": "ok"})
    p = _write_synthetic_parquet(rows, tmp_path)
    out = aggregate(p)
    pareto = out["pareto"]

    dom = pareto[
        (pareto["family"] == "discrete")
        & (pareto["baseline"] == "nbn-cat-ve")
        & (pareto["problem_id"] == "5")
    ].iloc[0]
    assert bool(dom["is_pareto"]) is True

    dominated = pareto[
        (pareto["family"] == "discrete")
        & (pareto["baseline"] == "nbn-cat-lw")
        & (pareto["problem_id"] == "5")
    ].iloc[0]
    assert bool(dominated["is_pareto"]) is False
    assert "nbn-cat-ve" in dominated["dominated_by"]


def test_aggregate_status_counts(tmp_path: Path) -> None:
    """The ``status`` DataFrame counts every status code per cell."""
    p = _write_synthetic_parquet([
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "nbn-cat-ve", "metric": "accuracy",
         "value": 0.05, "status": "ok"},
        {"family": "discrete", "n_nodes": 5, "seed": 1,
         "baseline": "nbn-cat-ve", "metric": "accuracy",
         "value": float("nan"), "status": "error"},
    ], tmp_path)
    out = aggregate(p)
    status = out["status"]
    row = status[
        (status["family"] == "discrete")
        & (status["baseline"] == "nbn-cat-ve")
        & (status["problem_id"] == "5")
    ].iloc[0]
    # ok and error each appear once.
    assert int(row.get("ok", 0)) == 1
    assert int(row.get("error", 0)) == 1


def test_aggregate_long_dataframe_schema(tmp_path: Path) -> None:
    """The ``long`` DataFrame has the documented schema columns."""
    p = _write_synthetic_parquet([
        {"family": "discrete", "n_nodes": 5, "seed": 0,
         "baseline": "nbn-cat-ve", "metric": "accuracy",
         "value": 0.05, "status": "ok"},
    ], tmp_path)
    out = aggregate(p)
    long = out["long"]
    expected_cols = {
        "family", "baseline", "problem_id", "metric",
        "mean", "std", "n_ok", "n_seeds",
        "applicable", "formatted",
    }
    assert expected_cols.issubset(set(long.columns))
