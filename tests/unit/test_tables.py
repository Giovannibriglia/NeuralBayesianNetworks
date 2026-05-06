"""Tests for v0.6c-C-3 table writers (``benchmarking._tables``).

Pin the four output formats: CSV / markdown / LaTeX / parquet.  CSV
must use ``na_rep='n/a'``; markdown must render without ``tabulate``;
LaTeX must escape underscores and bold Pareto-optimal cells.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmarking._tables import (
    _df_to_markdown,
    _escape_latex,
    write_csv,
    write_latex,
    write_markdown,
    write_parquet,
)


def _make_wide() -> pd.DataFrame:
    """Tiny wide DataFrame for round-trip tests."""
    return pd.DataFrame(
        {
            "nbn-cat-ve": ["0.0500", "0.0429", "n/a (not applicable)"],
            "pgmpy-mle-ve": ["n/a (not applicable)", "0.0612", "0.0581"],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("discrete", "accuracy", 5),
                ("discrete", "accuracy", 10),
                ("continuous_lg", "accuracy", 5),
            ],
            names=["family", "metric", "n_nodes"],
        ),
    )


def _make_pareto() -> pd.DataFrame:
    """Pareto DataFrame matching the wide above."""
    return pd.DataFrame([
        {"family": "discrete", "baseline": "nbn-cat-ve",
         "n_nodes": 5, "is_pareto": True, "dominated_by": ()},
        {"family": "discrete", "baseline": "nbn-cat-ve",
         "n_nodes": 10, "is_pareto": False, "dominated_by": ("pgmpy-mle-ve",)},
        {"family": "discrete", "baseline": "pgmpy-mle-ve",
         "n_nodes": 10, "is_pareto": True, "dominated_by": ()},
    ])


# ---------------------------------------------------------------------- #
# CSV
# ---------------------------------------------------------------------- #


def test_write_csv_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "out.csv"
    write_csv(_make_wide(), p)
    text = p.read_text()
    assert "nbn-cat-ve" in text
    assert "n/a (not applicable)" in text
    # Reload via pandas and confirm data round-trips.
    reloaded = pd.read_csv(p, index_col=[0, 1, 2])
    assert "nbn-cat-ve" in reloaded.columns


# ---------------------------------------------------------------------- #
# Markdown — must work without the optional ``tabulate`` library
# ---------------------------------------------------------------------- #


def test_df_to_markdown_renders_without_tabulate() -> None:
    """The hand-rolled renderer in ``_tables._df_to_markdown`` must not
    depend on the optional ``tabulate`` library."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=[10, 20])
    df.index.name = "n"
    out = _df_to_markdown(df)
    assert "| n | a | b |" in out
    assert "| --- | --- | --- |" in out
    assert "| 10 | 1 | 3 |" in out
    assert "| 20 | 2 | 4 |" in out


def test_write_markdown_groups_by_family_metric(tmp_path: Path) -> None:
    """One H2 section per ``(family, metric)`` combination."""
    p = tmp_path / "out.md"
    write_markdown(_make_wide(), p)
    text = p.read_text()
    assert "## discrete — accuracy" in text
    assert "## continuous_lg — accuracy" in text


# ---------------------------------------------------------------------- #
# LaTeX
# ---------------------------------------------------------------------- #


def test_escape_latex_handles_underscores_and_pm() -> None:
    """``_`` → ``\\_``; ``±`` → ``$\\pm$``."""
    assert _escape_latex("nbn_cat_lw") == "nbn\\_cat\\_lw"
    assert _escape_latex("0.0429 ± 0.005") == "0.0429 $\\pm$ 0.005"


def test_write_latex_uses_booktabs(tmp_path: Path) -> None:
    """LaTeX output uses ``\\toprule \\midrule \\bottomrule`` (booktabs)
    and wraps each (family, metric) in its own ``\\begin{table}`` block."""
    p = tmp_path / "out.tex"
    write_latex(_make_wide(), p, pareto_df=_make_pareto())
    text = p.read_text()
    assert "\\toprule" in text
    assert "\\midrule" in text
    assert "\\bottomrule" in text
    assert "\\begin{table}" in text
    assert "\\end{table}" in text


def test_write_latex_bolds_pareto_cells(tmp_path: Path) -> None:
    """A ``(family, baseline, n_nodes)`` cell with ``is_pareto=True``
    has its value wrapped in ``\\textbf{...}``."""
    p = tmp_path / "out.tex"
    write_latex(_make_wide(), p, pareto_df=_make_pareto())
    text = p.read_text()
    # nbn-cat-ve at n=5 discrete: pareto-optimal → bold.
    assert "\\textbf{0.0500}" in text
    # nbn-cat-ve at n=10 discrete: dominated → NOT bold.
    assert "\\textbf{0.0429}" not in text
    # pgmpy-mle-ve at n=10 discrete: pareto-optimal → bold.
    assert "\\textbf{0.0612}" in text


def test_write_latex_does_not_bold_na_cells(tmp_path: Path) -> None:
    """Cells whose value starts with ``n/a`` are never bolded, even if
    the Pareto entry says ``is_pareto=True`` (which shouldn't happen
    in practice but is defensive)."""
    p = tmp_path / "out.tex"
    write_latex(_make_wide(), p, pareto_df=_make_pareto())
    text = p.read_text()
    # The "n/a (not applicable)" cells must NOT be bolded.
    assert "\\textbf{n/a" not in text


# ---------------------------------------------------------------------- #
# Parquet round-trip
# ---------------------------------------------------------------------- #


def test_write_parquet_round_trips(tmp_path: Path) -> None:
    """Parquet snapshot can be reloaded with the same schema."""
    p = tmp_path / "out.parquet"
    wide = _make_wide()
    write_parquet(wide, p)
    reloaded = pd.read_parquet(p)
    # Reset_index dropped the MultiIndex into columns for parquet
    # compatibility — confirm the family / metric / n_nodes columns
    # are present along with the baseline columns.
    assert "family" in reloaded.columns
    assert "metric" in reloaded.columns
    assert "n_nodes" in reloaded.columns
    assert "nbn-cat-ve" in reloaded.columns


# ---------------------------------------------------------------------- #
# Integration: write_all writes to canonical paths
# ---------------------------------------------------------------------- #


def test_write_all_uses_canonical_paths(tmp_path: Path) -> None:
    """write_all writes to ``{output_dir}/tables/{prefix}_summary.{ext}``
    for all four formats."""
    from benchmarking._tables import write_all
    aggregated = {"wide": _make_wide(), "pareto": _make_pareto()}
    paths = write_all(
        aggregated, output_dir=str(tmp_path), output_prefix="testrun",
    )
    assert paths["csv"].name == "testrun_summary.csv"
    assert paths["md"].name == "testrun_summary.md"
    assert paths["tex"].name == "testrun_summary.tex"
    assert paths["parquet"].name == "testrun_summary.parquet"
    for fmt, p in paths.items():
        assert p.parent.name == "tables"
        assert p.exists() and p.stat().st_size > 0
