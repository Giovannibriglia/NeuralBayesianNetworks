"""Tests for the LaTeX-table export helper used by the runner / crash tests."""
from __future__ import annotations

from pathlib import Path

from benchmarking.latex import rows_to_latex, write_latex_table


def test_rows_to_latex_minimal():
    rows = [
        {"baseline": "nbn_ve", "tv": 0.0074, "ms_per_query": 5.15},
        {"baseline": "pgmpy", "tv": 0.0064, "ms_per_query": 1.12},
    ]
    out = rows_to_latex(rows, [("baseline", "Baseline"),
                               ("tv", "TV"), ("ms_per_query", "ms/q")])
    assert "\\toprule" in out and "\\midrule" in out and "\\bottomrule" in out
    assert "nbn\\_ve" in out
    assert "pgmpy" in out
    # Per-column formatting overrides the default
    out = rows_to_latex(rows, [("baseline", "Baseline"), ("tv", "TV")],
                       formats={"tv": ".6f"})
    assert "0.007400" in out


def test_write_latex_table(tmp_path: Path):
    rows = [{"a": 1.0, "b": "x_y"}]
    p = tmp_path / "out.tex"
    write_latex_table(p, rows, ["a", "b"], caption="test", label="tab:t1")
    text = p.read_text()
    assert "\\begin{table}" in text
    assert "\\caption{test}" in text
    assert "\\label{tab:t1}" in text
    assert "x\\_y" in text


def test_nan_renders_as_em_dash():
    rows = [{"a": float("nan")}]
    out = rows_to_latex(rows, ["a"])
    assert "—" in out
