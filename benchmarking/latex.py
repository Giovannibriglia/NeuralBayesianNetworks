"""LaTeX table export for benchmark / crash-test results.

Designed to be copy-pasted into a paper without further editing — uses
``booktabs`` style (`\\toprule`, `\\midrule`, `\\bottomrule`) and escapes
common LaTeX special characters in cell text.

Public API:
    rows_to_latex(rows, columns=...)        -> str
    write_latex_table(path, rows, ...)      -> None
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

# Minimal escaping for cell text (we don't try to be perfect — just safe).
_LATEX_REPL = {"_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#",
               "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\~{}", "^": r"\^{}"}


def _escape(s: Any) -> str:
    if not isinstance(s, str):
        return str(s)
    out = s
    for k, v in _LATEX_REPL.items():
        out = out.replace(k, v)
    return out


def _format_cell(v: Any, fmt: str | None) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        if fmt:
            return format(v, fmt)
        if abs(v) < 1e-3 or abs(v) > 1e4:
            return f"{v:.2e}"
        return f"{v:.3g}"
    return _escape(v)


def rows_to_latex(
    rows: Sequence[dict],
    columns: Sequence[str | tuple[str, str]] = (),
    *,
    caption: str | None = None,
    label: str | None = None,
    formats: dict[str, str] | None = None,
) -> str:
    """Convert a list of dict rows to a complete ``tabular`` LaTeX block.

    Parameters
    ----------
    rows:
        The data rows (only keys in ``columns`` are read).
    columns:
        Either column keys (``("baseline", "device", "tv")``) or
        (key, header) tuples (``("ms_per_query", "ms / query")``). The
        order is preserved.
    caption, label:
        Wrap the tabular in a ``table`` float with caption + label.
    formats:
        Optional ``{key: fmt_spec}`` for per-column number formatting,
        e.g. ``{"tv": ".4f", "ms_per_query": ".2f"}``.

    Returns
    -------
    str — a self-contained ``\\begin{table}…\\end{table}`` (or just the
    ``tabular`` block when no caption/label is provided).
    """
    formats = dict(formats or {})
    cols: list[tuple[str, str]] = []
    for c in columns:
        if isinstance(c, tuple):
            cols.append(c)
        else:
            cols.append((c, c.replace("_", " ")))

    keys = [k for k, _ in cols]
    headers = [_escape(h) for _, h in cols]

    align = "l" + "r" * (len(cols) - 1)
    lines: list[str] = []
    if caption or label:
        lines.append("\\begin{table}[h]")
        lines.append("  \\centering")
        if caption:
            lines.append(f"  \\caption{{{_escape(caption)}}}")
        if label:
            lines.append(f"  \\label{{{label}}}")
    lines.append("  \\begin{tabular}{" + align + "}")
    lines.append("    \\toprule")
    lines.append("    " + " & ".join(headers) + r" \\")
    lines.append("    \\midrule")
    for r in rows:
        cells = [_format_cell(r.get(k), formats.get(k)) for k in keys]
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    if caption or label:
        lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def write_latex_table(
    path: str | Path,
    rows: Sequence[dict],
    columns: Sequence[str | tuple[str, str]],
    **kwargs,
) -> Path:
    """Render ``rows`` to LaTeX and write to ``path``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(rows_to_latex(rows, columns, **kwargs))
    return p
