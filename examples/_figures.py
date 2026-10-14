"""One plotting style for every NBN example notebook.

Usage in a notebook (the notebooks run with ``examples/`` as working directory)::

    from _figures import BLUE, ORANGE, AQUA, GREY, INK, SINGLE, DOUBLE, save, use_style
    use_style()
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    ...
    save(fig, "09_sets")        # examples/figures/09_sets.pdf (vector) and .png (300 dpi)

Colours are the colour-blind-safe palette of Okabe & Ito (2008).  Figures
show inline as SVG (vector) unless ``use_style(inline="png")``.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STYLE = HERE / "nbn_style.mplstyle"
FIGURES = HERE / "figures"

# Okabe & Ito (2008), "Color universal design": distinguishable with every common
# colour-vision deficiency.  Names are what the notebooks use.
BLUE = "#0072B2"
ORANGE = "#E69F00"
AQUA = "#009E73"        # bluish green
VERMILLION = "#D55E00"
SKY = "#56B4E9"
MAGENTA = "#CC79A7"     # reddish purple
YELLOW = "#F0E442"      # light: use for fills, not thin lines
GREY = "#8C8C8C"
INK = "#000000"
PALETTE = [BLUE, ORANGE, AQUA, VERMILLION, SKY, MAGENTA, YELLOW, INK]

SINGLE = 3.4          # inches, one column of a two-column paper
DOUBLE = 7.0          # inches, full text width


def use_style(inline: str | None = None) -> None:
    """Apply the shared style.

    ``inline`` is the format of figures shown in the notebook: ``"svg"``
    (vector, the default) or ``"png"``; the environment variable
    ``NBN_FIGURE_FORMAT`` overrides the default.
    """
    plt.style.use(str(STYLE))
    fmt = inline or os.environ.get("NBN_FIGURE_FORMAT", "svg")
    try:  # inside Jupyter only
        from matplotlib_inline.backend_inline import set_matplotlib_formats

        set_matplotlib_formats(fmt)
    except Exception:  # pragma: no cover - outside a notebook
        pass


def save(fig, name: str) -> Path:
    """Write ``figures/<name>.pdf`` (vector) and ``figures/<name>.png`` (300 dpi); return the PDF path."""
    FIGURES.mkdir(exist_ok=True)
    pdf = FIGURES / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    return pdf
