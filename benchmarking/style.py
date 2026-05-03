"""Publication-ready matplotlib styling for NBN benchmark figures."""
from __future__ import annotations

PUB_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
}

# Tol-bright palette — colour-blind safe, stable across plots.
NBN_PALETTE = {
    "nbn":      "#4477AA",
    "nbn_lw":   "#66CCEE",
    "pgmpy":    "#EE6677",
    "gpytorch": "#228833",
    "pyro":     "#CCBB44",
    "ground":   "#444444",
}


def apply_style() -> None:
    """Apply NBN's publication style to the active matplotlib rcParams."""
    import matplotlib as mpl
    mpl.rcParams.update(PUB_STYLE)


def savefig_multi(fig, stem: str, formats=("pdf", "png", "svg")) -> list[str]:
    """Save a figure in multiple formats. Returns the list of written paths."""
    paths = []
    for fmt in formats:
        path = f"{stem}.{fmt}"
        fig.savefig(path)
        paths.append(path)
    return paths
