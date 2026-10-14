"""Every example notebook uses the shared figure style and stores at least one figure (#325).

Fast: reads the stored notebooks, executes nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("[0-9][0-9]_*.ipynb"))


def _code(nb: dict) -> str:
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_there_are_example_notebooks():
    assert len(NOTEBOOKS) >= 18
    assert (EXAMPLES / "_figures.py").is_file() and (EXAMPLES / "nbn_style.mplstyle").is_file()


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_uses_the_shared_style_and_has_a_figure(path):
    nb = json.loads(path.read_text())
    code = _code(nb)
    assert "from _figures import" in code and "use_style()" in code
    assert "plt.rcParams.update" not in code, "palette and rcParams come from _figures / nbn_style.mplstyle"
    images = [o for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", [])
              if {"image/png", "image/svg+xml"} & set(o.get("data", {}))]
    assert images, f"{path.name} stores no figure output"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_every_shown_figure_is_also_saved(path):
    code = _code(json.loads(path.read_text()))
    assert code.count("plt.show()") == code.count("save(fig"), "each figure gets save(fig, name) before plt.show()"
