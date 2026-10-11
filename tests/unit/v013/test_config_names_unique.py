"""Every shipped config has its own ``config_name`` (#141).

``config_name`` is the output-artifact prefix (``<config_name>_metrics.parquet``,
figure / table prefixes); three synthetic smoke configs (and two speed smoke
configs) shared one, so their artifacts were indistinguishable by filename.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

_CONFIGS = Path(__file__).resolve().parents[3] / "nbn" / "bench" / "configs"


def test_config_names_are_unique():
    names = {}
    for path in sorted(_CONFIGS.rglob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        if isinstance(doc, dict) and "config_name" in doc:
            names[path.relative_to(_CONFIGS).as_posix()] = doc["config_name"]
    assert len(names) > 20
    dup = {n: [p for p, m in names.items() if m == n]
           for n, c in Counter(names.values()).items() if c > 1}
    assert not dup, f"config_name shared by several configs: {dup}"
