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


def test_every_config_baseline_is_applicable_to_some_family():
    """A baseline whose label is in no family's applicability (e.g. pgmpy LG
    declared with param_method: mle -> 'pgmpy-mle-predict') silently becomes
    not_supported rows and its config edits never take effect."""
    import pytest

    from nbn.bench.core.applicability import BASELINE_FAMILY_APPLICABILITY
    from nbn.bench.core.config import BaselineSpec, build_adapter

    bad = []
    for path in sorted(_CONFIGS.rglob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        if not isinstance(doc, dict):
            continue
        for b in doc.get("baselines") or []:
            spec = BaselineSpec(
                library=b["library"], mechanism=b.get("mechanism"),
                param_method=b.get("param_method"),
                inference_method=b.get("inference_method"),
                extra_kwargs=b.get("extra_kwargs") or {},
            )
            try:
                name = build_adapter(spec, require_engine=False).name
            except ImportError as exc:  # optional library absent
                pytest.skip(f"optional dependency missing: {exc}")
            entry = BASELINE_FAMILY_APPLICABILITY.get(name)
            if entry is None or not entry.families:
                bad.append(f"{path.relative_to(_CONFIGS)}: {name}")
    assert not bad, bad
