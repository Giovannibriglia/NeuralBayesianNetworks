"""Tests for ``CrashTestConfig.from_yaml`` — particularly the required-
field guard introduced in v0.6c-A round 2 to prevent silent default
drift.

The bug class retired here: smoke-tuned defaults silently inheriting
into paper config.  Round 1's diagnosis localised it to
``nbn_lw_n_samples`` (paper YAML omitted the field; dataclass default
of 512 leaked in; W₁ jumped from 0.04 to ~1.0 on continuous accuracy
because the IS weight collapses below ~2000 samples).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchmarking._crash_test_utils import CrashTestConfig


def test_paper_yaml_has_nbn_lw_n_samples_explicit() -> None:
    """Regression for v0.6c-A Finding 1: paper YAML must set
    ``nbn_lw_n_samples`` explicitly so the smoke-tuned default
    (512) cannot silently leak."""
    path = Path("benchmarking/configs/inference_paper.yaml")
    data = yaml.safe_load(path.read_text())
    assert "nbn_lw_n_samples" in data, (
        "inference_paper.yaml is missing nbn_lw_n_samples; the v0.6c-A "
        "config-drift bug requires this field to be explicit."
    )
    assert data["nbn_lw_n_samples"] >= 4000, (
        f"paper config nbn_lw_n_samples={data['nbn_lw_n_samples']} is too "
        f"small; round-1 sweep showed the W₁ phase transition between "
        f"1024 and 2000 samples; minimum-correct value is 4000."
    )


def test_smoke_yaml_has_nbn_lw_n_samples_explicit() -> None:
    """Mirror check for smoke."""
    path = Path("benchmarking/configs/inference_smoke.yaml")
    data = yaml.safe_load(path.read_text())
    assert "nbn_lw_n_samples" in data
    assert data["nbn_lw_n_samples"] >= 4000


def _v2_inference_yaml(tmp_path: Path, **overrides) -> Path:
    """Build a minimal v0.6c-C-1a schema-v2 inference config dict.

    Each test customises via ``overrides`` and pops fields it wants
    absent.  Centralising the template here keeps the tests focused
    on the validation behaviour each one is pinning.
    """
    data = {
        "config_schema_version": 2,
        "mode": "inference",
        "families": ["discrete"],
        "n_nodes": [5],
        "n_seeds": 1,
        "n_queries_per_cell": 16,
        "n_train": 200,
        "n_reference": 500,
        "edge_density": 0.20,
        "max_in_degree": 4,
        "cardinality": 4,
        "fraction_continuous": 0.5,
        "baselines": [
            {"library": "nbn", "mechanism": "cat",
             "param_method": "mle", "inference_method": "lw"},
        ],
        "per_cell_timeout_s": 30,
        "output_dir": str(tmp_path),
        "output_prefix": "test",
        "nbn_lw_n_samples": 4000,
    }
    data.update(overrides)
    out = tmp_path / "config.yaml"
    out.write_text(yaml.safe_dump(data))
    return out


def test_from_yaml_rejects_missing_nbn_lw_n_samples(tmp_path: Path) -> None:
    """If an inference config omits ``nbn_lw_n_samples``,
    ``CrashTestConfig.from_yaml`` raises ``ValueError`` rather than
    silently defaulting."""
    data = {
        "config_schema_version": 2,
        "mode": "inference",
        "families": ["discrete"],
        "n_nodes": [5],
        "n_seeds": 1,
        "n_queries_per_cell": 16,
        "n_train": 200,
        "n_reference": 500,
        "edge_density": 0.20,
        "max_in_degree": 4,
        "cardinality": 4,
        "fraction_continuous": 0.5,
        "baselines": [
            {"library": "nbn", "mechanism": "cat",
             "param_method": "mle", "inference_method": "lw"},
        ],
        "per_cell_timeout_s": 30,
        "output_dir": str(tmp_path),
        "output_prefix": "test",
        # nbn_lw_n_samples deliberately missing
    }
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError, match="nbn_lw_n_samples"):
        CrashTestConfig.from_yaml(bad_yaml)


def test_from_yaml_accepts_minimal_valid_config(tmp_path: Path) -> None:
    """Sanity: a config with all required fields *including*
    ``nbn_lw_n_samples`` parses without error."""
    cfg = CrashTestConfig.from_yaml(_v2_inference_yaml(tmp_path))
    assert cfg.nbn_lw_n_samples == 4000
    assert len(cfg.baselines) == 1
    assert cfg.baselines[0]["library"] == "nbn"


def test_from_yaml_rejects_missing_schema_version(tmp_path: Path) -> None:
    """v0.6c-C-1a: a config without ``config_schema_version`` raises
    ``ValueError`` pointing the user at schema v2."""
    p = _v2_inference_yaml(tmp_path)
    data = yaml.safe_load(p.read_text())
    del data["config_schema_version"]
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="config_schema_version"):
        CrashTestConfig.from_yaml(p)


def test_from_yaml_rejects_schema_v1_with_migration_hint(tmp_path: Path) -> None:
    """A schema-v1 config (flat baseline list, no version) raises
    ``ValueError`` with a clear migration message."""
    p = tmp_path / "v1.yaml"
    p.write_text(yaml.safe_dump({
        "mode": "inference",
        "families": ["discrete"],
        "n_nodes": [5],
        "n_seeds": 1,
        "n_queries_per_cell": 16,
        "n_train": 200,
        "n_reference": 500,
        "edge_density": 0.20,
        "max_in_degree": 4,
        "cardinality": 4,
        "fraction_continuous": 0.5,
        "baselines": ["nbn_lw", "pgmpy"],   # flat-list of strings (v1)
        "per_cell_timeout_s": 30,
        "output_dir": str(tmp_path),
        "output_prefix": "test",
        "nbn_lw_n_samples": 4000,
    }))
    with pytest.raises(ValueError, match=r"config_schema_version"):
        CrashTestConfig.from_yaml(p)


def test_from_yaml_rejects_baseline_spec_missing_fields(tmp_path: Path) -> None:
    """A baseline entry missing required fields raises a helpful error."""
    bad_baseline = [{"library": "nbn", "mechanism": "cat"}]
    # Missing param_method (and inference_method for inference).
    p = _v2_inference_yaml(tmp_path, baselines=bad_baseline)
    with pytest.raises(ValueError, match="missing fields"):
        CrashTestConfig.from_yaml(p)


def test_from_yaml_rejects_baseline_spec_not_a_dict(tmp_path: Path) -> None:
    """A baseline entry that's a string (v1 holdover) raises with a
    schema-v1 migration hint."""
    p = _v2_inference_yaml(tmp_path, baselines=["nbn_lw"])
    with pytest.raises(ValueError, match=r"is not a dict|Schema v1"):
        CrashTestConfig.from_yaml(p)
