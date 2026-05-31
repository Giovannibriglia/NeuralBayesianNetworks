"""Tests for the v0.13 YAML → RunnerConfig parser.

Covers:
  - Each of the 7 canonical configs loads cleanly.
  - Missing version → error.
  - Wrong version → error with pointer to schema docs.
  - Unknown top-level field → error.
  - Default for ``metrics`` omitted produces AccuracyAndTiming.
  - ``--device`` override sets device on baselines without explicit device.
  - Per-baseline explicit device is not overridden by --device.
  - Missing required source field → error.
  - Unknown source field → error.
  - Unknown baseline field → error.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchmarking.core.yaml_config import load_runner_config
from benchmarking.measurements import AccuracyAndTiming, TimingOnly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, d: dict, filename: str = "cfg.yaml") -> Path:
    p = tmp_path / filename
    p.write_text(yaml.safe_dump(d))
    return p


def _minimal_valid(config_name: str = "test") -> dict:
    """Minimal valid v0.13 config dict."""
    return {
        "version": "v0.13",
        "benchmark": "synthetic",
        "config_name": config_name,
        "source": {
            "families": ["discrete"],
            "n_nodes_list": [5],
            "seeds": [0],
            "n_train": 100,
            "n_reference": 200,
            "edge_density": 0.20,
            "max_in_degree": 2,
            "cardinality": 4,
            "fraction_continuous": 0.0,
        },
        "baselines": [
            {"library": "nbn", "mechanism": "cat",
             "param_method": "mle", "inference_method": "ve"},
        ],
        "n_queries_per_cell": 2,
        "per_cell_timeout_s": 30.0,
    }


# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path("benchmarking/configs")
_INFERENCE_CONFIGS = [
    "inference_smoke.yaml",
    "inference_paper.yaml",
    "inference_paper_laptop.yaml",
    "inference_scalability.yaml",
]
_PARAM_CONFIGS = [
    "parameter_learning_smoke.yaml",
    "parameter_learning_paper.yaml",
    "parameter_learning_paper_laptop.yaml",
]


@pytest.mark.parametrize("cfg_name", _INFERENCE_CONFIGS)
def test_inference_config_loads_cleanly(cfg_name: str, tmp_path: Path) -> None:
    """Each inference config parses without error."""
    cfg = load_runner_config(
        _CONFIG_DIR / cfg_name,
        jsonl_path=tmp_path / "out.jsonl",
    )
    assert cfg.benchmark == "synthetic"
    assert cfg.config_name
    assert cfg.baselines
    assert cfg.n_queries_per_cell > 0
    assert cfg.per_cell_timeout_s > 0


@pytest.mark.parametrize("cfg_name", _PARAM_CONFIGS)
def test_param_learning_config_loads_cleanly(cfg_name: str, tmp_path: Path) -> None:
    """Each param-learning config is syntactically valid for v0.13."""
    cfg = load_runner_config(
        _CONFIG_DIR / cfg_name,
        jsonl_path=tmp_path / "out.jsonl",
    )
    assert cfg.benchmark == "synthetic"
    assert cfg.baselines


# ---------------------------------------------------------------------------
# Version checks
# ---------------------------------------------------------------------------

def test_missing_version_raises(tmp_path: Path) -> None:
    d = _minimal_valid()
    del d["version"]
    p = _write_yaml(tmp_path, d)
    with pytest.raises(ValueError, match="version"):
        load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")


def test_wrong_version_raises(tmp_path: Path) -> None:
    d = _minimal_valid()
    d["version"] = "v0.12"
    p = _write_yaml(tmp_path, d)
    with pytest.raises(ValueError, match="v0.13"):
        load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")


# ---------------------------------------------------------------------------
# Unknown/missing field validation
# ---------------------------------------------------------------------------

def test_unknown_top_level_field_raises(tmp_path: Path) -> None:
    d = _minimal_valid()
    d["typo_field"] = "oops"
    p = _write_yaml(tmp_path, d)
    with pytest.raises(ValueError, match="typo_field"):
        load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")


def test_missing_source_required_field_raises(tmp_path: Path) -> None:
    d = _minimal_valid()
    del d["source"]["families"]
    p = _write_yaml(tmp_path, d)
    with pytest.raises(ValueError, match="families"):
        load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")


def test_unknown_source_field_raises(tmp_path: Path) -> None:
    d = _minimal_valid()
    d["source"]["nbn_lw_n_samples"] = 512
    p = _write_yaml(tmp_path, d)
    with pytest.raises(ValueError, match="nbn_lw_n_samples"):
        load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")


def test_unknown_baseline_field_raises(tmp_path: Path) -> None:
    d = _minimal_valid()
    d["baselines"][0]["fit_epochs"] = 50
    p = _write_yaml(tmp_path, d)
    with pytest.raises(ValueError, match="fit_epochs"):
        load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")


# ---------------------------------------------------------------------------
# Metrics dispatch
# ---------------------------------------------------------------------------

def test_metrics_omitted_produces_accuracy_and_timing(tmp_path: Path) -> None:
    """Omitting 'metrics' defaults to AccuracyAndTiming."""
    d = _minimal_valid()
    # no 'metrics' key → default "all"
    p = _write_yaml(tmp_path, d)
    cfg = load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")
    assert isinstance(cfg.measurement, AccuracyAndTiming)


def test_metrics_all_produces_accuracy_and_timing(tmp_path: Path) -> None:
    d = _minimal_valid()
    d["metrics"] = "all"
    p = _write_yaml(tmp_path, d)
    cfg = load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")
    assert isinstance(cfg.measurement, AccuracyAndTiming)


def test_metrics_timing_produces_timing_only(tmp_path: Path) -> None:
    d = _minimal_valid()
    d["metrics"] = "timing"
    p = _write_yaml(tmp_path, d)
    cfg = load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")
    assert isinstance(cfg.measurement, TimingOnly)


def test_unknown_metrics_value_raises(tmp_path: Path) -> None:
    d = _minimal_valid()
    d["metrics"] = "accuracy_only"
    p = _write_yaml(tmp_path, d)
    with pytest.raises(ValueError, match="accuracy_only"):
        load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")


# ---------------------------------------------------------------------------
# --device override semantics
# ---------------------------------------------------------------------------

def test_device_override_applies_to_baselines_without_explicit_device(
    tmp_path: Path,
) -> None:
    """--device sets the device for baselines without an explicit device."""
    d = _minimal_valid()
    p = _write_yaml(tmp_path, d)
    cfg = load_runner_config(p, device_override="cpu",
                              jsonl_path=tmp_path / "out.jsonl")
    assert cfg.baselines[0].device == "cpu"


def test_device_override_does_not_override_explicit_baseline_device(
    tmp_path: Path,
) -> None:
    """Per-baseline device: in YAML takes priority over --device override."""
    d = _minimal_valid()
    d["baselines"][0]["device"] = "cpu"
    p = _write_yaml(tmp_path, d)
    # --device cuda would normally set cuda, but baseline has explicit cpu
    cfg = load_runner_config(p, device_override="cuda",
                              jsonl_path=tmp_path / "out.jsonl")
    assert cfg.baselines[0].device == "cpu"


# ---------------------------------------------------------------------------
# extra_kwargs
# ---------------------------------------------------------------------------

def test_extra_kwargs_parsed_into_baseline_spec(tmp_path: Path) -> None:
    d = _minimal_valid()
    d["baselines"][0]["extra_kwargs"] = {"n_samples": 512}
    p = _write_yaml(tmp_path, d)
    cfg = load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")
    assert cfg.baselines[0].extra_kwargs == {"n_samples": 512}
