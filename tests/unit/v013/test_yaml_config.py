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
    "synthetic/smoke_tests/inference_smoke.yaml",
    "synthetic/complete/inference_complete.yaml",
    "synthetic/smoke_tests/inference_smoke_topological.yaml",
    "synthetic/smoke_tests/inference_smoke_random_only.yaml",
    "synthetic/smoke_tests/scalability_smoke.yaml",
    "synthetic/complete/inference_scalability_complete.yaml",
]
_PARAM_CONFIGS = [
    "synthetic/smoke_tests/parameter_learning_smoke.yaml",
    "synthetic/complete/parameter_learning_complete.yaml",
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
    """Each param-learning config loads via the PL command path (#109).

    PL configs declare ``metrics: log_likelihood`` and are loaded with a
    ``measurement_override`` (what the ``param-learning`` command supplies);
    the override is authoritative and the config carries it through.
    """
    from benchmarking.measurements import ParamLearningMeasurement

    cfg = load_runner_config(
        _CONFIG_DIR / cfg_name,
        jsonl_path=tmp_path / "out.jsonl",
        measurement_override=ParamLearningMeasurement(),
    )
    assert cfg.benchmark == "synthetic"
    assert cfg.baselines
    assert type(cfg.measurement).__name__ == "ParamLearningMeasurement"


def test_param_learning_metrics_field_guardrail(tmp_path: Path) -> None:
    """metrics='log_likelihood' is valid ONLY under the PL override path.

    Locks the structural-dispatch contract (#109): the metrics field cannot
    swap command behavior. Without an override (the inference path), a
    'log_likelihood' metrics field is rejected; with an override, a non-
    'log_likelihood' metrics field is rejected.
    """
    from benchmarking.measurements import ParamLearningMeasurement

    pl_cfg = _CONFIG_DIR / "synthetic/smoke_tests/parameter_learning_smoke.yaml"

    # Inference path (no override) must reject metrics=log_likelihood.
    with pytest.raises(ValueError, match="unknown metrics"):
        load_runner_config(pl_cfg, jsonl_path=tmp_path / "a.jsonl")

    # PL override path requires the config to declare log_likelihood: an
    # inference config (metrics=all) passed with an override is rejected.
    inf_cfg = _CONFIG_DIR / "synthetic/smoke_tests/inference_smoke.yaml"
    with pytest.raises(ValueError, match="metrics='log_likelihood'"):
        load_runner_config(
            inf_cfg, jsonl_path=tmp_path / "b.jsonl",
            measurement_override=ParamLearningMeasurement(),
        )


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


# ---------------------------------------------------------------------------
# Selector dispatch (Phase 2 — string + dict forms)
# ---------------------------------------------------------------------------

def test_yaml_dispatch_uniform_string_still_works(tmp_path: Path) -> None:
    """Backward-compat: string selector resolves to UniformRandomSelector."""
    from benchmarking.selectors.uniform import UniformRandomSelector

    d = _minimal_valid()
    d["selector"] = "uniform_random"
    p = _write_yaml(tmp_path, d)
    cfg = load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")
    assert isinstance(cfg.selector, UniformRandomSelector)


def test_yaml_dispatch_selector_omitted_defaults_uniform(tmp_path: Path) -> None:
    """No selector field → UniformRandomSelector (legacy default)."""
    from benchmarking.selectors.uniform import UniformRandomSelector

    d = _minimal_valid()
    d.pop("selector", None)
    p = _write_yaml(tmp_path, d)
    cfg = load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")
    assert isinstance(cfg.selector, UniformRandomSelector)


def test_yaml_dispatch_topological_dict(tmp_path: Path) -> None:
    """Phase 2: dict selector with type='topological' builds TopologicalAllocator."""
    from benchmarking.selectors.topological import TopologicalAllocator

    d = _minimal_valid()
    d["selector"] = {
        "type": "topological",
        "target_allocation": {"hub": 0.30, "cut": 0.25, "terminal": 0.25, "random": 0.20},
    }
    p = _write_yaml(tmp_path, d)
    cfg = load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")
    assert isinstance(cfg.selector, TopologicalAllocator)
    assert cfg.selector.target_allocation["hub"] == 0.30


def test_yaml_dispatch_topological_string(tmp_path: Path) -> None:
    """String 'topological' builds TopologicalAllocator with defaults."""
    from benchmarking.selectors.topological import TopologicalAllocator

    d = _minimal_valid()
    d["selector"] = "topological"
    p = _write_yaml(tmp_path, d)
    cfg = load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")
    assert isinstance(cfg.selector, TopologicalAllocator)
    assert cfg.selector.target_allocation["hub"] == 0.30  # default


def test_yaml_dispatch_heaviest_dict(tmp_path: Path) -> None:
    """Phase 3: dict selector type='heaviest_by_role' builds HeaviestQueryByRole."""
    from benchmarking.selectors.heaviest import HeaviestQueryByRole

    d = _minimal_valid()
    d["selector"] = {"type": "heaviest_by_role"}
    p = _write_yaml(tmp_path, d)
    cfg = load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")
    assert isinstance(cfg.selector, HeaviestQueryByRole)


def test_yaml_dispatch_heaviest_string(tmp_path: Path) -> None:
    """String 'heaviest_by_role' builds HeaviestQueryByRole with defaults."""
    from benchmarking.selectors.heaviest import HeaviestQueryByRole

    d = _minimal_valid()
    d["selector"] = "heaviest_by_role"
    p = _write_yaml(tmp_path, d)
    cfg = load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")
    assert isinstance(cfg.selector, HeaviestQueryByRole)


def test_yaml_dispatch_heaviest_passes_overrides(tmp_path: Path) -> None:
    """n_evidence / max_retry from the YAML block reach the selector."""
    from benchmarking.selectors.heaviest import HeaviestQueryByRole

    d = _minimal_valid()
    d["selector"] = {
        "type": "heaviest_by_role",
        "n_evidence": {"prediction": 2, "diagnosis": 2},
        "max_retry": 5,
    }
    p = _write_yaml(tmp_path, d)
    cfg = load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")
    assert isinstance(cfg.selector, HeaviestQueryByRole)
    assert cfg.selector.n_evidence == {"prediction": 2, "diagnosis": 2}
    assert cfg.selector.max_retry == 5


def test_scalability_complete_config_loads(tmp_path: Path) -> None:
    """The paper config loads and exposes the Phase 3 selector + 600s budget."""
    from benchmarking.selectors.heaviest import HeaviestQueryByRole

    cfg = load_runner_config(
        _CONFIG_DIR / "synthetic/complete/inference_scalability_complete.yaml",
        jsonl_path=tmp_path / "out.jsonl",
    )
    assert isinstance(cfg.selector, HeaviestQueryByRole)
    assert cfg.per_cell_timeout_s == 600.0
    # 23 + 3 non-parametric -lw baselines (kde/knn/flexcode, #228 / PR 10).
    assert len(cfg.baselines) == 26


def _bnlearn_source(**overrides) -> dict:
    src = {
        "networks": ["asia"], "seeds": [0],
        "n_train": 100, "n_test": 20, "n_reference": 50,
    }
    src.update(overrides)
    return src


def test_yaml_dispatch_bnlearn(tmp_path: Path) -> None:
    """Phase 4: benchmark=bnlearn builds BnlearnProblemSource + BnlearnConfig."""
    from benchmarking.problems import BnlearnConfig, BnlearnProblemSource

    d = _minimal_valid()
    d["benchmark"] = "bnlearn"
    d["source"] = _bnlearn_source()
    p = _write_yaml(tmp_path, d)
    cfg = load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")
    assert cfg.benchmark == "bnlearn"
    assert isinstance(cfg.problem_source, BnlearnProblemSource)
    assert isinstance(cfg.source_config, BnlearnConfig)
    assert cfg.source_config.networks == ["asia"]


def test_yaml_dispatch_bnlearn_all_fields(tmp_path: Path) -> None:
    """All 5 BnlearnConfig fields thread through the YAML source block."""
    d = _minimal_valid()
    d["benchmark"] = "bnlearn"
    d["source"] = _bnlearn_source(
        networks=["asia", "alarm"], seeds=[0, 1, 2],
        n_train=1000, n_test=200, n_reference=500,
    )
    p = _write_yaml(tmp_path, d)
    sc = load_runner_config(p, jsonl_path=tmp_path / "out.jsonl").source_config
    assert sc.networks == ["asia", "alarm"]
    assert sc.seeds == [0, 1, 2]
    assert sc.n_train == 1000
    assert sc.n_test == 200
    assert sc.n_reference == 500


def test_yaml_dispatch_bnlearn_unknown_source_field_raises(tmp_path: Path) -> None:
    d = _minimal_valid()
    d["benchmark"] = "bnlearn"
    d["source"] = _bnlearn_source(edge_density=0.2)  # synthetic-only field
    p = _write_yaml(tmp_path, d)
    with pytest.raises(ValueError, match="edge_density"):
        load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")


def test_bnlearn_smoke_config_loads(tmp_path: Path) -> None:
    """The committed bnlearn smoke config loads cleanly."""
    from benchmarking.problems import BnlearnProblemSource

    cfg = load_runner_config(
        _CONFIG_DIR / "bnlearn/smoke_tests/inference_smoke.yaml",
        jsonl_path=tmp_path / "out.jsonl",
    )
    assert cfg.benchmark == "bnlearn"
    assert isinstance(cfg.problem_source, BnlearnProblemSource)
    assert cfg.source_config.networks == ["asia", "ecoli70", "sangiovese"]
    assert len(cfg.baselines) == 7


def test_yaml_dispatch_unknown_selector_raises(tmp_path: Path) -> None:
    """Unknown selector name raises ValueError, not silent fall-through."""
    d = _minimal_valid()
    d["selector"] = "not_a_real_selector"
    p = _write_yaml(tmp_path, d)
    with pytest.raises(ValueError, match="selector"):
        load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")


def test_yaml_dispatch_unknown_dict_type_raises(tmp_path: Path) -> None:
    """Unknown selector dict 'type' raises ValueError."""
    d = _minimal_valid()
    d["selector"] = {"type": "bogus"}
    p = _write_yaml(tmp_path, d)
    with pytest.raises(ValueError, match="selector"):
        load_runner_config(p, jsonl_path=tmp_path / "out.jsonl")
