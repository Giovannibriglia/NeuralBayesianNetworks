"""Tests for ``benchmarking._baseline_registry`` (v0.6c-C-1a).

Pin the canonical label scheme + applicability matrix so future
baselines added in v0.6c-C-1b/2/3 don't accidentally regress the
parquet's baseline-column conventions or the family applicability
gates.

v0.8-#26: ``nbn-neuralcat-ve`` is no longer deferred — the engine
now reads CPDs via ``mech.tabulate(parent_cards)``, so the
NeuralCategorical-VE combination works.  The corresponding
``test_nbn_neuralcat_ve_deferred_to_v07`` deferral-pinning test
was removed in the same commit.
"""
from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from benchmarking._baseline_registry import (
    BaselineSpec,
    _BASELINE_APPLICABILITY,
    _label_from_spec,
    is_applicable,
    known_labels,
)


# ---------------------------------------------------------------------- #
# Label generation — pgmpy uses param_method as the headline; NBN uses
# the mechanism name.
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("spec_dict, expected", [
    # pgmpy inference: param_method is the headline distinguisher.
    ({"library": "pgmpy", "mechanism": "discrete", "param_method": "mle",
      "inference_method": "ve"}, "pgmpy-mle-ve"),
    ({"library": "pgmpy", "mechanism": "discrete", "param_method": "bayes",
      "inference_method": "ve"}, "pgmpy-bayes-ve"),
    ({"library": "pgmpy", "mechanism": "lg", "param_method": "mle",
      "inference_method": "predict"}, "pgmpy-lg-predict"),
    # NBN inference: mechanism is the headline distinguisher.
    ({"library": "nbn", "mechanism": "cat", "param_method": "mle",
      "inference_method": "ve"}, "nbn-cat-ve"),
    ({"library": "nbn", "mechanism": "cat", "param_method": "mle",
      "inference_method": "lw"}, "nbn-cat-lw"),
    ({"library": "nbn", "mechanism": "neuralcat", "param_method": "mle",
      "inference_method": "lw"}, "nbn-neuralcat-lw"),
    ({"library": "nbn", "mechanism": "lg", "param_method": "mle",
      "inference_method": "lw"}, "nbn-lg-lw"),
    ({"library": "nbn", "mechanism": "mdn", "param_method": "mle",
      "inference_method": "lw"}, "nbn-mdn-lw"),
    ({"library": "nbn", "mechanism": "flow", "param_method": "mle",
      "inference_method": "lw"}, "nbn-flow-lw"),
    ({"library": "nbn", "mechanism": "hybrid", "param_method": "mle",
      "inference_method": "router"}, "nbn-hybrid-router"),
])
def test_label_from_inference_spec(spec_dict: dict, expected: str) -> None:
    assert _label_from_spec(spec_dict) == expected


@pytest.mark.parametrize("spec_dict, expected", [
    # Param-learning: drop the inference_method suffix.
    ({"library": "pgmpy", "mechanism": "discrete", "param_method": "mle"},
     "pgmpy-mle"),
    ({"library": "pgmpy", "mechanism": "discrete", "param_method": "bayes"},
     "pgmpy-bayes"),
    ({"library": "pgmpy", "mechanism": "lg", "param_method": "mle"},
     "pgmpy-lg"),
    ({"library": "nbn", "mechanism": "cat", "param_method": "mle"},
     "nbn-cat"),
    ({"library": "nbn", "mechanism": "neuralcat", "param_method": "mle"},
     "nbn-neuralcat"),
    ({"library": "nbn", "mechanism": "hybrid", "param_method": "mle"},
     "nbn-hybrid"),
])
def test_label_from_param_learning_spec(spec_dict: dict, expected: str) -> None:
    assert _label_from_spec(spec_dict) == expected


def test_label_accepts_baseline_spec_dataclass() -> None:
    """``_label_from_spec`` accepts both ``dict`` and ``BaselineSpec`` —
    the latter is the type-checked surface."""
    spec = BaselineSpec(
        library="nbn", mechanism="cat", param_method="mle",
        inference_method="lw",
    )
    assert _label_from_spec(spec) == "nbn-cat-lw"


def test_label_treats_inference_method_none_as_param_learning() -> None:
    """``inference_method=None`` and an absent key both yield the
    no-suffix label."""
    spec_a = {"library": "nbn", "mechanism": "cat", "param_method": "mle"}
    spec_b = {"library": "nbn", "mechanism": "cat", "param_method": "mle",
              "inference_method": None}
    assert _label_from_spec(spec_a) == "nbn-cat"
    assert _label_from_spec(spec_b) == "nbn-cat"


# ---------------------------------------------------------------------- #
# Applicability matrix
# ---------------------------------------------------------------------- #


def test_applicability_pgmpy_mle_ve_discrete_only() -> None:
    assert is_applicable("pgmpy-mle-ve", "discrete")
    assert not is_applicable("pgmpy-mle-ve", "continuous_lg")
    assert not is_applicable("pgmpy-mle-ve", "continuous_nongauss")
    assert not is_applicable("pgmpy-mle-ve", "hybrid")


def test_applicability_pgmpy_lg_predict_continuous_lg_only() -> None:
    assert is_applicable("pgmpy-lg-predict", "continuous_lg")
    assert not is_applicable("pgmpy-lg-predict", "discrete")
    assert not is_applicable("pgmpy-lg-predict", "continuous_nongauss")
    assert not is_applicable("pgmpy-lg-predict", "hybrid")


def test_applicability_nbn_mdn_lw_continuous_and_hybrid() -> None:
    # Bug B (issues #81 #24 #54) fixed by parent standardization in
    # MDNMechanism.fit_local — continuous_nongauss is now applicable again.
    assert is_applicable("nbn-mdn-lw", "continuous_lg")
    assert is_applicable("nbn-mdn-lw", "continuous_nongauss")
    assert is_applicable("nbn-mdn-lw", "hybrid")
    assert not is_applicable("nbn-mdn-lw", "discrete")
    # nbn-flow-lw uses MDN internally — same fix applies.
    assert is_applicable("nbn-flow-lw", "continuous_lg")
    assert is_applicable("nbn-flow-lw", "continuous_nongauss")
    assert is_applicable("nbn-flow-lw", "hybrid")
    assert not is_applicable("nbn-flow-lw", "discrete")


def test_applicability_nbn_hybrid_router_hybrid_only() -> None:
    assert is_applicable("nbn-hybrid-router", "hybrid")
    assert not is_applicable("nbn-hybrid-router", "discrete")
    assert not is_applicable("nbn-hybrid-router", "continuous_lg")
    assert not is_applicable("nbn-hybrid-router", "continuous_nongauss")


def test_applicability_unknown_label_returns_false() -> None:
    """A typo in a YAML produces a clean ``not_supported`` row, not a
    hard crash.  ``is_applicable`` must return False rather than raise."""
    assert not is_applicable("pgmpy-mle-mcmc", "discrete")
    assert not is_applicable("nonsense-label", "hybrid")


def test_known_labels_returns_sorted_list() -> None:
    labels = list(known_labels())
    assert labels == sorted(labels)
    assert len(labels) == len(set(labels))  # no duplicates
    assert "nbn-cat-ve" in labels
    assert "pgmpy-lg-predict" in labels


# ---------------------------------------------------------------------- #
# YAML round-trip: every spec in every shipping config produces a label
# in the registry.
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("config_path", [
    "benchmarking/configs/inference_smoke.yaml",
    "benchmarking/configs/inference_paper.yaml",
    "benchmarking/configs/parameter_learning_smoke.yaml",
    "benchmarking/configs/parameter_learning_paper.yaml",
])
def test_shipping_configs_use_known_or_legacy_labels(config_path: str) -> None:
    """Every spec in every shipping config must yield a label that is
    either:
    * in the registry's applicability matrix (NBN + pgmpy specs), or
    * a legacy label (gpytorch / pomegranate / pyro — kept as-is in
      v0.6c-C-1a; per-method dispatch lands in v0.6c-C-2).

    This test catches typos in YAML specs at test-collection time.
    """
    legacy_libraries = {"gpytorch", "pomegranate", "pyro"}
    data = yaml.safe_load(Path(config_path).read_text())
    for spec in data["baselines"]:
        label = _label_from_spec(spec)
        if spec["library"] in legacy_libraries:
            # Legacy libraries: registry doesn't know them yet; we
            # only check the label is computable.
            assert isinstance(label, str) and label
        else:
            assert label in _BASELINE_APPLICABILITY, (
                f"{config_path}: spec {spec!r} produced label {label!r} "
                f"which is not in the applicability matrix"
            )
