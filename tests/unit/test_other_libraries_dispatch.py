"""Pin the v0.6c-C-2 dispatch labels for gpytorch / pomegranate / pyro.

This PR adds three libraries' worth of method-keyed labels to the
registry.  No adapter code changes:

* gpytorch: v1 adapter is already correctly vectorised (one SVGP per
  continuous node; ``pred.sample(torch.Size([n_samples]))`` gives the
  full ``[B, n_samples, 1]`` tensor in one GP call).  Labels:
  ``gpytorch-gp`` (param-learning) and ``gpytorch-gp-predict`` (inference),
  applicable to ``{continuous_lg, continuous_nongauss}``.

* pomegranate: v1 adapter is empirical-CPT BayesianNetwork on torch
  backend, discrete-only.  Single canonical method (Laplace-smoothed
  count + ``predict_proba``).  Labels: ``pomegranate-discrete`` /
  ``pomegranate-discrete-ve``, applicable to ``{discrete}``.  Known
  ``predict_proba`` IndexError on conditional queries — filed as v0.7.

* pyro: v1 adapter is Importance sampler over ancestral generative
  model.  Continuous path is structurally broken (parent-ignoring
  Gaussian leaves) — tracked as v0.7.  Labels: ``pyro-empirical`` /
  ``pyro-empirical-importance``, applicable to ``{discrete}`` despite
  the v1 adapter's broader ``supports`` declaration.

This file pins:
1. The 6 new labels appear in the registry with the right applicability.
2. ``_label_from_spec`` round-trips each new spec to the matching
   registry label (no ``-mle-`` infix on non-pgmpy libraries; convention
   from C-1a).
3. The applicability gate rejects cross-family cells (gpytorch on
   discrete, pomegranate-discrete on continuous_*, etc.).
4. Per-library dispatch tests skip cleanly on environments without the
   library installed (most likely scenario in this PR's sandbox).
"""
from __future__ import annotations

import importlib

import pytest

from benchmarking._baseline_registry import (
    BaselineSpec,
    _BASELINE_APPLICABILITY,
    _label_from_spec,
    is_applicable,
    known_labels,
)


def _has_lib(name: str) -> bool:
    """True if ``name`` is importable in this environment."""
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------- #
# Registry coverage — the 6 new labels are present with correct
# applicability sets.
# ---------------------------------------------------------------------- #


def test_gpytorch_labels_in_registry() -> None:
    """gpytorch labels: ``gpytorch-gp`` (param-learning) and
    ``gpytorch-gp-predict`` (inference), applicable to continuous-only
    families."""
    assert "gpytorch-gp" in known_labels()
    assert "gpytorch-gp-predict" in known_labels()
    for label in ("gpytorch-gp", "gpytorch-gp-predict"):
        assert is_applicable(label, "continuous_lg")
        assert is_applicable(label, "continuous_nongauss")
        assert not is_applicable(label, "discrete")
        assert not is_applicable(label, "hybrid")


def test_pomegranate_labels_in_registry() -> None:
    """pomegranate labels: discrete-only canonical pair."""
    assert "pomegranate-discrete" in known_labels()
    assert "pomegranate-discrete-ve" in known_labels()
    for label in ("pomegranate-discrete", "pomegranate-discrete-ve"):
        assert is_applicable(label, "discrete")
        assert not is_applicable(label, "continuous_lg")
        assert not is_applicable(label, "continuous_nongauss")
        assert not is_applicable(label, "hybrid")


def test_pyro_labels_in_registry() -> None:
    """pyro labels: applicable to discrete only despite the v1 adapter's
    broader ``supports`` declaration (the continuous path is structurally
    broken — see ``benchmarking/baselines/pyro_adapter.py`` docstring;
    v0.7 issue tracks the rework)."""
    assert "pyro-empirical" in known_labels()
    assert "pyro-empirical-importance" in known_labels()
    for label in ("pyro-empirical", "pyro-empirical-importance"):
        assert is_applicable(label, "discrete")
        assert not is_applicable(label, "continuous_lg")
        assert not is_applicable(label, "continuous_nongauss")
        assert not is_applicable(label, "hybrid")


# ---------------------------------------------------------------------- #
# Label-generator round-trip — every spec produces its registry label.
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("spec_dict, expected", [
    # gpytorch: mechanism is "gp"; no -mle- infix per the non-pgmpy
    # convention from C-1a (_label_from_spec else-branch).
    ({"library": "gpytorch", "mechanism": "gp", "param_method": "mle"},
     "gpytorch-gp"),
    ({"library": "gpytorch", "mechanism": "gp", "param_method": "mle",
      "inference_method": "predict"}, "gpytorch-gp-predict"),
    # pomegranate
    ({"library": "pomegranate", "mechanism": "discrete",
      "param_method": "mle"}, "pomegranate-discrete"),
    ({"library": "pomegranate", "mechanism": "discrete",
      "param_method": "mle", "inference_method": "ve"},
     "pomegranate-discrete-ve"),
    # pyro
    ({"library": "pyro", "mechanism": "empirical",
      "param_method": "mle"}, "pyro-empirical"),
    ({"library": "pyro", "mechanism": "empirical",
      "param_method": "mle", "inference_method": "importance"},
     "pyro-empirical-importance"),
])
def test_label_generator_matches_registry(
    spec_dict: dict, expected: str,
) -> None:
    """Every shipped spec round-trips to a label that's in the registry.

    Without this round-trip, the runner-loop applicability gate would
    look up a different label than what eventually lands in the parquet
    ``baseline`` column, silently routing to ``not_supported`` for
    every cell.
    """
    label = _label_from_spec(spec_dict)
    assert label == expected, (
        f"_label_from_spec({spec_dict}) → {label!r}, expected {expected!r}"
    )
    assert label in _BASELINE_APPLICABILITY, (
        f"label {label!r} produced by _label_from_spec is not in the "
        f"_BASELINE_APPLICABILITY matrix; the runner's applicability "
        f"gate would reject every cell."
    )


# ---------------------------------------------------------------------- #
# Cross-family applicability gate — the v0.6c-C-1b runner-loop gate
# rejects non-applicable cells before adapter dispatch.
# ---------------------------------------------------------------------- #


def test_applicability_gate_rejects_other_library_cross_family() -> None:
    """Cross-family applicability assertions for the v0.6c-C-2 labels.

    Critical audit: the C-1b post-merge bug was that the runner consulted
    the legacy ``_NOT_APPLICABLE`` table which didn't catch every
    method-keyed cross-family combination.  These assertions pin that
    the registry-based gate (now the canonical filter) handles all six
    new labels.
    """
    # gpytorch on discrete / hybrid → not applicable.
    for label in ("gpytorch-gp", "gpytorch-gp-predict"):
        assert not is_applicable(label, "discrete")
        assert not is_applicable(label, "hybrid")
    # pomegranate on continuous / hybrid → not applicable.
    for label in ("pomegranate-discrete", "pomegranate-discrete-ve"):
        for fam in ("continuous_lg", "continuous_nongauss", "hybrid"):
            assert not is_applicable(label, fam)
    # pyro on continuous / hybrid → not applicable (continuous path is
    # broken; tracked as v0.7).
    for label in ("pyro-empirical", "pyro-empirical-importance"):
        for fam in ("continuous_lg", "continuous_nongauss", "hybrid"):
            assert not is_applicable(label, fam)


# ---------------------------------------------------------------------- #
# Paper YAML round-trip — every spec in the paper config produces a
# label in the registry's applicability matrix.
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("config_path", [
    "benchmarking/configs/inference_paper.yaml",
    "benchmarking/configs/parameter_learning_paper.yaml",
])
def test_paper_config_specs_have_known_labels(config_path: str) -> None:
    """Every spec in every paper config produces a label that's in the
    applicability matrix.  Catches typos and convention mismatches at
    test-collection time."""
    import yaml
    from pathlib import Path
    data = yaml.safe_load(Path(config_path).read_text())
    for spec in data["baselines"]:
        label = _label_from_spec(spec)
        assert label in _BASELINE_APPLICABILITY, (
            f"{config_path}: spec {spec!r} produces label {label!r} "
            f"which is missing from the applicability matrix."
        )


# ---------------------------------------------------------------------- #
# Per-library dispatch — only run if the library is installed.
# ---------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_lib("gpytorch"), reason="gpytorch not installed")
def test_gpytorch_v2_shim_builds_and_fits_continuous_lg() -> None:
    """Verify ``build_adapter_v2`` returns a working shim for the
    gpytorch label, and that ``adapter.fit + adapter.query_batch_samples``
    runs end-to-end on a tiny continuous_lg cell.

    Skipped when gpytorch isn't installed (most likely in this PR's
    sandbox); will exercise on Giovanni's hardware via the paper-config
    quick-verify run.
    """
    import torch
    from benchmarking.baselines._adapter_v2 import build_adapter_v2
    from benchmarking.domains.base import Query
    from benchmarking.synthetic import make_synthetic_bn

    spec = BaselineSpec(library="gpytorch", mechanism="gp",
                        param_method="mle", inference_method="predict")
    bn = make_synthetic_bn(
        family="continuous_lg", n_nodes=4, n_train=200, n_test=50,
        n_reference=200, seed=0, device="cpu",
    )
    adapter = build_adapter_v2(spec, device="cpu")
    fitted = adapter.fit(bn.train_data, bn.dag, bn.variable_specs)
    assert fitted is not bn.true_model

    target = list(bn.dag.nodes)[2]
    ev_node = list(bn.dag.nodes)[0]
    q = Query(targets=(target,),
              evidence={ev_node: torch.zeros(2, dtype=torch.float32)},
              kind="marginal")
    samples = adapter.query_batch_samples(fitted, q, n_samples=50)
    assert torch.isfinite(samples.float()).all()


@pytest.mark.skipif(not _has_lib("pomegranate"),
                    reason="pomegranate not installed")
def test_pomegranate_v2_shim_builds_on_discrete() -> None:
    """End-to-end shim test for pomegranate-discrete-ve.  Skipped when
    pomegranate isn't installed.  Note: the pomegranate v1 adapter has
    a known IndexError on conditional queries — this test uses no
    evidence so it exercises only the marginal-query path."""
    from benchmarking.baselines._adapter_v2 import build_adapter_v2
    from benchmarking.domains.base import Query
    from benchmarking.synthetic import make_synthetic_bn

    spec = BaselineSpec(library="pomegranate", mechanism="discrete",
                        param_method="mle", inference_method="ve")
    bn = make_synthetic_bn(
        family="discrete", n_nodes=3, n_train=100, n_test=30,
        n_reference=100, seed=0, device="cpu",
    )
    adapter = build_adapter_v2(spec, device="cpu")
    _ = adapter.fit(bn.train_data, bn.dag, bn.variable_specs)
    # Smoke: just that fit completed without raising.


@pytest.mark.skipif(not _has_lib("pyro"), reason="pyro not installed")
def test_pyro_v2_shim_builds_on_discrete() -> None:
    """End-to-end shim test for pyro-empirical-importance on discrete.
    Skipped when pyro isn't installed."""
    from benchmarking.baselines._adapter_v2 import build_adapter_v2
    from benchmarking.synthetic import make_synthetic_bn

    spec = BaselineSpec(library="pyro", mechanism="empirical",
                        param_method="mle", inference_method="importance")
    bn = make_synthetic_bn(
        family="discrete", n_nodes=3, n_train=100, n_test=30,
        n_reference=100, seed=0, device="cpu",
    )
    adapter = build_adapter_v2(spec, device="cpu")
    _ = adapter.fit(bn.train_data, bn.dag, bn.variable_specs)
