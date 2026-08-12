"""Regression tests for Pass-10 priority-1 — Gate B / Gate C
consolidation into the BaselineApplicability registry.

Pins three properties:

1. **Post-consolidation gate matches pre-consolidation matrix**:
   parametrized over (label, family) cartesian product, asserts
   ``is_applicable`` returns the expected boolean per the snapshot
   table.

2. **Pre-consolidation Gate B coverage preserved**: every
   (family, legacy_adapter) pair that USED to live in the deleted
   ``_NOT_APPLICABLE`` set is still rejected by the registry on every
   label that translates to that adapter.  Anchors the proof from
   /tmp/dual_gate_proof.py (Phase 1) as a permanent regression.

3. **Gate C → accuracy_supported translation**: the 2
   (family, label) pairs that used to live in ``_ACCURACY_NOT_APPLICABLE``
   now report ``accuracy_supported(label, family) == False`` via the
   registry accessor.  Other (label, family) pairs report True.

Also covers:
- Bucket-1 hybrid additions (nbn-mdn / nbn-mdn-lw / nbn-flow / nbn-flow-lw).
- Unknown-label fallback returns False from both accessors.
"""
from __future__ import annotations

import pytest

from benchmarking.core.applicability import (
    BaselineApplicability, BASELINE_FAMILY_APPLICABILITY as _BASELINE_APPLICABILITY,
    accuracy_supported, is_applicable, known_labels,
)


FAMILIES = ("discrete", "continuous_lg", "continuous_nongauss", "hybrid")


_EXPECTED_APPLICABILITY: dict[tuple[str, str], bool] = {
    # Param-learning labels
    ("pgmpy-mle", "discrete"): True,
    ("pgmpy-mle", "continuous_lg"): False,
    ("pgmpy-mle", "continuous_nongauss"): False,
    ("pgmpy-mle", "hybrid"): False,
    ("pgmpy-bayes", "discrete"): True,
    ("pgmpy-bayes", "continuous_lg"): False,
    ("pgmpy-bayes", "continuous_nongauss"): False,
    ("pgmpy-bayes", "hybrid"): False,
    ("pgmpy-lg", "discrete"): False,
    ("pgmpy-lg", "continuous_lg"): True,
    ("pgmpy-lg", "continuous_nongauss"): False,
    ("pgmpy-lg", "hybrid"): False,
    ("nbn-cat", "discrete"): True,
    ("nbn-cat", "continuous_lg"): False,
    ("nbn-cat", "continuous_nongauss"): False,
    ("nbn-cat", "hybrid"): False,
    # Laplace-smoothed (alpha=1) empirical CPTs — discrete-only like nbn-cat.
    ("nbn-cat-bayes", "discrete"): True,
    ("nbn-cat-bayes", "continuous_lg"): False,
    ("nbn-cat-bayes", "continuous_nongauss"): False,
    ("nbn-cat-bayes", "hybrid"): False,
    ("nbn-neuralcat", "discrete"): True,
    ("nbn-neuralcat", "continuous_lg"): False,
    ("nbn-neuralcat", "continuous_nongauss"): False,
    ("nbn-neuralcat", "hybrid"): False,
    ("nbn-lg", "discrete"): False,
    ("nbn-lg", "continuous_lg"): True,
    ("nbn-lg", "continuous_nongauss"): False,
    ("nbn-lg", "hybrid"): False,
    # Pass-10 priority-1: hybrid added.
    ("nbn-mdn", "discrete"): False,
    ("nbn-mdn", "continuous_lg"): True,
    ("nbn-mdn", "continuous_nongauss"): True,
    ("nbn-mdn", "hybrid"): True,
    # Pass-10 priority-1: hybrid added (FLIP to False if Section 6 fails).
    ("nbn-flow", "discrete"): False,
    ("nbn-flow", "continuous_lg"): True,
    ("nbn-flow", "continuous_nongauss"): True,
    ("nbn-flow", "hybrid"): True,
    ("nbn-hybrid", "discrete"): False,
    ("nbn-hybrid", "continuous_lg"): False,
    ("nbn-hybrid", "continuous_nongauss"): False,
    ("nbn-hybrid", "hybrid"): True,
    # Non-parametric continuous mechanisms (#223 / PR 8) — every continuous
    # family, like nbn-mdn / nbn-flow.
    ("nbn-kde", "discrete"): False,
    ("nbn-kde", "continuous_lg"): True,
    ("nbn-kde", "continuous_nongauss"): True,
    ("nbn-kde", "hybrid"): True,
    ("nbn-knn", "discrete"): False,
    ("nbn-knn", "continuous_lg"): True,
    ("nbn-knn", "continuous_nongauss"): True,
    ("nbn-knn", "hybrid"): True,
    ("nbn-flexcode", "discrete"): False,
    ("nbn-flexcode", "continuous_lg"): True,
    ("nbn-flexcode", "continuous_nongauss"): True,
    ("nbn-flexcode", "hybrid"): True,
    # Inference labels
    ("pgmpy-mle-ve", "discrete"): True,
    ("pgmpy-mle-ve", "continuous_lg"): False,
    ("pgmpy-mle-ve", "continuous_nongauss"): False,
    ("pgmpy-mle-ve", "hybrid"): False,
    ("pgmpy-bayes-ve", "discrete"): True,
    ("pgmpy-bayes-ve", "continuous_lg"): False,
    ("pgmpy-bayes-ve", "continuous_nongauss"): False,
    ("pgmpy-bayes-ve", "hybrid"): False,
    ("pgmpy-lg-predict", "discrete"): False,
    ("pgmpy-lg-predict", "continuous_lg"): True,
    ("pgmpy-lg-predict", "continuous_nongauss"): False,
    ("pgmpy-lg-predict", "hybrid"): False,
    ("nbn-cat-ve", "discrete"): True,
    ("nbn-cat-ve", "continuous_lg"): False,
    ("nbn-cat-ve", "continuous_nongauss"): False,
    ("nbn-cat-ve", "hybrid"): False,
    ("nbn-cat-lw", "discrete"): True,
    ("nbn-cat-lw", "continuous_lg"): False,
    ("nbn-cat-lw", "continuous_nongauss"): False,
    ("nbn-cat-lw", "hybrid"): False,
    ("nbn-cat-bayes-ve", "discrete"): True,
    ("nbn-cat-bayes-ve", "continuous_lg"): False,
    ("nbn-cat-bayes-ve", "continuous_nongauss"): False,
    ("nbn-cat-bayes-ve", "hybrid"): False,
    ("nbn-cat-bayes-lw", "discrete"): True,
    ("nbn-cat-bayes-lw", "continuous_lg"): False,
    ("nbn-cat-bayes-lw", "continuous_nongauss"): False,
    ("nbn-cat-bayes-lw", "hybrid"): False,
    ("nbn-neuralcat-ve", "discrete"): True,
    ("nbn-neuralcat-ve", "continuous_lg"): False,
    ("nbn-neuralcat-ve", "continuous_nongauss"): False,
    ("nbn-neuralcat-ve", "hybrid"): False,
    ("nbn-neuralcat-lw", "discrete"): True,
    ("nbn-neuralcat-lw", "continuous_lg"): False,
    ("nbn-neuralcat-lw", "continuous_nongauss"): False,
    ("nbn-neuralcat-lw", "hybrid"): False,
    ("nbn-lg-lw", "discrete"): False,
    ("nbn-lg-lw", "continuous_lg"): True,
    ("nbn-lg-lw", "continuous_nongauss"): False,
    ("nbn-lg-lw", "hybrid"): False,
    # Bug B fixed by parent standardization in MDNMechanism (issues #81 #24 #54).
    # Pass-10 priority-1: hybrid added; continuous_nongauss restored by this fix.
    ("nbn-mdn-lw", "discrete"): False,
    ("nbn-mdn-lw", "continuous_lg"): True,
    ("nbn-mdn-lw", "continuous_nongauss"): True,
    ("nbn-mdn-lw", "hybrid"): True,
    # nbn-flow-lw uses MDN internally — same fix applies transitively.
    ("nbn-flow-lw", "discrete"): False,
    ("nbn-flow-lw", "continuous_lg"): True,
    ("nbn-flow-lw", "continuous_nongauss"): True,
    ("nbn-flow-lw", "hybrid"): True,
    # Non-parametric continuous mechanisms, LW only (#224 / PR 9) — every
    # continuous family, like nbn-mdn-lw / nbn-flow-lw. (ais/avi deferred to #227.)
    ("nbn-kde-lw", "discrete"): False,
    ("nbn-kde-lw", "continuous_lg"): True,
    ("nbn-kde-lw", "continuous_nongauss"): True,
    ("nbn-kde-lw", "hybrid"): True,
    ("nbn-knn-lw", "discrete"): False,
    ("nbn-knn-lw", "continuous_lg"): True,
    ("nbn-knn-lw", "continuous_nongauss"): True,
    ("nbn-knn-lw", "hybrid"): True,
    ("nbn-flexcode-lw", "discrete"): False,
    ("nbn-flexcode-lw", "continuous_lg"): True,
    ("nbn-flexcode-lw", "continuous_nongauss"): True,
    ("nbn-flexcode-lw", "hybrid"): True,
    # Router widened to every family (fix/benchmark-baseline-enablement):
    # it dispatches all-discrete nets to VE and anything else to LW, so it
    # is applicable wherever its constituent engines are.
    ("nbn-hybrid-router", "discrete"): True,
    ("nbn-hybrid-router", "continuous_lg"): True,
    ("nbn-hybrid-router", "continuous_nongauss"): True,
    ("nbn-hybrid-router", "hybrid"): True,
    # Amortized neural-proposal IS (v0.14, #181): same accuracy class and
    # applicable families as LW (it is LW with a learned proposal).
    ("nbn-cat-ais", "discrete"): True,
    ("nbn-cat-ais", "continuous_lg"): False,
    ("nbn-cat-ais", "continuous_nongauss"): False,
    ("nbn-cat-ais", "hybrid"): False,
    ("nbn-neuralcat-ais", "discrete"): True,
    ("nbn-neuralcat-ais", "continuous_lg"): False,
    ("nbn-neuralcat-ais", "continuous_nongauss"): False,
    ("nbn-neuralcat-ais", "hybrid"): False,
    ("nbn-lg-ais", "discrete"): False,
    ("nbn-lg-ais", "continuous_lg"): True,
    ("nbn-lg-ais", "continuous_nongauss"): False,
    ("nbn-lg-ais", "hybrid"): False,
    ("nbn-mdn-ais", "discrete"): False,
    ("nbn-mdn-ais", "continuous_lg"): True,
    ("nbn-mdn-ais", "continuous_nongauss"): True,
    ("nbn-mdn-ais", "hybrid"): True,
    ("nbn-flow-ais", "discrete"): False,
    ("nbn-flow-ais", "continuous_lg"): True,
    ("nbn-flow-ais", "continuous_nongauss"): True,
    ("nbn-flow-ais", "hybrid"): True,
    # Amortized variational inference (v0.14, #182): bounded-ELBO engine,
    # same applicable families as LW / AIS.
    ("nbn-cat-avi", "discrete"): True,
    ("nbn-cat-avi", "continuous_lg"): False,
    ("nbn-cat-avi", "continuous_nongauss"): False,
    ("nbn-cat-avi", "hybrid"): False,
    ("nbn-neuralcat-avi", "discrete"): True,
    ("nbn-neuralcat-avi", "continuous_lg"): False,
    ("nbn-neuralcat-avi", "continuous_nongauss"): False,
    ("nbn-neuralcat-avi", "hybrid"): False,
    ("nbn-lg-avi", "discrete"): False,
    ("nbn-lg-avi", "continuous_lg"): True,
    ("nbn-lg-avi", "continuous_nongauss"): False,
    ("nbn-lg-avi", "hybrid"): False,
    ("nbn-mdn-avi", "discrete"): False,
    ("nbn-mdn-avi", "continuous_lg"): True,
    ("nbn-mdn-avi", "continuous_nongauss"): True,
    ("nbn-mdn-avi", "hybrid"): True,
    ("nbn-flow-avi", "discrete"): False,
    ("nbn-flow-avi", "continuous_lg"): True,
    ("nbn-flow-avi", "continuous_nongauss"): True,
    ("nbn-flow-avi", "hybrid"): True,
    # Other libraries
    # Gated out pending v0.8 BN-inference adapter (issue #96).
    ("gpytorch-gp", "discrete"): False,
    ("gpytorch-gp", "continuous_lg"): False,
    ("gpytorch-gp", "continuous_nongauss"): False,
    ("gpytorch-gp", "hybrid"): False,
    ("gpytorch-gp-predict", "discrete"): False,
    ("gpytorch-gp-predict", "continuous_lg"): False,
    ("gpytorch-gp-predict", "continuous_nongauss"): False,
    ("gpytorch-gp-predict", "hybrid"): False,
    ("pomegranate-discrete", "discrete"): True,
    ("pomegranate-discrete", "continuous_lg"): False,
    ("pomegranate-discrete", "continuous_nongauss"): False,
    ("pomegranate-discrete", "hybrid"): False,
    ("pomegranate-discrete-ve", "discrete"): True,
    ("pomegranate-discrete-ve", "continuous_lg"): False,
    ("pomegranate-discrete-ve", "continuous_nongauss"): False,
    ("pomegranate-discrete-ve", "hybrid"): False,
    ("pyro-empirical", "discrete"): True,
    ("pyro-empirical", "continuous_lg"): True,
    ("pyro-empirical", "continuous_nongauss"): False,
    # hybrid added: one-hot regression for discrete parents (issue #61).
    ("pyro-empirical", "hybrid"): True,
    ("pyro-empirical-importance", "discrete"): True,
    ("pyro-empirical-importance", "continuous_lg"): True,
    ("pyro-empirical-importance", "continuous_nongauss"): False,
    ("pyro-empirical-importance", "hybrid"): True,
}


_FROZEN_GATE_B_PAIRS: list[tuple[str, str]] = [
    ("discrete", "gpytorch"),
    ("continuous_lg", "pomegranate"),
    ("continuous_nongauss", "pomegranate"),
    ("continuous_lg", "nbn_ve"),
    ("continuous_nongauss", "nbn_ve"),
    ("continuous_nongauss", "pgmpy"),
    ("hybrid", "gpytorch"),
    ("hybrid", "pomegranate"),
    ("hybrid", "pgmpy"),
    ("hybrid", "nbn_ve"),
]


_ADAPTER_TO_LABELS: dict[str, list[str]] = {
    "gpytorch":    ["gpytorch-gp", "gpytorch-gp-predict"],
    "pomegranate": ["pomegranate-discrete", "pomegranate-discrete-ve"],
    "nbn_ve":      ["nbn-cat-ve", "nbn-neuralcat-ve"],
    "pgmpy":       ["pgmpy-mle", "pgmpy-bayes", "pgmpy-lg",
                    "pgmpy-mle-ve", "pgmpy-bayes-ve", "pgmpy-lg-predict"],
}


# Gate C pairs were (continuous_lg/nongauss, gpytorch-gp/gp-predict).
# gpytorch-gp and gpytorch-gp-predict are now gated out of all families
# (issue #96), so is_applicable returns False — the Gate C
# accuracy_supported precondition is no longer meaningful.  The snapshot
# table above covers the False applicability; the Gate C test is removed.
_FROZEN_GATE_C_PAIRS: list[tuple[str, str]] = []


@pytest.mark.parametrize(
    ("label", "family"),
    sorted(_EXPECTED_APPLICABILITY.keys()),
    ids=lambda lf: f"{lf}",
)
def test_is_applicable_matches_post_consolidation_snapshot(
    label: str, family: str,
) -> None:
    """Cartesian product (label, family) — snapshot test."""
    expected = _EXPECTED_APPLICABILITY[(label, family)]
    assert is_applicable(label, family) is expected, (
        f"is_applicable({label!r}, {family!r}) returned "
        f"{is_applicable(label, family)}, expected {expected}"
    )


def test_snapshot_covers_every_known_label() -> None:
    """The snapshot must cover every label in the registry × every family."""
    snapshot_labels = {label for (label, _) in _EXPECTED_APPLICABILITY}
    registry_labels = set(known_labels())
    missing = registry_labels - snapshot_labels
    extra = snapshot_labels - registry_labels
    assert not missing, f"snapshot missing labels: {missing}"
    assert not extra, f"snapshot has labels not in registry: {extra}"
    snapshot_pairs = set(_EXPECTED_APPLICABILITY.keys())
    expected_pairs = {(lab, fam) for lab in registry_labels for fam in FAMILIES}
    assert snapshot_pairs == expected_pairs, (
        f"snapshot pair coverage mismatch; "
        f"missing: {expected_pairs - snapshot_pairs}, "
        f"extra: {snapshot_pairs - expected_pairs}"
    )


@pytest.mark.parametrize(
    ("family", "adapter", "label"),
    [
        (family, adapter, label)
        for (family, adapter) in _FROZEN_GATE_B_PAIRS
        for label in _ADAPTER_TO_LABELS[adapter]
    ],
)
def test_frozen_gate_b_pairs_still_rejected(
    family: str, adapter: str, label: str,
) -> None:
    """Pre-consolidation Gate B coverage anchor."""
    assert not is_applicable(label, family), (
        f"Gate-B regression: (family={family!r}, adapter={adapter!r}) "
        f"used to be in _NOT_APPLICABLE; label {label!r} translates "
        f"to that adapter and must still be rejected by the registry."
    )


@pytest.mark.parametrize(
    ("family", "label"),
    _FROZEN_GATE_C_PAIRS,
)
def test_frozen_gate_c_pairs_have_accuracy_unsupported(
    family: str, label: str,
) -> None:
    """Pre-consolidation Gate C → accuracy_supported translation."""
    assert is_applicable(label, family), (
        f"precondition: ({label!r}, {family!r}) should be applicable "
        f"per the registry; if not, the test is vacuous."
    )
    assert not accuracy_supported(label, family), (
        f"accuracy_supported({label!r}, {family!r}) returned True; "
        f"this pair was in _ACCURACY_NOT_APPLICABLE pre-consolidation "
        f"and must remain accuracy-unsupported."
    )


def test_accuracy_supported_default_true_for_non_gpytorch() -> None:
    """Every non-gpytorch (label, family) pair where the label is
    applicable must report accuracy_supported=True."""
    for label in known_labels():
        if label.startswith("gpytorch-"):
            continue
        entry = _BASELINE_APPLICABILITY[label]
        for family in entry.families:
            assert accuracy_supported(label, family) is True, (
                f"non-gpytorch label {label!r} on family {family!r} "
                f"unexpectedly reports accuracy_supported=False"
            )


def test_unknown_label_returns_false_from_both_accessors() -> None:
    """Unknown-label fallback contract."""
    assert is_applicable("foo-bar-baz", "discrete") is False
    assert is_applicable("nbn-cat-typo", "continuous_lg") is False
    assert accuracy_supported("foo-bar-baz", "discrete") is False
    assert accuracy_supported("nbn-cat-typo", "hybrid") is False


def test_accuracy_supported_false_for_label_not_applicable_to_family() -> None:
    """Defensive contract: accuracy_supported is False for
    (label, family) pairs the label doesn't apply to."""
    assert accuracy_supported("nbn-cat", "continuous_lg") is False
    assert accuracy_supported("nbn-cat", "hybrid") is False
    assert accuracy_supported("gpytorch-gp", "discrete") is False
    assert accuracy_supported("gpytorch-gp", "hybrid") is False


def test_baseline_applicability_dataclass_is_frozen() -> None:
    """BaselineApplicability instances must be frozen."""
    entry = _BASELINE_APPLICABILITY["nbn-cat"]
    assert isinstance(entry, BaselineApplicability)
    with pytest.raises((AttributeError, TypeError)):
        entry.families = frozenset({"hybrid"})  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        entry.accuracy_supported = False  # type: ignore[misc]
