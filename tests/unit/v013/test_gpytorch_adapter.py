"""Tests for gpytorch adapter contracts in v0.13.

Decision 2 (Phase 0 hard gate): port both gpytorch UNIQUE tests from
tests/integration/test_inference_accuracy_oracle.py and
tests/integration/test_adapters_smoke.py.

Context
-------
The v0.13 nbn.bench.adapters directory does NOT yet contain a
gpytorch_adapter.py — the GPyTorch BN-inference adapter is deferred to v0.8
(issue #96).  The applicability registry (nbn.bench.core.applicability)
does hold the gpytorch labels with ``families=frozenset()`` and
``accuracy_supported=False``.

What is testable in v0.13
--------------------------
1. Registry correctly marks gpytorch labels:
   - ``families`` is empty (no benchmark family applicable, pending #96).
   - ``accuracy_supported`` is False for both gpytorch labels.
   - ``is_applicable("gpytorch-gp-predict", any_family)`` returns False.
   - ``accuracy_supported("gpytorch-gp-predict", any_family)`` returns False.

2. When a benchmark run encounters a gpytorch baseline, the runner emits a
   not_supported row (driven by is_applicable returning False).  This is
   covered by the Runner's TestNotSupportedPath in test_runner.py; the
   registry-level assertions below are the anchor.

3. gpytorch-specific discrete-evidence test (legacy test_gpytorch_adapter_
   skips_discrete_evidence): Cannot be ported yet — no v0.13 GPyTorchAdapter
   class exists.  Tracked in issue #96; port when the adapter lands.

Reference: docs/v0.13-benchmark-redesign.md §1c
"""
from __future__ import annotations

import pytest

from nbn.bench.core.applicability import (
    BASELINE_FAMILY_APPLICABILITY,
    accuracy_supported,
    is_applicable,
)

_GPYTORCH_LABELS = ["gpytorch-gp", "gpytorch-gp-predict"]
_ALL_FAMILIES = ["discrete", "continuous_lg", "continuous_nongauss", "hybrid"]


# ---------------------------------------------------------------------------
# Registry: applicability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", _GPYTORCH_LABELS)
def test_gpytorch_label_in_registry(label: str) -> None:
    """Both gpytorch labels must exist in the applicability registry."""
    assert label in BASELINE_FAMILY_APPLICABILITY, (
        f"{label!r} missing from BASELINE_FAMILY_APPLICABILITY — "
        f"add it to nbn/bench/core/applicability.py"
    )


@pytest.mark.parametrize("label", _GPYTORCH_LABELS)
def test_gpytorch_families_empty(label: str) -> None:
    """gpytorch labels have families=frozenset() (excluded from runs, issue #96)."""
    entry = BASELINE_FAMILY_APPLICABILITY[label]
    assert entry.families == frozenset(), (
        f"{label!r}.families must be frozenset() pending #96 BN-inference "
        f"adapter; got {entry.families}"
    )


@pytest.mark.parametrize("label", _GPYTORCH_LABELS)
@pytest.mark.parametrize("family", _ALL_FAMILIES)
def test_gpytorch_is_not_applicable_to_any_family(
    label: str, family: str,
) -> None:
    """is_applicable returns False for gpytorch on any standard family."""
    assert not is_applicable(label, family), (
        f"is_applicable({label!r}, {family!r}) must be False "
        f"(empty families set, pending #96)"
    )


# ---------------------------------------------------------------------------
# Registry: accuracy_supported=False
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", _GPYTORCH_LABELS)
def test_gpytorch_accuracy_not_supported_flag(label: str) -> None:
    """gpytorch labels must have accuracy_supported=False.

    GPyTorch SVGPs return the prior marginal independent of evidence, so
    distributional accuracy metrics (W₁, TV, JSD) cannot meaningfully
    compare their output to oracle posteriors.  The flag drives the runner
    to short-circuit accuracy rows to not_supported while still emitting
    the speed row.
    """
    entry = BASELINE_FAMILY_APPLICABILITY[label]
    assert not entry.accuracy_supported, (
        f"{label!r}.accuracy_supported must be False; got True. "
        f"GPyTorch SVGPs return prior marginals regardless of evidence."
    )


@pytest.mark.parametrize("label", _GPYTORCH_LABELS)
@pytest.mark.parametrize("family", _ALL_FAMILIES)
def test_gpytorch_accuracy_supported_returns_false(
    label: str, family: str,
) -> None:
    """accuracy_supported() returns False for gpytorch on any standard family."""
    assert not accuracy_supported(label, family), (
        f"accuracy_supported({label!r}, {family!r}) must be False"
    )
