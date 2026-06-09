"""Baseline applicability registry for v0.13.

Moved from ``benchmarking/_baseline_registry.py`` (v0.12).  The data
constant is renamed ``BASELINE_FAMILY_APPLICABILITY`` (no leading
underscore); all function signatures are preserved so call sites only
need to update their import path.

Reference: docs/v0.13-benchmark-redesign.md §1c
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Sequence


@dataclass(frozen=True)
class BaselineApplicability:
    """Per-label applicability + capability flags.

    ``families``: families on which the label can be dispatched.
    ``accuracy_supported``: False for baselines that emit honest speed
    measurements but cannot score posterior accuracy (e.g. GPyTorch SVGPs
    return the prior marginal independent of evidence).
    """

    families: FrozenSet[str]
    accuracy_supported: bool = True


# Single source of truth for label → applicable families.
# Replaces the v0.12 ``_BASELINE_APPLICABILITY`` dict in
# ``benchmarking/_baseline_registry.py``.
BASELINE_FAMILY_APPLICABILITY: dict[str, BaselineApplicability] = {
    # --- Parameter learning (no inference_method suffix) ---
    "pgmpy-mle":               BaselineApplicability(frozenset({"discrete"})),
    "pgmpy-bayes":             BaselineApplicability(frozenset({"discrete"})),
    "pgmpy-lg":                BaselineApplicability(frozenset({"continuous_lg"})),
    "nbn-cat":                 BaselineApplicability(frozenset({"discrete"})),
    "nbn-neuralcat":           BaselineApplicability(frozenset({"discrete"})),
    "nbn-lg":                  BaselineApplicability(frozenset({"continuous_lg"})),
    "nbn-mdn":                 BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-flow":                BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-hybrid":              BaselineApplicability(frozenset({"hybrid"})),

    # --- Inference ---
    "pgmpy-mle-ve":            BaselineApplicability(frozenset({"discrete"})),
    "pgmpy-bayes-ve":          BaselineApplicability(frozenset({"discrete"})),
    "pgmpy-lg-predict":        BaselineApplicability(frozenset({"continuous_lg"})),
    "nbn-cat-ve":              BaselineApplicability(frozenset({"discrete"})),
    "nbn-cat-lw":              BaselineApplicability(frozenset({"discrete"})),
    "nbn-neuralcat-ve":        BaselineApplicability(frozenset({"discrete"})),
    "nbn-neuralcat-lw":        BaselineApplicability(frozenset({"discrete"})),
    "nbn-lg-lw":               BaselineApplicability(frozenset({"continuous_lg"})),
    "nbn-mdn-lw":              BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-flow-lw":             BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-hybrid-router":       BaselineApplicability(frozenset({"hybrid"})),

    # Amortized neural-proposal IS (v0.14, #181) — same accuracy class /
    # applicable families as LW (it is LW with a learned proposal).
    "nbn-cat-ais":             BaselineApplicability(frozenset({"discrete"})),
    "nbn-neuralcat-ais":       BaselineApplicability(frozenset({"discrete"})),
    "nbn-lg-ais":              BaselineApplicability(frozenset({"continuous_lg"})),
    "nbn-mdn-ais":             BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-flow-ais":            BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),

    # --- Other libraries ---
    # gpytorch-gp / gpytorch-gp-predict: per-node SVGP regression.
    # accuracy_supported=False: predictions are independent of evidence
    # (prior marginal at parent=0), so distributional metrics are
    # meaningless.  See issue #96 for the full architectural analysis.
    # Excluded from benchmark applicability pending v0.8 BN-inference
    # adapter (issue #96).
    "gpytorch-gp": BaselineApplicability(
        frozenset(),
        accuracy_supported=False,
    ),
    "gpytorch-gp-predict": BaselineApplicability(
        frozenset(),
        accuracy_supported=False,
    ),
    "pomegranate-discrete":      BaselineApplicability(frozenset({"discrete"})),
    "pomegranate-discrete-ve":   BaselineApplicability(frozenset({"discrete"})),
    "pyro-empirical":            BaselineApplicability(
        frozenset({"discrete", "continuous_lg", "hybrid"})),
    "pyro-empirical-importance": BaselineApplicability(
        frozenset({"discrete", "continuous_lg", "hybrid"})),
}


def is_applicable(label: str, family: str) -> bool:
    """True iff the baseline is applicable to the family.

    Unknown labels return ``False`` — the runner emits a ``not_supported``
    row for unknown labels, so a typo in a YAML produces a clean parquet
    rather than a hard crash.
    """
    entry = BASELINE_FAMILY_APPLICABILITY.get(label)
    if entry is None:
        return False
    return family in entry.families


def accuracy_supported(label: str, family: str) -> bool:
    """True iff the (label, family) cell can emit a meaningful accuracy row.

    Returns ``False`` for unknown labels and for labels not applicable to
    the given family.
    """
    entry = BASELINE_FAMILY_APPLICABILITY.get(label)
    if entry is None or family not in entry.families:
        return False
    return entry.accuracy_supported


def known_labels() -> Sequence[str]:
    """Sorted list of all known baseline labels."""
    return sorted(BASELINE_FAMILY_APPLICABILITY.keys())


__all__ = [
    "BaselineApplicability",
    "BASELINE_FAMILY_APPLICABILITY",
    "is_applicable",
    "accuracy_supported",
    "known_labels",
]
