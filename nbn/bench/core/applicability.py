"""Baseline applicability registry for v0.13.

Moved from ``nbn/bench/_baseline_registry.py`` (v0.12).  The data
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
# ``nbn/bench/_baseline_registry.py``.
BASELINE_FAMILY_APPLICABILITY: dict[str, BaselineApplicability] = {
    # --- Parameter learning (no inference_method suffix) ---
    "pgmpy-mle":               BaselineApplicability(frozenset({"discrete"})),
    "pgmpy-bayes":             BaselineApplicability(frozenset({"discrete"})),
    "pgmpy-lg":                BaselineApplicability(frozenset({"continuous_lg"})),
    "nbn-cat":                 BaselineApplicability(frozenset({"discrete"})),
    # Laplace-smoothed (alpha=1) empirical CPTs — same table machinery as
    # nbn-cat, Bayesian estimator instead of MLE parity. Registered because
    # the alpha=1 competitors (pyro, pgmpy-bayes) beat nbn-cat on discrete
    # recovery TV/KL at paper scale; the smoothed mechanism already existed
    # in nbn but was never benchmarked.
    "nbn-cat-bayes":           BaselineApplicability(frozenset({"discrete"})),
    "nbn-neuralcat":           BaselineApplicability(frozenset({"discrete"})),
    "nbn-lg":                  BaselineApplicability(frozenset({"continuous_lg"})),
    "nbn-mdn":                 BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-flow":                BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-hybrid":              BaselineApplicability(frozenset({"hybrid"})),
    # Non-parametric continuous mechanisms (#223 / PR 8) — general conditional
    # density estimators, applicable to every continuous family like mdn/flow.
    "nbn-kde":                 BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-knn":                 BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-flexcode":            BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),

    # --- Inference ---
    "pgmpy-mle-ve":            BaselineApplicability(frozenset({"discrete"})),
    "pgmpy-bayes-ve":          BaselineApplicability(frozenset({"discrete"})),
    "pgmpy-lg-predict":        BaselineApplicability(frozenset({"continuous_lg"})),
    "nbn-cat-ve":              BaselineApplicability(frozenset({"discrete"})),
    "nbn-cat-lw":              BaselineApplicability(frozenset({"discrete"})),
    "nbn-cat-bayes-ve":        BaselineApplicability(frozenset({"discrete"})),
    "nbn-cat-bayes-lw":        BaselineApplicability(frozenset({"discrete"})),
    "nbn-neuralcat-ve":        BaselineApplicability(frozenset({"discrete"})),
    "nbn-neuralcat-lw":        BaselineApplicability(frozenset({"discrete"})),
    "nbn-lg-lw":               BaselineApplicability(frozenset({"continuous_lg"})),
    "nbn-mdn-lw":              BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-flow-lw":             BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    # Non-parametric continuous mechanisms (#224 / PR 9) — LW only. The amortized
    # engines (ais/avi) have no RecognitionNetwork proposal head for these
    # mechanism types (recognition_net._head_for raises); deferred to #227.
    "nbn-kde-lw":              BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-knn-lw":              BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-flexcode-lw":         BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    # The router is family-universal by construction: it dispatches
    # all-discrete networks to exact VE (treewidth permitting) and anything
    # else to LW. It was previously gated to {hybrid} only, which — with no
    # hybrid family in any synthetic config — meant the flagship "auto"
    # engine produced zero ok cells in every benchmark run (100%
    # not_supported in the 20260701 parquets). Widened so it competes
    # everywhere its constituent engines do.
    "nbn-hybrid-router":       BaselineApplicability(
        frozenset({"discrete", "continuous_lg", "continuous_nongauss", "hybrid"})),

    # Amortized neural-proposal IS (v0.14, #181) — same accuracy class /
    # applicable families as LW (it is LW with a learned proposal).
    "nbn-cat-ais":             BaselineApplicability(frozenset({"discrete"})),
    "nbn-neuralcat-ais":       BaselineApplicability(frozenset({"discrete"})),
    "nbn-lg-ais":              BaselineApplicability(frozenset({"continuous_lg"})),
    "nbn-mdn-ais":             BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-flow-ais":            BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),

    # Amortized variational inference (v0.14, #182) — bounded-ELBO engine;
    # same applicable families as LW / AIS.
    "nbn-cat-avi":             BaselineApplicability(frozenset({"discrete"})),
    "nbn-neuralcat-avi":       BaselineApplicability(frozenset({"discrete"})),
    "nbn-lg-avi":              BaselineApplicability(frozenset({"continuous_lg"})),
    "nbn-mdn-avi":             BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-flow-avi":            BaselineApplicability(
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
