"""The baseline applicability registry, pinned as one table.

``nbn.bench.core.applicability`` decides which baseline runs on which family
(``is_applicable``) and whether a cell may emit accuracy rows
(``accuracy_supported``).  Both are pinned here against a full
label x family snapshot, compared in one assertion so a change shows up as a
single readable diff rather than a few hundred parametrised failures.

History: this table replaced two gate sets (``_NOT_APPLICABLE`` and
``_ACCURACY_NOT_APPLICABLE``) in Pass 10; every pair those gates rejected is
a ``False`` row below.  gpytorch's labels stay registered with no family and
``accuracy_supported=False`` until a BN-inference adapter exists (#96) --
SVGPs return the prior marginal regardless of evidence.
"""
from __future__ import annotations

import pytest

from nbn.bench.core.applicability import (
    BASELINE_FAMILY_APPLICABILITY,
    BaselineApplicability,
    accuracy_supported,
    is_applicable,
    known_labels,
)

FAMILIES = ("discrete", "continuous_lg", "continuous_nongauss", "hybrid")

#: Labels whose registry entry has ``accuracy_supported=False`` (#96).
_ACCURACY_UNSUPPORTED = {"gpytorch-gp", "gpytorch-gp-predict"}


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
    # pomegranate's predict_proba is loopy BP: "-lbp" is the label since D10,
    # "-ve" its pre-D10 alias (kept so old result files keep their
    # applicability) — the two must stay identical.
    ("pomegranate-discrete-lbp", "discrete"): True,
    ("pomegranate-discrete-lbp", "continuous_lg"): False,
    ("pomegranate-discrete-lbp", "continuous_nongauss"): False,
    ("pomegranate-discrete-lbp", "hybrid"): False,
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


def test_registry_matches_the_snapshot():
    """Every (label, family): is_applicable and accuracy_supported, one diff."""
    labels = set(known_labels())
    snap_labels = {label for label, _ in _EXPECTED_APPLICABILITY}
    assert labels == snap_labels, (
        f"registry vs snapshot labels: missing {labels - snap_labels}, "
        f"extra {snap_labels - labels}"
    )
    expected_pairs = {(lab, fam) for lab in labels for fam in FAMILIES}
    assert set(_EXPECTED_APPLICABILITY) == expected_pairs

    want = {
        k: (v, v and k[0] not in _ACCURACY_UNSUPPORTED)
        for k, v in _EXPECTED_APPLICABILITY.items()
    }
    got = {k: (is_applicable(*k), accuracy_supported(*k)) for k in want}
    diff = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    assert not diff, f"(got, want) per (label, family): {diff}"


def test_entry_level_accuracy_flag():
    """Only gpytorch carries accuracy_supported=False on the entry itself."""
    flagged = {lab for lab, e in BASELINE_FAMILY_APPLICABILITY.items()
               if not e.accuracy_supported}
    assert flagged == _ACCURACY_UNSUPPORTED
    for lab in _ACCURACY_UNSUPPORTED:
        assert BASELINE_FAMILY_APPLICABILITY[lab].families == frozenset()


@pytest.mark.parametrize("label,family", [
    ("foo-bar-baz", "discrete"), ("nbn-cat-typo", "continuous_lg"),
    ("nbn-cat-typo", "hybrid"),
])
def test_unknown_label_is_rejected_by_both_accessors(label, family):
    assert is_applicable(label, family) is False
    assert accuracy_supported(label, family) is False


def test_baseline_applicability_dataclass_is_frozen():
    entry = BASELINE_FAMILY_APPLICABILITY["nbn-cat"]
    assert isinstance(entry, BaselineApplicability)
    with pytest.raises((AttributeError, TypeError)):
        entry.families = frozenset({"hybrid"})  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        entry.accuracy_supported = False  # type: ignore[misc]
