"""Baseline registry for v0.6c-C-1a — method-keyed labels and applicability.

A "baseline" in v0.6c-C+ is a structured tuple
``(library, mechanism, param_method, [inference_method])`` that the
runner derives a canonical string label from.  Every parquet
``baseline`` column value is one of these labels; figures legend by
them; the applicability matrix gates non-applicable cells.

v0.6c-C-1a (this file) ships:

* The label scheme (``_label_from_spec``).
* The applicability matrix (``_BASELINE_APPLICABILITY``).
* ``BaselineSpec`` dataclass + ``known_labels()``.

v0.6c-C-1b landed the runner refactor that consumes this registry
for fit-then-query semantics and gates dispatch via the applicability
matrix.  Pass-10 priority-1 consolidated the historically-parallel
``_NOT_APPLICABLE`` and ``_ACCURACY_NOT_APPLICABLE`` tables from
``crash_test_runner.py`` into ``BaselineApplicability`` entries below,
making this registry the single source of truth.

Adding a new baseline requires:

1. Add the label + applicable families to ``_BASELINE_APPLICABILITY``.
2. Wire the adapter to recognise the spec's ``mechanism`` /
   ``param_method`` / ``inference_method`` fields (v0.6c-C-1b).
3. Add the spec to the relevant YAML configs (smoke and/or paper).

Notes on what's deferred from v0.6c-C-1a:

* ``nbn-neuralcat-ve`` was deferred at v0.6c-C-1a — the §4.4
  diagnostic on PR #28 showed ``TensorVariableElimination._extract_factors``
  structurally required ``mech._logits``, but
  ``NeuralCategoricalMechanism`` computes logits per-parent via a
  neural-net forward pass and has no tabulated form.  v0.8-#26
  (PR landing this comment) closes the gap: the engine now reads
  CPDs via ``mech.tabulate(parent_cards)``, which enumerates parent
  configurations once for forward-based mechanisms.
  ``nbn-neuralcat-ve`` is now in the registry.

* gpytorch / pomegranate / pyro adapters keep their pre-PR labels
  for now; v0.6c-C-2 expands them with the same spec-based dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Mapping, Sequence


@dataclass(frozen=True)
class BaselineSpec:
    """Structured baseline identifier.

    ``inference_method`` is ``None`` for parameter-learning specs
    (param-learning crash test does not invoke an inference engine —
    accuracy is per-CPD; speed is direct CPD lookup, landing in
    v0.6c-C-1b).
    """

    library: str            # 'pgmpy' | 'nbn' | 'gpytorch' | 'pomegranate' | 'pyro'
    mechanism: str          # 'discrete' | 'lg' | 'cat' | 'neuralcat' | 'mdn' | 'flow' | 'hybrid' | 'gp'
    param_method: str       # 'mle' | 'bayes'
    inference_method: str | None = None  # 've' | 'lw' | 'predict' | 'router' | None


def _label_from_spec(spec: BaselineSpec | Mapping[str, str | None]) -> str:
    """Canonical string label.

    Inference baselines: ``"<library>-<param_method>-<inference_method>"``
    for pgmpy (where param_method names disambiguate mle/bayes); or
    ``"<library>-<mechanism>-<inference_method>"`` for libraries where
    the mechanism is the headline distinguisher (NBN, gpytorch, ...).

    Parameter-learning baselines: same, minus the trailing
    ``-<inference_method>`` segment.

    The scheme intentionally privileges the spec field that *varies*
    per library:

    * pgmpy has multiple param_methods on discrete (mle/bayes) and a
      single mechanism per family (DiscreteBN / LinearGaussianBN), so
      the param_method is the headline distinguisher (``pgmpy-mle-ve``,
      ``pgmpy-bayes-ve``, ``pgmpy-lg-predict``).
    * NBN has a single param_method (mle via gradient descent) and
      multiple mechanisms per family (cat/neuralcat/lg/mdn/flow), so
      the mechanism is the headline distinguisher (``nbn-cat-ve``,
      ``nbn-mdn-lw``, etc.).
    """
    if isinstance(spec, BaselineSpec):
        library, mechanism = spec.library, spec.mechanism
        param_method, inference_method = spec.param_method, spec.inference_method
    else:
        library = str(spec["library"])
        mechanism = str(spec["mechanism"])
        param_method = str(spec["param_method"])
        inference_method = spec.get("inference_method")  # type: ignore[arg-type]

    if library == "pgmpy":
        # pgmpy: param_method is the headline distinguisher on discrete
        # (mle vs bayes); mechanism collapses to lg for continuous.
        if mechanism == "lg":
            head = f"{library}-lg"
        else:
            head = f"{library}-{param_method}"
    else:
        # nbn / gpytorch / pomegranate / pyro: mechanism is the headline
        # distinguisher (or the only library-specific identifier).
        head = f"{library}-{mechanism}"

    if inference_method:
        return f"{head}-{inference_method}"
    return head


@dataclass(frozen=True)
class BaselineApplicability:
    """Per-label applicability + per-label capability flags.

    ``families``: the families on which the label can be dispatched
    (replaces the bare ``frozenset[str]`` value the registry previously
    held).

    ``accuracy_supported``: True (default) iff the label can produce a
    meaningful accuracy row.  GPyTorch SVGPs return the prior marginal
    independent of evidence, so they emit honest speed measurements
    but cannot score posterior accuracy — these set this flag to False
    and the inference cell short-circuits the accuracy row to
    ``not_supported`` while still emitting the speed row.  Replaces the
    old ``_ACCURACY_NOT_APPLICABLE`` set in ``crash_test_runner.py``.
    """

    families: FrozenSet[str]
    accuracy_supported: bool = True


# Family applicability + capability flags per baseline label.
# Single source of truth — replaces the legacy ``_NOT_APPLICABLE`` and
# ``_ACCURACY_NOT_APPLICABLE`` sets that lived in
# ``crash_test_runner.py`` (Pass-10 priority-1 consolidation, v0.10).
_BASELINE_APPLICABILITY: dict[str, BaselineApplicability] = {
    # --- Parameter learning (no inference_method suffix) ---
    "pgmpy-mle":               BaselineApplicability(frozenset({"discrete"})),
    "pgmpy-bayes":             BaselineApplicability(frozenset({"discrete"})),
    "pgmpy-lg":                BaselineApplicability(frozenset({"continuous_lg"})),
    "nbn-cat":                 BaselineApplicability(frozenset({"discrete"})),
    "nbn-neuralcat":           BaselineApplicability(frozenset({"discrete"})),
    "nbn-lg":                  BaselineApplicability(frozenset({"continuous_lg"})),
    # Pass-10 priority-1: hybrid added (verified WORKS on smoke n=5).
    "nbn-mdn":                 BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    # Pass-10 priority-1: hybrid added (verified WORKS on laptop with zuko;
    # Section 6's pre-flight verification gates this entry's hybrid inclusion).
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
    # Bug B (NaN-driven multinomial CUDA assert on continuous_nongauss) fixed
    # by parent standardization in MDNMechanism.fit_local (issue #81, #24, #54).
    # Extreme parent values (e.g. X21=3.3M from deep chain variance amplification)
    # caused log_scale overflow → backward NaN → CUDA assert at LW inference.
    # Standardizing parents to O(1) scale in fit_local + _params_from_parents
    # keeps log_scale finite.  Validated: continuous_nongauss n=50 seed=3 PASS.
    "nbn-mdn-lw":              BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    # nbn-flow-lw uses MDN internally — same fix applies transitively.
    "nbn-flow-lw":             BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss", "hybrid"})),
    "nbn-hybrid-router":       BaselineApplicability(frozenset({"hybrid"})),

    # --- Other libraries ---
    # gpytorch-gp / gpytorch-gp-predict: per-node SVGP regression baseline.
    # Fits one SVGP per non-root node modeling P(node | direct_parents).
    # At query time, evaluates the target's SVGP at evidence-derived parent
    # inputs via q.evidence.get(parent, 0.0).
    #
    # IMPORTANT: The benchmark's evidence is on descendants/laterals of the
    # target, never on its direct parents.  Parent inputs therefore default
    # to 0 and predictions are independent of the actual evidence values
    # (std ≈ 0.05 across all B batch items at any evidence setting).
    # Predictions are effectively the conditional mean at the prior parent
    # value (0), not a Bayesian-inference output.
    #
    # accuracy_supported=False is therefore correct: distributional metrics
    # (accuracy/jsd/tv/w1) cannot meaningfully compare these constant
    # predictions to oracle-conditional posteriors.
    #
    # Note on total_time_s: this measures the query phase only (~1s at
    # all n).  Fit time is ~96s at n=100 and scales linearly with n (not
    # counted in the benchmark timing column).  See issue #96 for the
    # full architectural analysis.
    "gpytorch-gp": BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss"}),
        accuracy_supported=False,
    ),
    "gpytorch-gp-predict": BaselineApplicability(
        frozenset({"continuous_lg", "continuous_nongauss"}),
        accuracy_supported=False,
    ),
    "pomegranate-discrete":      BaselineApplicability(frozenset({"discrete"})),
    "pomegranate-discrete-ve":   BaselineApplicability(frozenset({"discrete"})),
    "pyro-empirical":            BaselineApplicability(
        frozenset({"discrete", "continuous_lg"})),
    "pyro-empirical-importance": BaselineApplicability(
        frozenset({"discrete", "continuous_lg"})),
}


def is_applicable(label: str, family: str) -> bool:
    """True iff the baseline is applicable to the family.

    Unknown labels return ``False`` rather than raising — the runner
    emits a ``not_supported`` row for unknown labels, so a typo in a
    YAML produces a clean parquet rather than a hard crash.
    """
    entry = _BASELINE_APPLICABILITY.get(label)
    if entry is None:
        return False
    return family in entry.families


def accuracy_supported(label: str, family: str) -> bool:
    """True iff the (label, family) cell can emit a meaningful
    accuracy row.  Used by ``_inference_cell`` to short-circuit the
    accuracy row to ``not_supported`` while still emitting the speed
    row (Pass-9 Phase 3 Option I).

    Returns ``False`` for unknown labels and for labels not applicable
    to the given family — both cases are guarded by the runner's
    upstream ``is_applicable`` gate, but defensive False here keeps
    the contract clean.
    """
    entry = _BASELINE_APPLICABILITY.get(label)
    if entry is None or family not in entry.families:
        return False
    return entry.accuracy_supported


def known_labels() -> Sequence[str]:
    """Sorted list of all known baseline labels.  Useful for tests
    and for YAML validation."""
    return sorted(_BASELINE_APPLICABILITY.keys())


__all__ = [
    "BaselineApplicability",
    "BaselineSpec",
    "_label_from_spec",
    "accuracy_supported",
    "is_applicable",
    "known_labels",
    "_BASELINE_APPLICABILITY",
]
