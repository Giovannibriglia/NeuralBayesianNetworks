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

v0.6c-C-1b will land the runner refactor that consumes this registry
for fit-then-query semantics and gates dispatch via the applicability
matrix.  In 1a the registry is consumed only for parquet relabeling;
the legacy ``_NOT_APPLICABLE`` table in ``crash_test_runner.py`` still
controls dispatch.

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


# Family applicability per baseline label.  The applicability matrix
# is consumed by v0.6c-C-1b's runner refactor for early-skip dispatch;
# in v0.6c-C-1a it is only exercised by tests.
_BASELINE_APPLICABILITY: dict[str, FrozenSet[str]] = {
    # --- Parameter learning (no inference_method suffix) ---
    "pgmpy-mle":               frozenset({"discrete"}),
    "pgmpy-bayes":             frozenset({"discrete"}),
    "pgmpy-lg":                frozenset({"continuous_lg"}),
    "nbn-cat":                 frozenset({"discrete"}),
    "nbn-neuralcat":           frozenset({"discrete"}),
    "nbn-lg":                  frozenset({"continuous_lg"}),
    "nbn-mdn":                 frozenset({"continuous_lg", "continuous_nongauss"}),
    "nbn-flow":                frozenset({"continuous_lg", "continuous_nongauss"}),
    "nbn-hybrid":              frozenset({"hybrid"}),

    # --- Inference ---
    "pgmpy-mle-ve":            frozenset({"discrete"}),
    "pgmpy-bayes-ve":          frozenset({"discrete"}),
    "pgmpy-lg-predict":        frozenset({"continuous_lg"}),
    "nbn-cat-ve":              frozenset({"discrete"}),
    "nbn-cat-lw":              frozenset({"discrete"}),
    # v0.8-#26: nbn-neuralcat-ve was deferred at v0.6c-C-1a because
    # TensorVariableElimination structurally required a tabulated
    # _logits tensor; the engine now reads CPDs via
    # mech.tabulate(parent_cards), which enumerates parent
    # configurations once for forward-based mechanisms like
    # NeuralCategoricalMechanism.  Discrete-only.
    "nbn-neuralcat-ve":        frozenset({"discrete"}),
    "nbn-neuralcat-lw":        frozenset({"discrete"}),
    "nbn-lg-lw":               frozenset({"continuous_lg"}),
    "nbn-mdn-lw":              frozenset({"continuous_lg", "continuous_nongauss"}),
    "nbn-flow-lw":             frozenset({"continuous_lg", "continuous_nongauss"}),
    "nbn-hybrid-router":       frozenset({"hybrid"}),

    # --- Other libraries (v0.6c-C-2) ---
    # gpytorch: SVGP per continuous node (independent leaves on parents).
    # Already vectorised in the v1 adapter (pred.sample(torch.Size([n])))
    # so no per-method fan-out beyond gp / gp-predict.  GPs cannot
    # condition on discrete evidence, so applicability is continuous-only.
    "gpytorch-gp":               frozenset({"continuous_lg", "continuous_nongauss"}),
    "gpytorch-gp-predict":       frozenset({"continuous_lg", "continuous_nongauss"}),

    # pomegranate v1.x: empirical-CPT BayesianNetwork on torch backend.
    # Discrete-only.  Single canonical method (Laplace-smoothed counts +
    # predict_proba).  Known v1 bug: predict_proba raises IndexError on
    # conditional queries (filed as v0.7 issue).  Applicability matrix
    # marks discrete-only; cells with continuous targets/evidence skip.
    "pomegranate-discrete":      frozenset({"discrete"}),
    "pomegranate-discrete-ve":   frozenset({"discrete"}),

    # pyro: Importance sampler over an ancestral generative model.
    # Empirical CPT for discrete; linear-Gaussian conditional
    # ``Normal(beta_0 + sum_i beta_i * pa_i, sigma)`` for continuous_lg
    # (#32 fix).  Applicability covers discrete + continuous_lg;
    # continuous_nongauss and hybrid stay excluded — non-Gaussian
    # continuous would need SVI with a parameterised guide, and the
    # hybrid mixed-parent (discrete-parent / continuous-child) case
    # is tracked as a v0.8 follow-up.  Mechanism label "empirical"
    # describes the empirical-distribution-fit style; inference_method
    # "importance" is pyro.infer.Importance.
    "pyro-empirical":            frozenset({"discrete", "continuous_lg"}),
    "pyro-empirical-importance": frozenset({"discrete", "continuous_lg"}),
}


def is_applicable(label: str, family: str) -> bool:
    """True iff the baseline is applicable to the family.

    Unknown labels return ``False`` rather than raising — the runner
    can emit a ``not_supported`` row for unknown labels with a
    warning, so a typo in a YAML produces a clean parquet rather than
    a hard crash."""
    return family in _BASELINE_APPLICABILITY.get(label, frozenset())


def known_labels() -> Sequence[str]:
    """Sorted list of all known baseline labels.  Useful for tests
    and for YAML validation."""
    return sorted(_BASELINE_APPLICABILITY.keys())


__all__ = [
    "BaselineSpec",
    "_label_from_spec",
    "is_applicable",
    "known_labels",
    "_BASELINE_APPLICABILITY",
]
