"""Structural causal models on top of NBN.

* :class:`StructuralCausalModel` — an NBN with explicit noise, noise tying,
  declared hidden common causes, abduction and counterfactuals.
* Structural mechanisms — :class:`DiscreteAdditiveNoiseMechanism`
  (``x = clip(g(pa) + U)``) and :class:`AdditiveNoiseMechanism`
  (``x = g(pa) + U``, ``g`` linear or MLP).
* Diagnostics — noise sharing, tying, freshness, cross-node dependence,
  context (policy) invariance; Benjamini–Hochberg for many tests.
* Structural probability sets — :func:`noise_parameter_set` (a polytope on a
  shared discrete noise law) and :func:`cdf_band_set` (a band on a continuous
  noise CDF).
* Neural mechanisms — :class:`FlowNoiseMechanism` (``x = T(u; pa)``, exact
  abduction) and calibrated uncertainty for any continuous structural
  mechanism: :func:`pit_band_set` (DKW band on the probability integral
  transform), :func:`conformal_intervals` (split / quantile-regression
  conformal), :func:`noise_independence_test`.
* Discrete structural mechanisms from a categorical kernel —
  :class:`StructuralNeuralCategorical`, :class:`StructuralCategoricalTable`
  (declared ``coupling``: Gumbel-max or monotone), sharp counterfactual
  bounds over all compatible SCMs (:func:`counterfactual_bounds`,
  :func:`counterfactual_bounds_from_sets`) and :func:`conformal_sets`.

User guide: https://nbn.readthedocs.io/en/latest/user_guide/scm.html
"""
from nbn.scm._stats import benjamini_hochberg
from nbn.scm.calibration import (
    ConformalIntervals,
    ConformalSets,
    PITBandSet,
    check_held_out,
    conformal_intervals,
    conformal_sets,
    pit_band_set,
    pit_values,
)
from nbn.scm.categorical import (
    StructuralCategoricalTable,
    StructuralNeuralCategorical,
    counterfactual_bounds,
    counterfactual_bounds_from_sets,
)
from nbn.scm.diagnostics import (
    DiagnosticResult,
    calibration_by_bins_test,
    context_invariance_test,
    freshness_test,
    noise_independence_test,
    noise_sharing_test,
    noise_tying_test,
    residual_dependence_test,
    residuals,
)
from nbn.scm.episodes import transitions_from_episodes
from nbn.scm.flow import FlowNoiseMechanism, StandardNormalNoise
from nbn.scm.mechanisms import (
    AdditiveNoiseMechanism,
    DiscreteAdditiveNoiseMechanism,
    StructuralMechanism,
)
from nbn.scm.model import StructuralCausalModel
from nbn.scm.noise import GaussianMixtureNoise, GaussianNoise, OffsetNoise
from nbn.scm.sets import cdf_band_set, noise_parameter_set

__all__ = [
    "StructuralCausalModel",
    "StructuralMechanism",
    "DiscreteAdditiveNoiseMechanism",
    "AdditiveNoiseMechanism",
    "FlowNoiseMechanism",
    "StandardNormalNoise",
    "OffsetNoise",
    "GaussianNoise",
    "GaussianMixtureNoise",
    "DiagnosticResult",
    "noise_sharing_test",
    "noise_tying_test",
    "noise_independence_test",
    "calibration_by_bins_test",
    "freshness_test",
    "residual_dependence_test",
    "context_invariance_test",
    "residuals",
    "benjamini_hochberg",
    "transitions_from_episodes",
    "noise_parameter_set",
    "cdf_band_set",
    "pit_values",
    "pit_band_set",
    "check_held_out",
    "PITBandSet",
    "conformal_intervals",
    "ConformalIntervals",
    "StructuralNeuralCategorical",
    "StructuralCategoricalTable",
    "counterfactual_bounds",
    "counterfactual_bounds_from_sets",
    "conformal_sets",
    "ConformalSets",
]
