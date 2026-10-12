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

See ``docs/scm-design-draft.md``.
"""
from nbn.scm._stats import benjamini_hochberg
from nbn.scm.calibration import (
    ConformalIntervals,
    PITBandSet,
    conformal_intervals,
    pit_band_set,
    pit_values,
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
    "PITBandSet",
    "conformal_intervals",
    "ConformalIntervals",
]
