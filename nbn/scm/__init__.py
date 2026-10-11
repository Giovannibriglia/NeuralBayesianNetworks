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

See ``docs/scm-design-draft.md``.
"""
from nbn.scm._stats import benjamini_hochberg
from nbn.scm.diagnostics import (
    DiagnosticResult,
    context_invariance_test,
    freshness_test,
    noise_sharing_test,
    noise_tying_test,
    residual_dependence_test,
    residuals,
)
from nbn.scm.episodes import transitions_from_episodes
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
    "OffsetNoise",
    "GaussianNoise",
    "GaussianMixtureNoise",
    "DiagnosticResult",
    "noise_sharing_test",
    "noise_tying_test",
    "freshness_test",
    "residual_dependence_test",
    "context_invariance_test",
    "residuals",
    "benjamini_hochberg",
    "transitions_from_episodes",
    "noise_parameter_set",
    "cdf_band_set",
]
