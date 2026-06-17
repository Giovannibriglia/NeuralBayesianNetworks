"""Parametric and neural conditional mechanisms.

These mechanisms assume a fixed-form conditional family P(X | pa(X)) whose
parameters are either fit in closed form (e.g. :class:`LinearGaussianMechanism`,
:class:`CategoricalTableMechanism`) or learned by gradient descent
(e.g. :class:`MDNMechanism`, :class:`NeuralCategoricalMechanism`,
``NormalizingFlowMechanism``).  Contrast with
:mod:`nbn.mechanisms.non_parametric`, whose estimators let model complexity grow
with the data rather than committing to a parametric form.
"""
from __future__ import annotations

from nbn.mechanisms.parametric.binning_categorical import BinningCategoricalTable
from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism
from nbn.mechanisms.parametric.deterministic import DeterministicMechanism
from nbn.mechanisms.parametric.dirac_gaussian import DiracGaussianMechanism
from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism
from nbn.mechanisms.parametric.mdn import MDNMechanism
from nbn.mechanisms.parametric.neural_categorical import NeuralCategoricalMechanism

__all__ = [
    "CategoricalTableMechanism",
    "BinningCategoricalTable",
    "LinearGaussianMechanism",
    "MDNMechanism",
    "NeuralCategoricalMechanism",
    "DeterministicMechanism",
    "DiracGaussianMechanism",
]

# Optional: normalizing flow mechanism (requires the ``neural`` extra → zuko).
try:
    from nbn.mechanisms.parametric.normalizing_flow import (
        ConditionalFlowMechanism,
        NormalizingFlowMechanism,
    )

    __all__ += ["NormalizingFlowMechanism", "ConditionalFlowMechanism"]
except ImportError:
    pass
