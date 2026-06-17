"""Conditional mechanisms P(X | pa(X)) for NBN nodes.

Mechanisms are split into two families:

* :mod:`nbn.mechanisms.parametric` — fixed-form / neural CPDs (categorical
  table, linear-Gaussian, MDN, neural-categorical, normalizing-flow, ...).
* :mod:`nbn.mechanisms.non_parametric` — estimators whose complexity grows
  with the data (conditional KDE, k-NN, smoothed-empirical categorical,
  FlexCode orthogonal-series CDE).

All concrete mechanisms subclass :class:`~nbn.mechanisms.base.Mechanism` and
are re-exported here so existing ``from nbn.mechanisms import X`` and
``from nbn import X`` imports keep working after the parametric/non-parametric
reorganisation.
"""
from nbn.mechanisms.base import Mechanism

# --- Parametric / neural -------------------------------------------------
from nbn.mechanisms.parametric import (
    BinningCategoricalTable,
    CategoricalTableMechanism,
    DeterministicMechanism,
    DiracGaussianMechanism,
    LinearGaussianMechanism,
    MDNMechanism,
    NeuralCategoricalMechanism,
)

# --- Non-parametric ------------------------------------------------------
from nbn.mechanisms.non_parametric import (
    ConditionalKDEMechanism,
    FlexCodeMechanism,
    KNNConditionalMechanism,
    SmoothedEmpiricalCategoricalMechanism,
)

__all__ = [
    "Mechanism",
    # parametric
    "CategoricalTableMechanism",
    "BinningCategoricalTable",
    "LinearGaussianMechanism",
    "MDNMechanism",
    "NeuralCategoricalMechanism",
    "DeterministicMechanism",
    "DiracGaussianMechanism",
    # non-parametric
    "ConditionalKDEMechanism",
    "KNNConditionalMechanism",
    "SmoothedEmpiricalCategoricalMechanism",
    "FlexCodeMechanism",
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
