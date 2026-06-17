"""Non-parametric conditional density / distribution estimators.

Unlike :mod:`nbn.mechanisms.parametric`, these mechanisms do not commit to a
fixed-form conditional family; their effective complexity grows with the data.
They share the project's design contract (``nn.Module`` + a
``torch.distributions.Distribution`` from ``forward``), are pure-torch and
GPU-friendly, and bound peak memory by chunking over the training set so they
fit the 8 GB-VRAM target.

Members
-------
ConditionalKDEMechanism
    Nadaraya--Watson conditional KDE (continuous child).
KNNConditionalMechanism
    k-NN local conditional density (continuous child) / smoothed neighbour
    frequencies (discrete child).
SmoothedEmpiricalCategoricalMechanism
    Laplace/Lidstone-smoothed empirical categorical CPD (discrete child).
FlexCodeMechanism
    Orthogonal-series (cosine-basis) CDE for higher-dimensional parents
    (univariate continuous child).
"""
from __future__ import annotations

from nbn.mechanisms.non_parametric.conditional_kde import ConditionalKDEMechanism
from nbn.mechanisms.non_parametric.flexcode import FlexCodeMechanism
from nbn.mechanisms.non_parametric.knn_conditional import KNNConditionalMechanism
from nbn.mechanisms.non_parametric.smoothed_empirical_categorical import (
    SmoothedEmpiricalCategoricalMechanism,
)

__all__ = [
    "ConditionalKDEMechanism",
    "KNNConditionalMechanism",
    "SmoothedEmpiricalCategoricalMechanism",
    "FlexCodeMechanism",
]
