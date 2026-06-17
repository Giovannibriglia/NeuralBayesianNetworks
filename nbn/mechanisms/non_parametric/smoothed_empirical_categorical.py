"""Laplace/Lidstone-smoothed empirical categorical CPD (non-parametric).

This is the closed-form, tuning-light, *non-learned* discrete conditional
distribution: for each parent configuration it estimates

    P(y = c | pa) = (count(c, pa) + alpha) / (sum_c' count(c', pa) + alpha * K),

the Bayesian MAP estimate under a symmetric Dirichlet(alpha) prior
(``alpha = 1`` = Laplace's rule of succession; general ``alpha`` = Lidstone).
It is the empirical-frequency counterpart to the gradient-trained
:class:`~nbn.mechanisms.parametric.neural_categorical.NeuralCategoricalMechanism`
and the natural fair-comparison baseline against pgmpy / pomegranate CPTs.

Relationship to ``CategoricalTableMechanism``
---------------------------------------------
The count-and-smooth machinery already lives in
:class:`~nbn.mechanisms.parametric.categorical_table.CategoricalTableMechanism`,
whose ``fit_local`` is itself a closed-form Dirichlet(alpha) estimator (it is
*not* gradient-trained).  Rather than fork that carefully-tested core (which
also carries the #127 declared-cardinality fixes), this class is a thin
subclass that only (a) changes the default to Laplace smoothing
(``alpha = 1.0`` instead of the MLE-parity ``alpha = 0.0``) and (b) gives the
estimator its conventional non-parametric name and home.  Pass ``alpha=0`` to
recover the unsmoothed MLE, or any positive value for Lidstone.
"""
from __future__ import annotations

from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism


class SmoothedEmpiricalCategoricalMechanism(CategoricalTableMechanism):
    """Empirical conditional categorical with Laplace/Lidstone smoothing.

    Parameters
    ----------
    alpha:
        Symmetric Dirichlet pseudo-count.  Default ``1.0`` (Laplace).  Use a
        smaller positive value (e.g. ``0.5`` Jeffreys, ``0.01``) for lighter
        smoothing, or ``0.0`` for the raw MLE.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError(f"alpha (Lidstone pseudo-count) must be >= 0, got {alpha}")
        super().__init__(alpha=alpha)


__all__ = ["SmoothedEmpiricalCategoricalMechanism"]
