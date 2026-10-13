"""Probability sets: bounds on probabilities with a declared guarantee.

Builders (from data, for a discrete node's table ``P(X | pa)``):

* :func:`interval_set` — entry-wise intervals (min/max), PAC;
* :func:`l1_ball_set` — one L1 ball per row, PAC;
* :func:`dirichlet_set` — posterior draws, Bayesian;
* :func:`bootstrap_set` — episode bootstrap of any statistic, asymptotic
  (discrete or continuous models).

The first three have ``*_from_counts`` twins that take a count table ``[R, K]``
directly; :func:`count_table_for` counts a node against any parent tuple.

Structural sets that pool shared noise live in :mod:`nbn.scm`.
User guide: https://nbn.readthedocs.io/en/latest/user_guide/probability_sets.html
"""
from nbn.uncertainty.binomial import anytime_delta, binomial_interval, kl_bernoulli, union_bound
from nbn.uncertainty.builders import (
    bootstrap_set,
    dirichlet_set,
    dirichlet_set_from_counts,
    interval_set,
    interval_set_from_counts,
    l1_ball_set,
    l1_ball_set_from_counts,
    per_interval_delta,
)
from nbn.uncertainty.counts import count_table, count_table_for, parent_row_index
from nbn.uncertainty.sets import (
    CDFBandSet,
    IntervalSet,
    L1BallSet,
    NoiseParameterSet,
    ProbabilitySet,
    SampleSet,
    tighten_box_to_simplex,
)

__all__ = [
    "ProbabilitySet",
    "IntervalSet",
    "L1BallSet",
    "SampleSet",
    "NoiseParameterSet",
    "CDFBandSet",
    "interval_set",
    "l1_ball_set",
    "dirichlet_set",
    "bootstrap_set",
    "interval_set_from_counts",
    "l1_ball_set_from_counts",
    "dirichlet_set_from_counts",
    "binomial_interval",
    "kl_bernoulli",
    "union_bound",
    "anytime_delta",
    "per_interval_delta",
    "count_table",
    "count_table_for",
    "parent_row_index",
    "tighten_box_to_simplex",
]
