"""Probability sets: bounds on probabilities with a declared guarantee.

Builders (from data, for a discrete node's table ``P(X | pa)``):

* :func:`interval_set` — entry-wise intervals (min/max), PAC;
* :func:`l1_ball_set` — one L1 ball per row, PAC;
* :func:`dirichlet_set` — posterior draws, Bayesian;
* :func:`bootstrap_set` — episode bootstrap of any statistic, asymptotic
  (discrete or continuous models).

The first three have ``*_from_counts`` twins that take a count table ``[R, K]``
directly; :func:`count_table_for` counts a node against any parent tuple.

Rows of a possibly confounded action effect from several logging regimes
(Manski / intersection / IV linear program / sensitivity; the confidence box
first): :func:`confounded_interval_set`, with :func:`confounding_gate`.

Query bounds over the sets (every network whose rows lie in them), and PAC
answers from data: :func:`credal_query`, :func:`pac_sets`, :func:`pac_query`.

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
    interval_sets,
    l1_ball_set,
    l1_ball_set_from_counts,
    per_interval_delta,
)
from nbn.uncertainty.confounded import (
    ConfoundedSet,
    GateDecision,
    LPFailure,
    ProgramTooLarge,
    RefutedAssumption,
    confounded_interval_set,
    confounding_gate,
)
from nbn.uncertainty.counts import count_table, count_table_for, parent_grid, parent_row_index
from nbn.uncertainty.credal import CredalSets, QueryBounds, credal_query, pac_query, pac_sets
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
    "credal_query",
    "pac_query",
    "pac_sets",
    "QueryBounds",
    "CredalSets",
    "ProbabilitySet",
    "IntervalSet",
    "L1BallSet",
    "SampleSet",
    "NoiseParameterSet",
    "CDFBandSet",
    "interval_set",
    "interval_sets",
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
    "parent_grid",
    "tighten_box_to_simplex",
    "confounded_interval_set",
    "confounding_gate",
    "ConfoundedSet",
    "GateDecision",
    "RefutedAssumption",
    "LPFailure",
    "ProgramTooLarge",
]
