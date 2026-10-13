"""Structure uncertainty: several candidate parent sets per node.

* :class:`HypothesisSet` — candidate parent tuples per node; hypotheses are
  one candidate per node (``prod_j |P_j|``), factors are ``(node, parents)``
  pairs (``sum_j |P_j|``) fitted once each;
* :meth:`HypothesisSet.counts` — the factor cache (count once over the union
  of a node's candidates, marginalise for each);
* :meth:`HypothesisSet.factor_sets` — one probability set per factor, the
  budget split over the distinct factors;
* :meth:`HypothesisSet.model_for` — a fitted network for one hypothesis,
  from the cached counts.

User guide: https://nbn.readthedocs.io/en/latest/user_guide/probability_sets.html
"""
from nbn.structure.hypotheses import (
    DEFAULT_MAX_UNION_CELLS,
    FactorCounts,
    FactorSets,
    HypothesisSet,
)

__all__ = ["HypothesisSet", "FactorCounts", "FactorSets", "DEFAULT_MAX_UNION_CELLS"]
