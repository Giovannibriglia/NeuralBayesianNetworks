"""Structure uncertainty: several candidate parent sets per node.

* :class:`HypothesisSet` — candidate parent tuples per node; hypotheses are
  one candidate per node (``prod_j |P_j|``), factors are ``(node, parents)``
  pairs (``sum_j |P_j|``) fitted once each;
* :meth:`HypothesisSet.counts` — the factor cache (count once over the union
  of a node's candidates, marginalise for each);
* :meth:`HypothesisSet.factor_sets` — one probability set per factor, the
  budget split over the distinct factors;
* :meth:`HypothesisSet.model_for` — a fitted network for one hypothesis,
  from the cached counts;
* :func:`agreement_set` — one valid set per row of a node whose parents are
  uncertain (KWIK agreement rule, under a declared bound on the number of
  parents);
* :func:`export_factored_sets` / :func:`load_factored_sets` — a documented
  side file (manifest + arrays) with the sets and their sharing, for robust,
  interval and parametric MDP tools.

User guide: https://nbn.readthedocs.io/en/latest/user_guide/probability_sets.html
"""
from nbn.structure.agreement import AgreementSet, agreement_set
from nbn.structure.export import (
    FactoredSetsExport,
    export_factored_sets,
    joint_row_bounds,
    load_factored_sets,
)
from nbn.structure.hypotheses import (
    DEFAULT_MAX_UNION_CELLS,
    FactorCounts,
    FactorSets,
    HypothesisSet,
)

__all__ = [
    "HypothesisSet", "FactorCounts", "FactorSets", "DEFAULT_MAX_UNION_CELLS",
    "agreement_set", "AgreementSet",
    "export_factored_sets", "load_factored_sets", "joint_row_bounds", "FactoredSetsExport",
]
