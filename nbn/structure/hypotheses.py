"""Structure uncertainty: several candidate parent sets per node, fitted once each.

A :class:`HypothesisSet` holds, for every *target* node, a list of candidate
parent tuples.  A **hypothesis** picks one candidate per target; there are
``prod_j |P_j|`` of them.  A **factor** is one pair ``(node, parents)``; there
are only ``sum_j |P_j|``.  Every quantity that depends on one family only (its
counts, its MLE table, its probability set) is computed once per factor and
reused by every hypothesis that contains it.

Why this is exact
-----------------
With complete data, the likelihood of a Bayesian network is a product of one
local likelihood per family, provided the families' parameters are disjoint
(Koller & Friedman, *Probabilistic Graphical Models*, 2009, §17.2.2,
Prop. 17.1, pp. 724–725); for table CPDs each term depends only on the counts
``M[u, x]`` and the MLE is ``M[u, x] / M[u]`` (Eq. 17.5, p. 726).  So the fit
of a family does not depend on the rest of the hypothesis.

Counts are shared too: for each node the table is counted **once** over the
union of its candidates' parents and every candidate's table is obtained by
summing out the parents it does not use (K&F, Box 18.A, pp. 819–820:
caching and marginalising counts across families).  When the union table
would be too large, each candidate is counted directly.

Confidence budget
-----------------
:meth:`HypothesisSet.factor_sets` splits ``delta`` over the **distinct
factors** by a union bound: a factor's set is the same event in every
hypothesis that uses it, so all sets hold jointly with probability at least
``1 - delta``.  For every factor whose parents contain the true parents (or a
superset of them) the set then bounds the true conditional; a factor that
misses a true parent has no fixed conditional to bound (its set describes the
data's average over the missing parent; see below).  The book's PAC bound splits over the families of one DAG (K&F,
Thms 17.3–17.4, pp. 774–775); the split over all distinct factors is the
natural extension.

Superset rule (why a hypothesis with extra parents is still correct)
--------------------------------------------------------------------
In a 2-slice model with no intra-slice edges among the targets, for
``Pa_j ⊆ U``: ``P(S'_j | U) = P(S'_j | Pa_j)`` whatever the distribution of
the inputs, including inputs whose distribution is set by a behaviour policy,
wherever ``P(u) > 0`` (the 2-TBN is a conditional BN: K&F Def. 5.17, p. 191;
§6.2, Defs 6.1–6.3, pp. 201–204).  A candidate that **misses** a true parent
converges to the M-projection, which averages over the missing parent with
weights set by the data (K&F Thm 8.7, p. 281; Thm 17.2, p. 770): biased, and
dependent on the loggers.

Scope
-----
Discrete targets and discrete parents (count-based sets).  Continuous and
mixed factors: issue #318.  One set per row that combines the candidates of a
node: :func:`nbn.structure.agreement_set` (agreement rule).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Literal, Mapping, Sequence, Tuple, Union

import torch

from nbn.core.variables import Variable
from nbn.learning.weighting import validate_weights
from nbn.uncertainty.builders import (
    Sampling,
    dirichlet_set_from_counts,
    interval_set_from_counts,
    l1_ball_set_from_counts,
)
from nbn.uncertainty.binomial import IntervalMethod
from nbn.uncertainty.counts import count_table_for
from nbn.uncertainty.sets import ProbabilitySet

Factor = Tuple[str, Tuple[str, ...]]
Hypothesis = Dict[str, Tuple[str, ...]]
Builder = Literal["interval", "l1", "dirichlet"]

# Cells (rows x classes) above which a node's union table is not built and its
# candidates are counted one by one instead.
DEFAULT_MAX_UNION_CELLS = 1 << 24


def _build_variable(name: str, spec: Union[Tuple, Variable]) -> Variable:
    from nbn.core.network import _build_variable as build

    return build(name, spec)


class HypothesisSet:
    """Candidate parent sets per node, and the hypotheses they span.

    Parameters
    ----------
    variables:
        Variable specs by name, same format as
        :class:`~nbn.core.network.NeuralBayesianNetwork`
        (``("discrete", K)`` or a ``Variable``).
    candidates:
        ``node -> list of candidate parent tuples``.  Nodes that appear here
        are the *targets*; every other variable is an *input* (a root in every
        hypothesis).  Within a node, two candidates with the same parents in
        a different order are the same family and are rejected.
    fixed_edges:
        Optional ``(parent, child)`` edges that every hypothesis has: the
        parent is added to every candidate of the child (a child without
        candidates gets one candidate, the fixed parents).

    Notes
    -----
    The union of all candidate edges must be acyclic, so that every
    hypothesis is a DAG.  This holds for 2-slice models, where the
    candidates of next-state variables come from the previous slice.
    """

    def __init__(
        self,
        variables: Mapping[str, Union[Tuple, Variable]],
        candidates: Mapping[str, Sequence[Sequence[str]]],
        fixed_edges: Sequence[Tuple[str, str]] = (),
    ) -> None:
        self.variables: Dict[str, Variable] = {
            name: _build_variable(name, spec) for name, spec in variables.items()
        }
        cands: Dict[str, List[Tuple[str, ...]]] = {}
        for node, cs in candidates.items():
            if isinstance(cs, str) or any(isinstance(c, str) for c in cs):
                raise TypeError(
                    f"candidates['{node}'] must be a list of parent tuples, e.g. [('x', 'a')]; "
                    f"got {cs!r}."
                )
            cands[node] = [tuple(c) for c in cs]
        for node, cs in cands.items():
            self._check_node(node, cs)  # duplicates in the user's own list are an error
        for parent, child in fixed_edges:
            cands.setdefault(child, [()])
            merged = [c if parent in c else (*c, parent) for c in cands[child]]
            # A fixed edge can make two candidates the same family ((a,) and () + a): keep one.
            unique: Dict[frozenset, Tuple[str, ...]] = {}
            for c in merged:
                unique.setdefault(frozenset(c), c)
            cands[child] = list(unique.values())
        if not cands:
            raise ValueError("A HypothesisSet needs at least one node with candidate parent sets.")
        for node, cs in cands.items():
            self._check_node(node, cs)
        self.candidates: Dict[str, List[Tuple[str, ...]]] = cands
        self.targets: List[str] = list(cands)
        self.inputs: List[str] = [v for v in self.variables if v not in cands]
        self._check_acyclic()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_node(self, node: str, cs: List[Tuple[str, ...]]) -> None:
        if node not in self.variables:
            raise KeyError(f"Unknown node '{node}'.")
        if not cs:
            raise ValueError(f"Node '{node}' has no candidate parent set.")
        seen = set()
        for c in cs:
            if len(set(c)) != len(c):
                raise ValueError(f"Candidate {c} of '{node}' repeats a parent.")
            if node in c:
                raise ValueError(f"Candidate {c} of '{node}' contains the node itself.")
            for p in c:
                if p not in self.variables:
                    raise KeyError(f"Candidate {c} of '{node}': unknown parent '{p}'.")
            key = frozenset(c)
            if key in seen:
                raise ValueError(
                    f"Node '{node}' lists the parent set {sorted(c)} twice (order does not matter)."
                )
            seen.add(key)

    def _check_acyclic(self) -> None:
        from nbn.core.dag import DAG

        edges = sorted({(p, node) for node, cs in self.candidates.items() for c in cs for p in c})
        try:
            DAG(edges, extra_nodes=list(self.variables))
        except ValueError as err:
            raise ValueError(
                "The union of all candidate edges has a cycle, so some hypotheses "
                "would not be DAGs.  In a 2-slice model take candidates from the "
                "previous slice only."
            ) from err

    def _require_discrete(self) -> None:
        involved = set(self.targets) | {p for cs in self.candidates.values() for c in cs for p in c}
        continuous = sorted(v for v in involved if not self.variables[v].is_discrete)
        if continuous:
            raise ValueError(
                f"Count-based factors need discrete variables; {continuous} are continuous "
                "(continuous and mixed factors: issue #318)."
            )

    # ------------------------------------------------------------------
    # Sizes and enumeration
    # ------------------------------------------------------------------

    @property
    def factors(self) -> List[Factor]:
        """Every distinct ``(node, parents)``, targets in order, candidates in order."""
        return [(node, c) for node in self.targets for c in self.candidates[node]]

    @property
    def n_factors(self) -> int:
        """``sum_j |P_j|``: the number of fits (and of sets)."""
        return sum(len(cs) for cs in self.candidates.values())

    @property
    def n_hypotheses(self) -> int:
        """``prod_j |P_j|`` (a Python int; never materialised)."""
        return math.prod(len(cs) for cs in self.candidates.values())

    def hypotheses(self) -> Iterator[Hypothesis]:
        """Lazy iterator over hypotheses ``{target: parent tuple}``."""
        for choice in itertools.product(*(self.candidates[n] for n in self.targets)):
            yield dict(zip(self.targets, choice))

    def edges(self, hypothesis: Mapping[str, Sequence[str]]) -> List[Tuple[str, str]]:
        """The ``(parent, child)`` edges of one hypothesis (validated)."""
        self._check_hypothesis(hypothesis)
        return [(p, node) for node in self.targets for p in hypothesis[node]]

    def _check_hypothesis(self, hypothesis: Mapping[str, Sequence[str]]) -> None:
        if set(hypothesis) != set(self.targets):
            raise ValueError(
                f"A hypothesis picks one candidate for each of {self.targets}; got {sorted(hypothesis)}."
            )
        for node, parents in hypothesis.items():
            if isinstance(parents, str):
                raise TypeError(f"The parents of '{node}' must be a tuple of names, not {parents!r}.")
            if tuple(parents) not in self.candidates[node]:
                same = [c for c in self.candidates[node] if frozenset(c) == frozenset(parents)]
                if same:
                    raise ValueError(
                        f"{tuple(parents)} is the candidate {same[0]} of '{node}' in another order; "
                        f"use {same[0]} (table rows follow that order)."
                    )
                raise ValueError(f"{tuple(parents)} is not a candidate of '{node}'.")

    # ------------------------------------------------------------------
    # Counts (the factor cache)
    # ------------------------------------------------------------------

    def counts(
        self,
        data: Mapping[str, torch.Tensor],
        weights: torch.Tensor | None = None,
        max_union_cells: int = DEFAULT_MAX_UNION_CELLS,
        input_marginals: bool = True,
    ) -> FactorCounts:
        """Count tables ``[R, K]`` for every factor (and the inputs' marginals).

        For each target node the table is counted once over the union of its
        candidates' parents and summed down to each candidate; if the union
        table has more than ``max_union_cells`` cells, the candidates are
        counted one by one.  Tables stay on the data's device, in the widest
        float the device supports.  ``input_marginals`` also counts the
        marginal of every input that some candidate uses (needed only by
        :meth:`model_for`).
        """
        self._require_discrete()
        first = data[self.targets[0]]
        n_rows = first.reshape(-1).shape[0]
        weights = validate_weights(weights, n_rows, where="HypothesisSet.counts", device=first.device)
        tables: Dict[Factor, torch.Tensor] = {}
        cards: Dict[Factor, List[int]] = {}
        union_used: Dict[str, bool] = {}
        for node in self.targets:
            cs = self.candidates[node]
            union = tuple(dict.fromkeys(p for c in cs for p in c))
            union_cards = [int(self.variables[p].cardinality) for p in union]
            k = int(self.variables[node].cardinality)
            cells = math.prod(union_cards) * k
            direct_cells = sum(
                math.prod(int(self.variables[p].cardinality) for p in c) * k for c in cs
            )
            # One pass over the data plus a dense union table, against one pass per
            # candidate: the union pays off unless its table dwarfs the data.
            use_union = (
                len(cs) > 1 and cells <= max_union_cells
                and cells <= direct_cells + (len(cs) - 1) * n_rows
            )
            union_used[node] = use_union
            if use_union:
                full, _ = count_table_for(self.variables, data, node, union, weights)
                full = full.reshape(*union_cards, k)
                for c in cs:
                    drop = [d for d, p in enumerate(union) if p not in c]
                    t = full.sum(dim=drop) if drop else full
                    in_union_order = tuple(p for p in union if p in c)   # axes left after the sum
                    tables[(node, c)] = _reorder_parents(
                        t.reshape(-1, k), [int(self.variables[p].cardinality) for p in in_union_order],
                        in_union_order, c,
                    ).contiguous()
                    cards[(node, c)] = [int(self.variables[p].cardinality) for p in c]
            else:
                for c in cs:
                    tables[(node, c)], cards[(node, c)] = count_table_for(
                        self.variables, data, node, c, weights,
                    )
        marginals: Dict[str, torch.Tensor] = {}
        if input_marginals:
            used = {p for cs in self.candidates.values() for c in cs for p in c}
            for v in self.inputs:
                if v in used and v in data:
                    marginals[v], _ = count_table_for(self.variables, data, v, (), weights)
        return FactorCounts(
            hypothesis_set=self, tables=tables, parent_cards=cards,
            input_marginals=marginals, union_used=union_used, weighted=weights is not None,
        )

    # ------------------------------------------------------------------
    # Probability sets per factor
    # ------------------------------------------------------------------

    def factor_sets(
        self,
        data_or_counts: Union[Mapping[str, torch.Tensor], FactorCounts],
        builder: Builder = "interval",
        delta: float = 0.05,
        sampling: Sampling = "iid",
        method: IntervalMethod = "kl",
        weights: torch.Tensor | None = None,
        prior: float = 1.0,
        n_samples: int | None = None,
    ) -> FactorSets:
        """One probability set per factor, all holding jointly at ``1 - delta``.

        ``delta`` is split evenly over the :attr:`n_factors` distinct factors
        (union bound); each factor's set is then built by the matching
        ``*_from_counts`` builder of :mod:`nbn.uncertainty` at level
        ``1 - delta / n_factors``.  For ``builder="dirichlet"`` the level is a
        posterior credibility, not a frequentist guarantee, and ``n_samples``
        (default: enough draws for the per-entry quantiles at the split level,
        at least 4000, at most 200 000) is chosen per factor.

        **What the level means.**  For a factor whose parents contain the true
        parents (or a superset), the set bounds the true conditional.  For a
        factor that misses a true parent there is no fixed conditional to
        bound: the set describes the data's average over the missing parent,
        which depends on the behaviour policies (and with trajectory data is
        not even stationary).  So ``level`` speaks for the superset factors
        only; which factors those are is the structural question S3 answers.
        """
        if builder not in ("interval", "l1", "dirichlet"):
            raise ValueError(f"Unknown builder '{builder}'; use 'interval', 'l1' or 'dirichlet'.")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1); got {delta}.")
        if builder in ("interval", "l1") and (
            weights is not None
            or (isinstance(data_or_counts, FactorCounts) and data_or_counts.weighted)
        ):
            raise ValueError(
                "PAC sets need counts of independent draws; weighted counts do not carry the "
                "guarantee (use builder='dirichlet' for weighted pseudo-counts)."
            )
        if isinstance(data_or_counts, FactorCounts):
            if data_or_counts.hypothesis_set is not self:
                raise ValueError("These counts were built by another HypothesisSet.")
            if weights is not None:
                raise ValueError("weights apply when counting; pass them to counts().")
            fc = data_or_counts
        else:
            fc = self.counts(data_or_counts, weights=weights, input_marginals=False)
        per_factor = delta / self.n_factors
        sets: Dict[Factor, ProbabilitySet] = {}
        for f in self.factors:
            table = fc.tables[f]
            if builder == "interval":
                sets[f] = interval_set_from_counts(table, delta=per_factor, method=method, sampling=sampling)
            elif builder == "l1":
                sets[f] = l1_ball_set_from_counts(table, delta=per_factor, sampling=sampling)
            else:
                draws = n_samples if n_samples is not None else _auto_draws(per_factor, table.numel())
                sets[f] = dirichlet_set_from_counts(table, delta=per_factor, prior=prior, n_samples=draws)
        return FactorSets(
            hypothesis_set=self, sets=sets, counts=fc, builder=builder, level=1.0 - delta,
            delta_per_factor=per_factor, sampling=sampling,
        )

    # ------------------------------------------------------------------
    # A fitted model for one hypothesis
    # ------------------------------------------------------------------

    def model_for(
        self,
        hypothesis: Mapping[str, Sequence[str]],
        counts: FactorCounts,
        alpha: float = 0.0,
        device: str | None = None,
    ):
        """A fitted :class:`~nbn.core.network.NeuralBayesianNetwork` for one hypothesis.

        The network has the hypothesis' variables only: the targets and the
        inputs that are parents of some target (inputs used by no target are
        left out).  Every node gets a :class:`CategoricalTableMechanism` whose
        table is set from the cached counts (no pass over the data): the
        targets from their factor tables, the inputs from their marginal
        counts.  The result is
        the model that fitting this hypothesis' DAG directly with table
        mechanisms of smoothing ``alpha`` would give.
        """
        from nbn.core.network import NeuralBayesianNetwork
        from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism

        if counts.hypothesis_set is not self:
            raise ValueError("These counts were built by another HypothesisSet.")
        self._check_hypothesis(hypothesis)
        used = set(self.targets) | {p for parents in hypothesis.values() for p in parents}
        roots = [v for v in self.inputs if v in used]
        marginals = {v: counts.input_marginals.get(v) for v in roots}
        for v in roots:
            if marginals[v] is None:
                marginals[v] = _marginal_from_factors(counts, v)
        if device is None:
            first = next(iter(counts.tables.values()))
            device = str(first.device)
        variables = {v: spec for v, spec in self.variables.items() if v in used}
        model = NeuralBayesianNetwork(self.edges(hypothesis), variables, device=device)
        for node in model.dag.topological_order():
            mech = CategoricalTableMechanism(alpha=alpha)
            if node in self.candidates:
                parents = tuple(hypothesis[node])
                dag_parents = tuple(model.dag.parents(node))
                table = _reorder_parents(
                    counts.tables[(node, parents)], counts.parent_cards[(node, parents)],
                    parents, dag_parents,
                )
                mech.fit_from_counts(table, [int(self.variables[p].cardinality) for p in dag_parents])
            else:
                mech.fit_from_counts(marginals[node], [])
            model.set_mechanism(node, mech)
        model.to(model.device)
        return model

    def __repr__(self) -> str:
        return (
            f"HypothesisSet(targets={len(self.targets)}, inputs={len(self.inputs)}, "
            f"factors={self.n_factors}, hypotheses={self.n_hypotheses})"
        )


@dataclass(eq=False)
class FactorCounts:
    """Count tables per factor, from :meth:`HypothesisSet.counts`."""

    hypothesis_set: HypothesisSet = field(repr=False)
    tables: Dict[Factor, torch.Tensor] = field(repr=False)
    parent_cards: Dict[Factor, List[int]] = field(repr=False)
    input_marginals: Dict[str, torch.Tensor] = field(repr=False)
    # Per target: whether the union-and-marginalise path was used.
    union_used: Dict[str, bool] = field(default_factory=dict)
    # Whether the tables are weighted (then they carry no PAC guarantee).
    weighted: bool = False

    def __getitem__(self, factor: Factor) -> torch.Tensor:
        node, parents = factor
        return self.tables[(node, tuple(parents))]

    def for_hypothesis(self, hypothesis: Mapping[str, Sequence[str]]) -> Dict[str, torch.Tensor]:
        """The count table of every target under one hypothesis (no copy)."""
        self.hypothesis_set._check_hypothesis(hypothesis)
        return {n: self.tables[(n, tuple(p))] for n, p in hypothesis.items()}


@dataclass(eq=False)
class FactorSets:
    """Probability sets per factor, all holding jointly at :attr:`level`."""

    hypothesis_set: HypothesisSet = field(repr=False)
    sets: Dict[Factor, ProbabilitySet] = field(repr=False)
    counts: FactorCounts = field(repr=False)
    builder: str = "interval"
    level: float = 0.95
    # The union-bound split: every factor's set is built at 1 - delta_per_factor.
    delta_per_factor: float = 0.05
    sampling: str = "iid"

    @property
    def n_factors(self) -> int:
        return len(self.sets)

    def __getitem__(self, factor: Factor) -> ProbabilitySet:
        node, parents = factor
        return self.sets[(node, tuple(parents))]

    def sets_for(self, hypothesis: Mapping[str, Sequence[str]]) -> Dict[str, ProbabilitySet]:
        """The set of every target under one hypothesis (views into the cache, no copy)."""
        self.hypothesis_set._check_hypothesis(hypothesis)
        return {n: self.sets[(n, tuple(p))] for n, p in hypothesis.items()}


def _reorder_parents(
    table: torch.Tensor, cards: List[int], current: Tuple[str, ...], target: Tuple[str, ...],
) -> torch.Tensor:
    """Permute the parent axes of a ``[R, K]`` table from ``current`` to ``target`` order."""
    if current == target:
        return table
    if sorted(current) != sorted(target):
        raise ValueError(f"Parent sets differ: {current} vs {target}.")
    k = table.shape[-1]
    axes = [current.index(p) for p in target]
    return table.reshape(*cards, k).permute(*axes, len(cards)).reshape(-1, k).contiguous()


def _auto_draws(delta: float, n_entries: int) -> int:
    """Posterior draws so that >= 5 fall beyond each simultaneous quantile (capped)."""
    per_entry = delta / max(n_entries, 1)
    return int(min(200_000, max(4000, math.ceil(10.0 / per_entry))))


def _marginal_from_factors(counts: FactorCounts, v: str) -> torch.Tensor:
    """Counts ``[1, K_v]`` of input ``v``, summed from a factor table that has ``v`` as a parent.

    A factor table counts the same rows (with the same weights) as a direct
    count of ``v``, so summing out the child and the other parents is exact.
    """
    for (node, parents), table in counts.tables.items():
        if v in parents:
            cards = counts.parent_cards[(node, parents)]
            full = table.reshape(*cards, table.shape[-1])
            i = parents.index(v)
            drop = [d for d in range(full.dim()) if d != i]
            return full.sum(dim=drop).reshape(1, -1)
    raise ValueError(f"No counts for the input '{v}': it is a parent of no factor.")

