"""Agreement rule: one valid set per row of a node whose parents are uncertain.

With several candidate parent sets, taking the worst case over hypotheses
falls back to the largest candidate, which has no data on most of its rows.
The agreement rule of the KWIK framework ("knows what it knows") keeps every
candidate up to a declared size, discards the ones the data contradict, and
predicts a row only where the survivors have data; rows that no dataset
visited **jointly** can still get a set from the parent values each survivor
did visit.

The rule here
-------------
1. **Families.** Every candidate ``c`` (default: every parent set of size
   ``d = max_parents`` from the declared ``inputs``) and every one-variable
   extension ``c ∪ {v}``.  Counts come from the factor cache
   (:class:`~nbn.structure.HypothesisSet`), and each family gets an L1 ball
   per row at level ``delta / n_families`` (Weissman radius), so all balls
   hold together with probability at least ``1 - delta``.
2. **Refutation.** If ``c`` contains the true parents, so does ``c ∪ {v}``,
   and both give the same conditional ``P(x | Pa)`` on every row with data
   (a 2-slice model is a conditional BN: Koller & Friedman 2009, Def. 5.17,
   p. 191; §6.2, pp. 201–204).  On the joint event their balls then overlap
   on every row, so **disjoint balls prove that ``c`` misses a true parent**
   and ``c`` is discarded; a superset candidate is never discarded on that
   event.  This is the batch form of the structure test of Strehl, Diuk &
   Littman (AAAI 2007), "does adding a variable change the prediction?".
3. **One set per row** of the union ``U`` of the survivors' parents.  Where
   every survivor has data the set is a ball centred at the survivors' mean
   ``c̄`` with radius ``max_s (||c_s - c̄||_1 + r_s)``: it contains every
   survivor's ball, hence the truth, because under ``|Pa| <= d`` some survivor
   is a superset of the true parents.  Other rows are the whole simplex.
   ``known`` marks the rows where every survivor has data and the radius is
   at most ``epsilon`` (the KWIK "predict" flag).
4. If every candidate is refuted, the declared bound is contradicted
   (``|Pa| > d``, or a confounded logger); the result says so
   (``assumption_refuted``) and falls back to the flat model, all inputs as
   parents, which stays valid while ``Pa ⊆ inputs`` (one share of the budget
   is kept for it).

Literature: k-meteorologists (Diuk, Li & Leffler, ICML 2009); offline
structure learning for safe policy improvement with one behaviour policy
(Simão & Spaan, IJCAI 2019, Alg. 3 and Prop. 1, p. 3456); the KWIK framework
(Li, Littman, Walsh & Strehl, MLJ 82(3), 2011).  With several behaviour
policies that read only the observed state, pooling their counts keeps every
superset conditional the same, so the rule applies to the pooled data.

Declared assumptions (recorded in the result)
---------------------------------------------
* ``|Pa(node)| <= max_parents`` and ``Pa(node) ⊆ inputs``;
* no intra-slice edges into the node (2-slice model), Markov observed state;
* the same mechanism in every dataset (no hidden confounding of the action).

If an assumption is false an agreed row can be wrong.  Unseen rows (whole
simplex) break *graph preservation* (``lower = 0 < upper``), which some
interval-MDP solvers assume; they are flagged by ``set.unseen``.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import torch

from nbn.core.variables import Variable
from nbn.structure.hypotheses import HypothesisSet, _reorder_parents
from nbn.uncertainty.builders import Sampling, l1_ball_set_from_counts
from nbn.uncertainty.counts import count_table_for, parent_row_index
from nbn.uncertainty.sets import L1BallSet

Family = Tuple[str, ...]

ASSUMPTIONS = (
    "|Pa(node)| <= max_parents",
    "Pa(node) is a subset of inputs",
    "no intra-slice edges into the node; Markov observed state",
    "the same mechanism in every dataset (no hidden confounding)",
)


@dataclass(eq=False)
class AgreementSet:
    """Result of :func:`agreement_set` for one node."""

    node: str
    # Row space of the set: the union of the survivors' parents, C order.
    parents: Family
    parent_cards: List[int]
    # L1 ball per row; rows where some survivor has no data are the whole simplex.
    set: L1BallSet = field(repr=False)
    # Rows with a KWIK prediction: every survivor has data and radius <= epsilon.
    known: torch.Tensor = field(repr=False)
    # Rows of `parents` that appear in the data as such (joint coverage).
    covered_jointly: torch.Tensor = field(repr=False)
    candidates: List[Family] = field(default_factory=list)
    survivors: List[Family] = field(default_factory=list)
    # Discarded candidate -> evidence: the extension it disagrees with, the
    # number of rows where the balls are disjoint, and the largest gap.
    discarded: Dict[Family, dict] = field(default_factory=dict)
    max_parents: int = 0
    epsilon: float = 0.0
    level: float = 0.95
    n_families: int = 0
    delta_per_family: float = 0.0
    sampling: str = "iid"
    assumption_refuted: bool = False
    assumptions: Tuple[str, ...] = ASSUMPTIONS
    # Against the flat model (one row per joint value of all inputs), when that
    # row space has at most max_rows rows; else None.
    flat_rows: int | None = None
    flat_visited: int | None = None
    flat_with_set: int | None = None

    @property
    def n_rows_with_set(self) -> int:
        """Rows with a set narrower than the simplex."""
        return int((~self.set.unseen).sum())

    @property
    def n_rows_gained(self) -> int | None:
        """Joint input values never visited that still get a set ("structure buys coverage").

        Counted against the flat model, which has a set only on visited joint
        values of all inputs; ``None`` when that row space was too large.
        """
        if self.flat_with_set is None or self.flat_visited is None:
            return None
        return self.flat_with_set - self.flat_visited


def agreement_set(
    variables: Union[HypothesisSet, Mapping[str, Union[Tuple, Variable]]],
    data: Mapping[str, torch.Tensor],
    node: str,
    inputs: Sequence[str],
    max_parents: int,
    delta: float = 0.05,
    epsilon: float = 0.1,
    sampling: Sampling = "iid",
    candidates: Sequence[Sequence[str]] | None = None,
    max_rows: int = 1 << 20,
) -> AgreementSet:
    """One probability set per row of ``node``, valid without knowing its parents.

    Parameters
    ----------
    variables:
        Variable specs (as for :class:`HypothesisSet`), or a
        :class:`HypothesisSet` whose variables are used.
    data:
        Columns by name (unweighted counts of independent draws, or
        transitions of episodes with ``sampling="trajectory"``).
    node:
        The discrete child.
    inputs:
        The variables allowed as parents; the true parents must be among them.
    max_parents:
        Declared bound ``d`` on the number of true parents.
    delta:
        Failure probability of the whole result (all families at once).
        Split it across nodes when several nodes must hold together.
    epsilon:
        L1 radius below which a row counts as ``known``.
    candidates:
        Optional explicit candidate parent sets (default: all size-``d``
        subsets of ``inputs``).  The guarantee then needs a superset of the
        true parents among them.
    max_rows:
        Largest row space (union of the survivors' parents) built densely.

    Returns
    -------
    AgreementSet
    """
    if isinstance(variables, HypothesisSet):
        variables = variables.variables
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1); got {delta}.")
    if isinstance(inputs, str):
        raise TypeError(f"inputs must be a sequence of names, not the string {inputs!r}.")
    inputs = tuple(dict.fromkeys(inputs))
    if node in inputs:
        raise ValueError(f"'{node}' cannot be one of its own inputs.")
    if max_parents < 0:
        raise ValueError(f"max_parents must be >= 0; got {max_parents}.")
    d = min(int(max_parents), len(inputs))
    if candidates is None:
        cands: List[Family] = [tuple(c) for c in itertools.combinations(inputs, d)]
    else:
        cands = []
        for c in candidates:
            if isinstance(c, str):
                raise TypeError(f"Each candidate must be a tuple of names, not {c!r}.")
            if not set(c) <= set(inputs):
                raise ValueError(f"Candidate {tuple(c)} uses variables outside inputs {inputs}.")
            cands.append(tuple(c))
        if not cands:
            raise ValueError("candidates is empty.")

    # Families: candidates and their one-variable extensions, deduplicated as sets.
    families: Dict[frozenset, Family] = {}
    for c in cands:
        families.setdefault(frozenset(c), c)
    extensions: Dict[Family, List[Tuple[str, Family]]] = {}
    for c in cands:
        extensions[c] = []
        for v in inputs:
            if v in c:
                continue
            e = families.setdefault(frozenset((*c, v)), (*c, v))
            extensions[c].append((v, e))
    fam_list = list(families.values())
    # A one-node HypothesisSet gives the factor cache (count once, marginalise).
    spec = {v: variables[v] for v in (*inputs, node)}
    hs = HypothesisSet(spec, {node: fam_list})
    counts = hs.counts(data, input_marginals=False)
    cards = {v: int(hs.variables[v].cardinality) for v in (*inputs, node)}
    flat_rows = math.prod(cards[p] for p in inputs)
    flat_ok = flat_rows <= max_rows
    # One share of the budget is kept for the flat fallback (all inputs as parents).
    n_budget = len(fam_list) + (1 if flat_ok else 0)
    per_family = delta / n_budget
    balls: Dict[Family, L1BallSet] = {
        f: l1_ball_set_from_counts(counts[(node, f)], delta=per_family, sampling=sampling)
        for f in fam_list
    }
    k = cards[node]

    # 2. Refutation against the one-variable extensions.
    survivors: List[Family] = []
    discarded: Dict[Family, dict] = {}
    for c in cands:
        bc = balls[c]
        r_c = bc.unseen.numel()
        worst = None
        for v, e in extensions[c]:
            be = balls[e]
            order = (*c, v)
            e_cards = [cards[p] for p in e]
            center = _reorder_parents(be.center, e_cards, e, order).reshape(r_c, cards[v], k)
            radius = _reorder_parents(be.radius.unsqueeze(-1), e_cards, e, order).reshape(r_c, cards[v])
            gap = (center - bc.center.unsqueeze(1)).abs().sum(-1) - (radius + bc.radius.unsqueeze(1))
            n_conflicts = int((gap > 1e-12).sum())
            if n_conflicts:
                g = float(gap.max())
                if worst is None or g > worst["gap"]:
                    worst = {"against": e, "rows": n_conflicts, "gap": g}
        if worst is None:
            survivors.append(c)
        else:
            discarded[c] = worst

    first = data[node]
    device = first.device
    flat_visited_mask = None
    flat_counts = None
    if flat_ok:
        flat_counts, _ = count_table_for(hs.variables, data, node, inputs)
        flat_visited_mask = flat_counts.sum(-1) > 0

    if not survivors:
        # Every size-d candidate is contradicted: |Pa| > d, a parent outside the
        # inputs, or a confounded logger.  If Pa ⊆ inputs still holds, the flat
        # model (all inputs as parents) is a superset and stays valid.
        if flat_counts is not None:
            union: Family = inputs
            fallback = l1_ball_set_from_counts(flat_counts, delta=per_family, sampling=sampling)
            fallback.guarantee = "pac_under_model"
            fallback.level = 1.0 - delta
        else:
            union = cands[0]
            n_rows = math.prod(cards[p] for p in union)
            dtype = balls[cands[0]].center.dtype
            fallback = L1BallSet(
                level=1.0 - delta, guarantee="pac_under_model",
                unseen=torch.ones(n_rows, dtype=torch.bool, device=device),
                center=torch.full((n_rows, k), 1.0 / k, dtype=dtype, device=device),
                radius=torch.full((n_rows,), 2.0, dtype=dtype, device=device),
            )
        union_cards = [cards[p] for p in union]
        joint, _ = count_table_for(hs.variables, data, node, union)
        return AgreementSet(
            node=node, parents=union, parent_cards=union_cards, set=fallback,
            known=(~fallback.unseen) & (fallback.radius <= epsilon),
            covered_jointly=joint.sum(-1) > 0, candidates=cands, survivors=[],
            discarded=discarded, max_parents=d, epsilon=epsilon, level=1.0 - delta,
            n_families=len(fam_list), delta_per_family=per_family, sampling=sampling,
            assumption_refuted=True,
            flat_rows=flat_rows if flat_visited_mask is not None else None,
            flat_visited=int(flat_visited_mask.sum()) if flat_visited_mask is not None else None,
            flat_with_set=int((~fallback.unseen).sum()) if flat_visited_mask is not None else None,
        )

    # 3. One set per row of the union of the survivors' parents.
    union = tuple(p for p in inputs if any(p in s for s in survivors))
    union_cards = [cards[p] for p in union]
    n_rows = math.prod(union_cards)
    if n_rows > max_rows:
        raise ValueError(
            f"The survivors' parents {union} span {n_rows} rows (> max_rows={max_rows}); "
            "lower max_parents, restrict inputs, or raise max_rows."
        )
    grid = _grid(union_cards, device)
    centers, radii, seen = [], [], []
    for s in survivors:
        cols = [union.index(p) for p in s]
        rows = parent_row_index(grid[:, cols], [cards[p] for p in s])
        b = balls[s]
        centers.append(b.center[rows])
        radii.append(b.radius[rows])
        seen.append(~b.unseen[rows])
    center_stack = torch.stack(centers)                    # [S, R, K]
    radius_stack = torch.stack(radii)                      # [S, R]
    all_seen = torch.stack(seen).all(0)                    # [R]
    mean = center_stack.mean(0)
    radius = ((center_stack - mean).abs().sum(-1) + radius_stack).amax(0).clamp_max(2.0)
    mean = torch.where(all_seen.unsqueeze(-1), mean, torch.full_like(mean, 1.0 / k))
    radius = torch.where(all_seen, radius, torch.full_like(radius, 2.0))
    joint, _ = count_table_for(hs.variables, data, node, union)
    flat_with_set = None
    if flat_visited_mask is not None:
        full = _grid([cards[p] for p in inputs], device)
        proj = parent_row_index(full[:, [inputs.index(p) for p in union]], union_cards)
        flat_with_set = int(all_seen[proj].sum())
    result_set = L1BallSet(
        level=1.0 - delta, guarantee="pac_under_model", unseen=~all_seen,
        center=mean, radius=radius,
    )
    return AgreementSet(
        node=node, parents=union, parent_cards=union_cards, set=result_set,
        known=all_seen & (radius <= epsilon), covered_jointly=joint.sum(-1) > 0,
        candidates=cands, survivors=survivors, discarded=discarded, max_parents=d,
        epsilon=epsilon, level=1.0 - delta, n_families=len(fam_list),
        delta_per_family=per_family, sampling=sampling,
        flat_rows=flat_rows if flat_visited_mask is not None else None,
        flat_visited=int(flat_visited_mask.sum()) if flat_visited_mask is not None else None,
        flat_with_set=flat_with_set,
    )


def _grid(cards: List[int], device) -> torch.Tensor:
    """All parent configurations ``[prod(cards), len(cards)]`` in C order (last fastest)."""
    if not cards:
        return torch.zeros(1, 0, dtype=torch.long, device=device)
    axes = [torch.arange(c, device=device) for c in cards]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=-1)
