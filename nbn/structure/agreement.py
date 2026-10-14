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

**Union over a given family** (``refute=False``).  When the family of
candidates comes from elsewhere (the parent sets accepted by an invariance
test, a Bayesian credible family, a declared list), step 2 is skipped: no
extension families, every candidate survives, ``delta`` is split over the
candidates only, and step 3 gives the row-wise hull of their balls.  The
result is valid when some candidate contains the true parents; if the family
holds only with probability ``1 - delta_G`` (e.g. ICP at level ``delta_G``),
the result holds with probability at least ``1 - delta_G - delta`` (union
bound; the caller accounts for ``delta_G``).

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
    # The declared bound d, or None with explicit candidates (not used then).
    max_parents: int | None = 0
    epsilon: float = 0.0
    level: float = 0.95
    n_families: int = 0
    delta_per_family: float = 0.0
    sampling: str = "iid"
    assumption_refuted: bool = False
    # Whether candidates were tested against their one-variable extensions
    # (the agreement rule); False: the union over the given family.
    refute: bool = True
    # Declared assumptions; with explicit candidates the first one is "some
    # given candidate contains Pa(node)" instead of the bound on max_parents.
    assumptions: Tuple[str, ...] = ASSUMPTIONS
    # Coverage against the flat model (one row per joint value of all inputs),
    # on `coverage_rows` if given, else on every joint value (when there are at
    # most max_rows of them); None when not computed.
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
        values of all inputs; ``None`` when that row space was too large, or
        when every candidate was refuted and the flat fallback could not be
        built (then no set is compared).
        """
        if self.flat_with_set is None or self.flat_visited is None:
            return None
        return self.flat_with_set - self.flat_visited


def agreement_set(
    variables: Union[HypothesisSet, Mapping[str, Union[Tuple, Variable]]],
    data: Mapping[str, torch.Tensor],
    node: str,
    inputs: Sequence[str],
    max_parents: int | None = None,
    delta: float = 0.05,
    epsilon: float = 0.1,
    sampling: Sampling = "iid",
    candidates: Sequence[Sequence[str]] | None = None,
    max_rows: int = 1 << 20,
    max_families: int = 1 << 14,
    coverage_rows: torch.Tensor | None = None,
    refute: bool = True,
) -> AgreementSet:
    """One probability set per row of ``node``, valid without knowing its parents.

    Parameters
    ----------
    variables:
        Variable specs (as for :class:`HypothesisSet`), or a
        :class:`HypothesisSet`: its variables are used and, when ``candidates``
        is not given, its candidates for ``node``.
    data:
        Columns by name (unweighted counts of independent draws, or
        transitions of episodes with ``sampling="trajectory"``).
    node:
        The discrete child.
    inputs:
        The variables allowed as parents; the true parents must be among them.
    max_parents:
        Declared bound ``d`` on the number of true parents; required unless
        ``candidates`` is given (then it is not used, and
        :attr:`AgreementSet.max_parents` is ``None``).
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
        Largest row space (union of the survivors' parents, or all inputs for
        the flat comparison) built densely.
    max_families:
        Largest number of families (candidates plus extensions) to build;
        checked before any counting.
    coverage_rows:
        Optional ``[n, len(inputs)]`` joint input values (columns in the order
        of ``inputs``) on which the coverage counts ``flat_*`` are taken, e.g.
        the feasible state-action rows when the inputs encode actions with
        constraints (one-hot repair flags).  Default: every joint value, which
        can include combinations that never occur.
    refute:
        ``True`` (default): the agreement rule, which discards candidates that
        disagree with a one-variable extension.  ``False``: no refutation; the
        result is the row-wise union (hull) of the given ``candidates``'
        balls, with ``delta`` split over them only.  Use it for a family
        built elsewhere (ICP accepted sets, a credible family): valid when
        some candidate contains ``Pa(node)``.  Requires ``candidates`` (or a
        :class:`HypothesisSet` with candidates for ``node``).

    Returns
    -------
    AgreementSet
    """
    if isinstance(variables, HypothesisSet):
        if candidates is None and node in variables.candidates:
            candidates = variables.candidates[node]       # the user's own candidates
        variables = variables.variables
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1); got {delta}.")
    if isinstance(inputs, str):
        raise TypeError(f"inputs must be a sequence of names, not the string {inputs!r}.")
    inputs = tuple(inputs)
    if len(set(inputs)) != len(inputs):
        raise ValueError(f"inputs has repeated names: {list(inputs)}.")
    if node in inputs:
        raise ValueError(f"'{node}' cannot be one of its own inputs.")
    if not refute and candidates is None:
        raise ValueError("refute=False needs the family: pass candidates (or a HypothesisSet with "
                         f"candidates for '{node}').")
    if candidates is None and max_parents is None:
        raise ValueError("max_parents is required when candidates is not given.")
    if max_parents is not None and max_parents < 0:
        raise ValueError(f"max_parents must be >= 0; got {max_parents}.")
    d = len(inputs) if max_parents is None else min(int(max_parents), len(inputs))
    if candidates is None:
        # candidates plus their distinct one-variable extensions (deduplicated as sets)
        n_default = math.comb(len(inputs), d) + (math.comb(len(inputs), d + 1) if d < len(inputs) else 0)
        if n_default > max_families:
            raise ValueError(
                f"{len(inputs)} inputs with max_parents={d} give about {n_default} families "
                f"(> max_families={max_families}); restrict inputs or pass candidates."
            )
        cands: List[Family] = [tuple(c) for c in itertools.combinations(inputs, d)]
    else:
        cands = []
        seen_sets: set = set()
        for c in candidates:
            if isinstance(c, str):
                raise TypeError(f"Each candidate must be a tuple of names, not {c!r}.")
            if not set(c) <= set(inputs):
                raise ValueError(f"Candidate {tuple(c)} uses variables outside inputs {inputs}.")
            if frozenset(c) in seen_sets:
                raise ValueError(f"Candidate {tuple(c)} is listed twice (order does not matter).")
            seen_sets.add(frozenset(c))
            cands.append(tuple(c))
        if not cands:
            raise ValueError("candidates is empty.")

    # Families: candidates and their one-variable extensions, deduplicated as sets.
    families: Dict[frozenset, Family] = {frozenset(c): c for c in cands}
    extensions: Dict[Family, List[Tuple[str, Family]]] = {}
    for c in cands:
        extensions[c] = []
        if not refute:
            continue                                      # union over the given family: no extensions
        for v in inputs:
            if v in c:
                continue
            e = families.setdefault(frozenset((*c, v)), (*c, v))
            extensions[c].append((v, e))
    fam_list = list(families.values())
    if len(fam_list) > max_families:
        raise ValueError(f"{len(fam_list)} families (> max_families={max_families}).")
    # A one-node HypothesisSet gives the factor cache (count once, marginalise).
    spec = {v: variables[v] for v in (*inputs, node)}
    hs = HypothesisSet(spec, {node: fam_list})
    counts = hs.counts(data, keep_union_tables=True)
    cards = {v: int(hs.variables[v].cardinality) for v in (*inputs, node)}
    k = cards[node]
    flat_rows = math.prod(cards[p] for p in inputs)
    flat_ok = flat_rows <= max_rows
    flat_is_family = frozenset(inputs) in families
    # One share of the budget is kept for the flat fallback (all inputs as parents)
    # whenever it is not already a family; it does not depend on max_rows.  Without
    # refutation every candidate survives, so there is no fallback to pay for.
    per_family = delta / (len(fam_list) + (0 if flat_is_family or not refute else 1))
    balls: Dict[Family, L1BallSet] = {
        f: l1_ball_set_from_counts(counts[(node, f)], delta=per_family, sampling=sampling)
        for f in fam_list
    }

    # 2. Refutation against the one-variable extensions (one host sync in total).
    # Tolerance: float32 rounding (MPS) must not refute a ball that just touches.
    tol = 1e-6
    pairs: List[Tuple[Family, Family]] = []
    stats: List[torch.Tensor] = []
    for c in cands:
        bc = balls[c]
        r_c = bc.unseen.numel()
        for v, e in extensions[c]:
            be = balls[e]
            order = (*c, v)
            e_cards = [cards[p] for p in e]
            center = _reorder_parents(be.center, e_cards, e, order).reshape(r_c, cards[v], k)
            radius = _reorder_parents(be.radius.unsqueeze(-1), e_cards, e, order).reshape(r_c, cards[v])
            gap = (center - bc.center.unsqueeze(1)).abs().sum(-1) - (radius + bc.radius.unsqueeze(1))
            stats.append(torch.stack([(gap > tol).sum().to(gap.dtype), gap.max()]))
            pairs.append((c, e))
    values = torch.stack(stats).tolist() if stats else []
    worst: Dict[Family, dict] = {}
    for (c, e), (n_conflicts, g) in zip(pairs, values):
        if n_conflicts > 0 and (c not in worst or g > worst[c]["gap"]):
            worst[c] = {"against": e, "rows": int(n_conflicts), "gap": float(g)}
    survivors = [c for c in cands if c not in worst]
    discarded = {c: worst[c] for c in cands if c in worst}

    device = data[node].device
    input_cards = [cards[p] for p in inputs]
    flat_counts = None
    if flat_ok:
        if flat_is_family:
            flat_counts = counts[(node, families[frozenset(inputs)])]
            flat_counts = _reorder_parents(flat_counts, [cards[p] for p in families[frozenset(inputs)]],
                                           families[frozenset(inputs)], inputs)
        elif node in counts.union_tables and frozenset(counts.union_tables[node][0]) == frozenset(inputs):
            u_names, u_table = counts.union_tables[node]          # counted once already
            flat_counts = _reorder_parents(u_table, [cards[p] for p in u_names], u_names, inputs)
        else:
            flat_counts, _ = count_table_for(hs.variables, data, node, inputs)
    if coverage_rows is not None:
        coverage_rows = torch.as_tensor(coverage_rows, device=device).long()
        if coverage_rows.dim() != 2 or coverage_rows.shape[1] != len(inputs):
            raise ValueError(f"coverage_rows must be [n, {len(inputs)}] (columns in the order of inputs).")
        bad = (coverage_rows < 0) | (coverage_rows >= torch.tensor(input_cards, device=device))
        if coverage_rows.numel() and bool(bad.any()):
            raise ValueError("coverage_rows has values outside the inputs' declared states.")
        # Repeated rows count once.  Rows, not flat indices: the joint index of
        # all inputs can overflow int64 when the flat table is not built.
        coverage_rows = torch.unique(coverage_rows, dim=0)
        if flat_counts is not None:
            visited_flat = flat_counts.sum(-1) > 0
            flat_visited: int | None = int(visited_flat[parent_row_index(coverage_rows, input_cards)].sum())
        else:
            data_rows = torch.unique(torch.stack([data[p].reshape(-1).long() for p in inputs], -1), dim=0)
            _, seen_twice = torch.unique(torch.cat([coverage_rows, data_rows]), dim=0, return_counts=True)
            flat_visited = int((seen_twice == 2).sum())      # both lists are free of repeats
        n_coverage: int | None = int(coverage_rows.shape[0])
    else:
        flat_visited = int((flat_counts.sum(-1) > 0).sum()) if flat_counts is not None else None
        n_coverage = flat_rows if flat_ok else None

    def with_set_on_coverage(row_set_unseen: torch.Tensor, union: Family) -> int | None:
        """Coverage rows whose projection onto `union` has a set."""
        if coverage_rows is None:
            if not flat_ok:
                return None
            n_union = math.prod(cards[p] for p in union)
            return int((~row_set_unseen).sum()) * (flat_rows // n_union)
        proj = parent_row_index(coverage_rows[:, [inputs.index(p) for p in union]],
                                [cards[p] for p in union])
        return int((~row_set_unseen[proj]).sum())

    def joint_counts(union: Family) -> torch.Tensor:
        """Counts over `union`: summed down from the flat table when there is one."""
        if flat_counts is not None and union == tuple(p for p in inputs if p in union):
            full = flat_counts.reshape(*[cards[p] for p in inputs], k)
            drop = [i for i, p in enumerate(inputs) if p not in union]
            return (full.sum(dim=drop) if drop else full).reshape(-1, k)
        return count_table_for(hs.variables, data, node, union)[0]

    if not refute:
        assumptions: Tuple[str, ...] = (
            "some given candidate contains Pa(node) (union over the given family, no refutation)",
            *ASSUMPTIONS[2:],
        )
    elif candidates is None:
        assumptions = ASSUMPTIONS
    else:
        assumptions = (
            "some given candidate contains Pa(node) (explicit candidates: max_parents is not used)",
            *ASSUMPTIONS[1:],
        )
    common = dict(
        node=node, candidates=cands, discarded=discarded,
        max_parents=d if candidates is None else None, epsilon=epsilon,
        level=1.0 - delta, n_families=len(fam_list), delta_per_family=per_family,
        sampling=sampling, flat_rows=n_coverage, flat_visited=flat_visited, assumptions=assumptions,
        refute=refute,
    )

    if not survivors:
        # Every candidate is contradicted: |Pa| > d, a parent outside the inputs,
        # or a confounded logger.  If Pa ⊆ inputs still holds, the flat model (all
        # inputs as parents) is a superset and stays valid.
        if flat_counts is not None:
            union: Family = inputs
            fallback = l1_ball_set_from_counts(flat_counts, delta=per_family, sampling=sampling)
            fallback.guarantee = "pac_under_model"
            fallback.level = 1.0 - delta
        else:
            # The flat table is too large to build: no information (one simplex row,
            # no parents), never a refuted candidate's row space.
            union = ()
            dtype = balls[cands[0]].center.dtype
            fallback = L1BallSet(
                level=1.0 - delta, guarantee="pac_under_model",
                unseen=torch.ones(1, dtype=torch.bool, device=device),
                center=torch.full((1, k), 1.0 / k, dtype=dtype, device=device),
                radius=torch.full((1,), 2.0, dtype=dtype, device=device),
            )
        return AgreementSet(
            parents=union, parent_cards=[cards[p] for p in union], set=fallback,
            known=(~fallback.unseen) & (fallback.radius <= epsilon),
            covered_jointly=joint_counts(union).sum(-1) > 0, survivors=[],
            assumption_refuted=True,
            # Without the flat table there is no fallback set to compare.
            flat_with_set=with_set_on_coverage(fallback.unseen, union) if flat_counts is not None else None,
            **common,
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
    # Two streaming passes over the survivors: memory O(R K), whatever their number.
    def lifted(s: Family, table: torch.Tensor) -> torch.Tensor:
        """A survivor's [R_s, C] table broadcast to the union rows [R, C]."""
        in_u = tuple(p for p in union if p in s)
        t = _reorder_parents(table, [cards[p] for p in s], s, in_u)
        shape = [cards[p] if p in s else 1 for p in union] + [t.shape[-1]]
        return t.reshape(shape).expand(*union_cards, t.shape[-1]).reshape(n_rows, t.shape[-1])

    mean = torch.zeros(n_rows, k, dtype=balls[survivors[0]].center.dtype, device=device)
    all_seen = torch.ones(n_rows, dtype=torch.bool, device=device)
    for s in survivors:
        b = balls[s]
        mean += lifted(s, b.center)
        all_seen &= lifted(s, (~b.unseen).unsqueeze(-1).to(mean.dtype))[:, 0] > 0.5
    mean /= len(survivors)
    radius = torch.zeros(n_rows, dtype=mean.dtype, device=device)
    for s in survivors:
        b = balls[s]
        spread = (lifted(s, b.center) - mean).abs().sum(-1) + lifted(s, b.radius.unsqueeze(-1))[:, 0]
        radius = torch.maximum(radius, spread)
    radius = radius.clamp_max(2.0)
    mean = torch.where(all_seen.unsqueeze(-1), mean, torch.full_like(mean, 1.0 / k))
    radius = torch.where(all_seen, radius, torch.full_like(radius, 2.0))
    result_set = L1BallSet(
        level=1.0 - delta, guarantee="pac_under_model", unseen=~all_seen,
        center=mean, radius=radius,
    )
    return AgreementSet(
        parents=union, parent_cards=union_cards, set=result_set,
        known=all_seen & (radius <= epsilon), covered_jointly=joint_counts(union).sum(-1) > 0,
        survivors=survivors,
        flat_with_set=with_set_on_coverage(~all_seen, union),
        **common,
    )

