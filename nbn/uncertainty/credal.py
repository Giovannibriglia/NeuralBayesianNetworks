"""Sound query bounds over probability sets, and PAC query answers from data.

Why
---
The builders of :mod:`nbn.uncertainty` give, for every discrete node, a set
``R_pa`` of possible rows ``P(X | pa)`` (intervals, L1 balls) that contains
the true row with a declared probability.  Inference engines, however, answer
queries on *one* fitted model.  This module closes the gap: it computes
bounds on ``P(y | e)`` (and ``P(y | e, do)``) that hold for **every** network
whose rows lie in the given sets.  If the sets hold jointly with probability
``>= 1 - delta`` (union bound over the factors used), the bounds contain the
*true* answer with the same probability: the computation after the sets is
deterministic, so the confidence is spent only once, on the data
[the scheme of model-based statistical model checking: Ashok et al., CAV
2019, arXiv:1904.00185, proof of Thm 1; Meggendorfer et al. 2025, §2.2].
Every query computed from the same sets holds simultaneously.

How (credal variable elimination, outer bound)
----------------------------------------------
1. Keep the ancestral set of the target, the evidence and the do-nodes:
   the other nodes are barren and sum to one for *every* parameterisation
   [Koller & Friedman 2009, Ex. 3.8, p. 98; §9.3.1, p. 301].
2. Write ``P(y, e) = sum_{x_1} th_1(x_1) sum_{x_2} th_2(x_2 | pa_2) ...`` in
   topological order (K&F Eq. 9.6, p. 298) and evaluate it innermost first:
   variables are eliminated children before parents, so when ``X`` is
   eliminated its own row is the only free parameter in the product.
3. Upper message: ``M(pa, z) = max_{p in R_pa} sum_x p_x m(x, z)`` with the
   upper messages ``m >= 0`` of the descendants; lower: ``min`` with lower
   messages.  By induction, for any network ``th`` in the sets,
   ``m_th <= m_upper`` and ``M_th = sum_x th_x m_th <= sum_x th_x m_upper <=
   max_p sum_x p_x m_upper``.  Optimising separately for every ``z`` lets a
   row take different values in different contexts: a relaxation (sound, not
   tight), like dropping parameter dependencies in parameter lifting /
   rectangular robust MDPs [Badings, *Robust Verification of Stochastic
   Systems*, 2025, Lemma 3.32 and p. 149].  The soundness argument is ours,
   checked with the koller_claude and badings_claude experts.
4. The inner problems are solved exactly: for an interval row, start at the
   lower bounds and give the remaining mass to the largest ``m`` first (the
   ordering algorithm of interval MDPs: Haddad & Monmege, TCS 2018, Lemma 6;
   Suilen et al. 2024, Alg. 1); for an L1 ball, move ``min(r/2, 1 - c_x*)``
   of mass to the best entry from the worst ones (Strehl & Littman 2008;
   Suilen et al. 2024, Alg. 2).  Other sets are used through their outer box
   ``bounds()``.
5. ``P(y | e) = a / (a + b)`` with ``a = P(y, e)``, ``b = P(not y, e)`` is
   increasing in ``a`` and decreasing in ``b``, so
   ``[L_a / (L_a + U_b), U_a / (U_a + L_b)]`` is sound; ``b`` is bounded
   directly (the event ``Y != y``), which is tighter than summing per-class
   bounds.  Finally the class intervals are tightened with ``sum = 1``.

Tight credal inference is NP-hard in general (already for precise networks,
K&F Thm 9.4, p. 291, for posteriors), so a cheap sound bound must be loose
somewhere; the gap is measured in the tests and the docs, not assumed small.

Cost: one elimination per bound (``2 (2K + 1)`` for a ``K``-class target),
each like VE with a children-first order, so its width can exceed the best
VE width.  ``max_elements`` caps the largest intermediate table.

The guarantee also needs what the sets need: discrete variables, the correct
DAG (or parents that are supersets of the true ones), and the sampling
assumptions of the builders (``sampling="iid"`` or ``"trajectory"``).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import torch

from nbn.uncertainty.sets import L1BallSet, ProbabilitySet

_ORDER = ("pac", "pac_under_model", "asymptotic", "bayesian")


@dataclass(frozen=True)
class QueryBounds:
    """Bounds on ``P(target | evidence, do)`` over a set of networks.

    ``lower[y] <= P(y | e) <= upper[y]`` for every network whose rows lie in
    the sets; with probability ``>= level`` over the data when the sets came
    from data (``None`` when no set declares a level).  ``joint_*`` bound
    ``P(y, e)`` and ``evidence_*`` bound ``P(e)``.  ``point`` is the answer of
    the model passed (if any), for comparison.
    """

    lower: torch.Tensor
    upper: torch.Tensor
    level: float | None
    guarantee: str
    joint_lower: torch.Tensor
    joint_upper: torch.Tensor
    evidence_lower: float
    evidence_upper: float
    nodes: tuple
    point: torch.Tensor | None = None

    @property
    def max_width(self) -> float:
        return float((self.upper - self.lower).max())


class CredalSets(dict):
    """``{node: ProbabilitySet}`` with a joint ``level`` (see :func:`pac_sets`)."""

    def __init__(self, sets, level: float, guarantee: str, delta: float):
        super().__init__(sets)
        # the joint level holds for these sets only: an entry replaced or
        # added later falls back to the union bound in credal_query
        self._ids = {n: id(v) for n, v in self.items()}
        self.level = level
        self.guarantee = guarantee
        self.delta = delta


# ---------------------------------------------------------------------------
# inner problems
# ---------------------------------------------------------------------------
def _interval_opt(g: torch.Tensor, lo: torch.Tensor, up: torch.Tensor, maximise: bool):
    """``max`` / ``min`` of ``sum_x p_x g_x`` over ``{lo <= p <= up, sum p = 1}``
    along the last axis (``lo``, ``up`` broadcast to ``g``)."""
    order = torch.argsort(g, dim=-1, descending=maximise)
    gs = torch.gather(g, -1, order)
    los = torch.gather(lo.expand_as(g), -1, order)
    ups = torch.gather(up.expand_as(g), -1, order)
    cap = (ups - los).clamp_min(0.0)
    budget = (1.0 - los.sum(-1, keepdim=True)).clamp_min(0.0)
    before = torch.cumsum(cap, dim=-1) - cap
    extra = torch.minimum(cap, (budget - before).clamp_min(0.0))
    return (los * gs).sum(-1) + (extra * gs).sum(-1)


def _l1_opt(g: torch.Tensor, c: torch.Tensor, r: torch.Tensor, maximise: bool):
    """``max`` / ``min`` of ``sum_x p_x g_x`` over ``{||p - c||_1 <= r, p in simplex}``."""
    c = c.expand_as(g)
    sgn = 1.0 if maximise else -1.0
    h = sgn * g                                        # maximise sum p h
    best = torch.argmax(h, dim=-1, keepdim=True)
    c_best = torch.gather(c, -1, best)
    move = torch.minimum((r / 2.0).expand_as(c_best), 1.0 - c_best).clamp_min(0.0)
    order = torch.argsort(h, dim=-1)                   # worst first
    hs = torch.gather(h, -1, order)
    cap = torch.gather(c, -1, order)
    is_best = order == best
    cap = torch.where(is_best, torch.zeros_like(cap), cap)
    before = torch.cumsum(cap, dim=-1) - cap
    take = torch.minimum(cap, (move - before).clamp_min(0.0))
    val = (c * h).sum(-1) + (move * torch.gather(h, -1, best)).squeeze(-1) - (take * hs).sum(-1)
    return sgn * val


# ---------------------------------------------------------------------------
# credal elimination
# ---------------------------------------------------------------------------
def _parents_of(model_or_parents, node):
    if isinstance(model_or_parents, Mapping):
        return list(model_or_parents.get(node, ()))
    return list(model_or_parents.dag.parents(node))


def _ancestral(parents_fn, nodes):
    out, stack = set(), list(nodes)
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        stack.extend(parents_fn(n))
    return out


def _topo(parents_fn, nodes):
    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for p in parents_fn(n):
            if p in nodes:
                visit(p)
        order.append(n)
    for n in sorted(nodes):
        visit(n)
    return order


class _Row:
    """Row sets of one node, as float64 tensors ``[*parent_cards, K]``."""

    def __init__(self, pset: ProbabilitySet, pcards, k):
        shape = (*pcards, k)
        if isinstance(pset, L1BallSet):
            self.kind = "l1"
            c = pset.center.detach().cpu().double().reshape(shape).clamp_min(0.0)
            # the closed form needs sum(c) == 1 (a float32 centre may miss by ~1e-7)
            self.c = c / c.sum(-1, keepdim=True).clamp_min(1e-300)
            self.r = pset.radius.detach().cpu().double().reshape(*pcards, 1)
        else:
            self.kind = "box"
        lo, up = pset.bounds()
        self.lo = lo.detach().cpu().double().reshape(shape)
        self.up = up.detach().cpu().double().reshape(shape)
        # rows without data are the whole simplex, whatever the set stores
        # (e.g. prior quantiles of a Dirichlet set, a refit bootstrap box)
        unseen = getattr(pset, "unseen", None)
        n_rows = math.prod(pcards)
        if unseen is not None and unseen.numel() not in (0, n_rows):
            raise ValueError(f"credal_query: a set's unseen mask has {unseen.numel()} rows, "
                             f"not {n_rows}.")
        if (unseen is None or unseen.numel() == 0) and getattr(pset, "guarantee", None) in (
                "asymptotic", "bayesian"):
            warnings.warn(
                "credal_query: a sample-based set has no unseen mask, so rows without data "
                "keep their (prior- or refit-driven) box; pass unseen= to bootstrap_set.",
                stacklevel=3)
        if unseen is not None and unseen.numel() == n_rows:
            m = unseen.detach().cpu().bool().reshape(tuple(pcards))
            if bool(m.any()):
                self.lo[m] = 0.0
                self.up[m] = 1.0
                if self.kind == "l1":
                    self.r[m] = 2.0

    def opt(self, g, maximise):
        """``g``: ``[*pa, *z, K]`` -> ``[*pa, *z]``."""
        nz = g.dim() - self.lo.dim()
        view = (*self.lo.shape[:-1], *([1] * nz), self.lo.shape[-1])
        if self.kind == "l1":
            rv = (*self.r.shape[:-1], *([1] * nz), 1)
            return _l1_opt(g, self.c.reshape(view), self.r.reshape(rv), maximise)
        return _interval_opt(g, self.lo.reshape(view), self.up.reshape(view), maximise)

    def entry(self, v, maximise):
        """Extreme of the single entry ``p_v`` per row: ``[*pa]``."""
        return (self.up if maximise else self.lo)[..., v]


def _imul(alo, aup, blo, bup):
    """Interval product, element-wise (either factor may be signed)."""
    c = torch.stack([alo * blo, alo * bup, aup * blo, aup * bup])
    return c.amin(0), c.amax(0)


def _iproduct(mine, scope, cards, max_elements):
    n = math.prod(cards[v] for v in scope)
    if n > max_elements:
        raise MemoryError(f"credal elimination: an intermediate table of {n} entries "
                          f"exceeds max_elements={max_elements}.")
    lo = torch.ones([cards[v] for v in scope], dtype=torch.float64)
    up = lo.clone()
    for s, tlo, tup in mine:
        idx = [scope.index(v) for v in s]
        perm = sorted(range(len(s)), key=lambda i: idx[i])
        shape = [1] * len(scope)
        for i in perm:
            shape[idx[i]] = cards[s[i]]
        a = (tlo.permute(*perm) if s else tlo).reshape(shape)
        b = (tup.permute(*perm) if s else tup).reshape(shape)
        lo, up = _imul(lo, up, a, b)
    return lo, up


def _eliminate(nodes_topo, parents_fn, rows, cards, clamp, do_nodes, event, max_elements):
    """Bounds on ``sum_x P(x, clamp) g(x_target)`` over the sets, as
    ``(lo, up, log_scale)``: the bounds are ``lo * exp(log_scale)`` and
    ``up * exp(log_scale)``.

    Messages carry an interval ``[lo, up]`` per entry.  They are
    non-negative except downstream of the target when ``g`` is signed (the
    generalized Bayes rule); the inner problems are valid for signed
    messages because the rows are non-negative, and products use interval
    arithmetic.  Every new message is divided by its largest magnitude (one
    positive scalar for ``lo`` and ``up``, kept in ``log_scale``), so many
    evidence factors cannot underflow float64; ratios and signs are
    unchanged by the scale.

    Order: a node is eliminated once all its children are (so its own row is
    the only free parameter in the product); among those, the one with the
    smallest resulting scope first (greedy, independent of node names).
    """
    msgs = []                                  # (scope tuple, lo, up)
    tgt, ind = event
    nodes = set(nodes_topo)
    children = {n: set() for n in nodes}
    for n in nodes:
        for q in parents_fn(n):
            if q in nodes:
                children[q].add(n)
    done, log_scale = set(), 0.0

    def scope_of(x):
        sc = {x} | set(parents_fn(x)) if x not in do_nodes else {x}
        for s_, _, _ in msgs:
            if x in s_:
                sc |= set(s_)
        return sc

    while len(done) < len(nodes):
        ready = [n for n in nodes if n not in done and children[n] <= done]
        x = min(ready, key=lambda n: (math.prod(cards[v] for v in scope_of(n)), n))
        done.add(x)
        mine = [m for m in msgs if x in m[0]]
        msgs = [m for m in msgs if x not in m[0]]
        scope = [x]
        for s_, _, _ in mine:
            scope += [v for v in s_ if v not in scope]
        if x in do_nodes:
            if not mine:
                continue
            lo, up = _iproduct(mine, scope, cards, max_elements)
            i = scope.index(x)
            lo, up = lo.select(i, int(clamp[x])), up.select(i, int(clamp[x]))
            out_scope = tuple(v for v in scope if v != x)
        else:
            pa = list(parents_fn(x))
            z = [v for v in scope if v != x and v not in pa]
            full = pa + z + [x]
            if mine:
                glo, gup = _iproduct(mine, full, cards, max_elements)
            else:
                glo = torch.ones([cards[v] for v in full], dtype=torch.float64)
                gup = glo.clone()
            if x == tgt:
                w = ind.reshape(*([1] * (len(full) - 1)), -1)
                glo, gup = torch.minimum(glo * w, gup * w), torch.maximum(glo * w, gup * w)
            row = rows[x]
            if x in clamp:
                v = int(clamp[x])
                shape = (*row.lo.shape[:-1], *([1] * len(z)))
                lo, up = _imul(row.entry(v, False).reshape(shape),
                               row.entry(v, True).reshape(shape), glo[..., v], gup[..., v])
            else:
                lo, up = row.opt(glo, False), row.opt(gup, True)
            out_scope = tuple(pa + z)
        m = float(torch.maximum(lo.abs().max(), up.abs().max())) if lo.numel() else 0.0
        if m > 0.0 and math.isfinite(m):
            lo, up = lo / m, up / m
            log_scale += math.log(m)
        msgs.append((out_scope, lo, up))
    lo = torch.ones((), dtype=torch.float64)
    up = lo.clone()
    for _, tlo, tup in msgs:
        lo, up = _imul(lo, up, tlo.reshape(()), tup.reshape(()))
    return float(lo), float(up), log_scale


def _ratio(a, b):
    """``a / (a + b)`` for scaled non-negative numbers ``(mantissa, log_scale)``;
    ``None`` when both are 0."""
    (am, al), (bm, bl) = a, b
    if am <= 0.0 and bm <= 0.0:
        return None
    if am <= 0.0:
        return 0.0
    if bm <= 0.0:
        return 1.0
    t = math.log(bm) + bl - math.log(am) - al
    return 0.0 if t > 745.0 else 1.0 / (1.0 + math.exp(t))


def credal_query(
    model,
    sets: Mapping[str, ProbabilitySet],
    target: str,
    evidence: Mapping[str, object] | None = None,
    *,
    do: Mapping[str, object] | None = None,
    cards: Mapping[str, int] | None = None,
    max_elements: int = 2 ** 26,
    point: bool = True,
    tight: bool = True,
    bisect_iters: int = 30,
    tol: float = 1e-6,
) -> QueryBounds:
    """Bounds on ``P(target | evidence, do)`` over every network whose rows
    lie in ``sets`` (see the module docstring).

    ``model``: an NBN (its DAG gives the parents; its fitted answer is
    reported as ``point``) or a ``{node: parents}`` mapping.  ``sets``:
    ``{node: ProbabilitySet}`` with rows in the layout of ``count_table``
    (C order over the parents in DAG order), for every node in the ancestral
    set of the target, the evidence and the do-nodes (do-nodes excepted).
    ``level`` is ``1 - sum(1 - level_i)`` over the sets used (union bound),
    or the joint level of a :class:`CredalSets`.  ``tight``: refine the
    ratio bounds by the generalized Bayes rule (bisection on ``mu`` with
    ``min P(y,e) - mu P(e) >= 0``; about ``2 * bisect_iters`` more
    eliminations per class; the result is the better of the two, both sound).
    """
    evidence = {k: int(torch.as_tensor(v).reshape(-1)[0]) for k, v in (evidence or {}).items()}
    do = {k: int(torch.as_tensor(v).reshape(-1)[0]) for k, v in (do or {}).items()}
    if target in evidence or target in do:
        raise ValueError("credal_query: the target is observed or intervened.")
    if set(evidence) & set(do):
        raise ValueError(f"credal_query: {sorted(set(evidence) & set(do))} are both "
                         f"observed and intervened.")

    def parents_fn(n):
        return [] if n in do else _parents_of(model, n)

    keep = _ancestral(parents_fn, [target, *evidence, *do])
    topo = _topo(parents_fn, keep)
    need = [n for n in topo if n not in do]
    missing = [n for n in need if n not in sets]
    if missing:
        raise KeyError(f"credal_query: no probability set for {missing}.")
    card = {}
    for n in need:
        lo, _ = sets[n].bounds()
        card[n] = int(lo.shape[-1])
    for n in do:
        card[n] = int(cards[n]) if cards and n in cards else (
            int(model.mechanisms[n].n_classes) if not isinstance(model, Mapping) else
            max(do[n] + 1, 1))
    if cards:
        card.update({k: int(v) for k, v in cards.items()})
    rows = {}
    for n in need:
        pc = [card[p] for p in _parents_of(model, n)]
        rows[n] = _Row(sets[n], pc, card[n])
    clamp = {**evidence, **do}
    k = card[target]

    def bound(ind):
        return _eliminate(topo, parents_fn, rows, card, clamp, set(do), (target, ind),
                          max_elements)

    def scaled(v, end):                       # (mantissa, log_scale) of one end
        return (max(v[end], 0.0), v[2])

    def value(v, end):
        return v[end] * math.exp(v[2]) if v[end] != 0.0 else 0.0

    ones = torch.ones(k, dtype=torch.float64)
    pe = bound(ones)
    if pe[1] <= 0.0:
        raise ValueError("credal_query: the evidence has probability 0 under every network "
                         "in the sets; P(target | evidence) is undefined.")
    jl, ju, lo, up = [], [], [], []
    for y in range(k):
        e_y = torch.zeros(k, dtype=torch.float64)
        e_y[y] = 1.0
        a = bound(e_y)
        b_ = bound(ones - e_y)
        jl.append(value(a, 0))
        ju.append(value(a, 1))
        r_lo = _ratio(scaled(a, 0), scaled(b_, 1))
        r_up = _ratio(scaled(a, 1), scaled(b_, 0))
        l_y = 0.0 if r_lo is None else r_lo
        u_y = 1.0 if r_up is None else r_up
        if tight and u_y - l_y > tol:
            # generalized Bayes rule: P(y|e) >= mu  iff  min_th P(y,e) - mu P(e) >= 0
            # (P(e) > 0); a lower bound >= 0 of that minimum certifies mu.  The
            # sign of a scaled bound is the sign of its mantissa.
            g = lambda mu: e_y - mu * ones          # noqa: E731
            lo_m, hi_m = l_y, u_y
            for _ in range(bisect_iters):
                mid = 0.5 * (lo_m + hi_m)
                if bound(g(mid))[0] > 0.0:
                    lo_m = mid
                else:
                    hi_m = mid
                if hi_m - lo_m <= tol:
                    break
            new_l = lo_m
            lo_m, hi_m = l_y, u_y
            for _ in range(bisect_iters):
                mid = 0.5 * (lo_m + hi_m)
                if bound(g(mid))[1] < 0.0:
                    hi_m = mid
                else:
                    lo_m = mid
                if hi_m - lo_m <= tol:
                    break
            l_y, u_y = max(l_y, new_l), min(u_y, hi_m)
        lo.append(l_y)
        up.append(u_y)
    pe_lo, pe_up = value(pe, 0), value(pe, 1)
    slack = 1e-9
    lower = (torch.tensor(lo, dtype=torch.float64) - slack).clamp(0.0, 1.0)
    upper = (torch.tensor(up, dtype=torch.float64) + slack).clamp(0.0, 1.0)
    lower = torch.maximum(lower, (1.0 - (upper.sum() - upper)).clamp(min=0.0))
    upper = torch.minimum(upper, (1.0 - (lower.sum() - lower)).clamp(max=1.0))

    used = [sets[n] for n in need]

    def miss(s_):                 # failure probability of one set
        lv = float(s_.level)
        if getattr(s_, "simultaneous", True) is False:   # per-entry level
            lo_, _ = s_.bounds()
            return min(1.0, lo_.numel() * (1.0 - lv))
        return 1.0 - lv
    if isinstance(sets, CredalSets) and all(sets._ids.get(n) == id(sets[n]) for n in need):
        level = sets.level
    elif all(getattr(s_, "level", None) is not None for s_ in used):
        level = max(0.0, 1.0 - sum(miss(s_) for s_ in used))
    else:
        level = None
    gs = [getattr(s, "guarantee", "pac") for s in used]
    worst = max(gs, key=lambda g: _ORDER.index(g) if g in _ORDER else len(_ORDER))
    guarantee = "pac_under_model" if worst == "pac" else worst

    pt = None
    if point and not isinstance(model, Mapping):
        try:
            pt = model.query([target], evidence=evidence or None, do=do or None).detach() \
                .cpu().double().reshape(-1)
        except Exception:  # noqa: BLE001  the comparison is optional
            pt = None
    return QueryBounds(lower, upper, level, guarantee,
                       torch.tensor(jl, dtype=torch.float64),
                       torch.tensor(ju, dtype=torch.float64),
                       pe_lo, pe_up, tuple(need), pt)


# ---------------------------------------------------------------------------
# from data
# ---------------------------------------------------------------------------
def pac_sets(
    model,
    data: Mapping[str, torch.Tensor],
    nodes: Sequence[str] | None = None,
    delta: float = 0.05,
    *,
    kind: str = "interval",
    method: str = "kl",
    sampling: str = "iid",
) -> CredalSets:
    """PAC sets for ``nodes`` (default: every node) that hold **jointly** with
    probability ``>= 1 - delta``.

    The budget is split over all entries of all tables (``kind="interval"``:
    ``delta * entries_i / entries_total`` per table, each split again over its
    entries by :func:`interval_set_from_counts`) or over all rows
    (``kind="l1"``), a union bound fixed before the data are seen (it depends
    only on the DAG and the cardinalities).  Every query answered with
    :func:`credal_query` from these sets then holds simultaneously.
    """
    from nbn.uncertainty.builders import interval_set_from_counts, l1_ball_set_from_counts
    from nbn.uncertainty.counts import count_table

    nodes = list(nodes) if nodes is not None else list(model.dag.topological_order())
    tables = {n: count_table(model, data, n)[0] for n in nodes}
    if kind == "interval":
        sizes = {n: t.numel() for n, t in tables.items()}
    elif kind == "l1":
        sizes = {n: t.shape[0] for n, t in tables.items()}
    else:
        raise ValueError("kind must be 'interval' or 'l1'.")
    total = sum(sizes.values())
    out = {}
    for n, t in tables.items():
        d = delta * sizes[n] / total
        if kind == "interval":
            out[n] = interval_set_from_counts(t, d, method=method, sampling=sampling)
        else:
            out[n] = l1_ball_set_from_counts(t, d, sampling=sampling)
    return CredalSets(out, level=1.0 - delta, guarantee="pac_under_model", delta=delta)


def pac_query(
    model,
    data: Mapping[str, torch.Tensor],
    target: str,
    evidence: Mapping[str, object] | None = None,
    *,
    do: Mapping[str, object] | None = None,
    delta: float = 0.05,
    kind: str = "interval",
    method: str = "kl",
    sampling: str = "iid",
    max_elements: int = 2 ** 26,
    **query_kwargs,
) -> QueryBounds:
    """PAC answer from data: with probability ``>= 1 - delta`` over the data,
    ``lower <= P_true(target | evidence, do) <= upper``.

    Sets are built only for the nodes this query reads (its ancestral set),
    so the budget is not spent on the rest; for many queries, build the sets
    once with :func:`pac_sets` and call :func:`credal_query` (all answers then
    hold together).  Assumes: discrete variables, the DAG of ``model`` is the
    true one (or a supergraph), and the declared ``sampling``.
    """
    evidence = dict(evidence or {})
    do = dict(do or {})

    def parents_fn(n):
        return [] if n in do else list(model.dag.parents(n))

    need = [n for n in model.dag.topological_order()
            if n in _ancestral(parents_fn, [target, *evidence, *do]) and n not in do]
    sets = pac_sets(model, data, need, delta, kind=kind, method=method, sampling=sampling)
    return credal_query(model, sets, target, evidence, do=do, max_elements=max_elements,
                        **query_kwargs)


__all__ = ["QueryBounds", "CredalSets", "credal_query", "pac_sets", "pac_query"]
