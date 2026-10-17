"""Exact inference under a memory cap by conditioning, and bounded conditioning.

Why
---
Variable elimination costs about ``n K^(w+1)`` in time *and* in space, where
``w`` is the width of the elimination order [Koller & Friedman 2009, §9.4,
Thm 9.6, pp. 306–310].  When the largest intermediate factor does not fit the
memory budget, more memory buys only ``log_K`` of its size in width, and more
time buys nothing.  *Conditioning* trades time for space: instantiate a set
``C`` of variables, run VE once per instantiation ``c`` on the reduced
network, and sum [K&F, §9.5, Alg 9.5 and Eq. 9.11, p. 317]:

    P(y, e) = sum_c P(y, c, e).

With ``C`` chosen so that the reduced plan fits the budget, the time grows to
about ``n K^(|C| + w_C + 1)`` and the space stays about ``n K^(w_C + 1)``
[K&F, Thms 9.12–9.13, pp. 322–323].  Conditioning never does fewer operations
than plain VE [K&F, Thm 9.11, p. 320], so it is used only when plain VE is
refused.  The instantiations are independent: here they are the *rows* of one
batched elimination (the same kernels as ``TensorVariableElimination.
query_batch``), so threads, a GPU or a larger memory budget (more rows per
pass) all shorten the run.

Two modes
---------
* **Exact** (:func:`conditioned_query` without ``time_budget_s``): every
  instantiation is enumerated; the answer equals VE's.  The factors are the
  ones VE itself would use (Bayes-ball relevant CPTs), conditioned on ``C``.
* **Bounded conditioning** (``time_budget_s`` or ``enumerate_limit``): if the budget runs out
  before every instantiation is done, the answer is a pair of *sound*,
  deterministic bounds [K&F, §12.5, pp. 536–540; Horvitz, Suermondt & Cooper,
  UAI 1989, arXiv:1304.1512].  With the enumerated set ``S``,
  ``A_y = sum_{c in S} P(y, c, e)``, ``A = sum_y A_y`` and any upper bound
  ``U`` on the unexplored part ``sum_{c not in S} P(c, e)``:

      A_y / (A + U)  <=  P(y | e)  <=  (A_y + U) / (A + U).

  (K&F Eq. 12.36 states the upper end as ``(A_y + U) / A``; the form here is
  tighter and still sound: ``P(y|e) = (A_y + r_y) / (A + r)`` with
  ``0 <= r_y <= r <= U``, which is largest at ``r_y = r = U`` because
  ``A_y <= A``.)  Two bounds on the unexplored part are used, and the smaller
  one is kept:

  * ``U_prior = 1 - sum_{c in S} P(c)`` (the unexplored prior mass, since
    ``P(c, e) <= P(c)``; K&F §12.5);
  * ``U_lw = W (1 - sum_{c in S} Q(c))``, where ``Q`` is the
    likelihood-weighting proposal (evidence clamped, K&F Def. 12.1, p. 499)
    and ``W = prod_i max_pa P(e_i | pa)``: ``P(c, e) = Q(c) E_Q[w | c] <=
    W Q(c)`` because every LW weight is at most ``W`` (K&F Alg. 12.2, p. 493;
    p. 500).  This bound is ours (checked with the koller_claude expert, not
    stated in the book).  It is much smaller than ``U_prior`` when the
    evidence is unlikely (``W`` small) and no evidence is upstream of ``C``;
    with evidence upstream of ``C`` it can be larger, hence the minimum.

  Both ends move towards each other as ``S`` grows, and they meet when ``S``
  is complete.  For exact joint probabilities the factors are the CPTs of the
  *ancestral* set of the target and the evidence (sums over it are
  probabilities); ``P(c)`` and ``Q(c)`` come from two more eliminations over
  the ancestors of ``C``.  Instantiations are visited in decreasing order of
  their frequency in a forward sample of ``C`` (prior order).  K&F suggest
  visiting the instantiations likely *given e* first (p. 537); measured here
  (small dense networks, 6 seeds x 3 budgets) that order was sometimes wider,
  because it lowers the unexplored *prior* mass more slowly, while the prior
  order with ``min(U_prior, U_lw)`` was never wider than ``U_prior`` alone.
  ``_prior_order(..., ev_int=...)`` keeps the likelihood-weighted order
  available.  The bounds are useful only when a few instantiations carry most
  of the mass [K&F, Ex. 12.21, p. 539; pp. 540–541].

Choosing ``C`` (a greedy "w-cutset", cf. Bidyuk & Dechter, JAIR 28, 2007):
repeatedly add the variable of the plan's largest intermediate factor with the
most neighbours in the conditioned moral graph, until the plan's priced peak
(the same allocation model as VE's guard, ``nbn.inference._ve_memory``) fits
the budget; on MPS also until every intermediate tensor keeps at most 16
dimensions and ``2**27`` elements.

The result is about the model, not about the data: no statistical guarantee
is implied.  Numerics: probabilities are accumulated in log space in the
factors' dtype; the bounded mode widens the bounds by a relative forward
error bound (at least ``bound_slack``) so that rounding cannot make them
unsound.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping

import networkx as nx
import torch

from nbn.inference._elimination_order import _eliminate_inplace, _moral_adjacency, get_order
from nbn.inference._memory import OutOfMemoryError
from nbn.inference._validation import check_discrete_values
from nbn.inference._ve_memory import _factor_dtype_bytes, _max_fitting_rows, _plan_cost, _Scope

# Apple MPS: tensors of at most 16 dimensions; reductions wrong beyond 2**27
# elements (see ``tensor_ve._MPS_MAX_ELEMENTS``).
_MPS_MAX_DIMS = 16
_MPS_MAX_ELEMENTS = 2 ** 27
# Rows per pass are capped so that a time budget is checked often enough.
_FIRST_ROWS = 64


class ConditioningLimitError(OutOfMemoryError):
    """Conditioning would need more instantiations than allowed (or cannot fit).

    A subclass of the out-of-memory error, so ``HybridRouter`` treats it like
    a refused VE plan and answers with sampling instead.
    """


@dataclass(frozen=True)
class ConditionedAnswer:
    """Answer of :func:`conditioned_query`.

    ``exact``: every instantiation was enumerated and ``lower == upper ==
    estimate`` is the exact posterior.  Otherwise ``lower <= P(y | e) <=
    upper`` holds for every class (deterministically, not with a probability)
    and ``estimate`` is the posterior restricted to the enumerated part.
    """

    estimate: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    exact: bool
    cutset: tuple
    n_instantiations: int
    n_enumerated: int
    prior_mass_enumerated: float | None
    seconds: float

    @property
    def max_width(self) -> float:
        return float((self.upper - self.lower).max())


@dataclass
class _Plan:
    cutset: list
    order: list
    scopes: dict
    cost: object  # _PlanCost
    targets: tuple


# ---------------------------------------------------------------------- #
# Planning
# ---------------------------------------------------------------------- #

def _mutilated(model, do_keys) -> nx.DiGraph:
    g = model.dag.networkx_graph
    if do_keys:
        g = g.copy()
        g.remove_edges_from([(u, v) for (u, v) in list(g.in_edges(list(do_keys)))])
    return g


def _ancestral(graph: nx.DiGraph, nodes) -> set:
    out = set(nodes)
    for v in nodes:
        out |= nx.ancestors(graph, v)
    return out


def _scopes(model, factor_nodes) -> dict:
    """``node -> _Scope`` in the engine's (topological) factor order."""
    out = {}
    for node in model.dag.topological_order():
        if node in factor_nodes:
            var_scope = list(model.dag.parents(node)) + [node]
            out[node] = _Scope(var_scope, {n: int(model.mechanisms[n].n_classes)
                                           for n in var_scope})
    return out


def _cliques(adj: dict, order, cards) -> tuple[int, list]:
    """(most variables in one product, ``(elements, scope)`` of every product,
    largest first)."""
    g = {v: set(s) for v, s in adj.items()}
    most, found = 0, []
    for v in order:
        if v not in g:
            continue
        clique = [v, *sorted(g[v])]
        most = max(most, len(clique))
        found.append((math.prod(cards[u] for u in clique), clique))
        _eliminate_inplace(g, v)
    found.sort(key=lambda t: -t[0])
    return most, found


def plan_conditioning(
    model,
    factor_nodes,
    *,
    targets: tuple,
    clamped,
    budget: float | None,
    device,
    dtype_bytes: int,
    max_instantiations: int,
    cutset=(),
    order: str = "min_fill",
    forbid=(),
    rows: int = 1,
) -> _Plan:
    """Smallest greedy cutset ``C`` (containing ``cutset``) whose plan fits.

    ``factor_nodes``: nodes whose CPT is a factor.  ``clamped``: evidence and
    do-nodes (never eliminated, never in ``C``).  ``targets``: kept variables
    (``()`` to sum everything out).  ``forbid``: never put these in ``C``.
    ``rows``: the plan must fit with this many instantiations eliminated in
    one batch (collapsed sampling asks for a whole batch of particles).
    """
    do_graph = model.dag.networkx_graph
    scope_nodes = set(factor_nodes)
    for v in factor_nodes:
        scope_nodes.update(do_graph.predecessors(v))
    sub = do_graph.subgraph(scope_nodes)
    cards = {v: int(model.mechanisms[v].n_classes) for v in scope_nodes}
    scopes = _scopes(model, factor_nodes)
    dev = torch.device(device).type
    max_dims = _MPS_MAX_DIMS if dev == "mps" else None
    elem_cap = _MPS_MAX_ELEMENTS if dev == "mps" else None
    C = list(cutset)
    base_adj = _moral_adjacency(sub)
    if budget is not None:
        # Floor: every free variable clamped.  Below it (allocator granularity,
        # the answer's own buffers) no cutset helps: refuse at once.
        free = sorted(v for v in scope_nodes
                      if v not in targets and v not in clamped and v not in forbid)
        floor_clamp = tuple(sorted(set(clamped) | set(free)))
        floor = _plan_cost(list(free), scopes, evidence_keys=floor_clamp,
                           targets=tuple(targets),
                           target_cards={t: cards[t] for t in targets},
                           joint=None if len(targets) != 1 else False)
        if floor.peak(rows, dtype_bytes, dev) > budget:
            raise ConditioningLimitError(
                f"conditioning cannot fit {budget / 2 ** 20:.3f} MiB: even with every "
                f"variable clamped the plan needs "
                f"{floor.peak(rows, dtype_bytes, dev) / 2 ** 20:.3f} MiB.")
    while True:
        clamp = set(clamped) | set(C)
        plan_order = get_order(order, sub, targets=list(targets), evidence=sorted(clamp),
                               cards=cards)
        cost = _plan_cost(plan_order, scopes, evidence_keys=tuple(sorted(clamp)),
                          targets=tuple(targets),
                          target_cards={t: cards[t] for t in targets},
                          joint=None if len(targets) != 1 else False)
        adj = {v: set(s) for v, s in base_adj.items()}
        for e in clamp:
            for u in adj.pop(e, ()):
                adj[u].discard(e)
        most, cliques = _cliques(adj, plan_order, cards)
        ok = budget is None or cost.peak(rows, dtype_bytes, dev) <= budget
        if elem_cap is not None:
            ok = ok and max(cost.tensor) * rows <= elem_cap
        if max_dims is not None:
            ok = ok and most + 1 <= max_dims     # + the batch (instantiation) axis
        if ok:
            return _Plan(C, plan_order, scopes, cost, tuple(targets))
        # Greedy w-cutset step: the free variable shared by the most (and
        # largest) products of the plan, weighted by their sizes; ties to the
        # smaller cardinality, then the name (deterministic).
        blocked = set(targets) | clamp | set(forbid)
        score: dict = {}
        for elems, cl in cliques:
            for v in cl:
                if v not in blocked:
                    score[v] = score.get(v, 0.0) + elems / cards[v]
        if not score:
            raise ConditioningLimitError(
                "conditioning cannot shrink this plan further: every variable "
                "left is a target or clamped, and the plan still does not fit.")
        pick = max(score, key=lambda v: (score[v], -cards[v], v))
        C.append(pick)
        if math.prod(cards[c] for c in C) > max_instantiations:
            raise ConditioningLimitError(
                f"conditioning would need more than max_instantiations="
                f"{max_instantiations} instantiations (cutset {C}); refused.")


# ---------------------------------------------------------------------- #
# Execution
# ---------------------------------------------------------------------- #

def _eliminate_rows(factors: dict, clamp: dict, order, keep: str | None, rows: int):
    """Unnormalised log result of eliminating ``order`` over ``factors``.

    ``clamp``: ``{var: [rows] long tensor}``.  Returns ``[rows, K_keep]``
    (``keep`` a variable) or ``[rows]`` (``keep=None``: total log mass).  The
    same batched kernels as ``TensorVariableElimination.query_batch``.
    """
    from nbn.inference.tensor_ve import _condition_factor_batched, _multiply_batched

    conditioned = [_condition_factor_batched(f.log_values, f.variables, clamp)
                   for f in factors.values()]
    for var in order:
        rel = [f for f in conditioned if var in f[1]]
        if not rel:
            continue
        conditioned = [f for f in conditioned if var not in f[1]]
        prod_lv, prod_vars, prod_has_b = _multiply_batched(rel)
        dim = prod_vars.index(var) + (1 if prod_has_b else 0)
        conditioned.append((torch.logsumexp(prod_lv, dim=dim),
                            [v for v in prod_vars if v != var], prod_has_b))
        del prod_lv
    if not conditioned:
        raise RuntimeError("no factors left after elimination")
    lv, vars_, has_b = _multiply_batched(conditioned)
    off = 1 if has_b else 0
    for i in range(len(vars_) - 1, -1, -1):
        if vars_[i] != keep:
            lv = torch.logsumexp(lv, dim=i + off)
            vars_.pop(i)
    if keep is not None and keep not in vars_:
        raise RuntimeError(f"variable {keep!r} vanished during elimination")
    if not has_b:
        lv = lv.unsqueeze(0).expand(rows, *lv.shape)
    return lv


def _digits(idx: torch.Tensor, cards: list) -> list:
    """Mixed-radix digits of instantiation indices (first variable slowest)."""
    out = []
    rest = idx
    for c in reversed(cards):
        out.append(rest % c)
        rest = rest // c
    return list(reversed(out))


def _prior_order(model, factors, graph, cutset, cards, do_int, n: int, seed: int,
                 ev_int=None):
    """Instantiation indices of ``C`` by decreasing likelihood-weighted
    frequency (an estimate of ``P(c, e)``) in a forward sample with the
    evidence clamped; with no evidence, plain frequency (``P(c)``)."""
    ev_int = dict(ev_int or {})
    anc = _ancestral(graph, list(cutset) + list(ev_int))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    vals = {}
    logw = torch.zeros(n, dtype=torch.float64)
    for node in model.dag.topological_order():
        if node not in anc:
            continue
        if node in do_int:
            vals[node] = torch.full((n,), int(do_int[node]), dtype=torch.long)
            continue
        f = factors[node]
        probs = f.log_values.detach().float().cpu().exp()
        idx = tuple(vals[p] for p in f.variables[:-1])
        p = probs[idx] if idx else probs.expand(n, -1)
        if node in ev_int:
            v = int(ev_int[node])
            vals[node] = torch.full((n,), v, dtype=torch.long)
            logw = logw + torch.log(p[:, v].double().clamp_min(1e-300))
        else:
            vals[node] = torch.multinomial(p, 1, generator=gen).reshape(-1)
    code = torch.zeros(n, dtype=torch.long)
    for c, k in zip(cutset, cards):
        code = code * k + vals[c]
    uniq, inv = torch.unique(code, return_inverse=True)
    score = torch.zeros(len(uniq), dtype=torch.float64).index_add_(
        0, inv, torch.exp(logw - logw.max()))
    return uniq[torch.argsort(score, descending=True, stable=True)]


@torch.no_grad()
def conditioned_query(
    model,
    target: str,
    evidence: Mapping[str, object] | None = None,
    *,
    do: Mapping[str, object] | None = None,
    memory_budget: float | None = None,
    time_budget_s: float | None = None,
    enumerate_limit: int | None = None,
    max_seconds: float | None = None,
    max_instantiations: int = 1_000_000,
    max_rows: int | None = None,
    order: str = "min_fill",
    engine=None,
    bound_slack: float = 1e-6,
    seed: int = 0,
) -> ConditionedAnswer:
    """``P(target | evidence, do)`` by conditioning, within a memory budget.

    ``memory_budget``: bytes for one pass (default: the engine's budget for
    the model's device, see ``TensorVariableElimination.memory_budget_bytes``).
    ``time_budget_s``: if given, the bounded-conditioning mode (see the module
    docstring); without it every instantiation is enumerated and the answer is
    exact.  ``enumerate_limit``: stop after this many instantiations (a
    deterministic budget; also selects the bounded mode).  ``max_seconds``
    (exact mode only): give up when this time is spent; the answer is then
    ``exact=False`` with the trivial bounds ``[0, 1]`` and ``n_enumerated``
    reports the progress (the remaining time is about proportional to the
    remaining instantiations).
    ``max_instantiations``: refuse (``ConditioningLimitError``) a
    cutset with more instantiations.  ``max_rows``: cap on instantiations per
    pass (default: as many as fit the budget).
    """
    from nbn.inference.tensor_ve import TensorVariableElimination, _require_discrete

    t0 = time.perf_counter()
    deadline = None if time_budget_s is None else t0 + float(time_budget_s)
    give_up = None if max_seconds is None else t0 + float(max_seconds)
    eng = engine if engine is not None else TensorVariableElimination(memory_budget=memory_budget)
    evidence = dict(evidence or {})
    do = dict(do or {})
    eng._validate_do(model, do, evidence)
    _require_discrete(model)
    check_discrete_values(model, {**evidence, **do}, do_keys=do)
    if target in evidence or target in do:
        raise ValueError("conditioned_query: the target is observed or intervened.")
    device = model.device

    def _int(v):
        return int(torch.as_tensor(v).reshape(-1)[0])

    ev_int = {k: _int(v) for k, v in evidence.items()}
    do_int = {k: _int(v) for k, v in do.items()}
    budget = memory_budget if memory_budget is not None else eng.memory_budget_bytes(device)
    k = int(model.mechanisms[target].n_classes)
    dtype_bytes = _factor_dtype_bytes(model)
    graph = _mutilated(model, tuple(do_int))
    clamped = set(ev_int) | set(do_int)
    bounded = time_budget_s is not None or enumerate_limit is not None

    if not bounded:
        _, relevant = eng._plan_with_relevant(model, target, tuple(sorted(ev_int)),
                                              do_keys=tuple(sorted(do_int)), order=order)
        y_nodes = {n for n in relevant if n not in do_int}
    else:
        y_nodes = {n for n in _ancestral(graph, [target, *ev_int]) if n not in do_int}
    common = dict(clamped=clamped, budget=budget, device=device, dtype_bytes=dtype_bytes,
                  max_instantiations=max_instantiations, order=order)
    plan_y = plan_conditioning(model, y_nodes, targets=(target,), **common)
    plan_c = None
    if bounded and plan_y.cutset:
        # P(c) needs its own elimination (over the ancestors of C); both plans
        # must fit, and they share one cutset.
        for _ in range(8):
            c_nodes = {n for n in _ancestral(graph, plan_y.cutset) if n not in do_int}
            plan_c = plan_conditioning(model, c_nodes, targets=(),
                                       cutset=plan_y.cutset,
                                       forbid={target, *ev_int},
                                       **{**common, "clamped": set(do_int)})
            if plan_c.cutset == plan_y.cutset:
                break
            plan_y = plan_conditioning(model, y_nodes, targets=(target,),
                                       cutset=plan_c.cutset, **common)
        else:  # pragma: no cover - the cutset only grows and is bounded
            raise ConditioningLimitError("conditioning plans did not settle")
    plan_q = None
    if plan_c is not None and ev_int:
        # Q(c): the ancestors of C with the evidence clamped and its factors
        # dropped (the LW proposal); used only if it fits with the same cutset
        q_nodes = {n for n in _ancestral(graph, plan_y.cutset)
                   if n not in do_int and n not in ev_int}
        try:
            pq = plan_conditioning(model, q_nodes, targets=(), cutset=plan_y.cutset,
                                   forbid={target, *ev_int},
                                   **{**common, "clamped": set(do_int) | set(ev_int)})
            plan_q = pq if pq.cutset == plan_y.cutset else None
        except ConditioningLimitError:
            plan_q = None
    C = list(plan_y.cutset)
    cards = [int(model.mechanisms[c].n_classes) for c in C]
    n_total = math.prod(cards) if C else 1

    all_factors = eng._extract_factors(model)
    f_y = {n: all_factors[n] for n in plan_y.scopes}
    f_c = {n: all_factors[n] for n in plan_c.scopes} if plan_c is not None else None
    f_q = ({n: all_factors[n] for n in plan_q.scopes if n not in ev_int}
           if plan_q is not None else None)
    dev_type = torch.device(device).type

    def fit_rows(plan):
        lim = n_total if max_rows is None else min(n_total, int(max_rows))
        if budget is None:
            r = lim
        else:
            r = _max_fitting_rows(lambda c: plan.cost.peak(c, dtype_bytes, dev_type), lim, budget)
        if dev_type == "mps":
            per_row = max(1, plan.cost.tensor[1])
            r = min(r, max(1, (_MPS_MAX_ELEMENTS - plan.cost.tensor[0]) // per_row))
        return max(1, r)

    rows_cap = fit_rows(plan_y)
    if plan_c is not None:
        rows_cap = min(rows_cap, fit_rows(plan_c))
    if plan_q is not None:
        rows_cap = min(rows_cap, fit_rows(plan_q))

    if bounded and C:
        first = _prior_order(model, all_factors, graph, C, cards, do_int,
                             n=min(16384, 16 * n_total), seed=seed, ev_int=None)
        seen = torch.zeros(n_total, dtype=torch.bool)
        seen[first] = True
        visit = torch.cat([first, torch.nonzero(~seen).reshape(-1)])
    else:
        visit = None

    # Accumulators in float64 on the CPU (MPS has no float64); each pass is
    # reduced on the device first, so only [k] numbers cross per pass.
    log_acc = torch.full((k,), -math.inf, dtype=torch.float64)
    log_mass = torch.tensor(-math.inf, dtype=torch.float64)
    log_qmass = torch.tensor(-math.inf, dtype=torch.float64)
    done = 0
    rows = min(rows_cap, _FIRST_ROWS) if (bounded or give_up is not None) else rows_cap
    stop = n_total if enumerate_limit is None else max(1, min(n_total, int(enumerate_limit)))
    while done < stop:
        r = min(rows, stop - done)
        idx = torch.arange(done, done + r) if visit is None else visit[done:done + r]
        digits = [d.to(device) for d in _digits(idx, cards)]
        clamp = {kk: torch.full((r,), v, dtype=torch.long, device=device)
                 for kk, v in {**ev_int, **do_int}.items()}
        for c, d in zip(C, digits):
            clamp[c] = d
        lv = _eliminate_rows(f_y, clamp, plan_y.order, target, r)          # [r, k]
        log_acc = torch.logaddexp(log_acc, torch.logsumexp(lv, dim=0).cpu().double())
        if f_c is not None:
            cc = {kk: torch.full((r,), v, dtype=torch.long, device=device)
                  for kk, v in do_int.items()}
            for c, d in zip(C, digits):
                cc[c] = d
            lm = _eliminate_rows(f_c, cc, plan_c.order, None, r)          # [r]
            log_mass = torch.logaddexp(log_mass, torch.logsumexp(lm, dim=0).cpu().double())
        if f_q is not None:
            cq = {kk: torch.full((r,), v, dtype=torch.long, device=device)
                  for kk, v in {**ev_int, **do_int}.items()}
            for c, d in zip(C, digits):
                cq[c] = d
            lq = _eliminate_rows(f_q, cq, plan_q.order, None, r)          # [r]
            log_qmass = torch.logaddexp(log_qmass, torch.logsumexp(lq, dim=0).cpu().double())
        done += r
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if give_up is not None and done < stop and time.perf_counter() >= give_up:
            ones = torch.ones(k, dtype=torch.float64)
            return ConditionedAnswer(torch.full((k,), float("nan"), dtype=torch.float64),
                                     torch.zeros(k, dtype=torch.float64), ones, False,
                                     tuple(C), n_total, done, None,
                                     time.perf_counter() - t0)
        rows = min(rows_cap, 2 * rows)

    exact = done >= n_total
    a_y = torch.exp(log_acc)
    if exact or not bounded:
        est = torch.softmax(log_acc, dim=0)
        mass = 1.0 if (bounded and C) else None
        return ConditionedAnswer(est, est.clone(), est.clone(), True, tuple(C), n_total,
                                 done, mass, time.perf_counter() - t0)
    mass_s = float(torch.exp(log_mass)) if f_c is not None else 1.0
    a = float(a_y.sum())
    u = max(0.0, 1.0 - mass_s)
    if f_q is not None:
        from nbn.inference.certified import evidence_weight_bound
        w_bound = evidence_weight_bound(model, ev_int)
        u = min(u, w_bound * max(0.0, 1.0 - float(torch.exp(log_qmass))))
    # float slack, relative to the joint mass (an absolute slack made the
    # bounds vacuous when P(e) is tiny): forward error bound of the
    # elimination, eta = 4 u (F + sum_f max |log f|), at least bound_slack
    unit = torch.finfo(next(iter(f_y.values())).log_values.dtype).eps if f_y else 2.0 ** -52
    mx = sum(float(f.log_values[torch.isfinite(f.log_values)].abs().max())
             if torch.isfinite(f.log_values).any() else 0.0 for f in f_y.values())
    eta = max(bound_slack, 4.0 * unit * (len(f_y) + mx))
    u = u + eta * (a + u) + 1e-300
    a_y = a_y * (1.0 - eta)
    a = a * (1.0 - eta)
    lower = (a_y / (a + u)).clamp(0.0, 1.0)
    upper = ((a_y + u) / (a + u)).clamp(0.0, 1.0)
    lower = torch.maximum(lower, (1.0 - (upper.sum() - upper)).clamp(min=0.0))
    upper = torch.minimum(upper, (1.0 - (lower.sum() - lower)).clamp(max=1.0))
    est = a_y / a if a > 0 else torch.full((k,), float("nan"), dtype=torch.float64)
    return ConditionedAnswer(est, lower, upper, False, tuple(C), n_total, done, mass_s,
                             time.perf_counter() - t0)


__all__ = ["ConditionedAnswer", "ConditioningLimitError", "conditioned_query",
           "plan_conditioning"]
