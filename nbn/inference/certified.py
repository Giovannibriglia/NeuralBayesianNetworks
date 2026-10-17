"""Certified sampling answers for discrete queries: bounds with a declared level.

Exact inference is exponential in the tree-width [Koller & Friedman 2009,
§9.4, Thm 9.6 and Def 9.6, pp. 309–310], so on dense graphs only sampling
scales.  Plain (self-normalised) likelihood weighting comes with no
sample-size guarantee [K&F, §12.2.3.4, p. 504]; its effective sample size is a
heuristic.  :func:`certified_query` returns instead an interval for every
class of a discrete target that holds **simultaneously, with probability at
least 1 - delta**, under assumptions that are checked, not assumed.

Construction (ours, from standard pieces; audited against K&F and AfV)
---------------------------------------------------------------------
Run unnormalised likelihood weighting [K&F, Alg 12.2, p. 493] a fixed number
``M`` of times (``M`` is chosen before sampling; a data-dependent stopping
rule would need a different argument, cf. K&F Alg 12.3 / Thm 12.1, p. 502).
Particle ``m`` carries the weight ``w_m = prod_i P(e_i | pa_i)`` over the
evidence nodes and a sampled target value ``Y_m``, so ``E[w] = P(e)`` and
``E[w 1{Y=y}] = P(y, e)`` [K&F, Eqs. 12.11–12.12, pp. 496–497].  Every weight
lies in ``[0, W]`` with ``W = prod_i max_pa P(e_i | pa)`` (the largest CPD
entry of each evidence node *at its observed value*; it follows from the
weight update of Alg 12.2), so ``w/W`` and ``w 1{Y=y}/W`` are i.i.d. in
``[0, 1]``.  For i.i.d. variables in ``[0, 1]`` the Chernoff–Hoeffding bound
``P(mean >= q) <= exp(-M kl(q || mu))`` (``q > mu``, and symmetrically below)
holds [Hoeffding 1963, JASA 58(301), Thm 1], so inverting
``M kl(hat || q) <= log(2/delta')`` gives a ``1 - delta'`` interval for each of
the ``K + 1`` means (the KL-UCB inversion, Garivier & Cappé, COLT 2011,
https://arxiv.org/abs/1102.2490; the same code as
:func:`nbn.uncertainty.binomial_interval`).  A union bound with
``delta' = delta / (K + 1)`` makes all of them hold together (no independence
between the statistics is needed), and on that event, for every class ``y``::

    P(y | e) = P(y, e) / P(e)  in  [ lo_ye / hi_e ,  min(1, hi_ye / lo_e) ]

(all quantities non-negative; if ``lo_e = 0`` the upper ends are 1).  By
default each mean's interval is the intersection of that KL interval and an
empirical Bernstein interval [Maurer & Pontil, COLT 2009, Thm 4], each at
half of ``delta'``; both are valid, so their intersection is, and it is about
half as wide when the weights vary little.  Since
the ``P(y | e)`` sum to one, each interval is then tightened with
``1 - sum_{y' != y} upper_{y'}`` and ``1 - sum_{y' != y} lower_{y'}``, still on
the same event.  K&F note that normalised LW has "no formal analysis" of its
sample size (p. 504); this finite-sample statement is what this module adds.

The sample size needed for a given width grows like ``W / P(e)``, as in the
classical relative-error bounds [K&F, Eqs. 12.17–12.18, p. 501; Dagum &
Luby 1997].  Nothing depends on the tree-width.  The guarantee is about the
*model's* posterior: it says nothing about model error.

Scope: discrete target; evidence on discrete table mechanisms (their CPD is
available to compute ``W``); ``do`` nodes are clamped and carry no weight.
Continuous evidence has an unbounded density, so it is rejected.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping

import torch

from nbn.inference.base import InferenceEngine
from nbn.inference.state import get_inference_state
from nbn.uncertainty.binomial import binomial_interval


@dataclass(frozen=True)
class CertifiedAnswer:
    """Interval answer for ``P(target | evidence, do)``.

    ``lower[y] <= P(y | e) <= upper[y]`` for every class ``y`` simultaneously,
    with probability at least ``level`` over the sampling.  ``estimate`` is
    the self-normalised point estimate (inside the intervals).

    Anytime runs (``time_budget_s`` / ``target_width``) also report how many
    checkpoints were passed and the wall time; ``n_samples`` is then the
    sample size of the last checkpoint, the one the interval is certified at.
    """

    estimate: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    level: float
    n_samples: int
    weight_bound: float
    evidence_prob_lower: float
    evidence_prob_upper: float
    n_checkpoints: int = 1
    seconds: float | None = None
    collapsed: tuple = ()

    @property
    def max_width(self) -> float:
        return float((self.upper - self.lower).max())


def evidence_weight_bound(model, evidence: Mapping[str, object]) -> float:
    """``W = prod_i max_pa P(E_i = e_i | pa)``: an upper bound on every LW weight."""
    u = 1.0
    for node, val in evidence.items():
        mech = model.mechanisms[node]
        if not getattr(mech, "is_discrete", False):
            raise NotImplementedError(
                f"certified_query: evidence node '{node}' is continuous; its density "
                f"is not bounded, so no weight bound exists.  Discretise it or use "
                f"plain likelihood weighting (no guarantee)."
            )
        parents = model.dag.parents(node)
        cards = [model.mechanisms[p].n_classes for p in parents]
        probs = torch.softmax(mech.tabulate(cards).detach().cpu().double(), -1)  # MPS has no float64
        v = int(torch.as_tensor(val).reshape(-1)[0])
        u *= float(probs[..., v].max())
    return u


def _bounds_from_sums(sum_w, sum_wy, sq_w, sq_wy, done: int, delta: float, k: int,
                      bound: str):
    """Simultaneous ``1 - delta`` bounds on ``P(y | e)`` from the running sums of
    ``K + 1`` i.i.d. ``[0, 1]`` statistics (see the module docstring).

    Returns ``(lower, upper, lo_e, hi_e)``; ``lo_e, hi_e`` bound ``P(e) / W``.
    """
    m = torch.tensor(float(done), dtype=torch.float64)
    succ = torch.cat([sum_w.reshape(1), sum_wy])
    d_each = delta / (k + 1)
    if bound == "kl":
        lo, hi = binomial_interval(succ, m.expand(k + 1), d_each, method="kl")
    else:
        lo, hi = binomial_interval(succ, m.expand(k + 1), d_each / 2, method="kl")
        sq = torch.cat([sq_w.reshape(1), sq_wy])
        mean = succ / m
        var = ((sq - succ * succ / m) / (m - 1)).clamp_min(0.0) if done > 1 else torch.ones_like(mean)
        d_side = d_each / 4                      # two sides, half the per-mean budget
        rad = torch.sqrt(2 * var * math.log(2 / d_side) / m) \
            + 7 * math.log(2 / d_side) / (3 * max(done - 1, 1))
        lo = torch.maximum(lo, (mean - rad).clamp(0.0, 1.0))
        hi = torch.minimum(hi, (mean + rad).clamp(0.0, 1.0))
    lo_e, hi_e = float(lo[0]), float(hi[0])
    lo_y, hi_y = lo[1:].cpu(), hi[1:].cpu()
    lower = (lo_y / hi_e).clamp(0.0, 1.0) if hi_e > 0 else torch.zeros(k, dtype=torch.float64)
    upper = (hi_y / lo_e).clamp(0.0, 1.0) if lo_e > 0 else torch.ones(k, dtype=torch.float64)
    # The posterior sums to one: tighten each end with the others' ends.
    lower, upper = _sum_to_one(lower, upper)
    return lower, upper, lo_e, hi_e


def _sum_to_one(lower, upper):
    lower = torch.maximum(lower, (1.0 - (upper.sum() - upper)).clamp(min=0.0))
    upper = torch.minimum(upper, (1.0 - (lower.sum() - lower)).clamp(max=1.0))
    return lower, upper


@torch.no_grad()
def certified_query(
    model,
    target: str,
    evidence: Mapping[str, object] | None = None,
    *,
    n_samples: int = 200_000,
    delta: float = 0.05,
    do: Mapping[str, object] | None = None,
    chunk: int | None = None,
    bound: str = "kl+bernstein",
    time_budget_s: float | None = None,
    target_width: float | None = None,
    max_samples: int | None = None,
    first_checkpoint: int = 4096,
    collapse=None,
    memory_budget: float | None = None,
) -> CertifiedAnswer:
    """Interval answer to ``P(target | evidence, do(...))`` at level ``1 - delta``.

    See the module docstring for the construction and its guarantee.
    ``chunk`` bounds the particles per pass (default: the likelihood-weighting
    engine's memory-aware pass size, about 256 MiB of particle buffer).
    ``bound``: ``"kl"`` (Chernoff–Hoeffding KL interval per mean) or
    ``"kl+bernstein"`` (default): the KL interval and the empirical
    Bernstein interval, each at half the level, intersected.  Empirical
    Bernstein [Maurer & Pontil, COLT 2009, Thm 4, arXiv:0907.3740]: for
    i.i.d. ``Z`` in ``[0, 1]`` with unbiased sample variance ``V``, with
    probability ``>= 1 - d``, ``E Z - mean <= sqrt(2 V ln(2/d) / M) +
    7 ln(2/d) / (3 (M - 1))``; applied to ``1 - Z`` for the lower side.  It is
    much tighter when the weights vary little.

    **Anytime mode** (``time_budget_s`` and/or ``target_width`` given; the
    fixed ``n_samples`` is then not used).  Sampling continues until the
    certified interval is at most ``target_width`` wide (every class), the
    time is up, or ``max_samples`` would be exceeded.  Stopping on the data is
    made valid by checkpoints fixed in advance: ``n_j = first_checkpoint *
    2^(j-1)`` with level ``delta_j = delta / (j (j + 1))`` (``sum_j delta_j =
    delta``), so by a union bound *all* checkpoint intervals hold at once with
    probability ``>= 1 - delta``, whatever the stopping rule [the
    "stitching" argument of Howard, Ramdas, McAuliffe & Sekhon, Ann. Statist.
    49(2), 2021, arXiv:1810.08240, in its simplest form].  The answer is the
    intersection of the checkpoint intervals (valid on the same event), so a
    larger budget never gives a wider interval.  Samples drawn after the last
    completed checkpoint enter only the point estimate.

    **Collapsed mode** (``collapse=`` a list of variables, or ``"auto"``):
    Rao–Blackwellised ("cutset") sampling [K&F, §12.4, Eqs. 12.30–12.32,
    pp. 527–530; Bidyuk & Dechter, JAIR 28, 2007].  Only the ancestral
    closure ``X_p`` of the given variables is sampled (forward, with the
    evidence inside it clamped and weighted, as in LW); for each particle the
    rest of the query is computed *exactly* by variable elimination:
    ``P(y, e_d | x_p, e_p)``.  ``"auto"`` picks the smallest greedy cutset
    whose exact remainder fits ``memory_budget`` (``nbn.inference.
    conditioning``).  Each particle's statistics ``w_p P(y, e_d | x_p, e_p)``
    stay in ``[0, W]`` (``P(e_d | x_p, e_p) <= prod_{e in E_d} max_pa
    P(e | pa)``, the LW identity [K&F, Eq. 12.11]), so the same certificate
    applies.  Collapsing removes the sampling variance of the exact part; it
    is not guaranteed to lower the variance of an importance-weighted
    estimate (the weights change), so its benefit is measured, not assumed.
    Needs a discrete model.
    """
    if bound not in ("kl", "kl+bernstein"):
        raise ValueError("bound must be 'kl' or 'kl+bernstein'.")
    anytime = time_budget_s is not None or target_width is not None
    if anytime or collapse is not None:
        return _certified_general(
            model, target, evidence, delta=delta, do=do, chunk=chunk, bound=bound,
            n_samples=n_samples, anytime=anytime, time_budget_s=time_budget_s,
            target_width=target_width, max_samples=max_samples,
            first_checkpoint=first_checkpoint, collapse=collapse,
            memory_budget=memory_budget)
    from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine

    evidence = dict(evidence or {})
    do = dict(do or {})
    if target in evidence or target in do:
        raise ValueError("certified_query: the target is observed or intervened.")
    mech_t = model.mechanisms[target]
    if not getattr(mech_t, "is_discrete", False):
        raise NotImplementedError("certified_query needs a discrete target.")
    k = int(mech_t.n_classes)
    W = evidence_weight_bound(model, evidence) if evidence else 1.0
    if W <= 0.0:
        raise ValueError("certified_query: the evidence has probability 0 under every "
                         "parent configuration.")

    eng = LikelihoodWeightingEngine(n_samples=1, prune_barren=True)
    if chunk is None:
        from nbn.inference.state import get_inference_state as _gis
        st = _gis(model, [target], tuple(sorted(evidence)), tuple(sorted(do)), eng._cache)
        chunk = eng._particle_chunk(1, st.total_dim)

    def _t(v):
        t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        return t.reshape(1).to(model.device)

    ev = {kk: _t(v) for kk, v in evidence.items()}
    dd = {kk: _t(v) for kk, v in do.items()}
    sum_w = torch.zeros((), dtype=torch.float64)
    sum_wy = torch.zeros(k, dtype=torch.float64)
    sq_w = torch.zeros((), dtype=torch.float64)
    sq_wy = torch.zeros(k, dtype=torch.float64)
    done = 0
    while done < n_samples:
        s = min(chunk, n_samples - done)
        log_w, buf = eng._run(model, [target], ev, dd, s, prune=True)
        state = get_inference_state(model, [target], tuple(sorted(ev)), tuple(sorted(dd)),
                                    eng._cache)
        sl = state.node_slices[state.node_to_idx[target]]
        y = buf[0, :, sl].reshape(-1).long().cpu()
        w = torch.exp(log_w[0].cpu().double()) / W        # in [0, 1]; CPU: MPS has no float64
        w = w.clamp(0.0, 1.0)
        sum_w += w.sum()
        sum_wy += torch.bincount(y, weights=w, minlength=k)[:k]
        sq_w += (w * w).sum()
        sq_wy += torch.bincount(y, weights=w * w, minlength=k)[:k]
        done += s
    lower, upper, lo_e, hi_e = _bounds_from_sums(sum_w, sum_wy, sq_w, sq_wy, done, delta,
                                                 k, bound)
    est = sum_wy / sum_w if float(sum_w) > 0 else torch.full((k,), float("nan"),
                                                               dtype=torch.float64)
    return CertifiedAnswer(
        estimate=est, lower=lower, upper=upper, level=1.0 - delta, n_samples=done,
        weight_bound=W, evidence_prob_lower=lo_e * W, evidence_prob_upper=hi_e * W,
    )


# ---------------------------------------------------------------------- #
# Anytime and collapsed sampling
# ---------------------------------------------------------------------- #

class _PlainDraw:
    """Likelihood-weighting particles: ``w / W`` and ``w 1{Y = y} / W``."""

    def __init__(self, model, target, evidence, do, k, W, chunk):
        from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine

        self.model, self.target, self.k, self.W = model, target, k, W
        self.eng = LikelihoodWeightingEngine(n_samples=1, prune_barren=True)

        def _t(v):
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            return t.reshape(1).to(model.device)

        self.ev = {kk: _t(v) for kk, v in evidence.items()}
        self.dd = {kk: _t(v) for kk, v in do.items()}
        if chunk is None:
            st = get_inference_state(model, [target], tuple(sorted(evidence)),
                                     tuple(sorted(do)), self.eng._cache)
            chunk = self.eng._particle_chunk(1, st.total_dim)
        self.chunk = int(chunk)
        self.collapsed: tuple = ()

    def __call__(self, s):
        log_w, buf = self.eng._run(self.model, [self.target], self.ev, self.dd, s, prune=True)
        state = get_inference_state(self.model, [self.target], tuple(sorted(self.ev)),
                                    tuple(sorted(self.dd)), self.eng._cache)
        sl = state.node_slices[state.node_to_idx[self.target]]
        y = buf[0, :, sl].reshape(-1).long().cpu()
        w = (torch.exp(log_w[0].cpu().double()) / self.W).clamp(0.0, 1.0)
        wy = torch.zeros(s, self.k, dtype=torch.float64)
        wy[torch.arange(s), y] = w
        return w, wy


class _CollapsedDraw:
    """Cutset particles: sample the ancestral closure ``X_p`` of a cutset, the
    rest exactly by batched elimination (see ``certified_query``)."""

    def __init__(self, model, target, ev_int, do_int, k, W, chunk, collapse,
                 memory_budget):
        from nbn.inference.conditioning import (
            _ancestral, _eliminate_rows, _mutilated, plan_conditioning)
        from nbn.inference.tensor_ve import TensorVariableElimination, _require_discrete
        from nbn.inference._ve_memory import _factor_dtype_bytes, _max_fitting_rows

        _require_discrete(model)
        self._elim = _eliminate_rows
        self.model, self.target, self.k, self.W = model, target, k, W
        self.ev, self.do = ev_int, do_int
        device = model.device
        self.device = device
        ve = TensorVariableElimination(memory_budget=memory_budget)
        budget = memory_budget if memory_budget is not None else ve.memory_budget_bytes(device)
        graph = _mutilated(model, tuple(do_int))
        A = {n for n in _ancestral(graph, [target, *ev_int]) if n not in do_int}
        dtype_bytes = _factor_dtype_bytes(model)
        common = dict(budget=budget, device=device, dtype_bytes=dtype_bytes,
                      max_instantiations=float("inf"))
        clamped = set(ev_int) | set(do_int)
        if collapse == "auto":
            C = plan_conditioning(model, A, targets=(target,), clamped=clamped, **common).cutset
        else:
            C = list(collapse)
            unknown = [c for c in C if c not in model.mechanisms]
            if unknown:
                raise ValueError(f"collapse: unknown variables {unknown}.")
        Xp = {n for n in _ancestral(graph, C) if n not in do_int} if C else set()
        self.Xp = [n for n in model.dag.topological_order() if n in Xp]
        self.collapsed = tuple(sorted(C))
        inner = A - Xp
        keep = (target,) if target not in Xp else ()
        plan = plan_conditioning(model, inner, targets=keep,
                                 clamped=clamped | Xp, **{**common, "max_instantiations": 1})
        self.order, self.keep = plan.order, (keep[0] if keep else None)
        factors = ve._extract_factors(model)
        self.factors = factors
        self.inner = {n: factors[n] for n in plan.scopes}
        dev = torch.device(device).type
        lim = int(chunk) if chunk is not None else 65536
        if budget is not None and self.inner:
            lim = max(1, _max_fitting_rows(lambda c: plan.cost.peak(c, dtype_bytes, dev),
                                           lim, budget))
        self.chunk = lim

    def __call__(self, s):
        dev = self.device
        # do-nodes are outside X_p but may be parents of its nodes
        vals = {kk: torch.full((s,), v, dtype=torch.long, device=dev)
                for kk, v in self.do.items()}
        log_w1 = torch.zeros(s, device=dev)
        for node in self.Xp:
            f = self.factors[node]
            pa_idx = tuple(vals[p] for p in f.variables[:-1])
            logp = f.log_values[pa_idx] if pa_idx else f.log_values.expand(s, -1)
            if node in self.do:
                vals[node] = torch.full((s,), self.do[node], dtype=torch.long, device=dev)
            elif node in self.ev:
                v = self.ev[node]
                vals[node] = torch.full((s,), v, dtype=torch.long, device=dev)
                log_w1 = log_w1 + logp[:, v]
            else:
                vals[node] = torch.multinomial(logp.float().exp(), 1).reshape(-1)
        clamp = {kk: torch.full((s,), v, dtype=torch.long, device=dev)
                 for kk, v in {**self.ev, **self.do}.items()}
        clamp.update(vals)
        if self.inner:
            lv = self._elim(self.inner, clamp, self.order, self.keep, s)
        else:
            lv = torch.zeros(s, device=dev)
        if self.keep is not None:
            stat = torch.exp((log_w1[:, None] + lv).cpu().double()) / self.W   # [s, k]
            stat = stat.clamp(0.0, 1.0)
            w = stat.sum(1).clamp(0.0, 1.0)
        else:
            w = (torch.exp((log_w1 + lv).cpu().double()) / self.W).clamp(0.0, 1.0)
            stat = torch.zeros(s, self.k, dtype=torch.float64)
            stat[torch.arange(s), vals[self.target].cpu()] = w
        return w, stat


def _certified_general(model, target, evidence, *, delta, do, chunk, bound, n_samples,
                       anytime, time_budget_s, target_width, max_samples,
                       first_checkpoint, collapse, memory_budget) -> CertifiedAnswer:
    t0 = time.perf_counter()
    deadline = None if time_budget_s is None else t0 + float(time_budget_s)
    evidence = dict(evidence or {})
    do = dict(do or {})
    if target in evidence or target in do:
        raise ValueError("certified_query: the target is observed or intervened.")
    mech_t = model.mechanisms[target]
    if not getattr(mech_t, "is_discrete", False):
        raise NotImplementedError("certified_query needs a discrete target.")
    k = int(mech_t.n_classes)
    W = evidence_weight_bound(model, evidence) if evidence else 1.0
    if W <= 0.0:
        raise ValueError("certified_query: the evidence has probability 0 under every "
                         "parent configuration.")
    if collapse is None:
        draw = _PlainDraw(model, target, evidence, do, k, W, chunk)
    else:
        def _int(v):
            return int(torch.as_tensor(v).reshape(-1)[0])
        draw = _CollapsedDraw(model, target, {a: _int(b) for a, b in evidence.items()},
                              {a: _int(b) for a, b in do.items()}, k, W, chunk, collapse,
                              memory_budget)
    sums = [torch.zeros((), dtype=torch.float64), torch.zeros(k, dtype=torch.float64),
            torch.zeros((), dtype=torch.float64), torch.zeros(k, dtype=torch.float64)]
    done = 0

    def take(n):
        nonlocal done
        while done < n:
            s = min(draw.chunk, n - done)
            w, wy = draw(s)
            sums[0] += w.sum()
            sums[1] += wy.sum(0)
            sums[2] += (w * w).sum()
            sums[3] += (wy * wy).sum(0)
            done += s
            if deadline is not None and time.perf_counter() >= deadline and done < n:
                return False
        return True

    if not anytime:   # collapsed, fixed sample size
        take(int(n_samples))
        lower, upper, lo_e, hi_e = _bounds_from_sums(*sums, done, delta, k, bound)
        est = sums[1] / sums[0] if float(sums[0]) > 0 else torch.full((k,), float("nan"),
                                                                      dtype=torch.float64)
        return CertifiedAnswer(est, lower, upper, 1.0 - delta, done, W, lo_e * W, hi_e * W,
                               1, time.perf_counter() - t0, draw.collapsed)
    lower = torch.zeros(k, dtype=torch.float64)
    upper = torch.ones(k, dtype=torch.float64)
    lo_e_best, hi_e_best = 0.0, 1.0
    j, n_cert = 0, 0
    n_next = max(2, int(first_checkpoint))
    while True:
        if max_samples is not None and n_next > int(max_samples):
            break
        if not take(n_next):
            break                       # time ran out before this checkpoint
        j += 1
        lo_j, up_j, lo_e, hi_e = _bounds_from_sums(*sums, done, delta / (j * (j + 1)), k,
                                                   bound)
        lower, upper = _sum_to_one(torch.maximum(lower, lo_j), torch.minimum(upper, up_j))
        lo_e_best, hi_e_best = max(lo_e_best, lo_e), min(hi_e_best, hi_e)
        n_cert = done
        if target_width is not None and float((upper - lower).max()) <= target_width:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break
        n_next *= 2
    est = sums[1] / sums[0] if float(sums[0]) > 0 else torch.full((k,), float("nan"),
                                                                  dtype=torch.float64)
    return CertifiedAnswer(est, lower, upper, 1.0 - delta, n_cert, W, lo_e_best * W,
                           hi_e_best * W, j, time.perf_counter() - t0, draw.collapsed)


class CertifiedEngine(InferenceEngine):
    """Engine wrapper: answers a discrete query with ``certified_query``.

    ``query`` returns the point estimate (``[K]`` probabilities, the same
    self-normalised estimate plain likelihood weighting gives) and keeps the
    interval in ``last_certificate``; ``query_batch`` loops over rows and keeps
    ``last_certificates``.  Used by ``HybridRouter(fallback="certified")``.

    ``time_budget_s`` / ``target_width`` / ``max_samples`` select the anytime
    mode and ``collapse`` the collapsed mode of ``certified_query`` (per
    query; all ``None`` keeps the fixed ``n_samples`` run).
    """

    def __init__(self, n_samples: int = 200_000, delta: float = 0.05,
                 bound: str = "kl+bernstein", *, time_budget_s: float | None = None,
                 target_width: float | None = None, max_samples: int | None = None,
                 collapse=None, memory_budget: float | None = None) -> None:
        super().__init__()
        self.n_samples = int(n_samples)
        self.delta = float(delta)
        self.bound = bound
        self.time_budget_s = time_budget_s
        self.target_width = target_width
        self.max_samples = max_samples
        self.collapse = collapse
        self.memory_budget = memory_budget
        self.last_certificate: CertifiedAnswer | None = None
        self.last_certificates: list[CertifiedAnswer] = []

    @staticmethod
    def applicable(model, targets, evidence, do=None) -> bool:
        """One discrete target, discrete evidence, at most one row per value."""
        if targets is None or len(targets) != 1:
            return False
        if not getattr(model.mechanisms[targets[0]], "is_discrete", False):
            return False
        for node, v in (evidence or {}).items():
            if not getattr(model.mechanisms[node], "is_discrete", False):
                return False
        return True

    def _one(self, model, target, evidence, do):
        extra = {}
        if self.time_budget_s is not None:
            extra["time_budget_s"] = self.time_budget_s
        if self.target_width is not None:
            extra["target_width"] = self.target_width
        if self.max_samples is not None:
            extra["max_samples"] = self.max_samples
        if self.collapse is not None:
            extra["collapse"] = self.collapse
            extra["memory_budget"] = self.memory_budget
        a = certified_query(model, target, evidence, n_samples=self.n_samples,
                            delta=self.delta, do=do, bound=self.bound, **extra)
        return a

    def query(self, model, targets, evidence=None, do=None, **kwargs):
        a = self._one(model, targets[0], evidence or {}, do)
        self.last_certificate = a
        self.last_certificates = [a]
        return a.estimate.to(device=model.device, dtype=torch.float32)

    def query_batch(self, model, targets, evidence, do=None, **kwargs):
        evidence = evidence or {}
        do = do or {}
        rows = max([1] + [torch.as_tensor(v).reshape(-1).shape[0]
                          for v in list(evidence.values()) + list(do.values())])

        def row(d, i):
            out = {}
            for k_, v in d.items():
                t = torch.as_tensor(v).reshape(-1)
                out[k_] = t[i if t.shape[0] > 1 else 0]
            return out

        answers = [self._one(model, targets[0], row(evidence, i), row(do, i) or None)
                   for i in range(rows)]
        self.last_certificates = answers
        self.last_certificate = answers[-1]
        return torch.stack([a.estimate for a in answers]).to(device=model.device,
                                                              dtype=torch.float32)


__all__: list[str] = ["CertifiedAnswer", "CertifiedEngine", "certified_query",
                      "evidence_weight_bound"]
