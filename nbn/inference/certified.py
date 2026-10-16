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

(all quantities non-negative; if ``lo_e = 0`` the upper ends are 1).  Since
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

from dataclasses import dataclass
from typing import Mapping

import torch

from nbn.inference.state import get_inference_state
from nbn.uncertainty.binomial import binomial_interval


@dataclass(frozen=True)
class CertifiedAnswer:
    """Interval answer for ``P(target | evidence, do)``.

    ``lower[y] <= P(y | e) <= upper[y]`` for every class ``y`` simultaneously,
    with probability at least ``level`` over the sampling.  ``estimate`` is
    the self-normalised point estimate (inside the intervals).
    """

    estimate: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    level: float
    n_samples: int
    weight_bound: float
    evidence_prob_lower: float
    evidence_prob_upper: float

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
        probs = torch.softmax(mech.tabulate(cards).detach().double(), -1)
        v = int(torch.as_tensor(val).reshape(-1)[0])
        u *= float(probs[..., v].max())
    return u


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
) -> CertifiedAnswer:
    """Interval answer to ``P(target | evidence, do(...))`` at level ``1 - delta``.

    See the module docstring for the construction and its guarantee.
    ``chunk`` bounds the particles per pass (default: the likelihood-weighting
    engine's memory-aware pass size, about 256 MiB of particle buffer).
    """
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
    done = 0
    while done < n_samples:
        s = min(chunk, n_samples - done)
        log_w, buf = eng._run(model, [target], ev, dd, s, prune=True)
        state = get_inference_state(model, [target], tuple(sorted(ev)), tuple(sorted(dd)),
                                    eng._cache)
        sl = state.node_slices[state.node_to_idx[target]]
        y = buf[0, :, sl].reshape(-1).long().cpu()
        w = torch.exp(log_w[0].double()).cpu() / W        # in [0, 1] up to rounding
        w = w.clamp(0.0, 1.0)
        sum_w += w.sum()
        sum_wy += torch.bincount(y, weights=w, minlength=k)[:k]
        done += s
    m = torch.tensor(float(done), dtype=torch.float64)
    succ = torch.cat([sum_w.reshape(1), sum_wy])
    lo, hi = binomial_interval(succ, m.expand(k + 1), delta / (k + 1), method="kl")
    lo_e, hi_e = float(lo[0]), float(hi[0])
    lo_y, hi_y = lo[1:].cpu(), hi[1:].cpu()
    lower = (lo_y / hi_e).clamp(0.0, 1.0) if hi_e > 0 else torch.zeros(k, dtype=torch.float64)
    upper = (hi_y / lo_e).clamp(0.0, 1.0) if lo_e > 0 else torch.ones(k, dtype=torch.float64)
    # The posterior sums to one: tighten each end with the others' ends.
    lower = torch.maximum(lower, (1.0 - (upper.sum() - upper)).clamp(min=0.0))
    upper = torch.minimum(upper, (1.0 - (lower.sum() - lower)).clamp(max=1.0))
    est = sum_wy / sum_w if float(sum_w) > 0 else torch.full((k,), float("nan"),
                                                               dtype=torch.float64)
    return CertifiedAnswer(
        estimate=est, lower=lower, upper=upper, level=1.0 - delta, n_samples=done,
        weight_bound=W, evidence_prob_lower=lo_e * W, evidence_prob_upper=hi_e * W,
    )


__all__: list[str] = ["CertifiedAnswer", "certified_query", "evidence_weight_bound"]
