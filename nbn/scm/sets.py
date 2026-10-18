"""Probability sets that use the structural form of the mechanisms.

Shared noise couples many probabilities through few parameters, so these sets
are much tighter than per-entry intervals — at the price of a model
assumption, hence the guarantee ``"pac_under_model"``.  Check the assumption
with :func:`nbn.scm.noise_sharing_test` / :func:`noise_tying_test`, and fit the
locations on other data than the sets when you can.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from nbn.scm.fitting import counts_by_location
from nbn.scm.mechanisms import AdditiveNoiseMechanism, DiscreteAdditiveNoiseMechanism
from nbn.uncertainty.binomial import IntervalMethod, binomial_interval
from nbn.uncertainty.builders import Sampling, per_interval_delta
from nbn.uncertainty.counts import count_table
from nbn.uncertainty.sets import CDFBandSet, NoiseParameterSet
from nbn.utils.device import to_device


@torch.no_grad()
def noise_parameter_set(
    model,
    data: Mapping[str, torch.Tensor],
    nodes: Sequence[str],
    delta: float = 0.05,
    method: IntervalMethod = "kl",
    sampling: Sampling = "iid",
) -> NoiseParameterSet:
    """Confidence polytope for the offset law shared by ``nodes``.

    Every entry ``P(x = k | r)`` of every member is a sum of noise
    probabilities, ``sum_{m in S} q_m``, where ``S`` is the set of offsets that
    land on ``k`` from the location ``g_r``.  Entries with the same ``S`` — in
    any row, in any tied node — are one *expression*: all their draws are
    Bernoulli trials with the same success probability, so their counts pool
    into one binomial.  One interval per distinct expression, with a union
    bound over all expressions the model can produce, gives the polytope.

    This is the construction of Schnitzer, Abate & Parker (2026) — intervals
    per distinct parametric expression with a union bound — applied to the
    noise law of a structural model.  The expressions depend on the fitted
    locations, so the guarantee holds when those are right
    (``"pac_under_model"``); fit them on other data when you can.
    """
    data = to_device(dict(data), model.device)
    mechs = [model.mechanisms[n] for n in nodes]
    if not all(isinstance(m, DiscreteAdditiveNoiseMechanism) and m.is_fitted for m in mechs):
        raise TypeError("noise_parameter_set needs fitted DiscreteAdditiveNoiseMechanism nodes.")
    if any(m.noise is not mechs[0].noise for m in mechs):
        raise ValueError(f"{list(nodes)} do not share one noise law (tie them first).")

    # Pool each member by (location, value): rows with one location share every expression.
    patterns, successes, trials, used, unseen = [], [], [], [], {}
    for node, mech in zip(nodes, mechs):
        counts, _ = count_table(model, data, node)
        mech.check_reachable(counts)
        g, k = mech.n_locations, mech.n_classes
        by_location = counts_by_location(counts, mech.locations, g)        # [G, K]
        seen = counts.sum(-1) > 0
        location_used = torch.zeros(g, dtype=torch.bool, device=counts.device)
        location_used[mech.locations[seen]] = True
        used.append(location_used.unsqueeze(-1).expand(g, k).reshape(-1))
        patterns.append(mech.transfer().reshape(g * k, -1))                # [G * K, M]
        successes.append(by_location.reshape(-1))
        trials.append(by_location.sum(-1, keepdim=True).expand(g, k).reshape(-1))
        unseen[node] = counts.sum(-1) == 0

    expressions, inverse = torch.unique(torch.cat(patterns), dim=0, return_inverse=True)
    pooled_k = torch.zeros(expressions.shape[0], dtype=successes[0].dtype, device=inverse.device)
    pooled_n = torch.zeros_like(pooled_k)
    pooled_k.scatter_add_(0, inverse, torch.cat(successes))
    pooled_n.scatter_add_(0, inverse, torch.cat(trials))

    # Budget.  i.i.d. rows: every non-empty expression of a location used by
    # some observed row; given the parent values (hence the locations used)
    # this set is fixed, so the union bound is valid under the model, and the
    # expressions of unused locations get [0, 1] without spending budget.
    # Trajectories: which rows are visited depends on earlier child draws, so
    # a data-dependent count would void the union bound; every non-empty
    # expression the model can produce is budgeted (fixed in advance).  An
    # empty expression (a value unreachable from its location) is exactly 0.
    empty = expressions.sum(-1) == 0
    if sampling == "trajectory":
        n_budgeted = max(int((~empty).sum()), 1)
    else:
        in_use = torch.zeros(expressions.shape[0], dtype=torch.bool, device=inverse.device)
        in_use[inverse[torch.cat(used)]] = True
        n_budgeted = max(int((in_use & ~empty).sum()), 1)
    per_expr = per_interval_delta(delta, n_budgeted, pooled_n, sampling)
    lower, upper = binomial_interval(pooled_k, pooled_n, per_expr, method=method)
    lower = torch.where(empty, torch.zeros_like(lower), lower)
    upper = torch.where(empty, torch.zeros_like(upper), upper)

    index, start = {}, 0
    for node, mech in zip(nodes, mechs):
        g, k = mech.n_locations, mech.n_classes
        by_location = inverse[start:start + g * k].reshape(g, k)
        index[node] = by_location[mech.locations]                           # [R, K]
        start += g * k
    return NoiseParameterSet(
        expressions=expressions.to(lower.dtype), lower=lower, upper=upper, index=index,
        level=1.0 - delta, unseen=unseen,
    )


@torch.no_grad()
def cdf_band_set(
    model, data: Mapping[str, torch.Tensor], node: str, delta: float = 0.05, *,
    fit_ids: torch.Tensor | None = None, ids: torch.Tensor | None = None,
) -> CDFBandSet:
    """DKW band on the noise CDF of a 1-D continuous additive node.

    ``eps = sqrt(log(2 / delta) / (2 n))`` (Dvoretzky–Kiefer–Wolfowitz with
    Massart's constant).  Valid for i.i.d. residuals given the location ``g``:
    pass ``data`` that were *not* used to fit ``g`` (a held-out split), or the
    guarantee is only approximate.  ``fit_ids`` / ``ids``: optional row ids
    of the fit rows and of ``data``; a shared id raises
    (:func:`nbn.scm.calibration.check_held_out`).  The band records the
    mechanism it was built from (``bound_to``); ``CDFBandSet.check`` or the
    ``mechanism=`` argument of its bound methods refuse another one.
    """
    if not 0.0 < float(delta) < 1.0:
        raise ValueError(f"delta must lie in (0, 1); got {delta}.")
    from nbn.scm.calibration import _fingerprint, check_held_out
    check_held_out(fit_ids, ids, None, int(data[node].shape[0]))
    data = to_device(dict(data), model.device)
    mech = model.mechanisms[node]
    if not isinstance(mech, AdditiveNoiseMechanism) or not mech.is_fitted:
        raise TypeError("cdf_band_set needs a fitted AdditiveNoiseMechanism.")
    residual = mech.residual(data[node], model.parent_values(data, node))
    if residual.shape[1] != 1:
        raise ValueError("cdf_band_set supports 1-D nodes only.")
    n = residual.shape[0]
    if n == 0:
        raise ValueError("cdf_band_set needs at least one data row.")
    return CDFBandSet(
        sorted_residuals=residual[:, 0].sort().values,
        epsilon=math.sqrt(math.log(2.0 / delta) / (2.0 * n)),
        level=1.0 - delta,
        bound_to=_fingerprint(mech),
    )
