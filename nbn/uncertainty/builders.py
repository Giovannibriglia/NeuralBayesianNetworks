"""Build probability sets for the table of a discrete node.

Each builder reads counts from data (see :func:`count_table`), so it does not
depend on how the node's mechanism was fitted.  ``sampling`` declares how the
data were collected:

``"iid"``
    Independent rows.  Given the parent values, the child values of a parent
    configuration are i.i.d., so an interval at the observed count is valid.
``"trajectory"``
    Rows are transitions of episodes.  The number of visits of a parent
    configuration depends on earlier draws, so every interval is made valid
    for all sample sizes at once (:func:`anytime_delta`).

Factored next states
--------------------
For an MDP whose next state has several variables, each variable's table is
one marginal ``P(s'_i | pa_i)`` and the kernel is their product.  Build one set
per variable and split the global budget between them (union bound): with
``n`` next-state variables, call each builder with ``delta / n``.  This is the
factored construction of Schnitzer, Abate & Parker, *Efficient Solution and
Learning of Robust Factored MDPs* (AAAI 2026, arXiv:2508.00707, §4.1), which
also pools counts per parent configuration ("dependency identifier"); solving
over the product of the per-variable sets is their §3 (exact vertex products,
McCormick relaxation) and happens downstream, not in NBN.
"""
from __future__ import annotations

import copy
import math
from typing import Callable, Dict, Literal

import torch
from torch.distributions import Dirichlet

from nbn.uncertainty.binomial import (
    IntervalMethod,
    anytime_delta,
    binomial_interval,
    union_bound,
)
from nbn.uncertainty.counts import count_table
from nbn.uncertainty.sets import IntervalSet, L1BallSet, SampleSet

Sampling = Literal["iid", "trajectory"]


def interval_set(
    model,
    data: Dict[str, torch.Tensor],
    node: str,
    delta: float = 0.05,
    method: IntervalMethod = "kl",
    sampling: Sampling = "iid",
) -> IntervalSet:
    """Entry-wise intervals for ``P(node | parents)``, valid for all entries at once.

    The budget ``delta`` is split over all ``R * K`` entries (union bound);
    the result is then tightened to the simplex.
    """
    counts, _ = count_table(model, data, node)
    trials = counts.sum(dim=-1, keepdim=True).expand_as(counts)
    per_entry = per_interval_delta(delta, counts.numel(), trials, sampling)
    lower, upper = binomial_interval(counts, trials, per_entry, method=method)
    return IntervalSet(
        level=1.0 - delta, guarantee="pac", unseen=trials[:, 0] == 0,
        lower=lower, upper=upper,
    )


def l1_ball_set(
    model,
    data: Dict[str, torch.Tensor],
    node: str,
    delta: float = 0.05,
    sampling: Sampling = "iid",
) -> L1BallSet:
    """One L1 ball per row, valid for all rows at once.

    Radius from the bound ``P(||p_hat - p||_1 >= eps) <= (2^K - 2) exp(-n eps^2 / 2)``
    on the empirical distribution of ``n`` i.i.d. draws over ``K`` values
    (Weissman, Ordentlich, Seroussi, Verdú, Weinberger, 2003, Thm 2.1 —
    HP Labs report HPL-2003-97, original not checked; the same radius,
    ``sqrt(2 (ln(2^a - 2) - ln delta) / n)``, is used by Schnitzer et al.,
    AAAI 2026, §4.1), solved for ``eps``.  Rows without data get radius
    2, i.e. the whole simplex.
    """
    counts, _ = count_table(model, data, node)
    trials = counts.sum(dim=-1)
    n_rows, k = counts.shape
    per_row = per_interval_delta(delta, n_rows, trials, sampling)
    if k == 1:  # a single value has probability 1: nothing to estimate
        radius = torch.zeros_like(trials)
    else:
        log_patterns = k * math.log(2.0) + math.log1p(-(2.0 ** (1 - k)))  # log(2^K - 2)
        radius = torch.sqrt(2.0 * (log_patterns + torch.log(1.0 / per_row)) / trials.clamp_min(1))
    radius = torch.where(trials == 0, torch.full_like(radius, 2.0), radius.clamp_max(2.0))
    center = counts / trials.clamp_min(1).unsqueeze(-1)
    center = torch.where(trials.unsqueeze(-1) == 0, torch.full_like(center, 1.0 / k), center)
    return L1BallSet(
        level=1.0 - delta, guarantee="pac", unseen=trials == 0,
        center=center, radius=radius,
    )


def dirichlet_set(
    model,
    data: Dict[str, torch.Tensor],
    node: str,
    delta: float = 0.05,
    prior: float = 1.0,
    n_samples: int = 4000,
) -> SampleSet:
    """Posterior draws of the table under independent ``Dirichlet(prior)`` rows.

    Bayesian credibility, not a frequentist guarantee; useful as genuine
    posterior weights and as a sanity check of the PAC sets.  Seed with
    ``torch.manual_seed`` for reproducible draws.
    """
    counts, _ = count_table(model, data, node)
    # MPS has no float64 Dirichlet sampler; float32 draws are plenty here.
    posterior = Dirichlet((counts + prior).to(torch.float32))
    return SampleSet(
        level=1.0 - delta, guarantee="bayesian", unseen=counts.sum(-1) == 0,
        samples=posterior.sample((n_samples,)),
    )


def bootstrap_set(
    model,
    data: Dict[str, torch.Tensor],
    statistic: Callable[[object], torch.Tensor],
    groups: torch.Tensor,
    n_boot: int = 200,
    delta: float = 0.05,
    generator: torch.Generator | None = None,
) -> SampleSet:
    """Episode (cluster) bootstrap of any statistic of a fitted model.

    Works for discrete and continuous models alike.  Each replicate refits a
    copy of ``model`` with per-row weights equal to the number of times the
    row's group (episode) was drawn, which is the same as refitting on the
    resampled data (NBN's weights have replication semantics).  Every
    mechanism of the model must support weights.

    Parameters
    ----------
    statistic:
        Maps a fitted model to a tensor (e.g. a probability table, a query).
    groups:
        ``[N]`` integer group id of each data row (episode id).  Use
        ``torch.arange(N)`` for i.i.d. rows.
    """
    groups = groups.reshape(-1).long()
    group_ids, inverse = torch.unique(groups, return_inverse=True)
    n_groups = group_ids.numel()
    draws = []
    for _ in range(n_boot):
        picks = torch.randint(n_groups, (n_groups,), generator=generator)
        multiplicity = torch.bincount(picks, minlength=n_groups).to(torch.float32)
        weights = multiplicity.to(inverse.device)[inverse]
        replicate = copy.deepcopy(model)
        replicate.fit(data, weights=weights, consolidate=False)
        with torch.no_grad():
            draws.append(statistic(replicate).detach())
    samples = torch.stack(draws)
    return SampleSet(
        level=1.0 - delta, guarantee="asymptotic",
        unseen=torch.zeros(0, dtype=torch.bool), samples=samples,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def per_interval_delta(
    delta: float, n_intervals: int, trials: torch.Tensor, sampling: Sampling
) -> torch.Tensor:
    """Level of each of ``n_intervals`` intervals so that all hold jointly at ``1 - delta``."""
    base = union_bound(delta, n_intervals)
    if sampling == "iid":
        return torch.full_like(trials, base, dtype=trials.dtype)
    if sampling == "trajectory":
        return anytime_delta(base, trials)
    raise ValueError(f"Unknown sampling '{sampling}'; use 'iid' or 'trajectory'.")
