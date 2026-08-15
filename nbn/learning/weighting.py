"""Per-sample weights: validation and the one reduction convention.

Weights are **multiplicities**, not a probability distribution — they need not
sum to 1, and ``w = [2, 1]`` means "the first row counts twice".  The
acceptance criterion throughout is *replication equivalence*: fitting with
integer weights ``w`` on ``D`` must equal fitting on ``D`` with each row
repeated ``w_i`` times.

Reduction
---------
The gradient-trained mechanisms reduce their per-row loss as a **weighted
mean**::

    L = sum_i w_i * nll_i / sum_i w_i

rather than a weighted sum.  This makes the effective step size invariant to
the weights' overall magnitude, which matters for the motivating use: an EM
M-step whose responsibilities happen to sum to 12 for one latent stratum and
0.8 for another must not take steps differing by 15x.  It also reduces
*exactly* to ``nll.mean()`` when every weight is 1, which is what keeps the
unweighted default byte-identical.

The closed-form families are indifferent to this choice — ``sum w`` cancels in
both the normal equations and the count normalisation — so the convention only
binds the gradient path.

Precision
---------
Weights arriving from an EM E-step are per-group responsibilities broadcast to
many rows: long runs of repeated values, some very small.  Summing thousands
of those in float32 loses digits exactly where the normaliser needs them, so
accumulation happens in float64 and casts back at the end.
"""
from __future__ import annotations

import torch

#: Guards ``0/0`` if a caller slips an all-zero vector past validation.
_MIN_TOTAL_WEIGHT = 1e-12


def validate_weights(
    weights: torch.Tensor | None,
    n_rows: int,
    *,
    where: str,
) -> torch.Tensor | None:
    """Return ``weights`` as a validated 1-D float64 tensor, or ``None``.

    Parameters
    ----------
    weights:
        ``None`` (unweighted) or a tensor broadcastable to ``[N]``.
    n_rows:
        Number of data rows the weights must align with.
    where:
        Caller name, quoted in error messages.
    """
    if weights is None:
        return None
    if not isinstance(weights, torch.Tensor):
        weights = torch.as_tensor(weights)
    w = weights.reshape(-1)
    if w.numel() != n_rows:
        raise ValueError(
            f"{where}: weights has {w.numel()} entries but the data has "
            f"{n_rows} rows; weights are per-sample and must align with them."
        )
    w = w.to(torch.float64)
    if not torch.isfinite(w).all():
        raise ValueError(f"{where}: weights contains NaN or infinite values.")
    if (w < 0).any():
        raise ValueError(
            f"{where}: weights must be non-negative (they are multiplicities, "
            f"not log-weights); found a minimum of {float(w.min())}."
        )
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError(
            f"{where}: weights sum to {total}; at least one row must carry "
            f"weight, otherwise there is nothing to fit."
        )
    return w


def weighted_mean(
    values: torch.Tensor, weights: torch.Tensor | None
) -> torch.Tensor:
    """``sum(w*v) / sum(w)``, or ``values.mean()`` when ``weights`` is None.

    Differentiable in ``values``.  Accumulated in float64 and cast back to
    ``values.dtype``, so a long run of small weights does not lose the
    normaliser's precision; the cast keeps the loss (and therefore the
    gradients) in the dtype the caller's parameters use.
    """
    if weights is None:
        return values.mean()
    v = values.reshape(-1)
    w = weights.reshape(-1).to(device=v.device, dtype=torch.float64)
    num = (w * v.to(torch.float64)).sum()
    den = w.sum().clamp_min(_MIN_TOTAL_WEIGHT)
    return (num / den).to(values.dtype)


def weighted_moments(
    t: torch.Tensor, weights: torch.Tensor | None, *, unbiased: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column ``(mean, std)`` of ``[N, D]``, weighted if ``weights`` given.

    Data-derived standardisation statistics have to be weighted along with
    everything else, or replication equivalence fails in a way that is easy to
    miss: the fit converges, but to a model standardised against a data
    distribution the caller did not ask for.  Accumulated in float64.

    ``unbiased`` follows the *frequency-weight* convention — the denominator
    is ``sum(w) - 1`` rather than ``n - 1`` — because that is what replicating
    a row ``w_i`` times produces.  Passing the population form (``sum(w)``)
    where the unweighted call site used ``n - 1`` is a 2%-scale error on small
    samples: big enough to move a standardised model, small enough to read as
    numerical noise.
    """
    if weights is None:
        return t.mean(0), t.std(0, unbiased=unbiased)
    td = t.to(torch.float64)
    wd = weights.reshape(-1, 1).to(device=t.device, dtype=torch.float64)
    tot = wd.sum().clamp_min(_MIN_TOTAL_WEIGHT)
    mean = (wd * td).sum(0) / tot
    denom = (tot - 1.0).clamp_min(_MIN_TOTAL_WEIGHT) if unbiased else tot
    var = (wd * (td - mean).pow(2)).sum(0) / denom
    return mean.to(t.dtype), var.sqrt().to(t.dtype)


def select(
    weights: torch.Tensor | None, idx: torch.Tensor
) -> torch.Tensor | None:
    """Index ``weights`` by a minibatch's row indices.

    Every gradient-trained mechanism shuffles *internally* (each owns its own
    ``torch.randperm``), so the weights have to be indexed with the very same
    ``idx`` inside that loop.  Routing every such site through this helper
    makes the pairing greppable; a desynchronised weight vector is worse than
    no weighting at all, because the fit still converges — to the wrong thing.
    """
    return None if weights is None else weights.reshape(-1)[idx]
