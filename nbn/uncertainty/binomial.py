"""Finite-sample confidence intervals for binomial proportions.

Every function is a batched tensor operation: ``successes`` and ``trials`` may
have any (common) shape and live on any device.  The default method inverts the
Chernoff (KL) tail bound by bisection, so it needs nothing beyond torch and runs
on CPU, CUDA and MPS alike.

Two-sided intervals spend ``delta / 2`` on each side.

Budget helpers
--------------
A probability set is "safe" when *all* its intervals hold at once.  The helpers
below turn a global budget ``delta`` into per-interval levels:

* :func:`union_bound` splits ``delta`` evenly over a fixed number of intervals;
* :func:`anytime_delta` makes one interval valid for every sample size ``n``
  simultaneously (``delta_n = delta / (n (n + 1))``, which sums to ``delta``).
  Use it when the number of draws is itself random and depends on earlier
  draws, as for the visit counts of a state-action pair along trajectories.
  The argument couples each row with its own i.i.d. stack of draws and takes
  a union over sample sizes, as in Agarwal et al., *RL: Theory and
  Algorithms*, Lemma 6.2; that proof uses a finite range of ``n``, the summable
  schedule here removes the need for a horizon cap.
"""
from __future__ import annotations

from typing import Literal

import torch

from nbn.utils.device import accum_dtype

IntervalMethod = Literal["kl", "hoeffding", "clopper_pearson"]

_BISECTION_STEPS = 60  # interval width 2**-60: far below float64 resolution


def kl_bernoulli(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL divergence ``KL(Bern(p) || Bern(q))``, with ``0 log 0 = 0``."""
    eps = torch.finfo(q.dtype).tiny
    q = q.clamp(eps, 1.0 - eps)
    return torch.xlogy(p, p / q) + torch.xlogy(1.0 - p, (1.0 - p) / (1.0 - q))


def union_bound(delta: float, n_intervals: int) -> float:
    """Per-interval level so that ``n_intervals`` intervals hold jointly w.p. ``>= 1 - delta``."""
    _check_delta(delta)
    if n_intervals < 1:
        raise ValueError(f"n_intervals must be >= 1, got {n_intervals}.")
    return delta / n_intervals


def anytime_delta(delta: float | torch.Tensor, trials: torch.Tensor) -> torch.Tensor:
    """Level ``delta / (n (n + 1))`` that makes an interval valid for every ``n`` at once."""
    n = trials.clamp_min(1)
    return delta / (n * (n + 1))


def binomial_interval(
    successes: torch.Tensor,
    trials: torch.Tensor,
    delta: float | torch.Tensor,
    method: IntervalMethod = "kl",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-sided ``1 - delta`` confidence interval for a binomial proportion.

    Parameters
    ----------
    successes, trials:
        Tensors of the same shape (counts; may be float).  ``trials == 0``
        yields the uninformative interval ``[0, 1]``.
    delta:
        Miscoverage level, a float or a tensor broadcastable to ``trials``.
    method:
        ``"kl"`` (default): Chernoff/KL inversion, pure torch.
        ``"hoeffding"``: ``p_hat +- sqrt(log(2/delta) / (2 n))``, closed form,
        looser.
        ``"clopper_pearson"``: exact Beta quantiles, needs SciPy, CPU round-trip.

    Returns
    -------
    (lower, upper), tensors shaped like ``trials`` in the accumulation dtype of
    the device (float64 where available).
    """
    if isinstance(delta, float):
        _check_delta(delta)
    dtype = accum_dtype(trials.device)
    k = successes.to(dtype)
    n = trials.to(dtype)
    if bool((k < 0).any()) or bool((k > n).any()):
        raise ValueError("successes must lie in [0, trials].")
    delta_t = torch.as_tensor(delta, dtype=dtype, device=n.device).expand_as(n)

    if method == "kl":
        lower, upper = _kl_interval(k, n, delta_t)
    elif method == "hoeffding":
        lower, upper = _hoeffding_interval(k, n, delta_t)
    elif method == "clopper_pearson":
        lower, upper = _clopper_pearson_interval(k, n, delta_t)
    else:
        raise ValueError(f"Unknown interval method '{method}'.")

    unseen = n == 0
    lower = torch.where(unseen, torch.zeros_like(lower), lower)
    upper = torch.where(unseen, torch.ones_like(upper), upper)
    return lower, upper


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def _kl_interval(
    k: torch.Tensor, n: torch.Tensor, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert ``n * KL(p_hat || p) <= log(2 / delta)`` on each side of ``p_hat``.

    Chernoff: ``P(p_hat <= x) <= exp(-n KL(x || p))`` for ``x <= p`` (and
    symmetrically above), so each side of the interval fails with
    probability at most ``delta / 2``.  ``KL(p_hat || .)`` is monotone on each
    side of ``p_hat``, so bisection finds the end points.  This is the
    confidence bound inverted by KL-UCB (Garivier & Cappé, COLT 2011,
    https://arxiv.org/abs/1102.2490).
    """
    p_hat = k / n.clamp_min(1)
    radius = torch.log(2.0 / delta) / n.clamp_min(1)

    def inside(q: torch.Tensor) -> torch.Tensor:
        return kl_bernoulli(p_hat, q) <= radius

    # Upper end in [p_hat, 1]: points below it are inside the set.
    lo, hi = p_hat.clone(), torch.ones_like(p_hat)
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        ok = inside(mid)
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
    upper = hi  # the outer end of the bracket keeps the interval conservative

    # Lower end in [0, p_hat]: points above it are inside the set.
    lo, hi = torch.zeros_like(p_hat), p_hat.clone()
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        ok = inside(mid)
        lo, hi = torch.where(ok, lo, mid), torch.where(ok, mid, hi)
    lower = lo

    return lower, upper


def _hoeffding_interval(
    k: torch.Tensor, n: torch.Tensor, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    p_hat = k / n.clamp_min(1)
    half_width = torch.sqrt(torch.log(2.0 / delta) / (2.0 * n.clamp_min(1)))
    return (p_hat - half_width).clamp_min(0.0), (p_hat + half_width).clamp_max(1.0)


def _clopper_pearson_interval(
    k: torch.Tensor, n: torch.Tensor, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from scipy.stats import beta
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "method='clopper_pearson' needs SciPy (pip install 'nbn[bounds]'); "
            "method='kl' is SciPy-free and nearly as tight."
        ) from exc
    k_np, n_np, d_np = (t.detach().cpu().double().numpy() for t in (k, n, delta))
    lower = beta.ppf(d_np / 2.0, k_np, n_np - k_np + 1.0)
    # isf, not ppf(1 - d/2): 1 - d/2 rounds to 1.0 for d < 1.1e-16 (anytime levels at
    # large n), which made the upper end 1 (sound but vacuous).
    upper = beta.isf(d_np / 2.0, k_np + 1.0, n_np - k_np)
    lower = torch.from_numpy(lower).nan_to_num(0.0)  # k == 0 -> NaN -> 0
    upper = torch.from_numpy(upper).nan_to_num(1.0)  # k == n -> NaN -> 1
    return lower.to(n), upper.to(n)


def _check_delta(delta: float) -> None:
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1), got {delta}.")
