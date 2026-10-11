"""Probability sets: what NBN knows about a probability table, with a guarantee.

A *probability set* describes every table of probabilities that is compatible
with the data at a declared level.  All sets share one small interface:

``bounds()``
    ``(lower, upper)`` for every probability — the min/max view, always
    available, possibly looser than the set itself.
``contains(P)``
    Whether a table ``P`` (shape ``[R, K]``) lies in the set.
``level``
    The declared confidence ``1 - delta``.
``guarantee``
    How to read ``level``: ``"pac"`` (finite-sample frequentist, for all
    entries at once), ``"pac_under_model"`` (the same, if the declared noise
    model holds), ``"bayesian"`` (posterior credibility) or ``"asymptotic"``
    (bootstrap).

Tables are ``[R, K]``: one row per parent configuration, one column per value
of the child.  All tensors stay on the device they were built on.

Two sets describe something other than one table and have their own API:
:class:`NoiseParameterSet` (a polytope on a shared noise law; its
``table_set(node)`` gives the per-table box) and :class:`CDFBandSet` (a band on
a continuous noise CDF).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import torch

Guarantee = Literal["pac", "pac_under_model", "bayesian", "asymptotic"]


def tighten_box_to_simplex(
    lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shrink a box using the fact that each row of probabilities sums to one.

    ``p_k >= 1 - sum_{j != k} upper_j`` and ``p_k <= 1 - sum_{j != k} lower_j``.
    Every distribution inside the original box is inside the tightened one, so
    the step is sound; when the box meets the simplex, one pass gives the
    exact per-entry range.  An empty intersection (``sum lower > 1`` or
    ``sum upper < 1``) shows up as ``lower > upper``; see
    :meth:`IntervalSet.empty_rows`.
    """
    upper_others = upper.sum(dim=-1, keepdim=True) - upper
    lower_others = lower.sum(dim=-1, keepdim=True) - lower
    new_lower = torch.maximum(lower, 1.0 - upper_others).clamp(0.0, 1.0)
    new_upper = torch.minimum(upper, 1.0 - lower_others).clamp(0.0, 1.0)
    return new_lower, new_upper


@dataclass(kw_only=True)
class ProbabilitySet:
    """Base class: a set of ``[R, K]`` probability tables with a guarantee."""

    level: float
    guarantee: Guarantee
    # True for rows without any data: their bounds are the whole simplex.
    unseen: torch.Tensor = field(repr=False)

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def contains(self, table: torch.Tensor, atol: float = 1e-6) -> bool:
        return bool(self.row_contains(table, atol).all())

    def row_contains(self, table: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
        """Per-row membership, shape ``[R]``.  Default: the box test."""
        lower, upper = self.bounds()
        table = table.to(lower)
        return ((table >= lower - atol) & (table <= upper + atol)).all(dim=-1)

    def width(self) -> torch.Tensor:
        """``upper - lower`` for every entry."""
        lower, upper = self.bounds()
        return upper - lower


@dataclass(kw_only=True)
class IntervalSet(ProbabilitySet):
    """A box: ``lower <= P <= upper`` entry-wise, tightened to the simplex."""

    lower: torch.Tensor = field(repr=False)
    upper: torch.Tensor = field(repr=False)

    def __post_init__(self) -> None:
        self.lower, self.upper = tighten_box_to_simplex(self.lower, self.upper)

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lower, self.upper

    def empty_rows(self) -> torch.Tensor:
        """Rows whose box does not meet the simplex, shape ``[R]``."""
        return (self.lower > self.upper + 1e-12).any(dim=-1)


@dataclass(kw_only=True)
class L1BallSet(ProbabilitySet):
    """Rows within an L1 ball: ``||P_r - center_r||_1 <= radius_r``."""

    center: torch.Tensor = field(repr=False)
    radius: torch.Tensor = field(repr=False)

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Moving mass m onto one entry removes m elsewhere, so the L1 distance
        # is 2m: each entry can move by at most radius / 2.
        half = (self.radius / 2.0).unsqueeze(-1)
        lower = (self.center - half).clamp(0.0, 1.0)
        upper = (self.center + half).clamp(0.0, 1.0)
        return tighten_box_to_simplex(lower, upper)

    def row_contains(self, table: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
        distance = (table.to(self.center) - self.center).abs().sum(dim=-1)
        return distance <= self.radius + atol


@dataclass(kw_only=True)
class SampleSet(ProbabilitySet):
    """A cloud of tables (posterior draws or bootstrap refits) and its box.

    ``bounds()`` uses the ``delta / 2`` and ``1 - delta / 2`` quantiles of every
    entry; with ``simultaneous=True`` the level is Bonferroni-split over all
    entries, so the box covers all entries at once (for the sampling law).
    """

    samples: torch.Tensor = field(repr=False)  # [S, R, K]
    simultaneous: bool = True

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        delta = 1.0 - self.level
        if self.simultaneous:
            delta = delta / self.samples[0].numel()
        n_tail = self.samples.shape[0] * delta / 2.0
        if n_tail < 5:
            warnings.warn(
                f"SampleSet: only {n_tail:.2g} draws expected beyond each quantile; the box "
                "is close to the sample min/max and does not support the declared level.  "
                "Draw more samples or use simultaneous=False.",
                UserWarning, stacklevel=2,
            )
        lower = torch.quantile(self.samples, delta / 2.0, dim=0)
        upper = torch.quantile(self.samples, 1.0 - delta / 2.0, dim=0)
        return lower, upper


@dataclass(kw_only=True)
class NoiseParameterSet:
    """A polytope of noise laws shared by several tabular mechanisms.

    The noise law is a probability vector ``q`` over ``M`` noise values.  Each
    child probability of each tied mechanism is a sum of entries of ``q`` — an
    *expression*.  The set is

        { q in simplex : lower_e <= sum_{m in S_e} q_m <= upper_e  for every e },

    with one confidence interval per distinct expression ``S_e`` (rows that
    share an expression pool their counts).  Downstream, ``q`` is the shared
    parameter of a parametric MDP and this polytope is its region.

    Attributes
    ----------
    expressions:
        ``[E, M]`` 0/1 matrix, one row per distinct expression.
    lower, upper:
        ``[E]`` interval end points.
    index:
        For every tied mechanism (by node name), a ``[R, K]`` long tensor that
        maps each table entry to its expression row.
    """

    expressions: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    index: dict[str, torch.Tensor]
    level: float
    guarantee: Guarantee = "pac_under_model"
    unseen: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    def contains(self, q: torch.Tensor, atol: float = 1e-6) -> bool:
        """Whether the noise law ``q`` (shape ``[M]``) satisfies every constraint."""
        q = q.detach().to(self.expressions)
        values = self.expressions @ q
        on_simplex = bool((q >= -atol).all()) and abs(float(q.sum()) - 1.0) <= 1e-6
        return on_simplex and bool(
            ((values >= self.lower - atol) & (values <= self.upper + atol)).all()
        )

    def table_set(self, node: str) -> IntervalSet:
        """The box this polytope implies for one tied mechanism's table.

        A box per entry forgets that all entries move with one ``q``: it is a
        rectangular superset of the polytope's image, valid but looser.  Rows
        without data get the whole simplex (their location is a guess).
        """
        idx = self.index[node]
        unseen = self.unseen.get(node, torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device))
        hidden = unseen.to(idx.device).unsqueeze(-1)
        lower = torch.where(hidden, torch.zeros_like(self.lower[idx]), self.lower[idx])
        upper = torch.where(hidden, torch.ones_like(self.upper[idx]), self.upper[idx])
        return IntervalSet(level=self.level, guarantee=self.guarantee, unseen=unseen,
                           lower=lower, upper=upper)

    def tightened(self) -> NoiseParameterSet:
        """Shrink every interval to its exact min/max over the polytope (needs SciPy).

        Each end point is a small linear programme over ``q``.  The result
        describes the same polytope with redundant slack removed, so the boxes
        returned by :meth:`table_set` get tighter.
        """
        try:
            from scipy.optimize import linprog
        except ImportError as exc:  # pragma: no cover
            raise ImportError("tightened() needs SciPy (pip install 'nbn[scm]').") from exc
        import numpy as np

        e_mat = self.expressions.detach().cpu().double().numpy()
        lo = self.lower.detach().cpu().double().numpy()
        hi = self.upper.detach().cpu().double().numpy()
        m = e_mat.shape[1]
        a_ub = np.vstack([e_mat, -e_mat])
        b_ub = np.concatenate([hi, -lo])
        a_eq, b_eq = np.ones((1, m)), np.ones(1)
        new_lo, new_hi = lo.copy(), hi.copy()
        for e in range(e_mat.shape[0]):
            for sign, store in ((1.0, new_lo), (-1.0, new_hi)):
                res = linprog(sign * e_mat[e], A_ub=a_ub, b_ub=b_ub, A_eq=a_eq,
                              b_eq=b_eq, bounds=[(0.0, 1.0)] * m, method="highs")
                if res.status != 0:
                    raise RuntimeError(
                        "The noise polytope is empty: the intervals are mutually "
                        "inconsistent, which suggests the shared-noise model is "
                        "wrong for these data (see noise_sharing_test)."
                    )
                store[e] = sign * res.fun
        as_tensor = lambda a: torch.as_tensor(a, dtype=self.lower.dtype, device=self.lower.device)  # noqa: E731
        return NoiseParameterSet(
            expressions=self.expressions, lower=torch.maximum(self.lower, as_tensor(new_lo)),
            upper=torch.minimum(self.upper, as_tensor(new_hi)), index=self.index,
            level=self.level, guarantee=self.guarantee, unseen=self.unseen,
        )


@dataclass(kw_only=True)
class CDFBandSet:
    """A uniform band on the noise CDF of a continuous additive mechanism.

    For ``X = g(pa) + U`` with a fixed location ``g``, the residuals are
    i.i.d. draws of ``U``.  The Dvoretzky–Kiefer–Wolfowitz inequality with
    Massart's constant (Massart, Ann. Probab. 18(3):1269–1283, 1990) gives ``sup_t |F_hat(t) - F(t)| <= eps`` with
    probability ``>= 1 - delta``, ``eps = sqrt(log(2 / delta) / (2 n))``.
    Because the noise is shared by every parent configuration, the band holds
    for all of them at once.  The guarantee is conditional on ``g``
    (``"pac_under_model"``): fit ``g`` on other data than the residuals.
    """

    sorted_residuals: torch.Tensor  # [n], ascending
    epsilon: float
    level: float
    guarantee: Guarantee = "pac_under_model"

    def noise_cdf_bounds(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bounds on ``P(U <= t)`` for every entry of ``t`` (any shape)."""
        n = self.sorted_residuals.numel()
        r = self.sorted_residuals.to(t)
        f_hat = torch.searchsorted(r, t.contiguous(), right=True).to(t.dtype) / n
        return (f_hat - self.epsilon).clamp(0.0, 1.0), (f_hat + self.epsilon).clamp(0.0, 1.0)

    def interval_probability_bounds(
        self, a: torch.Tensor, b: torch.Tensor, location: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bounds on ``P(a < X <= b | pa)`` given ``location = g(pa)``, batched.

        ``P(a < X <= b) = F(b - g) - F(a - g)``.  Inside the band each CDF value
        is known up to ``eps``, so the difference is known up to ``2 eps``.
        """
        lo_b, hi_b = self.noise_cdf_bounds(b - location)
        lo_a, hi_a = self.noise_cdf_bounds(a - location)
        return (lo_b - hi_a).clamp(0.0, 1.0), (hi_b - lo_a).clamp(0.0, 1.0)
