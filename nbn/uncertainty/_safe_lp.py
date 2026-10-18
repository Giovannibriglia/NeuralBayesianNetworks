"""Rigorous lower bounds for LP optima from an approximate dual (weak duality).

Why
---
A floating-point LP solver returns an optimum ``f`` that can be off by its
feasibility and optimality tolerances (HiGHS: about ``1e-7``), so an end point
``f - slack`` with a fixed slack is a guess, not a bound.  Weak duality turns
*any* dual vector into a bound whose only error is the rounding of a few dot
products, which can be bounded a priori [Neumaier & Shcherbina, "Safe bounds
in linear and mixed-integer linear programming", Math. Program. 99(2):283–296,
2004, https://doi.org/10.1007/s10107-003-0433-3]; the derivation below is the
textbook one, written out here so that it can be checked.

For ``min c^T x`` subject to ``A_ub x <= b_ub``, ``A_eq x = b_eq`` and a box
``l <= x <= u`` that every feasible point satisfies (given or implied), take
any ``y_ub <= 0`` and any ``y_eq``, and let ``r = c - A_ub^T y_ub - A_eq^T y_eq``.
For every feasible ``x``

    c^T x = y_ub^T A_ub x + y_eq^T A_eq x + r^T x
          >= y_ub^T b_ub + y_eq^T b_eq + sum_j min(r_j l_j, r_j u_j),

since ``y_ub <= 0`` and ``A_ub x <= b_ub``.  The right-hand side is a lower
bound on the optimum whatever the quality of ``y`` (a poor ``y`` gives a loose,
still valid, bound).  Floating point: each computed dot product of length
``n`` is within ``gamma_n sum |terms|`` of the exact one (``gamma_n = n u /
(1 - n u)``; Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd
ed., SIAM 2002, §3.1), so subtracting those magnitudes (doubled, for
the final additions) keeps the bound rigorous without directed rounding;
an absolute term covers underflow (Higham's ``eta``, ibid. §2.1).
"""
from __future__ import annotations

import numpy as np

_U = np.finfo(np.float64).eps / 2


def _gamma(n: int) -> float:
    nu = n * _U
    return nu / (1.0 - nu)


def _abs(a):
    return abs(a)                     # dense or scipy.sparse


def safe_lower_bound(c, a_ub, b_ub, a_eq, b_eq, y_ub, y_eq, lo, hi) -> float:
    """A rigorous lower bound on ``min c^T x`` over the program described in
    the module docstring, from approximate duals ``y_ub`` (clamped to ``<= 0``)
    and ``y_eq``; ``lo`` / ``hi`` are finite bounds on every variable."""
    c = np.asarray(c, np.float64)
    lo = np.broadcast_to(np.asarray(lo, np.float64), c.shape)
    hi = np.broadcast_to(np.asarray(hi, np.float64), c.shape)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("safe_lower_bound needs a finite box on every variable.")
    parts_b, parts_r, parts_abs_r = 0.0, c.copy(), np.abs(c)
    n_terms = 1
    mags = 0.0
    for a, b, y, clamp in ((a_ub, b_ub, y_ub, True), (a_eq, b_eq, y_eq, False)):
        if a is None or y is None or np.size(y) == 0:
            continue
        y = np.asarray(y, np.float64).reshape(-1)
        if clamp:
            y = np.minimum(y, 0.0)
        b = np.asarray(b, np.float64).reshape(-1)
        parts_b += float(y @ b)
        mags += float(np.abs(y) @ np.abs(b))
        aty = np.asarray(a.T @ y).reshape(-1)
        parts_r = parts_r - aty
        parts_abs_r = parts_abs_r + np.asarray(_abs(a).T @ np.abs(y)).reshape(-1)
        n_terms += a.shape[0]
    r = parts_r
    box = float(np.minimum(r * lo, r * hi).sum())
    xmax = np.maximum(np.abs(lo), np.abs(hi))
    # rounding of y^T b, of each r_j (n_terms products), and of the box sum
    err = (_gamma(n_terms + 2) * mags
           + _gamma(n_terms + 2) * float(parts_abs_r @ xmax)
           + _gamma(c.size + 2) * float(np.abs(np.minimum(r * lo, r * hi)).sum()))
    # gamma terms bound relative errors only; each of the (at most) n_ops
    # subnormal roundings adds an absolute error <= the smallest subnormal
    n_ops = 2 * n_terms + 2 * c.size + 4 + sum(
        int(getattr(a, "nnz", np.size(a))) for a in (a_ub, a_eq) if a is not None)
    eta = n_ops * float(np.finfo(np.float64).smallest_subnormal)
    return parts_b + box - 2.0 * err - eta


def safe_min(res, c, a_ub, b_ub, a_eq, b_eq, lo, hi) -> float | None:
    """Rigorous lower bound on the optimum of a solved ``scipy.optimize.linprog``
    (HiGHS) problem, from its marginals; ``None`` when the solver gave none."""
    try:
        y_ub = None if a_ub is None else res.ineqlin.marginals
        y_eq = None if a_eq is None else res.eqlin.marginals
    except AttributeError:
        return None
    if (a_ub is not None and y_ub is None) or (a_eq is not None and y_eq is None):
        return None
    try:
        v = safe_lower_bound(c, a_ub, b_ub, a_eq, b_eq, y_ub, y_eq, lo, hi)
    except ValueError:
        return None
    return v if np.isfinite(v) else None
