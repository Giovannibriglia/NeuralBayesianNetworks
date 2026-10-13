"""Sound rows for a possibly confounded action effect, from several logging regimes (#315).

When a logged action may depend on a **hidden** variable that also moves the
next state (a confounded logger, ``A <- W -> X'``), the fitted table
``P(X' | parents)`` is not the interventional row ``P(X' | do(A=a), u)`` that
planning needs (``u`` = the node's other parents).  This module turns data
from several regimes (behaviour policies) into per-entry intervals on the
interventional rows, valid together with probability ``>= 1 - delta``.

Steps (one parent configuration ``u`` at a time, batched where closed form)
---------------------------------------------------------------------------
1. **Confidence first.**  A box on every regime's observables: intervals on
   every cell ``P_k(A=a, X'=y | u)`` and every complement
   ``P_k(A=a, X' != y | u)``, simultaneous by Bonferroni over a budget fixed
   before seeing the data (configurations x regimes x ``2 |A| |X'|`` events),
   anytime-valid in the visit counts for ``sampling="trajectory"``.
2. **Rows from every law in the box** (outer envelope):

   * ``"manski"`` / ``"intersection"`` — per regime the natural row
     ``[P_k(a, y), 1 - P_k(a, X' != y)]`` (Manski 1990; Bareinboim, *Causal
     AI* v0.27, Thm 5.6.3, p. 309), intersected over regimes: the natural
     IV bound (Thm 5.6.4, p. 311).  Valid, not sharp in general (Pearl,
     *Causality* 2e, §8.2, Eq. 8.16, p. 268).
   * ``"lp"`` — the IV linear program over response types (Balke & Pearl
     1997; Pearl 2e §8.2, LP (8.13), pp. 264–267), generalised to finite
     ``A``, ``X'`` and number of regimes; with the box it is **valid, not
     sharp** (sharp only for the true law: Bareinboim, pp. 323–324).
     Regimes in ``declared_unconfounded`` (e.g. randomised loggers) add
     ``pi(a) mu_t`` constraints.  Needs SciPy.
   * ``"sensitivity"`` — per-regime rows under a marginal sensitivity model
     with parameter ``lam`` (``lam = 1``: no confounding; ``lam -> inf``:
     Manski), intersected.  Needs SciPy for non-point boxes.
   * ``"auto"`` — ``"lp"`` if the program is small enough and SciPy is
     installed, else ``"intersection"``.
3. **Refutation.**  An empty intersection or an infeasible LP means the
   instrument assumptions are violated for that configuration (the
   instrumental inequality, Pearl 2e §8.4, Eq. 8.22, pp. 274–275); it is
   reported (``refuted``), never silently widened.
4. **No data for an action** in every regime: the whole simplex.

Declared assumptions (in the result)
------------------------------------
* **Exclusion:** the regime acts on ``X'`` only through ``A``; every
  environment mechanism is the same in every regime.
* **Independence:** the regime is independent of the hidden variable given
  ``u``; with trajectory data this needs **fresh** confounding (a new ``W_t``
  at every step) and stationary Markov loggers.  Persistent hidden state
  breaks per-step rows (history-indexed bounds are out of scope:
  Bareinboim, Thm 9.2.5, pp. 591–592).

Gate (:func:`confounding_gate`)
-------------------------------
Rows of a node are treated as **possibly confounded** unless the user
declares them unconfounded.  ``context_invariance_test`` can only *raise*
suspicion: a rejection forces the confounded rows; a non-rejection never
certifies a node (there is no statistical test for confounding: Pearl 2e
§6.2, p. 182; Peters et al., *ECI*, Remark 6.40, pp. 113–114).

Port of the SCM → MDP project's ``confounding/`` reference package (its
tests, including Pearl's printed IV bounds, are ported to NBN's test suite).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import List, Literal, Mapping, Sequence, Tuple

import numpy as np
import torch

from nbn.uncertainty.binomial import IntervalMethod, anytime_delta, binomial_interval
from nbn.uncertainty.counts import count_table_for, parent_grid, parent_row_index
from nbn.uncertainty.sets import IntervalSet

Method = Literal["auto", "lp", "intersection", "manski", "sensitivity"]
Sampling = Literal["iid", "trajectory"]
TOL = 1e-9
#: outward widening of LP optima per unit of primal mass (above HiGHS' 1e-7 tolerances)
SOLVER_SLACK = 1e-6
#: relaxation of the box right-hand sides, so a point box survives rounding
RHS_SLACK = 1e-7
#: below this action probability the sensitivity LP is badly scaled
P_MIN = 1e-6

ASSUMPTIONS = (
    "exclusion: the regime acts on the child only through the action; same environment in every regime",
    "independence: the regime is independent of the hidden variable given the other parents",
    "fresh confounding for trajectory data (a new hidden value at every step); stationary Markov loggers",
    "declared_unconfounded regimes do not read the hidden variable",
)


class RefutedAssumption(ValueError):
    """The data contradict the declared model (empty intersection or infeasible IV program)."""


class ProgramTooLarge(RuntimeError):
    """The response-type space ``|X'| ** |A|`` exceeds the caller's limit."""


class LPFailure(RuntimeError):
    """The LP solver stopped without an optimum (numerical trouble, iteration limit)."""


# ---------------------------------------------------------------------------
# One configuration: the box and the IV program (NumPy, SciPy for the LP)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeBox:
    """Simultaneous box for one parent configuration and all regimes.

    Shapes ``[K, |A|, |X'|]``: ``cell_*`` bound ``P_k(a, y)``, ``comp_*`` bound
    ``P_k(a, X' != y)``; ``n[k]`` = visits of regime ``k``.
    """

    cell_lo: np.ndarray
    cell_hi: np.ndarray
    comp_lo: np.ndarray
    comp_hi: np.ndarray
    n: np.ndarray

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.cell_lo.shape  # type: ignore[return-value]

    @staticmethod
    def point(P: np.ndarray) -> RegimeBox:
        """Zero-width box at population values ``P[k, a, y]``."""
        P = np.asarray(P, float)
        comp = P.sum(-1, keepdims=True) - P
        return RegimeBox(P, P, comp, comp, np.full(P.shape[0], np.inf))

    def contains(self, P: np.ndarray, tol: float = 1e-12) -> bool:
        P = np.asarray(P, float)
        comp = P.sum(-1, keepdims=True) - P
        return bool(np.all(self.cell_lo - tol <= P) and np.all(self.cell_hi + tol >= P)
                    and np.all(self.comp_lo - tol <= comp) and np.all(comp <= self.comp_hi + tol))


def natural_rows(box: RegimeBox) -> Tuple[np.ndarray, np.ndarray]:
    """Intersection of the per-regime Manski rows, ``[|A|, |X'|]`` each (before the simplex)."""
    lo = box.cell_lo.max(axis=0)
    hi = (1.0 - box.comp_lo).min(axis=0)
    return np.clip(lo, 0, 1), np.clip(hi, 0, 1)


def _scipy():
    try:
        from scipy import sparse
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "method='lp' and 'sensitivity' need SciPy (pip install 'nbn[bounds]'); "
            "method='intersection' is SciPy-free."
        ) from exc
    return sparse, linprog


class IVProgram:
    """Feasible set of response-type masses for one configuration, built once.

    Response types ``t = (y_a)_{a in A}`` (the child under each action) have
    masses ``mu_t``, the same in every regime; regime ``k`` sends a part
    ``x[k, t, a] >= 0`` of type ``t`` to action ``a`` with
    ``sum_a x[k, t, a] = mu_t``.  Observables enter as a box.  Exact for the
    population (point box) and for objectives that depend on ``mu`` only.
    """

    def __init__(self, box: RegimeBox, declared_unconfounded: Sequence[int] = (),
                 max_types: int = 20_000, max_vars: int = 200_000) -> None:
        sparse, _ = _scipy()
        K, nA, nY = box.shape
        n_types = nY ** nA
        if n_types > max_types:
            raise ProgramTooLarge(f"{n_types} response types > max_types={max_types}")
        if n_types * (1 + K * nA) > max_vars:
            raise ProgramTooLarge(f"{n_types * (1 + K * nA)} variables > max_vars={max_vars}")
        self.slack = SOLVER_SLACK * (1 + K)
        self.nA, self.nY, self.K = nA, nY, K
        self.types = np.array(list(itertools.product(range(nY), repeat=nA)), dtype=int)
        nT = len(self.types)
        self.nT = nT
        n_var = nT + K * nT * nA

        def x_index(k: int, t: np.ndarray, a: int) -> np.ndarray:
            return nT + (k * nT + t) * nA + a

        t_all = np.arange(nT)
        eq_r: List[int] = []
        eq_c: List[int] = []
        eq_v: List[float] = []
        b_eq: List[float] = []
        ub_r: List[int] = []
        ub_c: List[int] = []
        ub_v: List[float] = []
        b_ub: List[float] = []

        def add(rows, cols, vals, r, c, v):
            rows.extend([r] * len(c))
            cols.extend(c)
            vals.extend(v)

        add(eq_r, eq_c, eq_v, 0, list(t_all), [1.0] * nT)
        b_eq.append(1.0)
        for k in range(K):
            for t in range(nT):
                r = len(b_eq)
                add(eq_r, eq_c, eq_v, r, [t] + [int(x_index(k, t, a)) for a in range(nA)], [-1.0] + [1.0] * nA)
                b_eq.append(0.0)

        def two_sided(cols: List[int], lo: float, hi: float) -> None:
            r = len(b_ub)
            add(ub_r, ub_c, ub_v, r, cols, [1.0] * len(cols))
            b_ub.append(hi)
            add(ub_r, ub_c, ub_v, r + 1, cols, [-1.0] * len(cols))
            b_ub.append(-lo)

        for k in range(K):
            for a in range(nA):
                for y in range(nY):
                    hit = t_all[self.types[:, a] == y]
                    miss = t_all[self.types[:, a] != y]
                    two_sided([int(i) for i in x_index(k, hit, a)], box.cell_lo[k, a, y], box.cell_hi[k, a, y])
                    if len(miss):
                        two_sided([int(i) for i in x_index(k, miss, a)], box.comp_lo[k, a, y], box.comp_hi[k, a, y])

        for k in declared_unconfounded:
            if not 0 <= k < K:
                raise IndexError(f"declared regime {k} out of range")
            pi_lo = np.max(box.cell_lo[k] + box.comp_lo[k], axis=-1)
            pi_hi = np.min(box.cell_hi[k] + box.comp_hi[k], axis=-1)
            for a in range(nA):
                for t in range(nT):
                    xi = int(x_index(k, t, a))
                    r = len(b_ub)
                    add(ub_r, ub_c, ub_v, r, [xi, t], [1.0, -min(pi_hi[a], 1.0)])
                    b_ub.append(0.0)
                    add(ub_r, ub_c, ub_v, r + 1, [xi, t], [-1.0, max(pi_lo[a], 0.0)])
                    b_ub.append(0.0)

        self.A_eq = sparse.csr_matrix((eq_v, (eq_r, eq_c)), shape=(len(b_eq), n_var))
        self.b_eq = np.array(b_eq)
        self.A_ub = sparse.csr_matrix((ub_v, (ub_r, ub_c)), shape=(len(b_ub), n_var))
        self.b_ub = np.array(b_ub) + RHS_SLACK
        self.n_var = n_var

    def bounds(self, c_mu: np.ndarray) -> Tuple[float, float]:
        """Min and max of ``sum_t c_mu[t] mu_t`` over the feasible set (widened outward)."""
        _, linprog = _scipy()
        c = np.zeros(self.n_var)
        c[: self.nT] = c_mu
        out = []
        for sign in (1.0, -1.0):
            res = linprog(sign * c, A_ub=self.A_ub, b_ub=self.b_ub, A_eq=self.A_eq, b_eq=self.b_eq,
                          bounds=(0, None), method="highs")
            if res.status == 2:
                raise RefutedAssumption("no shared environment fits the regimes (IV model refuted)")
            if res.status != 0:
                raise LPFailure(f"LP failed: {res.message}")
            out.append(sign * res.fun)
        return out[0] - self.slack, out[1] + self.slack

    def indicator(self, a: int, y: int) -> np.ndarray:
        """Coefficients of ``P(X'=y | do(A=a)) = sum_{t: t_a = y} mu_t``."""
        return (self.types[:, a] == y).astype(float)

    def rows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-entry bounds ``[|A|, |X'|]`` on ``P(X' | do(a))`` for every action."""
        lo, hi = np.zeros((self.nA, self.nY)), np.zeros((self.nA, self.nY))
        for a in range(self.nA):
            for y in range(self.nY):
                lo[a, y], hi[a, y] = self.bounds(self.indicator(a, y))
        return np.clip(lo, 0, 1), np.clip(hi, 0, 1)


def sensitivity_row(p_ay: np.ndarray, p_a: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """Population row of one regime under a marginal sensitivity model (``lam >= 1``).

    ``1 / pi(a | W)`` lies in ``[w_lo, w_hi]``, ``w_lo = 1 + (1/P(a) - 1)/lam``,
    ``w_hi = 1 + (1/P(a) - 1) lam``; with the balancing constraint
    ``sum_y P(a, y) w_y = 1`` and ``R = P(a) - P(a, y)``:
    ``lo_y = max(P(a,y) w_lo, 1 - R w_hi)``, ``hi_y = min(P(a,y) w_hi, 1 - R w_lo)``.
    ``lam = 1`` gives ``P(y | a)``; ``lam -> inf`` the Manski row.
    """
    if lam < 1:
        raise ValueError("lam must be >= 1")
    p_ay = np.asarray(p_ay, float)
    if p_a <= 0:
        raise ValueError("P(A=a) must be > 0 for a sensitivity row (no data for this action)")
    gap = 1.0 / p_a - 1.0
    w_lo, w_hi = 1.0 + gap / lam, 1.0 + gap * lam
    rest = p_a - p_ay
    lo = np.maximum(p_ay * w_lo, 1.0 - rest * w_hi)
    hi = np.minimum(p_ay * w_hi, 1.0 - rest * w_lo)
    return np.clip(lo, 0, 1), np.clip(hi, 0, 1)


def sensitivity_row_box(box: RegimeBox, k: int, a: int, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """Union over ``box`` of the sensitivity row of regime ``k`` for action ``a``.

    Never refutes: on any LP failure, and for rare actions, it returns the
    box's Manski row, which contains every sensitivity row.
    """
    if lam < 1:
        raise ValueError("lam must be >= 1")
    nY = box.shape[2]
    cl, ch = box.cell_lo[k, a], box.cell_hi[k, a]
    ml, mh = box.comp_lo[k, a], box.comp_hi[k, a]
    p_lo = max(float(np.max(cl + ml)), 0.0)
    p_hi = min(float(np.min(ch + mh)), 1.0)
    if p_hi <= 0:
        return np.zeros(nY), np.ones(nY)
    if np.array_equal(cl, ch) and np.array_equal(ml, mh):
        return sensitivity_row(cl, float(cl.sum()), lam)
    fallback = (np.clip(cl, 0, 1), np.clip(1.0 - ml, 0, 1))
    if p_hi < P_MIN:
        return fallback
    _, linprog = _scipy()
    g_lo = (1.0 - p_hi) / (p_hi * lam)
    g_hi = np.inf if p_lo < P_MIN else lam * (1.0 - p_lo) / p_lo
    A_ub: List[np.ndarray] = []
    b_ub: List[float] = []
    for y in range(nY):
        e = np.zeros(2 * nY)
        e[y] = 1
        A_ub += [e, -e]
        b_ub += [ch[y], -cl[y]]
        comp = np.zeros(2 * nY)
        comp[:nY] = 1
        comp[y] = 0
        A_ub += [comp, -comp]
        b_ub += [mh[y], -ml[y]]
        r = np.zeros(2 * nY)
        r[nY + y], r[y] = -1, g_lo
        A_ub.append(r)
        b_ub.append(0.0)
        if np.isfinite(g_hi):
            r = np.zeros(2 * nY)
            r[nY + y], r[y] = 1, -g_hi
            A_ub.append(r)
            b_ub.append(0.0)
    A_eq, b_eq = [np.ones(2 * nY)], [1.0]
    b = np.array(b_ub) + 1e-12
    lo, hi = np.zeros(nY), np.zeros(nY)
    for y in range(nY):
        c = np.zeros(2 * nY)
        c[y] = c[nY + y] = 1
        vals = []
        for sign in (1.0, -1.0):
            res = linprog(sign * c, A_ub=np.array(A_ub), b_ub=b, A_eq=np.array(A_eq), b_eq=b_eq,
                          bounds=(0, None), method="highs")
            if res.status != 0:
                return fallback
            vals.append(sign * res.fun)
        lo[y], hi[y] = vals[0] - SOLVER_SLACK, vals[1] + SOLVER_SLACK
    return np.clip(lo, 0, 1), np.clip(hi, 0, 1)


def _tighten(lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simplex tightening per action row; raises when a row's box misses the simplex."""
    lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    if np.any(lo.sum(-1) > 1 + TOL) or np.any(hi.sum(-1) < 1 - TOL):
        raise RefutedAssumption("box does not meet the simplex")
    new_lo = np.maximum(lo, 1.0 - (hi.sum(-1, keepdims=True) - hi))
    new_hi = np.minimum(hi, 1.0 - (lo.sum(-1, keepdims=True) - lo))
    return np.clip(new_lo, 0, 1), np.clip(new_hi, 0, 1)


def rows_from_box(box: RegimeBox, method: Method = "auto", declared_unconfounded: Sequence[int] = (),
                  lam: float | None = None, max_types: int = 20_000,
                  max_vars: int = 200_000) -> Tuple[np.ndarray, np.ndarray, str]:
    """All action rows ``[|A|, |X'|]`` of one configuration from its box, and the method used."""
    _check_method(method, lam)
    K, nA, nY = box.shape
    if method == "sensitivity":
        assert lam is not None
        per = [[sensitivity_row_box(box, k, a, lam) for k in range(K)] for a in range(nA)]
        lo = np.array([np.max([r[0] for r in rows], axis=0) for rows in per])
        hi = np.array([np.min([r[1] for r in rows], axis=0) for rows in per])
        if np.any(lo > hi + TOL):
            raise RefutedAssumption("empty intersection of per-regime sensitivity rows")
        return (*_tighten(lo, np.maximum(hi, lo)), "sensitivity")
    nat_lo, nat_hi = natural_rows(box)
    if np.any(nat_lo > nat_hi + TOL):
        raise RefutedAssumption("empty intersection of per-regime rows")
    nat_hi = np.maximum(nat_hi, nat_lo)
    if method == "manski":
        return (*_tighten(nat_lo, nat_hi), "manski")
    if method in ("auto", "lp"):
        try:
            prog = IVProgram(box, declared_unconfounded, max_types=max_types, max_vars=max_vars)
            lp_lo, lp_hi = prog.rows()
            # the LP is inside the natural rows up to solver slack; intersecting removes the slack
            lo, hi = np.maximum(lp_lo, nat_lo), np.minimum(lp_hi, nat_hi)
            if np.any(lo > hi + TOL):
                raise RefutedAssumption("LP and natural rows do not meet")
            return (*_tighten(lo, np.maximum(hi, lo)), "lp")
        except (ProgramTooLarge, ImportError):
            if method == "lp":
                raise
        except LPFailure:
            if method == "lp":
                raise
            return (*_tighten(nat_lo, nat_hi), "intersection (lp failed)")
    return (*_tighten(nat_lo, nat_hi), "intersection")


def _check_method(method: str, lam: float | None) -> None:
    if method not in ("auto", "lp", "intersection", "manski", "sensitivity"):
        raise ValueError("method must be 'auto', 'lp', 'intersection', 'manski' or 'sensitivity'.")
    if method == "sensitivity" and lam is None:
        raise ValueError("method='sensitivity' needs lam.")
    if method != "sensitivity" and lam is not None:
        raise ValueError("lam is only used by method='sensitivity'.")


# ---------------------------------------------------------------------------
# The node-level builder (torch I/O, batched boxes)
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class ConfoundedSet:
    """Interval rows on ``P(node | do(action), other parents)`` with provenance."""

    node: str
    action: str
    # Table layout: rows over the node's parents in DAG order (C order), as the
    # node's own count table; the action is one of the parents.
    parents: Tuple[str, ...]
    set: IntervalSet = field(repr=False)
    # Per configuration of the other parents (C order over `others`):
    others: Tuple[str, ...] = ()
    method_used: List[str] = field(default_factory=list)
    refuted: torch.Tensor = field(default=None, repr=False)  # type: ignore[assignment]
    counts: torch.Tensor = field(default=None, repr=False)   # [n_configs, K] visits per regime
    n_regimes: int = 1
    delta: float = 0.05
    sampling: str = "iid"
    interval_method: str = "kl"
    declared_unconfounded: Tuple[int, ...] = ()
    assumptions: Tuple[str, ...] = ASSUMPTIONS
    notes: List[str] = field(default_factory=list)


def confounded_interval_set(
    model,
    data: Mapping[str, torch.Tensor],
    node: str,
    action: str,
    regime: str,
    method: Method = "auto",
    delta: float = 0.05,
    sampling: Sampling = "iid",
    declared_unconfounded: Sequence[int] = (),
    lam: float | None = None,
    n_regimes: int | None = None,
    interval_method: IntervalMethod = "kl",
    on_refuted: Literal["flag", "raise"] = "flag",
    max_types: int = 20_000,
    max_vars: int = 200_000,
) -> ConfoundedSet:
    """Sound interval rows on ``P(node | do(action), u)`` from several logging regimes.

    Parameters
    ----------
    model:
        A model (or anything with ``variables`` and ``dag``) whose DAG gives
        the node's parents; ``action`` must be one of them, ``u`` = the others.
        All must be discrete.
    data:
        Columns by name, including the ``regime`` column (integer regime id
        ``0 .. n_regimes - 1`` of each row).
    method:
        ``"auto"``, ``"lp"``, ``"intersection"`` (= ``"manski"`` rows
        intersected over regimes), ``"manski"`` (same rows; kept for clarity
        with one regime) or ``"sensitivity"`` (needs ``lam``).
    delta:
        Total failure probability (all configurations, regimes and entries).
    sampling:
        ``"trajectory"`` makes every interval anytime-valid in its visit count.
    declared_unconfounded:
        Regimes declared not to read the hidden variable (used by the LP).
    n_regimes:
        Number of regimes; fixes the Bonferroni budget (default: largest id + 1).
        Must not depend on the outcomes.
    interval_method:
        Box intervals: ``"kl"`` (torch, SciPy-free; default) or
        ``"clopper_pearson"`` (as the reference package) or ``"hoeffding"``.
    on_refuted:
        ``"flag"`` (the configuration gets the simplex and ``refuted=True``)
        or ``"raise"``.

    Returns
    -------
    ConfoundedSet
        ``set`` is an ``IntervalSet`` in the node's table layout with
        ``guarantee="pac_under_model"`` (under the declared assumptions);
        ``refuted`` and ``method_used`` are per configuration of ``u``.
    """
    _check_method(method, lam)
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1); got {delta}.")
    if on_refuted not in ("flag", "raise"):
        raise ValueError("on_refuted must be 'flag' or 'raise'.")
    parents = tuple(model.dag.parents(node))
    if action not in parents:
        raise ValueError(f"'{action}' is not a parent of '{node}' (parents: {parents}).")
    others = tuple(p for p in parents if p != action)
    variables = model.variables
    for v in (node, *parents):
        if not variables[v].is_discrete:
            raise ValueError(f"'{v}' must be discrete for confounded rows.")
    if regime not in data:
        raise KeyError(f"No regime column '{regime}' in the data.")
    regimes = data[regime].reshape(-1).long()
    if regimes.numel() and int(regimes.min()) < 0:
        raise ValueError("regime ids must be non-negative integers.")
    K = int(n_regimes) if n_regimes is not None else (int(regimes.max()) + 1 if regimes.numel() else 1)
    if regimes.numel() and int(regimes.max()) >= K:
        raise ValueError(f"regime ids must be < n_regimes={K}.")
    bad = [k for k in declared_unconfounded if not 0 <= int(k) < K]
    if bad:
        raise ValueError(f"declared_unconfounded regimes {bad} out of range [0, {K}).")

    n_a = int(variables[action].cardinality)
    n_y = int(variables[node].cardinality)
    other_cards = [int(variables[p].cardinality) for p in others]
    n_u = math.prod(other_cards)

    # Counts N[u, k, a, y]: count the node against (others, regime, action).
    reg_var = _RegimeVar(K)
    spec = dict(variables)
    spec["__regime__"] = reg_var
    cols = dict(data)
    cols["__regime__"] = regimes
    table, _ = count_table_for(spec, cols, node, (*others, "__regime__", action))
    N = table.reshape(n_u, K, n_a, n_y)

    # 1. The box, batched (every cell and complement of every configuration and regime).
    n_events = n_u * K * 2 * n_a * n_y
    delta_each = delta / n_events
    trials = N.sum(dim=(2, 3), keepdim=True).expand_as(N)
    d = torch.full_like(trials, delta_each)
    if sampling == "trajectory":
        d = anytime_delta(delta_each, trials)
    elif sampling != "iid":
        raise ValueError("sampling must be 'iid' or 'trajectory'.")
    comp = N.sum(-1, keepdim=True) - N
    cell_lo, cell_hi = binomial_interval(N, trials, d, method=interval_method)
    comp_lo, comp_hi = binomial_interval(comp, trials, d, method=interval_method)
    to_np = lambda t: t.detach().cpu().double().numpy()  # noqa: E731
    c_lo, c_hi, m_lo, m_hi = map(to_np, (cell_lo, cell_hi, comp_lo, comp_hi))
    visits = N.sum(dim=(2, 3))                                  # [n_u, K]
    v_np = to_np(visits)

    lo = np.zeros((n_u, n_a, n_y))
    hi = np.ones((n_u, n_a, n_y))
    used: List[str] = ["unvisited"] * n_u
    refuted = np.zeros(n_u, dtype=bool)
    notes: List[str] = []
    declared = [int(k) for k in declared_unconfounded]
    for u in range(n_u):
        seen = np.flatnonzero(v_np[u] > 0)
        if len(seen) == 0:
            continue
        box = RegimeBox(c_lo[u, seen], c_hi[u, seen], m_lo[u, seen], m_hi[u, seen], v_np[u, seen])
        dec = [int(np.flatnonzero(seen == k)[0]) for k in declared if k in seen]
        try:
            row_lo, row_hi, used[u] = rows_from_box(box, method, dec, lam, max_types, max_vars)
        except RefutedAssumption as err:
            if on_refuted == "raise":
                raise RefutedAssumption(f"configuration {u} of {others}: {err}") from err
            used[u] = "refuted"
            refuted[u] = True
            notes.append(f"configuration {u} of {others}: {err}")
            continue
        if used[u] == "intersection (lp failed)":
            notes.append(f"configuration {u}: LP solver failed; used the intersection rows (sound, possibly wider)")
        lo[u], hi[u] = row_lo, row_hi

    # Layout: rows over `parents` in DAG order (C order), as the node's table.
    device = N.device
    grid = parent_grid([int(variables[p].cardinality) for p in parents])
    u_index = parent_row_index(grid[:, [parents.index(p) for p in others]], other_cards) if others \
        else torch.zeros(grid.shape[0], dtype=torch.long)
    a_index = grid[:, parents.index(action)]
    lo_t = torch.as_tensor(lo, dtype=N.dtype)[u_index, a_index]
    hi_t = torch.as_tensor(hi, dtype=N.dtype)[u_index, a_index]
    action_taken = torch.as_tensor(N.sum(dim=(1, 3)).cpu() > 0)  # [n_u, n_a]: some regime took a
    unseen = ~action_taken[u_index, a_index] | torch.as_tensor(refuted)[u_index]
    result = IntervalSet(level=1.0 - delta, guarantee="pac_under_model", unseen=unseen.to(device),
                         lower=lo_t.to(device), upper=hi_t.to(device))
    return ConfoundedSet(
        node=node, action=action, parents=parents, set=result, others=others, method_used=used,
        refuted=torch.as_tensor(refuted), counts=visits, n_regimes=K, delta=delta, sampling=sampling,
        interval_method=interval_method, declared_unconfounded=tuple(declared), notes=notes,
    )


class _RegimeVar:
    """A minimal discrete variable spec for the regime column."""

    is_discrete = True

    def __init__(self, k: int) -> None:
        self.cardinality = k


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateDecision:
    """Whether a node's action rows must be treated as possibly confounded, and why."""

    use_confounded_rows: bool
    reason: str


def confounding_gate(declared_unconfounded: bool, invariance=None, alpha: float = 0.05) -> GateDecision:
    """Possibly confounded unless declared unconfounded; a rejected invariance test overrides.

    ``invariance`` is an optional :class:`nbn.scm.DiagnosticResult` of
    ``context_invariance_test`` (regime as context).  A rejection raises
    suspicion and forces the confounded rows even for a declared node; a
    non-rejection never certifies a node (no test for confounding exists).
    """
    rejected = invariance is not None and bool(invariance.rejected(alpha))
    if rejected:
        return GateDecision(True, f"context invariance rejected (p = {invariance.p_value:.3g} < {alpha}): "
                                  "confounding or a changed mechanism; use the confounded rows")
    if declared_unconfounded:
        return GateDecision(False, "declared unconfounded (e.g. randomised loggers); the invariance test "
                                   "did not reject, which does not prove it")
    return GateDecision(True, "not declared unconfounded: rows are possibly confounded")


__all__ = [
    "ConfoundedSet", "GateDecision", "IVProgram", "LPFailure", "ProgramTooLarge", "RefutedAssumption",
    "RegimeBox", "confounded_interval_set", "confounding_gate", "natural_rows", "rows_from_box",
    "sensitivity_row", "sensitivity_row_box",
]
