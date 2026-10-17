"""Calibrated uncertainty for continuous structural mechanisms.

Neural mechanisms fit ``P(X | pa)`` only approximately.  The tools here put a
finite-sample statement on top of *any* fitted 1-D continuous structural
mechanism with an exact ``cdf`` and ``quantile`` (:class:`AdditiveNoiseMechanism`,
:class:`FlowNoiseMechanism`), using held-out *calibration* rows that were not
used to fit it.  Every result records which mechanism it was calibrated for
and refuses to be used with another (or a refitted or edited) one; a
pickled or copied model keeps its calibration.

Rows must be independent and identically distributed (exchangeable for the
conformal part).  Transitions from one episode are not: pass ``groups``
(episode ids) and only one randomly chosen row per group is kept.

:func:`pit_band_set` — a band on conditional probabilities
    Apply the fitted CDF to each held-out row: ``W = F_hat(X | pa)`` (the
    probability integral transform; the CDF must be continuous and strictly
    increasing, as it is for flows and Gaussian / mixture noise).  The ``W``
    are i.i.d. with some CDF ``G``.  The Dvoretzky–Kiefer–Wolfowitz
    inequality with Massart's constant gives ``sup_t |G_hat(t) - G(t)| <= eps``
    with probability ``>= 1 - delta``, ``eps = sqrt(log(2/delta) / (2 n))``
    (Massart, Ann. Probab. 18(3), 1990).

    * Unconditionally, this bounds the *marginal* law of ``W``: the
      probability of the parent-dependent regions ``{F_hat(X | pa) <= t}``.
    * **If** ``W`` is independent of the parents (the fitted model is
      correctly calibrated in every region of the parent space), then
      ``P(X <= x | pa) = G(F_hat(x | pa))`` for every ``pa``, and the band
      bounds every conditional probability at once.  This is an assumption
      (``guarantee="pac_if_pit_independent"``): :func:`nbn.scm.noise_independence_test`
      and :func:`nbn.scm.calibration_by_bins_test` can only *detect* a
      violation, and should be run on a separate split.

    The band also recalibrates: ``G_hat`` corrects a miscalibrated fit.

:func:`pit_band_set` for discrete children
    Uses the randomized PIT ``W = F_hat(x - 1 | pa) + V p_hat(x | pa)``
    (Dunn & Smyth, J. Comput. Graph. Stat. 5(3), 1996).  Since
    ``{X <= k} = {W <= F_hat(k | pa)}``, the same band bounds
    ``P(X <= k | pa)`` under the same assumption; class order is the index
    order (meaningful for ordered classes).

:func:`conformal_sets` — prediction sets for discrete children
    Split conformal with a cumulative (adaptive) score: sort classes by
    decreasing ``p_hat``; the score of class ``y`` is the mass of the classes
    ranked above it, plus ``u`` times its own probability (Romano, Sesia &
    Candès, *Classification with Valid and Adaptive Coverage*, NeurIPS 2020).
    ``score="exclusive"`` (``u = 0``, the default), ``"inclusive"``
    (``u = 1``; Kochenderfer et al., *Algorithms for Validation*, Ex. 12.4,
    p. 342) or ``"randomized"`` (``u ~ U(0, 1)`` per row).  Any of them gives
    marginal coverage ``>= 1 - alpha`` under exchangeability (*ibid.*,
    §12.2.3, p. 340).

:func:`conformal_intervals` — prediction intervals
    Split conformal prediction: intervals for ``X | pa`` that cover a new
    exchangeable row with probability ``>= 1 - alpha`` *on average over the
    parents* (marginal, not conditional), in finite samples, whatever the
    model (Lei, G'Sell, Rinaldo, Tibshirani & Wasserman, JASA 113(523), 2018).
    ``method="cqr"`` conformalises the mechanism's own conditional quantiles,
    so the intervals adapt to the parents (Romano, Patterson & Candès,
    *Conformalized Quantile Regression*, NeurIPS 2019); ``method="residual"``
    uses ``|x - median|``.  For a discrete-noise counterpart see
    :func:`nbn.scm.cdf_band_set` (additive noise, band on the noise CDF).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Literal, Mapping, get_args

import torch

from nbn.scm._stats import require_cpu_generator
from nbn.scm.mechanisms import StructuralMechanism
from nbn.utils.device import supports_float64, to_device, to_device_dtype

Guarantee = Literal["pac_if_pit_independent", "marginal"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mechanism(model, node: str, need: tuple[str, ...] = ("cdf", "quantile")) -> StructuralMechanism:
    mech = model.mechanisms[node] if node in model.mechanisms else None  # noqa: SIM401
    if not (isinstance(mech, StructuralMechanism) and all(hasattr(mech, n) for n in need)):
        raise TypeError(
            f"'{node}' needs a fitted structural mechanism with {'/'.join(need)} "
            f"(AdditiveNoiseMechanism, FlowNoiseMechanism or, for discrete children where "
            f"applicable, StructuralNeuralCategorical / StructuralCategoricalTable); "
            f"got {type(mech).__name__}."
        )
    return mech


def _fingerprint(mechanism) -> str:
    """A digest of what the mechanism computes: its class, its scalar settings and its weights.

    Two mechanisms with the same digest give the same answers, so a pickled or
    copied model keeps its calibration, while a refit, ``load_state_dict``,
    in-place edit, swapped noise law or changed setting (e.g. ``min_scale``)
    does not.  Values are hashed in float64 on the CPU, so moving the model
    between devices or float32 <-> float64 keeps the digest.
    """
    digest = hashlib.sha256()
    for name, module in mechanism.named_modules():
        digest.update(f"{name}:{type(module).__module__}.{type(module).__qualname__};".encode())
        for key, value in sorted(vars(module).items()):
            if key != "training" and isinstance(value, (bool, int, float, str, type(None))):
                digest.update(f"{key}={value!r};".encode())
    for key, tensor in mechanism.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            continue
        t = tensor.detach()
        t = to_device_dtype(t, "cpu", torch.float64) if t.is_floating_point() else t.cpu()
        digest.update(f"{key}:{tuple(t.shape)};".encode())
        digest.update(t.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _calibration_rows(
    model, data: Mapping[str, torch.Tensor], node: str,
    groups: torch.Tensor | None, generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Finite ``(x [N], parents [N, D_pa] | None)`` on the model's device, one row per group if asked."""
    require_cpu_generator(generator, "calibration (generator=)")
    data = to_device(dict(data), model.device)
    x = data[node]
    parents = model.parent_values(data, node)
    if x.shape[0] == 0:
        raise ValueError(f"no calibration rows for '{node}'.")
    if groups is not None:
        if groups.numel() != x.shape[0]:
            raise ValueError(
                f"groups has {groups.numel()} entries for {x.shape[0]} calibration rows; "
                "pass one episode id per row."
            )
        keep = _one_row_per_group(groups.reshape(-1).cpu(), generator).to(x.device)
        x = x[keep]
        parents = None if parents is None else parents[keep]
    finite = torch.isfinite(x.reshape(x.shape[0], -1)).all(-1)
    if parents is not None:
        finite &= torch.isfinite(parents).all(-1)
    if not bool(finite.all()):
        raise ValueError(
            f"{int((~finite).sum())} calibration rows of '{node}' contain NaN or inf; "
            "drop them before calibrating."
        )
    return x, parents


def _one_row_per_group(groups: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """Indices of one uniformly chosen row per distinct group id (sorted)."""
    n = groups.numel()
    order = torch.randperm(n, generator=generator)
    _, group_of = torch.unique(groups[order], return_inverse=True)
    # The first row of each group in a random order is a uniform choice within the group.
    first = torch.full((int(group_of.max()) + 1,), n, dtype=torch.long)
    first = first.scatter_reduce(0, group_of, torch.arange(n), reduce="amin")
    return order[first].sort().values


def _check_bound(stored: str, mechanism) -> None:
    if stored != _fingerprint(mechanism):
        raise ValueError(
            "This calibration result belongs to another mechanism, or the mechanism was "
            "modified (refitted, warm-started, loaded) after calibration; calibrate again."
        )


# ---------------------------------------------------------------------------
# PIT band
# ---------------------------------------------------------------------------


@torch.no_grad()
def pit_values(model, data: Mapping[str, torch.Tensor], node: str,
               groups: torch.Tensor | None = None,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """The probability integral transform of every (kept) row, ``[N]``.

    ``F_hat(x | pa)`` for continuous children; for discrete children the
    randomized version ``F_hat(x - 1 | pa) + V p_hat(x | pa)``, ``V ~ U(0, 1)``
    (drawn with ``generator``), which is exactly uniform under a correct kernel.
    """
    x, parents = _calibration_rows(model, data, node, groups, generator)
    mech = _mechanism(model, node, need=("cdf",))
    if getattr(mech, "is_discrete", False):  # W is exactly U(0, 1) under a correct kernel
        return mech.randomized_pit(x, parents, generator)
    return mech.cdf(x, parents)


@dataclass(kw_only=True)
class PITBandSet:
    """A uniform band on the calibration map ``G`` of one fitted continuous mechanism.

    ``cdf_bounds`` returns ``[G_hat(F_hat(x | pa)^-) - eps, G_hat(F_hat(x | pa)) + eps]``
    (strict count for the lower end, see :meth:`bounds_from_cdf`).
    With probability ``>= level`` the band contains ``G`` everywhere; it bounds
    ``P(X <= x | pa)`` for every ``pa`` *if* the PIT is independent of the
    parents (see the module docstring).
    """

    node: str
    sorted_pit: torch.Tensor  # [n] held-out PIT values, ascending
    epsilon: float
    level: float
    guarantee: Guarantee = "pac_if_pit_independent"
    _bound_to: str = field(default="", repr=False)

    def calibrated(self, cdf_values: torch.Tensor) -> torch.Tensor:
        """``G_hat(F_hat)``: the empirical CDF of the held-out PIT (NaN stays NaN)."""
        pit = self.sorted_pit.to(cdf_values)
        g = torch.searchsorted(pit, cdf_values.contiguous(), right=True).to(cdf_values.dtype) / pit.numel()
        return torch.where(torch.isnan(cdf_values), cdf_values, g)

    def bounds_from_cdf(self, cdf_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bounds on ``P(X <= x | pa)`` given ``F_hat(x | pa)`` (any shape).

        ``{W < F_hat(x)} ⊆ {X <= x} ⊆ {W <= F_hat(x)}`` for any non-decreasing
        ``F_hat`` (also the randomized PIT of a discrete node), so the lower
        end uses the *strict* empirical count and the upper end the
        inclusive one; DKW bounds both limits at once.  (Using the inclusive
        count for both ends was unsound where ``F_hat`` is flat, e.g. a
        Gaussian CDF that rounds to 1 far in the tail, or a class with
        ``p_hat`` ~ 0: the band then excluded the truth.)
        """
        pit = self.sorted_pit.to(cdf_values)
        v = cdf_values.contiguous()
        n = pit.numel()
        g_strict = torch.searchsorted(pit, v, right=False).to(cdf_values.dtype) / n
        g_incl = torch.searchsorted(pit, v, right=True).to(cdf_values.dtype) / n
        nan = torch.isnan(cdf_values)
        lo = torch.where(nan, cdf_values, (g_strict - self.epsilon).clamp(0.0, 1.0))
        up = torch.where(nan, cdf_values, (g_incl + self.epsilon).clamp(0.0, 1.0))
        return lo, up

    def cdf_bounds(self, mechanism, x: torch.Tensor, parents: torch.Tensor | None):
        """Bounds on ``P(X <= x | pa)`` for a batch of rows, ``[B]`` each."""
        _check_bound(self._bound_to, mechanism)
        with torch.no_grad():
            return self.bounds_from_cdf(mechanism.cdf(x, parents))

    def interval_probability_bounds(self, mechanism, a: torch.Tensor, b: torch.Tensor,
                                    parents: torch.Tensor | None):
        """Bounds on ``P(a < X <= b | pa)``: each end is known up to ``eps``."""
        lo_b, hi_b = self.cdf_bounds(mechanism, b, parents)
        lo_a, hi_a = self.cdf_bounds(mechanism, a, parents)
        return (lo_b - hi_a).clamp(0.0, 1.0), (hi_b - lo_a).clamp(0.0, 1.0)


def pit_band_set(model, calibration_data: Mapping[str, torch.Tensor], node: str,
                 delta: float = 0.05, groups: torch.Tensor | None = None,
                 generator: torch.Generator | None = None) -> PITBandSet:
    """DKW–Massart band on the PIT of held-out ``calibration_data`` (not the fit data)."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1), got {delta}.")
    pit = pit_values(model, calibration_data, node, groups, generator)
    n = pit.numel()
    if n == 0:
        raise ValueError("pit_band_set needs at least one calibration row.")
    return PITBandSet(
        node=node,
        sorted_pit=pit.sort().values,
        epsilon=math.sqrt(math.log(2.0 / delta) / (2.0 * n)),
        level=1.0 - delta,
        _bound_to=_fingerprint(_mechanism(model, node, need=("cdf",))),
    )


# ---------------------------------------------------------------------------
# Conformal intervals
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class ConformalIntervals:
    """Split-conformal prediction intervals for one continuous node.

    ``interval(mechanism, parents)`` covers a new exchangeable row with
    probability ``>= level`` *marginally* (averaged over the parents).
    ``correction`` is the conformal quantile of the calibration scores; it is
    ``inf`` when there are too few calibration rows for the requested level.
    """

    node: str
    method: Literal["cqr", "residual"]
    alpha: float
    correction: float
    n_calibration: int
    level: float
    guarantee: Guarantee = "marginal"
    _bound_to: str = field(default="", repr=False)

    def interval(self, mechanism, parents: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        _check_bound(self._bound_to, mechanism)
        with torch.no_grad():
            lo, hi = _base_interval(mechanism, parents, self.method, self.alpha)
        return lo - self.correction, hi + self.correction


def conformal_intervals(model, calibration_data: Mapping[str, torch.Tensor], node: str,
                        alpha: float = 0.1,
                        method: Literal["cqr", "residual"] = "cqr",
                        groups: torch.Tensor | None = None,
                        generator: torch.Generator | None = None) -> ConformalIntervals:
    """Calibrate intervals of level ``1 - alpha`` on held-out ``calibration_data``.

    Scores: ``max(q_lo - x, x - q_hi)`` for ``"cqr"`` (the mechanism's
    ``alpha/2`` and ``1 - alpha/2`` conditional quantiles), ``|x - median|`` for
    ``"residual"``.  The correction is the ``ceil((n + 1)(1 - alpha))``-th
    smallest score — the ``ceil((n+1)(1-alpha))/n`` empirical quantile — which
    gives finite-sample marginal coverage ``>= 1 - alpha``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
    if method not in ("cqr", "residual"):
        raise ValueError(f"method must be 'cqr' or 'residual', got {method!r}.")
    mech = _mechanism(model, node)
    x, parents = _calibration_rows(model, calibration_data, node, groups, generator)
    with torch.no_grad():
        lo, hi = _base_interval(mech, parents, method, alpha)
        x = x.reshape(-1).to(lo.dtype)
        scores = torch.maximum(lo - x, x - hi)
    n = scores.numel()
    rank = math.ceil((n + 1) * (1.0 - alpha))
    correction = math.inf if rank > n else float(scores.sort().values[rank - 1])
    return ConformalIntervals(node=node, method=method, alpha=alpha, correction=correction,
                              n_calibration=n, level=1.0 - alpha, _bound_to=_fingerprint(mech))


def _base_interval(mechanism, parents, method: str, alpha: float):
    if method == "cqr":
        return mechanism.quantile(alpha / 2.0, parents), mechanism.quantile(1.0 - alpha / 2.0, parents)
    median = mechanism.quantile(0.5, parents)
    return median, median


# ---------------------------------------------------------------------------
# Conformal prediction sets (discrete children)
# ---------------------------------------------------------------------------


ConformalScore = Literal["exclusive", "inclusive", "randomized"]
CONFORMAL_SCORES: tuple[str, ...] = get_args(ConformalScore)


def check_conformal_score(score: str) -> None:
    """Raise ``ValueError`` unless ``score`` is one of :data:`CONFORMAL_SCORES`."""
    if score not in CONFORMAL_SCORES:
        raise ValueError(f"score must be one of {CONFORMAL_SCORES}; got {score!r}.")


@dataclass(kw_only=True)
class ConformalSets:
    """Split-conformal prediction sets for one discrete node (cumulative score).

    ``sets(mechanism, parents)`` returns a ``[B, K]`` boolean mask; the set
    contains a new exchangeable row's class with probability ``>= level``
    *marginally*.  ``threshold`` is ``inf`` (every class) when there are too
    few calibration rows for the requested level.

    ``score`` is the one used at calibration (see :func:`conformal_sets`).
    Discrete scores tie, so coverage can exceed ``level`` by more than
    ``1/(n+1)``: only the lower bound is guaranteed (``"randomized"`` breaks
    the ties).  Float noise: the probabilities of one row can differ by about
    one unit in the last place between batch sizes.  Membership is
    ``score <= threshold`` up to ``K`` such units (a score sums up to ``K``
    probabilities), so a class at the threshold is not dropped by that noise.
    One case remains batch-dependent: two classes of a row whose
    probabilities differ by about one unit can swap order, which moves a whole
    probability between their scores (measure zero for continuous outputs).
    """

    node: str
    alpha: float
    threshold: float
    n_calibration: int
    level: float
    score: ConformalScore = "exclusive"
    guarantee: Guarantee = "marginal"
    _bound_to: str = field(default="", repr=False)

    def __setstate__(self, state: dict) -> None:
        # Sets pickled before the ``score`` field existed were calibrated with the inclusive score.
        state.setdefault("score", "inclusive")
        self.__dict__.update(state)

    def sets(self, mechanism, parents: torch.Tensor | None, batch: int | None = None,
             generator: torch.Generator | None = None) -> torch.Tensor:
        """``[B, K]`` membership mask.

        For ``score="randomized"`` one ``u`` is drawn per row, in row order, from
        ``generator`` (the global RNG if None): the same rows give the same sets
        only with a generator in the same state.  Pass a freshly seeded generator
        to every call whose sets must agree.
        """
        _check_bound(self._bound_to, mechanism)
        with torch.no_grad():
            probs = mechanism.probs(parents, batch)
            u = _score_u(self.score, probs.shape[0], generator)
            scores = _cumulative_scores(probs, u)
            bound = _upper_threshold(self.threshold, probs.dtype, probs.shape[-1])
            return (scores <= bound).to(probs.device)


def conformal_sets(model, calibration_data: Mapping[str, torch.Tensor], node: str,
                   alpha: float = 0.1, groups: torch.Tensor | None = None,
                   generator: torch.Generator | None = None,
                   score: ConformalScore = "exclusive") -> ConformalSets:
    """Calibrate prediction sets of level ``1 - alpha`` for a discrete node on held-out rows.

    Classes are ranked by decreasing ``p_hat``; the score of class ``y`` is the
    mass ranked before it plus ``u * p_hat(y)``:

    ``"exclusive"`` (default, ``u = 0``)
        The mass of the classes with a strictly larger probability: the most
        probable classes score exactly ``0`` and are always in the set, so a
        set is never empty, a confident row gets one class whatever the float
        precision, and exactly tied top classes are all kept.
        Ties at ``0`` make it conservative (coverage above ``1 - alpha``) for
        very accurate classifiers.
    ``"inclusive"`` (``u = 1``)
        The cumulative mass up to and including ``y`` in a stable descending
        sort, the score NBN used before (Kochenderfer et al., *Algorithms for
        Validation*, Ex. 12.4, p. 342).  When probabilities saturate, a row
        more confident than the threshold gets an empty set (NBN #322); prefer
        ``"exclusive"`` or ``"randomized"``.
    ``"randomized"`` (``u ~ U(0, 1)``, one draw per row)
        The randomized adaptive score of Romano, Sesia & Candès (NeurIPS
        2020): ties are broken, so coverage is close to ``1 - alpha`` from
        above, and sets are random (pass ``generator`` for reproducibility;
        :meth:`ConformalSets.sets` takes one too).  A set can be empty.

    Threshold: the ``ceil((n + 1)(1 - alpha))``-th smallest calibration
    score.  A new row's set is every class whose score is at most the
    threshold.  Every score gives marginal coverage ``>= 1 - alpha`` for
    exchangeable rows (*ibid.*, §12.2.3, p. 340); none gives conditional
    coverage.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
    check_conformal_score(score)
    mech = _mechanism(model, node, need=("probs", "class_index"))
    x, parents = _calibration_rows(model, calibration_data, node, groups, generator)
    with torch.no_grad():
        idx = mech.class_index(x)
        probs = mech.probs(parents, idx.shape[0])
        u = _score_u(score, probs.shape[0], generator)
        scores = _cumulative_scores(probs, u)
        scores = scores.gather(-1, idx.reshape(-1, 1).to(scores.device)).squeeze(-1)
    n = scores.numel()
    rank = math.ceil((n + 1) * (1.0 - alpha))
    threshold = math.inf if rank > n else float(scores.sort().values[rank - 1])
    return ConformalSets(node=node, alpha=alpha, threshold=threshold, n_calibration=n,
                         level=1.0 - alpha, score=score, _bound_to=_fingerprint(mech))


def _score_u(score: str, n_rows: int, generator: torch.Generator | None) -> torch.Tensor | float:
    """The weight ``u`` of the observed class's own probability in the score."""
    check_conformal_score(score)
    if score == "exclusive":
        return 0.0
    if score == "inclusive":
        return 1.0
    require_cpu_generator(generator, 'conformal_sets(score="randomized")')
    return torch.rand(n_rows, 1, generator=generator, dtype=torch.float64)


def _cumulative_scores(probs: torch.Tensor, u: torch.Tensor | float = 1.0) -> torch.Tensor:
    """Cumulative scores ``s[b, k]`` (see :func:`conformal_sets`).

    ``u = 0`` (exclusive): the mass of the classes with a strictly larger
    probability, by exact comparison, so exactly tied classes share a score
    (a uniform row scores 0 everywhere) and near-ties are ordered by value.
    ``u > 0`` (inclusive, randomized): the mass ranked before ``k`` in a
    stable descending sort plus ``u * p_k``, the previous NBN definition
    (exact ties broken by class index).  Float64 (on the CPU for MPS, which
    has no float64), probabilities normalised to sum to one; ``O(B K log K)``
    time and ``O(B K)`` memory.  ``u`` is a float or a ``[B, 1]`` tensor.
    """
    device = probs.device if supports_float64(probs.device) else torch.device("cpu")
    p = probs.detach().to(device=device, dtype=torch.float64)
    p = p / p.sum(-1, keepdim=True)
    sorted_p, order = torch.sort(p, dim=-1, descending=True, stable=True)
    before = torch.cat([torch.zeros_like(sorted_p[..., :1]), sorted_p.cumsum(-1)[..., :-1]], -1)
    if isinstance(u, float) and u == 0.0:
        # mass strictly above each value: position of its first equal in the descending sort
        first = torch.searchsorted(-sorted_p.contiguous(), -p.contiguous(), right=False)
        return before.gather(-1, first)
    u_t = u if isinstance(u, float) else u.to(device)
    scores_sorted = before + u_t * sorted_p
    if isinstance(u, float) and u == 1.0:
        scores_sorted[..., -1] = 1.0                                  # no float drift at the end
    return torch.empty_like(p).scatter_(-1, order, scores_sorted)


def _upper_threshold(threshold: float, dtype: torch.dtype, n_classes: int = 1) -> float:
    """``threshold`` plus ``n_classes`` units in the last place of ``dtype`` at ``threshold``.

    A cumulative score sums up to ``K`` probabilities, each of which can move
    by about one unit between batch sizes; the margin keeps such a class in
    the set.  Classes within the margin of the threshold are kept (they cannot
    be told apart from noise); at ``threshold = 0`` the margin is negligible.
    """
    if math.isinf(threshold):
        return threshold
    dt = dtype if dtype.is_floating_point else torch.float32
    t = torch.tensor(abs(threshold), dtype=dt)
    spacing = float(torch.nextafter(t, torch.tensor(math.inf, dtype=dt)).double() - t.double())
    return threshold + max(1, int(n_classes)) * spacing
