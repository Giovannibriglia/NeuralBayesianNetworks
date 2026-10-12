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
    Split conformal with the cumulative (adaptive) score: sort classes by
    decreasing ``p_hat`` and add the probabilities up to and including the
    observed class (Romano, Sesia & Candès, *Classification with Valid and
    Adaptive Coverage*, NeurIPS 2020; Kochenderfer et al., *Algorithms for
    Validation*, Ex. 12.4, p. 342).  Marginal coverage ``>= 1 - alpha``.

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
from typing import Literal, Mapping

import torch

from nbn.scm._stats import require_cpu_generator
from nbn.scm.mechanisms import StructuralMechanism
from nbn.utils.device import to_device, to_device_dtype

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
    require_cpu_generator(generator, "calibration with groups=")
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
    if hasattr(mech, "randomized_pit"):  # discrete child: W is exactly U(0, 1) under a correct kernel
        return mech.randomized_pit(x, parents, generator)
    return mech.cdf(x, parents)


@dataclass(kw_only=True)
class PITBandSet:
    """A uniform band on the calibration map ``G`` of one fitted continuous mechanism.

    ``cdf_bounds`` returns ``[G_hat(F_hat(x | pa)) - eps, G_hat(F_hat(x | pa)) + eps]``.
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
        """Bounds on ``P(X <= x | pa)`` given ``F_hat(x | pa)`` (any shape)."""
        g = self.calibrated(cdf_values)
        return (g - self.epsilon).clamp(0.0, 1.0), (g + self.epsilon).clamp(0.0, 1.0)

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


@dataclass(kw_only=True)
class ConformalSets:
    """Split-conformal prediction sets for one discrete node (cumulative score).

    ``sets(mechanism, parents)`` returns a ``[B, K]`` boolean mask; the set
    contains a new exchangeable row's class with probability ``>= level``
    *marginally*.  ``threshold`` is ``inf`` (every class) when there are too
    few calibration rows for the requested level.
    """

    node: str
    alpha: float
    threshold: float
    n_calibration: int
    level: float
    guarantee: Guarantee = "marginal"
    _bound_to: str = field(default="", repr=False)

    def sets(self, mechanism, parents: torch.Tensor | None, batch: int | None = None) -> torch.Tensor:
        _check_bound(self._bound_to, mechanism)
        with torch.no_grad():
            probs = mechanism.probs(parents, batch)
            return _cumulative_scores(probs) <= self.threshold


def conformal_sets(model, calibration_data: Mapping[str, torch.Tensor], node: str,
                   alpha: float = 0.1, groups: torch.Tensor | None = None,
                   generator: torch.Generator | None = None) -> ConformalSets:
    """Calibrate prediction sets of level ``1 - alpha`` for a discrete node on held-out rows.

    Score of a row: the total probability of the classes ranked (by
    decreasing ``p_hat``) up to and including the observed one.  Threshold:
    the ``ceil((n + 1)(1 - alpha))``-th smallest score.  A new row's set is
    every class whose score is at most the threshold.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
    mech = _mechanism(model, node, need=("probs", "class_index"))
    x, parents = _calibration_rows(model, calibration_data, node, groups, generator)
    with torch.no_grad():
        idx = mech.class_index(x)
        scores = _cumulative_scores(mech.probs(parents, idx.shape[0]))
        scores = scores.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    n = scores.numel()
    rank = math.ceil((n + 1) * (1.0 - alpha))
    threshold = math.inf if rank > n else float(scores.sort().values[rank - 1])
    return ConformalSets(node=node, alpha=alpha, threshold=threshold, n_calibration=n,
                         level=1.0 - alpha, _bound_to=_fingerprint(mech))


def _cumulative_scores(probs: torch.Tensor) -> torch.Tensor:
    """``s[b, k]`` = mass of the classes ranked at or above ``k`` (decreasing probability)."""
    order = torch.argsort(probs, dim=-1, descending=True, stable=True)
    cumulative = probs.gather(-1, order).cumsum(-1)
    cumulative = cumulative / cumulative[..., -1:]  # the last class scores exactly 1 (no float drift)
    return torch.empty_like(probs).scatter_(-1, order, cumulative)
