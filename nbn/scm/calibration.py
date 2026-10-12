"""Calibrated uncertainty for continuous structural mechanisms.

Neural mechanisms fit ``P(X | pa)`` only approximately.  The tools here put a
finite-sample statement on top of *any* fitted 1-D continuous structural
mechanism that exposes ``cdf`` and ``quantile`` (:class:`AdditiveNoiseMechanism`,
:class:`FlowNoiseMechanism`), using held-out *calibration* data that were not
used to fit the mechanism.  Each result says exactly which guarantee it has.

:func:`pit_band_set` — a band on conditional probabilities
    The probability integral transform ``W = F_hat(X | pa)`` of held-out rows
    is i.i.d. with some CDF ``G``.  The Dvoretzky–Kiefer–Wolfowitz inequality
    with Massart's constant bounds ``G`` uniformly: ``sup_t |G_hat(t) - G(t)|
    <= eps`` with probability ``>= 1 - delta``, ``eps = sqrt(log(2/delta) /
    (2 n))`` (Massart, Ann. Probab. 18(3), 1990).  If ``W`` is independent of
    the parents — the fitted model's noise is independent of its inputs, which
    :func:`nbn.scm.noise_independence_test` checks — then
    ``P(X <= x | pa) = G(F_hat(x | pa))`` for every ``pa``, so the band bounds
    every conditional probability at once (guarantee ``"pac_under_model"``).
    The band also *recalibrates*: when the fit is perfect ``G`` is the
    identity; otherwise ``G_hat`` corrects it.

:func:`conformal_intervals` — prediction intervals
    Split conformal prediction: intervals for ``X | pa`` whose coverage is at
    least ``1 - alpha`` *on average over* ``pa`` (marginal, not conditional),
    in finite samples, for exchangeable calibration and test rows, whatever
    the model (Lei, G'Sell, Rinaldo, Tibshirani & Wasserman, *Distribution-Free
    Predictive Inference for Regression*, JASA 113(523), 2018).  ``method="cqr"``
    conformalises the mechanism's own conditional quantiles, so the intervals
    adapt to the parents (Romano, Patterson & Candès, *Conformalized Quantile
    Regression*, NeurIPS 2019); ``method="residual"`` uses ``|x - median|``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

import torch

from nbn.scm.mechanisms import StructuralMechanism
from nbn.utils.device import to_device


def _mechanism(model, node: str) -> StructuralMechanism:
    mech = model.mechanisms[node] if node in model.mechanisms else None  # noqa: SIM401
    if not (isinstance(mech, StructuralMechanism) and hasattr(mech, "cdf") and hasattr(mech, "quantile")):
        raise TypeError(
            f"'{node}' needs a fitted continuous structural mechanism with cdf/quantile "
            f"(AdditiveNoiseMechanism or FlowNoiseMechanism), got {type(mech).__name__}."
        )
    return mech


@torch.no_grad()
def pit_values(model, data: Mapping[str, torch.Tensor], node: str) -> torch.Tensor:
    """``F_hat(x | pa)`` for every row, ``[N]`` (the probability integral transform)."""
    data = to_device(dict(data), model.device)
    return _mechanism(model, node).cdf(data[node], model.parent_values(data, node))


@dataclass(kw_only=True)
class PITBandSet:
    """A uniform band on the calibration map ``G`` of a fitted continuous mechanism.

    ``P(X <= x | pa)`` lies in ``[G_hat(F_hat(x | pa)) - eps, G_hat(F_hat(x | pa)) + eps]``
    for every ``x`` and ``pa`` at once, with probability ``>= level``, provided
    the calibration rows were not used for fitting and the PIT is independent
    of the parents (``guarantee="pac_under_model"``).
    """

    sorted_pit: torch.Tensor  # [n] held-out PIT values, ascending
    epsilon: float
    level: float
    guarantee: str = "pac_under_model"

    def calibrated(self, cdf_values: torch.Tensor) -> torch.Tensor:
        """``G_hat(F_hat)``: the empirical CDF of the held-out PIT at ``cdf_values``."""
        pit = self.sorted_pit.to(cdf_values)
        return torch.searchsorted(pit, cdf_values.contiguous(), right=True).to(cdf_values.dtype) / pit.numel()

    def bounds_from_cdf(self, cdf_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bounds on ``P(X <= x | pa)`` given ``F_hat(x | pa)`` (any shape)."""
        g = self.calibrated(cdf_values)
        return (g - self.epsilon).clamp(0.0, 1.0), (g + self.epsilon).clamp(0.0, 1.0)

    def cdf_bounds(self, mechanism, x: torch.Tensor, parents: torch.Tensor | None):
        """Bounds on ``P(X <= x | pa)`` for a batch of rows, ``[B]`` each."""
        with torch.no_grad():
            return self.bounds_from_cdf(mechanism.cdf(x, parents))

    def interval_probability_bounds(self, mechanism, a: torch.Tensor, b: torch.Tensor,
                                    parents: torch.Tensor | None):
        """Bounds on ``P(a < X <= b | pa)``: each end is known up to ``eps``."""
        lo_b, hi_b = self.cdf_bounds(mechanism, b, parents)
        lo_a, hi_a = self.cdf_bounds(mechanism, a, parents)
        return (lo_b - hi_a).clamp(0.0, 1.0), (hi_b - lo_a).clamp(0.0, 1.0)


def pit_band_set(model, calibration_data: Mapping[str, torch.Tensor], node: str,
                 delta: float = 0.05) -> PITBandSet:
    """DKW–Massart band on the PIT of held-out ``calibration_data`` (not the fit data)."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1), got {delta}.")
    pit = pit_values(model, calibration_data, node)
    n = pit.numel()
    if n == 0:
        raise ValueError("pit_band_set needs at least one calibration row.")
    return PITBandSet(
        sorted_pit=pit.sort().values,
        epsilon=math.sqrt(math.log(2.0 / delta) / (2.0 * n)),
        level=1.0 - delta,
    )


@dataclass(kw_only=True)
class ConformalIntervals:
    """Split-conformal prediction intervals for one continuous node.

    ``interval(mechanism, parents)`` covers a new exchangeable row with
    probability ``>= level`` *marginally* (averaged over the parents).
    ``correction`` is the conformal quantile of the calibration scores; it is
    ``inf`` when there are too few calibration rows for the requested level.
    """

    method: Literal["cqr", "residual"]
    alpha: float
    correction: float
    n_calibration: int
    level: float
    guarantee: str = "marginal"

    def interval(self, mechanism, parents: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            lo, hi = _base_interval(mechanism, parents, self.method, self.alpha)
        return lo - self.correction, hi + self.correction


def conformal_intervals(model, calibration_data: Mapping[str, torch.Tensor], node: str,
                        alpha: float = 0.1,
                        method: Literal["cqr", "residual"] = "cqr") -> ConformalIntervals:
    """Calibrate intervals of level ``1 - alpha`` on held-out ``calibration_data``.

    Scores: ``max(q_lo - x, x - q_hi)`` for ``"cqr"`` (the mechanism's
    ``alpha/2`` and ``1 - alpha/2`` conditional quantiles), ``|x - median|`` for
    ``"residual"``.  The correction is the ``ceil((n + 1)(1 - alpha))``-th
    smallest score, which gives finite-sample marginal coverage ``>= 1 - alpha``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
    if method not in ("cqr", "residual"):
        raise ValueError(f"method must be 'cqr' or 'residual', got {method!r}.")
    data = to_device(dict(calibration_data), model.device)
    mech = _mechanism(model, node)
    with torch.no_grad():
        x = data[node].reshape(-1).to(torch.float32)
        lo, hi = _base_interval(mech, model.parent_values(data, node), method, alpha)
        scores = torch.maximum(lo - x, x - hi)
    n = scores.numel()
    rank = math.ceil((n + 1) * (1.0 - alpha))
    correction = math.inf if rank > n else float(scores.sort().values[rank - 1])
    return ConformalIntervals(method=method, alpha=alpha, correction=correction,
                              n_calibration=n, level=1.0 - alpha)


def _base_interval(mechanism, parents, method: str, alpha: float):
    if method == "cqr":
        return mechanism.quantile(alpha / 2.0, parents), mechanism.quantile(1.0 - alpha / 2.0, parents)
    median = mechanism.quantile(0.5, parents)
    return median, median
