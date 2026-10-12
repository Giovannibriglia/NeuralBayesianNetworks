"""Discrete structural mechanisms from a fitted categorical kernel, and counterfactual bounds.

A categorical mechanism learns ``P(X | pa)``.  Data fix only that kernel, so
a counterfactual ("what would ``X`` have been under ``pa'``, given that we saw
``X = x`` under ``pa``?") depends on how the noise is coupled across parent
values.  This module makes that choice explicit and also gives the answer
that holds for *every* compatible model.

Declared couplings (``coupling=``):

``"gumbel_max"``
    ``X := argmax_k (log p(k | pa) + G_k)``, ``G_k`` i.i.d. standard Gumbel.
    Abduction samples the Gumbels exactly from their posterior given ``X``
    (top-down truncated-Gumbel sampling).  Counterfactually stable (Oberst &
    Sontag, *Counterfactual Off-Policy Evaluation with Gumbel-Max Structural
    Causal Models*, ICML 2019, arXiv:1905.05824); one choice among many that
    fit the same kernel (Lorberbom et al., NeurIPS 2021, arXiv:2111.06888).

``"monotone"``
    ``X := min{k : F(k | pa) >= U}``, ``U ~ Uniform(0, 1)``: the inverse-CDF
    (comonotone) coupling for *ordered* categories, the discrete analogue of
    :class:`nbn.scm.FlowNoiseMechanism`.  Abduction:
    ``U ~ Uniform(F(x - 1 | pa), F(x | pa)]``.

Bounds over all compatible models (:func:`counterfactual_bounds`): with
``p = P(x | pa)`` and ``q = P(. | pa')`` for ``pa' != pa``, the set of
counterfactual rows ``P(X_{pa'} = . | X = x, pa)`` that some SCM attains is
exactly ``{r : sum r = 1, max(0, (p + q_k - 1) / p) <= r_k <= min(p, q_k) / p}``
For two outcomes these are the probability-of-necessity bounds, sharp under
exogeneity (Pearl, *Causality*, 2nd ed., Thm 9.2.11 and Cor. 9.2.12, p. 290);
in MDPs, Lally, Kazemi & Paoletti, *Robust Counterfactual Inference in Markov
Decision Processes* (AAMAS 2026, arXiv:2502.13731).  For ``K > 2`` outcomes
the attainability of every row is our argument: a response-function noise
(Bareinboim, *Causal AI*, pp. 311–313) can realise any coupling ``pi`` of the
two marginals, the row is ``pi(x, .) / p``, and any first row ``r`` with
``0 <= p r <= q`` and ``sum r = 1`` completes to a coupling (the leftover
marginals both have mass ``1 - p``).  For ``pa' = pa`` the answer is the point
``X = x`` (consistency; Bareinboim, *Causal AI*, Lemma 5.3.1, p. 268).

Float note: the monotone CDF is summed in float64 on CPU/CUDA (float32 on
MPS).  Probabilities below the resolution that abduction needs (2 eps of the
neural kernel's dtype; for tables, of the CDF dtype) are refused rather than
mapped to a wrong class: e.g. table cells floored at ``1e-12`` on MPS.
"""
from __future__ import annotations

import copy
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.distributions import Gumbel

from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism
from nbn.mechanisms.parametric.neural_categorical import NeuralCategoricalMechanism
from nbn.scm._stats import require_cpu_generator
from nbn.scm.mechanisms import StructuralMechanism
from nbn.uncertainty.sets import tighten_box_to_simplex
from nbn.utils.batching import ensure_2d
from nbn.utils.device import accum_dtype, to_device_dtype

Coupling = Literal["gumbel_max", "monotone"]
_MARGIN_EPS = 2  # monotone abduction keeps this many float eps away from the CDF jumps
_COUPLINGS = ("gumbel_max", "monotone")


# ===========================================================================
# Counterfactual bounds (pure functions on probability rows)
# ===========================================================================


def counterfactual_bounds(
    p_observed: torch.Tensor,
    q_counterfactual: torch.Tensor,
    x_observed: torch.Tensor,
    same_parents: torch.Tensor | bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sharp bounds on ``P(X_{pa'} = k | X = x, pa)`` over all compatible SCMs.

    ``p_observed`` ``[B, K]`` is ``P(. | pa)``, ``q_counterfactual`` ``[B, K]``
    is ``P(. | pa')``, ``x_observed`` ``[B]`` the observed class indices and
    ``same_parents`` ``[B]`` (or a bool) marks rows with ``pa' = pa``.
    Returns ``(lower, upper)``, each ``[B, K]``: an interval-MDP row.  For
    ``pa' != pa`` every row of the box that sums to one is attained by some
    SCM; for ``pa' = pa`` the answer is the point mass on ``x``.
    """
    p_x = _observed_probability(p_observed, x_observed)
    q = q_counterfactual
    lower = ((p_x + q - 1.0) / p_x).clamp(0.0, 1.0)
    upper = (q / p_x).clamp(0.0, 1.0)
    return _point_where_same(lower, upper, x_observed, same_parents)


def counterfactual_bounds_from_sets(
    p_observed_lower: torch.Tensor,
    q_lower: torch.Tensor,
    q_upper: torch.Tensor,
    x_observed: torch.Tensor,
    same_parents: torch.Tensor | bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-entry hull of :func:`counterfactual_bounds` over a set of kernels.

    The kernels range over a rectangular confidence set: ``P(x | pa)`` at
    least ``p_observed_lower[b, x_b]`` (``[B, K]``, only the observed entry is
    used) and ``P(k | pa')`` in ``[q_lower, q_upper]`` (``[B, K]`` each).
    Both bounds of :func:`counterfactual_bounds` are monotone (the lower one
    increasing in ``p`` and ``q_k``, the upper one decreasing in ``p`` and
    increasing in ``q_k``), so the hull is attained at ``p = p_lower`` and
    ``q_k`` at its interval ends (after the ``q`` box is tightened to the
    simplex, so each end is attained by a distribution in the set).  If
    ``P(x | pa) = 0`` is allowed, the row is the limit ``p -> 0``: upper
    ``1[q_upper > 0]``, lower ``1[q_lower >= 1]``.

    This is a sound, per-entry interval row for an interval MDP: it contains
    every counterfactual row of every kernel in the set, and each end is
    attained when the set is a product of per-row boxes (``pa' != pa``).
    (The hull over a confidence set is our construction; the bounds per
    kernel are the attributed ones above.)
    """
    q_lower, q_upper = tighten_box_to_simplex(q_lower, q_upper)
    if bool((q_lower > q_upper + 1e-12).any()):
        raise ValueError("the q box does not meet the probability simplex in some row: no kernel is in the set.")
    p_x = _observed_probability(p_observed_lower, x_observed, allow_zero=True)
    safe = p_x.clamp_min(torch.finfo(p_x.dtype).tiny)
    lower = ((safe + q_lower - 1.0) / safe).clamp(0.0, 1.0)
    upper = (q_upper / safe).clamp(0.0, 1.0)
    impossible = (p_x <= 0).expand_as(lower)
    lower = torch.where(impossible, (q_lower >= 1.0).to(lower.dtype), lower)
    upper = torch.where(impossible, (q_upper > 0.0).to(upper.dtype), upper)
    return _point_where_same(lower, upper, x_observed, same_parents)


def _observed_probability(p: torch.Tensor, x: torch.Tensor, allow_zero: bool = False) -> torch.Tensor:
    if p.dim() != 2:
        raise ValueError(f"probability rows must be [B, K]; got {tuple(p.shape)}.")
    idx = x.reshape(-1, 1).long().to(p.device)
    p_x = p.gather(-1, idx)  # [B, 1]
    if not allow_zero and bool((p_x <= 0).any()):
        raise ValueError("an observed class has probability 0 under its parents: nothing to condition on.")
    return p_x


def _point_where_same(lower, upper, x, same_parents):
    same = torch.as_tensor(same_parents, device=lower.device, dtype=torch.bool)
    if same.dim() == 0:
        same = same.expand(lower.shape[0])
    point = torch.nn.functional.one_hot(x.reshape(-1).long().to(lower.device), lower.shape[-1])
    point = point.to(lower.dtype)
    same = same.reshape(-1, 1)
    return torch.where(same, point, lower), torch.where(same, point, upper)


# ===========================================================================
# Structural categorical mechanisms
# ===========================================================================


class CouplingNoise(nn.Module):
    """The fixed noise law of a declared coupling (no parameters).

    ``gumbel_max``: ``K`` i.i.d. standard Gumbels; ``monotone``: one
    Uniform(0, 1).  ``sample`` draws on the device and dtype of a zero-size,
    non-persistent anchor buffer that moves with the mechanism.
    """

    def __init__(self, coupling: str, n_classes: int) -> None:
        super().__init__()
        self.coupling = coupling
        self.n_classes = int(n_classes)
        self.register_buffer("_anchor", torch.empty(0), persistent=False)

    @property
    def dim(self) -> int:
        return self.n_classes if self.coupling == "gumbel_max" else 1

    def sample(self, n: int) -> torch.Tensor:
        u = torch.rand(n, self.dim, device=self._anchor.device, dtype=self._anchor.dtype)
        if self.coupling == "gumbel_max":
            return -torch.log(-torch.log(u.clamp_min(torch.finfo(u.dtype).tiny)))
        return u


class _CategoricalStructural(StructuralMechanism):
    """Structural reading of a fitted categorical kernel (mixin; see module docstring)."""

    coupling: str

    def _init_coupling(self, coupling: str) -> None:
        if coupling not in _COUPLINGS:
            raise ValueError(f"coupling must be one of {_COUPLINGS}, got {coupling!r}.")
        self.coupling = coupling
        self.noise = CouplingNoise(coupling, self._k() if self._k() > 0 else 1)

    def with_coupling(self, coupling: Coupling) -> _CategoricalStructural:
        """A copy with the same fitted kernel and another declared coupling."""
        clone = copy.deepcopy(self)
        clone._init_coupling(coupling)
        clone.noise.n_classes = clone._k()
        clone.noise.to(self._kernel_device())
        return clone

    def fit_local(self, *args: Any, **kwargs: Any) -> dict:
        metrics = super().fit_local(*args, **kwargs)  # type: ignore[safe-super]
        self.noise.n_classes = self._k()  # K may be set by the fit (tables)
        self.noise.to(self._kernel_device())
        return metrics

    # ------------------------------------------------------------------
    # Kernel access
    # ------------------------------------------------------------------

    def probs(self, parents: torch.Tensor | None, batch: int | None = None) -> torch.Tensor:
        """``P(. | pa)`` as ``[B, K]`` (``batch`` rows for a root).

        Parents are passed to the kernel as float32 (the neural kernel's
        network runs in float32; a float64 *neural* kernel is not supported).
        """
        if parents is not None and parents.shape[-1] > 0:
            parents = to_device_dtype(ensure_2d(parents), self._kernel_device(), torch.float32)
            if parents.dim() != 2:
                raise ValueError(f"parents must be [B, D_pa]; got {tuple(parents.shape)}.")
        probs = self.forward(parents).probs
        if batch is not None and probs.shape[0] == 1 and batch != 1:
            probs = probs.expand(batch, -1)
        return probs

    def class_index(self, x: torch.Tensor) -> torch.Tensor:
        """Class indices ``[B]`` of labels ``x`` (``[B]`` or ``[B, 1]``, values ``0 .. K-1``)."""
        x = x.reshape(-1).to(self._kernel_device())
        idx = x.long()
        if not bool(((idx.to(x.dtype) == x) & (idx >= 0) & (idx < self._k())).all()):
            raise ValueError(f"x contains values that are not classes 0..{self._k() - 1} of this mechanism.")
        return idx

    @staticmethod
    def class_value(idx: torch.Tensor) -> torch.Tensor:
        """Labels ``[B, 1]`` (float, as NBN stores discrete values) of class indices ``idx``."""
        return idx.to(torch.float32).reshape(-1, 1)

    # ------------------------------------------------------------------
    # Structural interface
    # ------------------------------------------------------------------

    def location(self, parents: torch.Tensor | None) -> torch.Tensor:
        """``f(pa, u_0)`` at the central noise value: the mode (``gumbel_max``,
        ``G = 0``) or the median class (``monotone``, ``U = 1/2``); ``[B, 1]``."""
        probs = self.probs(parents)
        if self.coupling == "gumbel_max":
            centre = torch.zeros_like(probs)
        else:
            centre = torch.full_like(probs[:, :1], 0.5)
        return self.structural(parents, centre)

    def structural(self, parents: torch.Tensor | None, noise: torch.Tensor) -> torch.Tensor:
        """``x = f(pa, u)``: ``noise`` is ``[B, K]`` Gumbels or ``[B, 1]`` uniforms."""
        noise = ensure_2d(noise)
        probs = self.probs(parents, noise.shape[0])
        if self.coupling == "gumbel_max":
            noise = noise.to(probs.device, probs.dtype)
            if noise.shape[-1] != probs.shape[-1]:
                raise ValueError(f"gumbel_max noise must be [B, K={probs.shape[-1]}].")
            idx = (_safe_log(probs) + noise).argmax(-1)
        else:
            if noise.shape[-1] != 1:
                raise ValueError("monotone noise must be [B, 1] uniforms.")
            cdf = self._cdf_rows(probs)
            noise = to_device_dtype(noise, cdf.device, cdf.dtype)
            idx = torch.searchsorted(cdf.contiguous(), noise.contiguous()).reshape(-1)
            idx = idx.clamp_max(probs.shape[-1] - 1)  # rounding: F(K - 1) may be 1 - eps
        return self.class_value(idx)

    @torch.no_grad()
    def abduct(self, x: torch.Tensor, parents: torch.Tensor | None, n_samples: int = 1) -> torch.Tensor:
        """Exact posterior draws of the noise given ``X = x``: ``[n_samples, B, D_u]``.

        Every draw reproduces ``x`` through :meth:`structural`.  Monotone
        draws stay a margin ``m`` inside ``(F(x-1), F(x)]`` (the kernel may give
        slightly different probabilities on another batch size), so a
        counterfactual probability can be off by at most ``2 m / P(x | pa)``;
        ``m`` is 2 eps of the kernel's float dtype for neural kernels and of
        the CDF dtype (float64 on CPU/CUDA) for table lookups.
        """
        idx = self.class_index(x)
        probs = self.probs(parents, idx.shape[0])
        p_x = probs.gather(-1, idx.unsqueeze(-1))  # [B, 1]
        if bool((p_x <= 0).any()):
            raise ValueError("an observed class has probability 0 under its parents: no noise produces it.")
        if self.coupling == "gumbel_max":
            return _gumbel_posterior(probs, idx, n_samples)
        cdf = self._cdf_rows(probs)
        high = cdf.gather(-1, idx.unsqueeze(-1))
        prev = torch.where(idx.unsqueeze(-1) > 0, cdf.gather(-1, (idx - 1).clamp_min(0).unsqueeze(-1)),
                           torch.zeros_like(high))
        jitter_dtype = cdf.dtype if self._deterministic_lookup else probs.dtype
        margin = _MARGIN_EPS * torch.finfo(jitter_dtype).eps
        if bool((high - prev <= 2 * margin).any()):
            raise ValueError(
                f"an observed class has probability below the {jitter_dtype} resolution of the CDF "
                f"(<= {2 * margin:.1e}; e.g. a table cell floored at 1e-12): the monotone coupling "
                "cannot represent it reliably. Use a float64 kernel, table smoothing (alpha > 0) "
                "or coupling='gumbel_max'."
            )
        v = torch.rand(n_samples, *high.shape, device=high.device, dtype=high.dtype)
        return prev + margin + v * (high - prev - 2 * margin)  # ~Uniform(F(x-1), F(x)]

    # ------------------------------------------------------------------
    # CDF, randomized PIT, counterfactual bounds
    # ------------------------------------------------------------------

    def cdf(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        """``P(X <= x | pa)`` in the class order, ``[B]``."""
        idx = self.class_index(x)
        cdf = self.probs(parents, idx.shape[0]).cumsum(-1)
        return cdf.gather(-1, idx.unsqueeze(-1)).squeeze(-1).clamp(0.0, 1.0)

    @torch.no_grad()
    def randomized_pit(self, x: torch.Tensor, parents: torch.Tensor | None,
                       generator: torch.Generator | None = None) -> torch.Tensor:
        """``W = F(x - 1 | pa) + V P(x | pa)``, ``V ~ U(0, 1)``: exactly Uniform(0, 1) under a correct kernel.

        The randomized quantile transform of Dunn & Smyth, *Randomized quantile
        residuals* (J. Comput. Graph. Stat. 5(3):236–244, 1996).  ``generator``
        must be a CPU generator (the uniforms are drawn on the CPU).
        """
        require_cpu_generator(generator, "randomized_pit")
        idx = self.class_index(x)
        probs = self.probs(parents, idx.shape[0])
        high = probs.cumsum(-1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        p_x = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        v = torch.rand(idx.shape[0], generator=generator, dtype=probs.dtype).to(probs.device)
        return (high - p_x + v * p_x).clamp(0.0, 1.0)

    def counterfactual_bounds(self, x: torch.Tensor, parents: torch.Tensor | None,
                              counterfactual_parents: torch.Tensor | None,
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bounds on ``P(X_{pa'} = k | X = x, pa)`` over every SCM with this kernel, ``[B, K]`` each."""
        idx = self.class_index(x)
        b = idx.shape[0]
        p = self.probs(parents, b)
        q = self.probs(counterfactual_parents, b)
        return counterfactual_bounds(p, q, idx, _same_rows(parents, counterfactual_parents, b, p.device))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _deterministic_lookup = False  # tables: the same probabilities for any batch size

    def _cdf_rows(self, probs: torch.Tensor) -> torch.Tensor:
        """Row CDFs in the device's widest float dtype (float64 on CPU/CUDA)."""
        return probs.to(accum_dtype(probs.device)).cumsum(-1)

    def _kernel_device(self) -> torch.device:
        for t in list(self.parameters()) + list(self.buffers()):
            return t.device
        return torch.device("cpu")

    def _k(self) -> int:
        return int(self.n_classes)


class StructuralNeuralCategorical(_CategoricalStructural, NeuralCategoricalMechanism):
    """:class:`NeuralCategoricalMechanism` read as ``X := f(pa, U)`` under a declared ``coupling``.

    Same constructor, fitting, ``state_dict`` and inference behaviour as the
    plain mechanism; adds ``structural``, ``abduct``, ``cdf``,
    ``randomized_pit`` and ``counterfactual_bounds``.
    """

    def __init__(self, *args: Any, coupling: Coupling = "gumbel_max", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._init_coupling(coupling)


class StructuralCategoricalTable(_CategoricalStructural, CategoricalTableMechanism):
    """:class:`CategoricalTableMechanism` read as ``X := f(pa, U)`` under a declared ``coupling``."""

    _deterministic_lookup = True

    def __init__(self, *args: Any, coupling: Coupling = "gumbel_max", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._init_coupling(coupling)


# ===========================================================================
# Internals
# ===========================================================================


def _safe_log(probs: torch.Tensor) -> torch.Tensor:
    return torch.where(probs > 0, probs.clamp_min(torch.finfo(probs.dtype).tiny).log(),
                       torch.full_like(probs, float("-inf")))


def _gumbel_posterior(probs: torch.Tensor, idx: torch.Tensor, n_samples: int) -> torch.Tensor:
    r"""Gumbels ``G`` with ``argmax(log p + G) = idx``, drawn from their exact posterior.

    The maximum ``M = max_k(log p_k + G_k)`` is standard Gumbel (``sum p = 1``)
    and independent of the argmax; the other perturbed values are Gumbels with
    location ``log p_k`` truncated below ``M`` (Oberst & Sontag 2019, after
    Maddison, Tarlow & Minka, *A\* Sampling*, arXiv:1411.0030).  Classes with ``p_k = 0`` are not informed by the
    observation and keep their prior.
    """
    b, k = probs.shape
    log_p = _safe_log(probs)
    std = Gumbel(torch.zeros((), device=probs.device, dtype=probs.dtype),
                 torch.ones((), device=probs.device, dtype=probs.dtype))
    top = std.sample((n_samples, b, 1))                               # M
    raw = log_p + std.sample((n_samples, b, k))                       # untruncated
    truncated = -torch.logaddexp(-top, -raw)                          # Gumbel(log p) | <= M
    chosen = torch.nn.functional.one_hot(idx, k).to(torch.bool).expand(n_samples, b, k)
    value = torch.where(chosen, top.expand(n_samples, b, k), truncated)
    noise = value - log_p
    prior = std.sample((n_samples, b, k))
    return torch.where(torch.isfinite(log_p).expand_as(noise), noise, prior)


def _same_rows(parents, counterfactual_parents, b: int, device) -> torch.Tensor:
    """``[B]`` True where ``pa' = pa`` (roots, or empty parent vectors, always match)."""
    empty = [t is None or t.shape[-1] == 0 for t in (parents, counterfactual_parents)]
    if all(empty):
        return torch.ones(b, dtype=torch.bool, device=device)
    if any(empty):
        raise ValueError("parents and counterfactual_parents must both be given (or both be None).")
    pa, cf = ensure_2d(parents).to(device), ensure_2d(counterfactual_parents).to(device)
    dtype = torch.promote_types(pa.dtype, cf.dtype)
    return (pa.to(dtype).expand(b, -1) == cf.to(dtype).expand(b, -1)).all(-1)
