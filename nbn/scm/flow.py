"""Invertible neural mechanisms: ``X := T(U; pa)`` with a conditional flow.

:class:`FlowNoiseMechanism` is NBN's conditional neural-spline flow read as a
structural equation.  The flow maps the (standardised) value to a standard
normal noise, so

* ``abduct``: ``u = T^{-1}(x; pa)`` is exact (no posterior, no sampling);
* ``structural``: ``x = T(u; pa)`` re-runs the mechanism with a given noise;
* for a 1-D child the transform is monotone increasing in ``u`` (rational-
  quadratic splines, identity outside their interval), so
  ``P(X <= x | pa) = Phi(T^{-1}(x; pa))`` and conditional quantiles are
  ``T(Phi^{-1}(p); pa)`` — exact and batched.

Counterfactuals from this mechanism are answers *under the declared bijective
model*.  Data identify only ``P(X | pa)``: two SCMs can agree on every
observational and interventional distribution and still disagree on
counterfactuals (Peters, Janzing & Schölkopf, *Elements of Causal Inference*,
Ex. 6.19).  Within the class of mechanisms bijective in a scalar noise,
counterfactuals are identifiable in the settings studied by Nasr-Esfahany,
Alizadeh & Shah, *Counterfactual Identifiability of Bijective Causal Models*
(ICML 2023, PMLR 202:25733-25754).  Flows as structural mechanisms follow
Pawlowski, Castro & Glocker, *Deep Structural Causal Models for Tractable
Counterfactual Inference* (NeurIPS 2020, arXiv:2006.06485); the spline
transform is Durkan et al., *Neural Spline Flows* (NeurIPS 2019,
arXiv:1906.04032).

The structural methods take 2-D tensors only (``x`` ``[B, D]``, parents
``[B, D_pa]`` or ``[1, D_pa]``), reject non-finite inputs instead of
sanitising them, and compute in the flow's dtype (float64 inputs are cast).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal

from nbn.mechanisms.parametric.normalizing_flow import NormalizingFlowMechanism
from nbn.scm.mechanisms import StructuralMechanism
from nbn.utils.batching import ensure_2d


class StandardNormalNoise(nn.Module):
    """The fixed noise law ``N(0, I)`` of a flow mechanism (no parameters, no buffers)."""

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = int(dim)

    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.randn(n, self.dim, device=device)

    def cdf(self, u: torch.Tensor) -> torch.Tensor:
        return Normal(torch.zeros((), dtype=u.dtype, device=u.device), 1.0).cdf(u)

    def icdf(self, p: torch.Tensor) -> torch.Tensor:
        return Normal(torch.zeros((), dtype=p.dtype, device=p.device), 1.0).icdf(p)


class FlowNoiseMechanism(NormalizingFlowMechanism, StructuralMechanism):
    """``X := T(U; pa)``, ``U ~ N(0, I)``: a conditional flow as a structural equation.

    Same constructor, fitting, weights, warm start, standardisation and
    serialisation as :class:`NormalizingFlowMechanism` (its ``state_dict`` is
    interchangeable with one); this class adds the structural interface
    (``abduct``, ``structural``, ``location``) and, for 1-D children, the
    exact conditional ``cdf`` and ``quantile`` used by :mod:`nbn.scm.calibration`.

    The noise law is fixed (standard normal), so flow mechanisms cannot be
    tied: each one already uses the same canonical law.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.noise = StandardNormalNoise(getattr(self, "d_x", 1))

    def fit_local(self, *args: Any, **kwargs: Any) -> dict:
        metrics = super().fit_local(*args, **kwargs)
        self.noise.dim = int(self.d_x)  # the fit infers d_x from the data
        return metrics

    # ------------------------------------------------------------------
    # Structural interface
    # ------------------------------------------------------------------

    def abduct(self, x: torch.Tensor, parents: torch.Tensor | None, n_samples: int = 1) -> torch.Tensor:
        """The exact noise ``u = T^{-1}(x; pa)`` repeated: ``[n_samples, B, D]``."""
        u = self._to_noise(x, parents)
        return u.unsqueeze(0).expand(n_samples, *u.shape)

    def structural(self, parents: torch.Tensor | None, noise: torch.Tensor) -> torch.Tensor:
        """``x = T(u; pa)`` for ``u`` of shape ``[B, D]``."""
        noise = self._finite_2d(noise, "noise")
        ctx = self._context(parents, noise.shape[0])
        z = self.forward(ctx)._flow_dist.transform.inv(noise.to(self._param_dtype()))
        return self._unstd_x(z)

    def location(self, parents: torch.Tensor | None) -> torch.Tensor:
        """``T(0; pa)`` — the conditional median for a 1-D child; ``[B, D]``."""
        b = 1 if parents is None else ensure_2d(parents).shape[0]
        zeros = torch.zeros(b, self.output_dim, device=self._param_device(), dtype=self._param_dtype())
        return self.structural(parents, zeros)

    # ------------------------------------------------------------------
    # Exact conditional CDF and quantiles (1-D children)
    # ------------------------------------------------------------------

    def cdf(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        """``P(X <= x | pa) = Phi(T^{-1}(x; pa))``, shape ``[B]`` (1-D children only)."""
        self._require_scalar("cdf")
        return self.noise.cdf(self._to_noise(x, parents)[:, 0])

    def quantile(self, p: torch.Tensor | float, parents: torch.Tensor | None) -> torch.Tensor:
        """Conditional ``p``-quantile ``T(Phi^{-1}(p); pa)``, shape ``[B]`` (1-D children only).

        ``p`` is clamped to ``[eps, 1 - eps]`` of the flow's dtype, so the
        quantiles at 0 and 1 are finite (the extreme representable ones).
        """
        self._require_scalar("quantile")
        dtype, device = self._param_dtype(), self._param_device()
        b = 1 if parents is None else ensure_2d(parents).shape[0]
        p_t = torch.as_tensor(p, dtype=dtype, device=device).reshape(-1)
        if p_t.numel() == 1:
            p_t = p_t.expand(b)
        eps = torch.finfo(dtype).eps
        u = self.noise.icdf(p_t.clamp(eps, 1.0 - eps)).unsqueeze(-1)
        return self.structural(parents, u)[:, 0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _param_device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def _to_noise(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        x = self._finite_2d(x, "x")
        ctx = self._context(parents, x.shape[0])
        return self.forward(ctx)._flow_dist.transform(self._std_x(x))

    def _context(self, parents: torch.Tensor | None, batch: int) -> torch.Tensor | None:
        """Validated 2-D parents; ``None`` for roots (the base is then broadcast)."""
        if parents is None or parents.shape[-1] == 0:
            return None
        if parents.dim() > 2:
            raise ValueError(
                f"{type(self).__name__}: structural methods take parents [B, D_pa]; got "
                f"{tuple(parents.shape)}. Flatten particle dimensions first."
            )
        parents = self._finite_2d(parents, "parents")
        if parents.shape[0] not in (1, batch):
            raise ValueError(f"parents have {parents.shape[0]} rows for {batch} values.")
        return parents

    @staticmethod
    def _finite_2d(t: torch.Tensor, what: str) -> torch.Tensor:
        t = ensure_2d(t)
        if t.dim() > 2:
            raise ValueError(f"{what} must be [B, D]; got {tuple(t.shape)}.")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{what} contains NaN or inf; abduction needs finite values.")
        return t

    def _require_scalar(self, what: str) -> None:
        if self.output_dim != 1:
            raise ValueError(
                f"{type(self).__name__}.{what} needs a 1-D child (got {self.output_dim} dims): "
                "a multivariate flow has no scalar CDF."
            )
