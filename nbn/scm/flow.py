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
``[B, D_pa]`` or ``[1, D_pa]``), reject non-finite inputs (the plain
``log_prob`` / ``forward`` path keeps NBN's sanitising), and move inputs to the
flow's device and dtype.  Their outputs carry gradients with respect to the
flow's weights; wrap calls in ``torch.no_grad()`` when only values are needed.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal

from nbn.mechanisms.parametric.normalizing_flow import NormalizingFlowMechanism
from nbn.scm.mechanisms import StructuralMechanism
from nbn.utils.batching import ensure_2d
from nbn.utils.device import to_device_dtype


class StandardNormalNoise(nn.Module):
    """The fixed noise law ``N(0, I)`` of a flow mechanism (no parameters).

    ``sample`` draws on the device and dtype of a zero-size, non-persistent
    anchor buffer, which moves with the mechanism (``.to(...)``, ``.double()``,
    ``torch.load(map_location=...)``) and is not part of the ``state_dict``.
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("_anchor", torch.empty(0), persistent=False)

    def sample(self, n: int) -> torch.Tensor:
        anchor = self._buffers.get("_anchor")  # absent in pickles made before it existed
        if anchor is None:
            return torch.randn(n, self.dim)
        return torch.randn(n, self.dim, device=anchor.device, dtype=anchor.dtype)

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
        # The fit infers d_x and the device from the data: keep the noise law in step.
        self.noise.dim = int(self.d_x)
        self.noise.to(device=self._param_device(), dtype=self._param_dtype())
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
        z = self.forward(ctx)._flow_dist.transform.inv(noise)
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

    def _finite_2d(self, t: torch.Tensor, what: str) -> torch.Tensor:
        """``t`` as ``[B, D]`` on the flow's device and dtype; NaN/inf rejected."""
        t = ensure_2d(t)
        if t.dim() > 2:
            raise ValueError(f"{what} must be [B, D]; got {tuple(t.shape)}.")
        t = to_device_dtype(t, self._param_device(), self._param_dtype())
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{what} contains NaN or inf; abduction needs finite values.")
        return t

    def _require_scalar(self, what: str) -> None:
        if self.output_dim != 1:
            raise ValueError(
                f"{type(self).__name__}.{what} needs a 1-D child (got {self.output_dim} dims): "
                "a multivariate flow has no scalar CDF."
            )
