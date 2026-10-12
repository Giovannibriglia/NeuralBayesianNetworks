"""Invertible neural mechanisms: ``X := T(U; pa)`` with a conditional flow.

:class:`FlowNoiseMechanism` is NBN's conditional neural-spline flow read as a
structural equation.  The flow maps the (standardised) value to a standard
normal noise, so

* ``abduct``: ``u = T^{-1}(x; pa)`` is exact (no posterior, no sampling);
* ``structural``: ``x = T(u; pa)`` re-runs the mechanism with a given noise;
* for a 1-D child the transform is monotone increasing in ``u``, so
  ``P(X <= x | pa) = Phi(T^{-1}(x; pa))`` and conditional quantiles are
  ``T(Phi^{-1}(p); pa)`` — exact, batched, differentiable.

Counterfactuals from this mechanism are answers *under the declared bijective
model*.  Data identify only ``P(X | pa)``: two SCMs can agree on every
observational and interventional distribution and still disagree on
counterfactuals (Peters, Janzing & Schölkopf, *Elements of Causal Inference*,
Ex. 6.19).  Within the class of mechanisms bijective in a scalar noise,
counterfactuals are identifiable in the settings studied by Nasr-Esfahany,
Alizadeh & Shah, *Counterfactual Identifiability of Bijective Causal Models*
(ICML 2023, PMLR 202:25733-25754).  Using flows as structural mechanisms
follows Pawlowski, Castro & Glocker, *Deep Structural Causal Models for
Tractable Counterfactual Inference* (NeurIPS 2020, arXiv:2006.06485); the
spline transform is Durkan et al., *Neural Spline Flows* (NeurIPS 2019,
arXiv:1906.04032).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from nbn.mechanisms.parametric.normalizing_flow import NormalizingFlowMechanism
from nbn.scm.mechanisms import StructuralMechanism
from nbn.utils.batching import ensure_2d


class StandardNormalNoise(nn.Module):
    """The fixed noise law ``N(0, I)`` of a flow mechanism (nothing to fit)."""

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("_anchor", torch.zeros(()))  # follows .to(device)

    def sample(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.dim, device=self._anchor.device)

    def cdf(self, u: torch.Tensor) -> torch.Tensor:
        return Normal(0.0, 1.0).cdf(u)

    def icdf(self, p: torch.Tensor) -> torch.Tensor:
        return Normal(0.0, 1.0).icdf(p)


class FlowNoiseMechanism(NormalizingFlowMechanism, StructuralMechanism):
    """``X := T(U; pa)``, ``U ~ N(0, I)``: a conditional flow as a structural equation.

    Same constructor, fitting, weights, warm start, standardisation and
    serialisation as :class:`NormalizingFlowMechanism`; this class only adds
    the structural interface (``abduct``, ``structural``, ``location``) and,
    for 1-D children, the exact conditional ``cdf`` and ``quantile`` used by
    the calibration tools in :mod:`nbn.scm.calibration`.

    The noise law is fixed (standard normal), so flow mechanisms cannot be
    tied: each one already shares the same canonical noise law.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.noise = StandardNormalNoise(getattr(self, "d_x", 1))

    # ------------------------------------------------------------------
    # Structural interface
    # ------------------------------------------------------------------

    def abduct(self, x: torch.Tensor, parents: torch.Tensor | None, n_samples: int = 1) -> torch.Tensor:
        """The exact noise ``u = T^{-1}(x; pa)`` repeated: ``[n_samples, B, D]``."""
        u = self._to_noise(x, parents)
        return u.unsqueeze(0).expand(n_samples, *u.shape)

    def structural(self, parents: torch.Tensor | None, noise: torch.Tensor) -> torch.Tensor:
        """``x = T(u; pa)`` for ``u`` of shape ``[B, D]``."""
        flow = self.forward(self._batched_parents(parents, noise.shape[0]))._flow_dist
        z = flow.transform.inv(noise.to(self._param_dtype()))
        return self._unstd_x(z)

    def location(self, parents: torch.Tensor | None) -> torch.Tensor:
        """The conditional median ``T(0; pa)`` (the flow's "deterministic part")."""
        b = 1 if parents is None else ensure_2d(parents).shape[0]
        zeros = torch.zeros(b, self.output_dim, device=self.noise._anchor.device,
                            dtype=self._param_dtype())
        return self.structural(parents, zeros)

    # ------------------------------------------------------------------
    # Exact conditional CDF and quantiles (1-D children)
    # ------------------------------------------------------------------

    def cdf(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        """``P(X <= x | pa) = Phi(T^{-1}(x; pa))``, shape ``[B]`` (1-D children only)."""
        self._require_scalar("cdf")
        return self.noise.cdf(self._to_noise(x, parents)[:, 0])

    def quantile(self, p: torch.Tensor | float, parents: torch.Tensor | None) -> torch.Tensor:
        """Conditional ``p``-quantile ``T(Phi^{-1}(p); pa)``, shape ``[B]`` (1-D children only)."""
        self._require_scalar("quantile")
        b = 1 if parents is None else ensure_2d(parents).shape[0]
        p_t = torch.as_tensor(p, dtype=torch.float32).reshape(-1)
        p_t = p_t.expand(b) if p_t.numel() == 1 else p_t
        u = self.noise.icdf(p_t.clamp(1e-12, 1 - 1e-12)).to(self.noise._anchor.device).unsqueeze(-1)
        return self.structural(parents, u)[:, 0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_noise(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        x = ensure_2d(x)
        flow = self.forward(self._batched_parents(parents, x.shape[0]))._flow_dist
        return flow.transform(self._std_x(x))

    def _batched_parents(self, parents: torch.Tensor | None, batch: int) -> torch.Tensor | None:
        """Root nodes have no context; the flow's base is then broadcast over the batch."""
        if parents is None or parents.shape[-1] == 0:
            return None
        return ensure_2d(parents)

    def _require_scalar(self, what: str) -> None:
        if self.output_dim != 1:
            raise ValueError(
                f"{type(self).__name__}.{what} needs a 1-D child (got {self.output_dim} dims): "
                "a multivariate flow has no scalar CDF."
            )
