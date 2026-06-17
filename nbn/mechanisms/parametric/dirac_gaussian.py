"""Dirac-tight-Gaussian mechanism for continuous do-interventions.

Per the user's request, do(X = v) on a continuous variable is implemented as
``P(X | parents) = N(mu = v, sigma = sigma_dirac)`` where ``sigma_dirac`` is
small enough (default 1e-4) to behave like a Dirac delta in practice but
keeps log_prob and KL computations finite (and differentiable).

This is functionally equivalent to a delta distribution but plays nicely with
the rest of the inference machinery, which assumes finite-density CPDs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from nbn import config
from nbn.mechanisms.base import Mechanism
from nbn.utils.batching import ensure_2d


class DiracGaussianMechanism(Mechanism):
    """A continuous do-intervention CPD: a tight isotropic Gaussian at ``value``.

    Parameters
    ----------
    value:
        Intervention target value, scalar or tensor of shape ``[D_x]`` or
        ``[B, D_x]``.  Stored as an ``nn.Parameter`` so that downstream
        autograd can propagate through the intervention value (useful for
        causality-driven RL).
    sigma:
        Standard deviation of the tight Gaussian.  Defaults to
        ``nbn.config.DIRAC_GAUSSIAN_SIGMA``.  Lower → closer to delta but
        more numerical sensitivity.
    output_dim:
        Dimensionality of X.  Inferred from ``value`` if it is a vector.

    Notes
    -----
    Differences from ``DeterministicMechanism``:

    * ``DiracGaussianMechanism`` returns a real ``Normal`` distribution, so
      ``rsample`` works (autograd-friendly).
    * ``log_prob(value, parents)`` evaluates to ``-log(σ √(2π))`` at the
      intervention value — finite and large, never ``inf``.
    """

    is_discrete: bool = False

    def __init__(
        self,
        value: float | torch.Tensor,
        sigma: float | None = None,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(float(value))
        if value.dim() == 0:
            value = value.view(1)
        # Last dim is the event dim
        d = int(value.shape[-1]) if value.dim() >= 1 else 1
        self.output_dim = max(output_dim, d)
        self._value = nn.Parameter(value.float().clone())
        self.sigma = float(sigma) if sigma is not None else float(config.DIRAC_GAUSSIAN_SIGMA)

    def forward(self, parents: torch.Tensor | None) -> Independent:
        """Return ``N(value, sigma·I)`` with batch_shape derived from parents."""
        if parents is None:
            b = 1
        else:
            b = ensure_2d(parents).shape[0]
        loc = self._value.view(1, -1).expand(b, self.output_dim)
        scale = torch.full_like(loc, self.sigma)
        return Independent(Normal(loc, scale), 1)

    def sample(self, parents: torch.Tensor | None, n: int = 1) -> torch.Tensor:
        b = 1 if parents is None else ensure_2d(parents).shape[0]
        loc = self._value.view(1, 1, -1).expand(b, n, self.output_dim)
        return loc + self.sigma * torch.randn_like(loc)

    def log_prob(
        self, x: torch.Tensor, parents: torch.Tensor | None
    ) -> torch.Tensor:
        squeeze_s = False
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_s = True
        b, s, d = x.shape
        loc = self._value.view(1, 1, -1).expand(b, s, d)
        # Independent Normal log-prob, summed over event dim
        var = self.sigma ** 2
        lp = -0.5 * (((x - loc) ** 2) / var
                     + 2 * torch.log(torch.tensor(self.sigma, device=x.device))
                     + torch.log(torch.tensor(2 * 3.141592653589793, device=x.device)))
        lp = lp.sum(-1)
        return lp.squeeze(1) if squeeze_s else lp

    def fit_local(
        self, x: torch.Tensor, parents: torch.Tensor | None, **kwargs
    ) -> dict:
        """No-op: the value is set externally (it's a do-intervention)."""
        return {"sigma": self.sigma, "value": float(self._value.detach().mean())}
