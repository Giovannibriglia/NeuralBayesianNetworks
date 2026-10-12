"""Noise laws ``P_U`` of structural mechanisms.

A noise law is an ``nn.Module`` so that it moves with ``.to(device)``, appears
in ``state_dict`` and can be **shared** by several mechanisms (one module
object, several owners): that is how noise tying is represented.  Sharing a
law is not sharing a draw — every mechanism still draws its own ``U``.

Discrete
    :class:`OffsetNoise` — a categorical law over integer offsets
    ``-m, ..., m``.
Continuous
    :class:`GaussianNoise` — ``N(0, diag(sigma^2))``.
    :class:`GaussianMixtureNoise` — a diagonal Gaussian mixture, for skewed or
    multi-modal residuals.

Continuous laws expose ``distribution(location)``: the law of
``X = location + U`` as a batched ``torch.distributions.Distribution``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    MixtureSameFamily,
    Normal,
)

from nbn.learning.weighting import validate_weights, weighted_moments, weighted_quantile


class OffsetNoise(nn.Module):
    """Categorical noise over the integer offsets from ``-max_offset`` to ``max_offset``."""

    def __init__(self, max_offset: int) -> None:
        super().__init__()
        if max_offset < 0:
            raise ValueError(f"max_offset must be >= 0, got {max_offset}.")
        self.max_offset = int(max_offset)
        # Start as a point mass at offset 0 (smoothed), i.e. "no noise".
        init = torch.full((self.n_values,), -10.0)
        init[self.max_offset] = 0.0
        self.logits = nn.Parameter(init)

    @property
    def n_values(self) -> int:
        return 2 * self.max_offset + 1

    @property
    def offsets(self) -> torch.Tensor:
        return torch.arange(-self.max_offset, self.max_offset + 1, device=self.logits.device)

    @property
    def probs(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    def set_probs(self, probs: torch.Tensor) -> None:
        """Overwrite the law in place (keeps the Parameter object, so sharing survives)."""
        with torch.no_grad():
            self.logits.copy_(torch.log(probs.clamp_min(1e-12)).to(self.logits))

    def sample(self, n: int) -> torch.Tensor:
        """``[n]`` integer offsets."""
        idx = Categorical(probs=self.probs).sample((n,))
        return self.offsets[idx]


class GaussianNoise(nn.Module):
    """Zero-mean diagonal Gaussian noise ``N(0, diag(sigma^2))``."""

    def __init__(self, dim: int = 1, min_scale: float = 1e-4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.min_scale = float(min_scale)
        self.log_scale = nn.Parameter(torch.zeros(self.dim))

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp_min(self.min_scale)

    def fit(self, residuals: torch.Tensor, weights: torch.Tensor | None = None) -> dict:
        """Closed-form (weighted) maximum likelihood around zero: ``sigma^2 = E[u^2]``."""
        w = validate_weights(weights, residuals.shape[0], where="GaussianNoise.fit",
                             device=residuals.device)
        sq = residuals.pow(2)
        var = sq.mean(0) if w is None else (w.unsqueeze(-1) * sq).sum(0) / w.sum()
        with torch.no_grad():
            self.log_scale.copy_(0.5 * torch.log(var.clamp_min(self.min_scale**2)))
        return {"noise_scale": self.scale.detach()}

    def distribution(self, location: torch.Tensor) -> Distribution:
        return Independent(Normal(location, self.scale.to(location)), 1)

    def sample(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.dim, device=self.log_scale.device) * self.scale

    def cdf(self, u: torch.Tensor) -> torch.Tensor:
        """``P(U <= u)`` for a 1-D law, elementwise."""
        _require_1d(self.dim, "cdf")
        return Normal(0.0, self.scale[0].to(u)).cdf(u)

    def icdf(self, p: torch.Tensor) -> torch.Tensor:
        """The ``p``-quantile of a 1-D law, elementwise."""
        _require_1d(self.dim, "icdf")
        return Normal(0.0, self.scale[0].to(p)).icdf(p)


class GaussianMixtureNoise(nn.Module):
    """Diagonal Gaussian mixture noise, fitted to residuals by (weighted) EM."""

    def __init__(
        self,
        dim: int = 1,
        n_components: int = 3,
        n_iter: int = 200,
        min_scale: float = 1e-4,
        tol: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_components = int(n_components)
        self.n_iter = int(n_iter)
        self.min_scale = float(min_scale)
        self.tol = float(tol)
        self.mix_logits = nn.Parameter(torch.zeros(self.n_components))
        self.means = nn.Parameter(torch.zeros(self.n_components, self.dim))
        self.log_scales = nn.Parameter(torch.zeros(self.n_components, self.dim))

    def _components(self, location: torch.Tensor) -> Independent:
        # location [B, D] -> component locations [B, C, D]
        loc = location.unsqueeze(-2) + self.means.to(location)
        scale = self.log_scales.exp().clamp_min(self.min_scale).to(location)
        return Independent(Normal(loc, scale.expand_as(loc)), 1)

    def distribution(self, location: torch.Tensor) -> Distribution:
        mix = Categorical(logits=self.mix_logits.to(location).expand(*location.shape[:-1], -1))
        return MixtureSameFamily(mix, self._components(location))

    def fit(self, residuals: torch.Tensor, weights: torch.Tensor | None = None) -> dict:
        """Weighted EM on the residuals (batched over rows and components)."""
        u = residuals
        n = u.shape[0]
        w = validate_weights(weights, n, where="GaussianMixtureNoise.fit", device=u.device)
        w = torch.ones(n, device=u.device, dtype=u.dtype) if w is None else w.to(u.dtype)
        self._initialise_from_quantiles(u, w)
        zeros = torch.zeros(n, self.dim, device=u.device, dtype=u.dtype)
        previous = -math.inf
        for _ in range(self.n_iter):
            with torch.no_grad():
                log_joint = self._components(zeros).log_prob(u.unsqueeze(-2)) + torch.log_softmax(
                    self.mix_logits, -1
                )  # [N, C]
                log_norm = torch.logsumexp(log_joint, dim=-1)
                resp = torch.exp(log_joint - log_norm.unsqueeze(-1)) * w.unsqueeze(-1)  # [N, C]
                mass = resp.sum(0).clamp_min(1e-12)  # [C]
                means = resp.T @ u / mass.unsqueeze(-1)
                var = resp.T @ u.pow(2) / mass.unsqueeze(-1) - means.pow(2)
                self.mix_logits.copy_(torch.log(mass / mass.sum()))
                self.means.copy_(means)
                self.log_scales.copy_(0.5 * torch.log(var.clamp_min(self.min_scale**2)))
                objective = float((w * log_norm).sum() / w.sum())
            if objective - previous < self.tol:
                break
            previous = objective
        return {"noise_mean_log_likelihood": objective}

    def _initialise_from_quantiles(self, u: torch.Tensor, w: torch.Tensor) -> None:
        """Deterministic start: components at evenly spaced weighted quantiles.

        Weighted statistics keep the fit replication-equivalent: integer
        weights start EM from the same point as the replicated data.
        """
        levels = torch.linspace(0.1, 0.9, self.n_components, device=u.device, dtype=u.dtype)
        means = weighted_quantile(u, levels, w)                       # [C, D]
        _, std = weighted_moments(u, w)
        spread = std.to(u.dtype).clamp_min(self.min_scale) / self.n_components
        with torch.no_grad():
            self.mix_logits.zero_()
            self.means.copy_(means)
            self.log_scales.copy_(torch.log(spread).expand(self.n_components, -1))

    def sample(self, n: int) -> torch.Tensor:
        zeros = torch.zeros(n, self.dim, device=self.means.device)
        return self.distribution(zeros).sample()

    def cdf(self, u: torch.Tensor) -> torch.Tensor:
        """``P(U <= u)`` for a 1-D mixture, elementwise: ``sum_c w_c Phi((u - mu_c) / s_c)``."""
        _require_1d(self.dim, "cdf")
        weights = torch.softmax(self.mix_logits, -1).to(u)
        means = self.means[:, 0].to(u)
        scales = self.log_scales[:, 0].exp().clamp_min(self.min_scale).to(u)
        return (weights * Normal(means, scales).cdf(u.unsqueeze(-1))).sum(-1)

    @torch.no_grad()
    def icdf(self, p: torch.Tensor, steps: int = 60) -> torch.Tensor:
        """The ``p``-quantile of a 1-D mixture by bisection (the CDF is monotone; no gradient)."""
        _require_1d(self.dim, "icdf")
        scales = self.log_scales[:, 0].exp().clamp_min(self.min_scale)
        span = (self.means[:, 0].abs() + 12.0 * scales).max().to(p)
        lo, hi = torch.full_like(p, -1.0) * span, torch.full_like(p, 1.0) * span
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            below = self.cdf(mid) < p
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        return 0.5 * (lo + hi)


def _require_1d(dim: int, what: str) -> None:
    if dim != 1:
        raise ValueError(f"{what} is defined for 1-D noise only (this law has {dim} dimensions).")
