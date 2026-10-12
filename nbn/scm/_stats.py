"""Small torch-only statistics helpers (tail probabilities, multiple testing)."""
from __future__ import annotations

import math

import torch


def chi2_sf(statistic: torch.Tensor | float, df: torch.Tensor | float) -> torch.Tensor:
    """Upper tail ``P(chi2_df >= statistic)``."""
    x = torch.as_tensor(statistic, dtype=torch.float64)
    k = torch.as_tensor(df, dtype=torch.float64)
    return torch.special.gammaincc(k / 2.0, (x / 2.0).clamp_min(0.0))


def normal_two_sided_p(z: torch.Tensor | float) -> torch.Tensor:
    """``P(|N(0, 1)| >= |z|)``."""
    z = torch.as_tensor(z, dtype=torch.float64)
    return torch.special.erfc(z.abs() / math.sqrt(2.0))


def kolmogorov_sf(lam: float, terms: int = 100) -> float:
    """``P(K >= lam)`` for the Kolmogorov distribution (asymptotic KS law).

    Uses the alternating series for ``lam >= 1`` and the Jacobi-theta form
    ``P(K <= lam) = sqrt(2 pi) / lam * sum_j exp(-(2j - 1)^2 pi^2 / (8 lam^2))``
    below, where the alternating series converges badly.
    """
    if lam <= 0.0:
        return 1.0
    j = torch.arange(1, terms + 1, dtype=torch.float64)
    if lam < 1.0:
        cdf = math.sqrt(2.0 * math.pi) / lam * torch.exp(
            -((2.0 * j - 1.0) ** 2) * math.pi**2 / (8.0 * lam**2)
        ).sum()
        return float((1.0 - cdf).clamp(0.0, 1.0))
    series = 2.0 * ((-1.0) ** (j - 1) * torch.exp(-2.0 * j**2 * lam**2)).sum()
    return float(series.clamp(0.0, 1.0))


def ks_two_sample(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov statistic and asymptotic p-value (1-D samples)."""
    a = a.reshape(-1).detach().cpu().double().sort().values
    b = b.reshape(-1).detach().cpu().double().sort().values
    grid = torch.cat([a, b])
    cdf_a = torch.searchsorted(a, grid, right=True).double() / a.numel()
    cdf_b = torch.searchsorted(b, grid, right=True).double() / b.numel()
    d = float((cdf_a - cdf_b).abs().max())
    n_eff = a.numel() * b.numel() / (a.numel() + b.numel())
    root = math.sqrt(n_eff)
    # Stephens' small-sample correction of the asymptotic law.
    return d, kolmogorov_sf((root + 0.12 + 0.11 / root) * d)


def ks_uniform(w: torch.Tensor) -> tuple[float, float]:
    """One-sample KS statistic of ``w`` against Uniform(0, 1) and its asymptotic p-value."""
    w = w.reshape(-1).detach().cpu().double().sort().values
    n = w.numel()
    i = torch.arange(1, n + 1, dtype=torch.float64)
    d = float(torch.maximum(i / n - w, w - (i - 1) / n).max())
    root = math.sqrt(n)
    return d, kolmogorov_sf((root + 0.12 + 0.11 / root) * d)


def require_cpu_generator(generator: torch.Generator | None, where: str) -> None:
    """Index sampling here runs on the CPU; a device generator would fail deep inside torch."""
    if generator is not None and generator.device.type != "cpu":
        raise ValueError(f"{where} needs a CPU torch.Generator (got one on {generator.device}).")


def benjamini_hochberg(p_values: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """Boolean mask of hypotheses rejected at false-discovery rate ``alpha``."""
    p = torch.as_tensor(p_values, dtype=torch.float64).reshape(-1)
    m = p.numel()
    order = torch.argsort(p)
    thresholds = alpha * torch.arange(1, m + 1, dtype=torch.float64) / m
    passed = p[order] <= thresholds
    reject = torch.zeros(m, dtype=torch.bool)
    if bool(passed.any()):
        last = int(torch.nonzero(passed).max())
        reject[order[: last + 1]] = True
    return reject
