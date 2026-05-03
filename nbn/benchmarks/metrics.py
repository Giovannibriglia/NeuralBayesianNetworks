from __future__ import annotations

import torch


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(p || q) for discrete probability tensors of identical shape."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen–Shannon divergence (symmetric KL)."""
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m, eps) + kl_divergence(q, m, eps))


def marginal_mae(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between two marginal probability tensors."""
    return (p - q).abs().mean()
