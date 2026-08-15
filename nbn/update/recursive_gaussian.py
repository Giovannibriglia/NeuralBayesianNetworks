"""Recursive least-squares for linear-Gaussian CPDs via normal equations.

The closed-form ridge fit of ``P(x | pa) = N(W pa + b, diag(sigma^2))`` depends
on the data only through the normal-equation sufficient statistics
``A = Z^T Z``, ``B = Z^T x``, ``c = sum x^2`` and ``N`` (with ``Z = [pa, 1]``).
Persisting ``(A, B, c, N)`` makes incremental update exact: ``accumulate`` adds
the new-data statistics, and ``solve`` recovers ``(W, b, log sigma)`` — a
chunked update reproduces a single pooled fit to ridge-solver precision, with
no rehearsal and O(state) memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NormalEquationState:
    A: torch.Tensor; B: torch.Tensor; c: torch.Tensor; N: torch.Tensor; input_dim: int

    def to(self, device):
        return NormalEquationState(self.A.to(device), self.B.to(device),
                                   self.c.to(device), self.N.to(device), self.input_dim)


def design_matrix(parents, x):
    n = x.shape[0]; device, dtype = x.device, x.dtype
    ones = torch.ones(n, 1, device=device, dtype=dtype)
    if parents is None or parents.numel() == 0 or parents.shape[-1] == 0:
        return ones
    pa = parents.reshape(n, -1).to(device=device, dtype=dtype)
    return torch.cat([pa, ones], dim=1)


def batch_statistics(parents, x, weights=None):
    """Normal-equation sufficient statistics, optionally per-sample weighted.

    ``weights`` is a non-negative ``[N]`` tensor of multiplicities.  Weighting
    enters as ``A = z^T W z``, ``B = z^T W x``, ``c = sum w*x^2`` and
    ``N = sum w``, i.e. exactly the statistics that replicating row ``i``
    ``w_i`` times would produce.  Accumulated in float64 before casting back:
    these are sums over every row, and weights from an EM E-step are long runs
    of repeated, sometimes tiny values.
    """
    x2 = x.reshape(x.shape[0], -1)
    z = design_matrix(parents, x2)
    input_dim = 0 if z.shape[1] == 1 else z.shape[1] - 1
    zd, xd = z.to(torch.float64), x2.to(torch.float64)
    if weights is None:
        A = zd.transpose(0, 1) @ zd
        B = zd.transpose(0, 1) @ xd
        c = (xd * xd).sum(dim=0)
        N = torch.tensor(float(x2.shape[0]), dtype=torch.float64, device=x2.device)
    else:
        w = weights.reshape(-1, 1).to(device=x2.device, dtype=torch.float64)
        zw = zd * w
        A = zw.transpose(0, 1) @ zd
        B = zw.transpose(0, 1) @ xd
        c = (w * xd * xd).sum(dim=0)
        N = w.sum()
    dt = x2.dtype
    return NormalEquationState(
        A.to(dt), B.to(dt), c.to(dt), N.to(dt), input_dim,
    )


def accumulate(prior, batch, *, forgetting: float = 1.0):
    g = forgetting
    return NormalEquationState(g*prior.A+batch.A, g*prior.B+batch.B,
                               g*prior.c+batch.c, g*prior.N+batch.N, prior.input_dim)


def solve(state, *, ridge: float = 1e-6, min_scale: float = 1e-3):
    A, B, c, N = state.A, state.B, state.c, state.N
    p1 = A.shape[0]; d_x = B.shape[1]; device, dtype = A.device, A.dtype
    reg = ridge * torch.eye(p1, device=device, dtype=dtype)
    if p1 >= 1:
        reg[-1, -1] = 0.0  # do not regularise the intercept
    theta = torch.linalg.solve(A + reg, B)
    if state.input_dim == 0:
        weight = torch.zeros(0, d_x, device=device, dtype=dtype); bias = theta[-1]
    else:
        weight = theta[:-1]; bias = theta[-1]
    tB = (theta * B).sum(dim=0)
    tAt = (theta * (A @ theta)).sum(dim=0)
    sse = (c - 2.0 * tB + tAt).clamp_min(0.0)
    n = float(N.clamp_min(1.0))
    std = (sse / n).clamp_min(min_scale * min_scale).sqrt().clamp_min(min_scale)
    return weight, bias, torch.log(std)
