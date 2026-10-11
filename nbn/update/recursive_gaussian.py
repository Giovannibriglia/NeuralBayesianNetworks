"""Recursive least-squares for linear-Gaussian CPDs via *centred* statistics.

The closed-form ridge fit of ``P(x | pa) = N(W pa + b, diag(sigma^2))`` depends
on the data only through the weighted mass ``N``, the means ``mean_z`` /
``mean_x`` and the centred (co)variance sums ``Czz = sum w (z-mz)(z-mz)^T``,
``Czx = sum w (z-mz)(x-mx)^T`` and ``cxx = sum w (x-mx)^2``.  Persisting those
makes incremental update exact: ``accumulate`` merges two states with Chan et
al.'s parallel formula, and ``solve`` recovers ``(W, b, log sigma)`` — a
chunked update reproduces a single pooled fit to solver precision, with no
rehearsal and O(state) memory.

Why centred
-----------
The previous version persisted the *raw* normal equations ``A = Z^T Z``,
``B = Z^T x``, ``c = sum x^2`` in the data dtype (float32, and float32 is all
MPS has) and computed the residual as ``c - 2 theta^T B + theta^T A theta``.
With data at an offset of 1000 and noise 0.1 every term is ~1e9 and their
difference ~1e1: float32 keeps ~7 digits, so sigma came out as the
``min_scale`` floor or as noise.  Centred sums are O(spread^2), never
O(offset^2), so the same arithmetic is stable in float32.

Persisted layout (compatibility)
--------------------------------
``LinearGaussianMechanism`` stores the state in four buffers named after the
raw normal equations (``_neq_A``, ``_neq_B``, ``_neq_c``, ``_neq_N``) and
round-trips it through :class:`NormalEquationState`.  To keep those buffers,
their shapes and every existing checkpoint working, the centred statistics are
*packed* into the same shapes (``p`` = number of parent columns)::

    A[:p, :p] = Czz        A[:p, p] = A[p, :p] = mean_z      A[p, p] = 0
    B[:p, :]  = Czx        B[p, :]  = mean_x
    c         = cxx        N        = sum(w)

``A[p, p]`` doubles as the layout marker: in the legacy raw layout it is
``sum(w * 1 * 1) = N > 0``; in the centred layout it is always 0.  A legacy
state (e.g. a checkpoint written before this change) is converted on first
use by :func:`unpack`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch

from nbn.utils.device import accum_dtype, linalg_device


@dataclass
class NormalEquationState:
    """Sufficient statistics of a linear-Gaussian fit, packed as described in
    the module docstring.  Field names are kept from the raw-normal-equation
    layout for backward compatibility; read them through :func:`unpack`."""

    A: torch.Tensor; B: torch.Tensor; c: torch.Tensor; N: torch.Tensor; input_dim: int
    # Layout known at construction (True for everything this module builds);
    # None = read the A[p, p] marker, which costs a device sync.
    centred: bool | None = None

    def to(self, device):
        return NormalEquationState(self.A.to(device), self.B.to(device),
                                   self.c.to(device), self.N.to(device), self.input_dim,
                                   self.centred)


def design_matrix(parents, x):
    n = x.shape[0]; device, dtype = x.device, x.dtype
    ones = torch.ones(n, 1, device=device, dtype=dtype)
    if parents is None or parents.numel() == 0 or parents.shape[-1] == 0:
        return ones
    pa = parents.reshape(n, -1).to(device=device, dtype=dtype)
    return torch.cat([pa, ones], dim=1)


def _pack(N, mz, mx, Czz, Czx, cxx, input_dim, dtype) -> NormalEquationState:
    p = mz.shape[0]
    A = torch.zeros(p + 1, p + 1, device=Czz.device, dtype=Czz.dtype)
    A[:p, :p] = Czz
    A[:p, p] = mz
    A[p, :p] = mz
    B = torch.cat([Czx, mx.reshape(1, -1)], dim=0)
    return NormalEquationState(
        A.to(dtype), B.to(dtype), cxx.to(dtype), N.to(dtype), input_dim, True,
    )


def is_centred(state: NormalEquationState) -> bool:
    """True for the centred layout (``A[p, p] == 0``), False for legacy raw."""
    if state.centred is not None:
        return state.centred
    return bool(state.A[-1, -1] == 0)


def unpack(state: NormalEquationState, dtype: torch.dtype | None = None):
    """Return ``(N, mean_z, mean_x, Czz, Czx, cxx)`` in ``dtype``.

    Accepts both layouts.  A legacy raw state is converted here; the
    conversion cannot restore digits the raw float32 sums already lost, but
    every later update is stable.
    """
    dt = dtype or state.A.dtype
    A, B, c, N = (t.to(dt) for t in (state.A, state.B, state.c, state.N))
    p = A.shape[0] - 1
    if is_centred(state):
        return N, A[:p, p], B[p], A[:p, :p], B[:p], c
    _warn_legacy_once()
    n = N.clamp_min(torch.finfo(dt).tiny)
    mz = A[:p, p] / n
    mx = B[p] / n
    Czz = A[:p, :p] - n * torch.outer(mz, mz)
    Czx = B[:p] - n * torch.outer(mz, mx)
    cxx = c - n * mx * mx
    return N, mz, mx, Czz, Czx, cxx


_LEGACY_WARNED = False


def _warn_legacy_once() -> None:
    """The conversion is exact algebra, but the raw float32 sums it starts
    from already lost digits at large data offsets; only a refit recovers
    them.  Warn once per process, not on every update."""
    global _LEGACY_WARNED
    if _LEGACY_WARNED:
        return
    _LEGACY_WARNED = True
    warnings.warn(
        "Converting linear-Gaussian sufficient statistics saved in the old "
        "raw normal-equation layout.  Those float32 sums lose precision when "
        "the data has a large offset (sigma can be badly off); refit the "
        "model (model.fit) on the original data to rebuild exact statistics.",
        UserWarning,
        stacklevel=3,
    )


def batch_statistics(parents, x, weights=None):
    """Centred sufficient statistics of one batch, optionally weighted.

    ``weights`` is a non-negative ``[N]`` tensor of multiplicities: the
    statistics equal those of the data with row ``i`` replicated ``w_i``
    times (``N = sum w``).  Accumulated in the device's widest dtype
    (float64, float32 on MPS) and centred *before* the squares are formed, so
    a large common offset never enters a sum of squares.
    """
    x2 = x.reshape(x.shape[0], -1)
    z = design_matrix(parents, x2)[:, :-1]           # parent columns only
    input_dim = z.shape[1]
    acc = accum_dtype(x2.device)
    zd, xd = z.to(acc), x2.to(acc)
    if weights is None:
        w = torch.ones(x2.shape[0], 1, dtype=acc, device=x2.device)
    else:
        w = weights.reshape(-1, 1).to(device=x2.device, dtype=acc)
    N = w.sum()
    n = N.clamp_min(torch.finfo(acc).tiny)
    mz = (w * zd).sum(0) / n
    mx = (w * xd).sum(0) / n
    zc, xc = zd - mz, xd - mx
    zw = zc * w
    Czz = zw.transpose(0, 1) @ zc
    Czx = zw.transpose(0, 1) @ xc
    cxx = (w * xc * xc).sum(0)
    return _pack(N, mz, mx, Czz, Czx, cxx, input_dim, x2.dtype)


def accumulate(prior, batch, *, forgetting: float = 1.0):
    """Merge ``batch`` into the ``forgetting``-discounted ``prior``.

    Chan et al.'s parallel update: equal to adding the raw sums
    ``forgetting * prior + batch``, but on centred statistics — the cross
    term ``delta delta^T Na Nb / N`` carries the shift between the two means,
    so nothing of order ``offset^2`` is ever formed.
    """
    acc = accum_dtype(prior.A.device)
    g = float(forgetting)
    Na, mza, mxa, Czza, Czxa, cxxa = unpack(prior, acc)
    Nb, mzb, mxb, Czzb, Czxb, cxxb = unpack(batch.to(prior.A.device), acc)
    Na = g * Na
    N = Na + Nb
    n = N.clamp_min(torch.finfo(acc).tiny)
    dz, dx = mzb - mza, mxb - mxa
    f = Na * Nb / n
    mz = mza + dz * (Nb / n)
    mx = mxa + dx * (Nb / n)
    Czz = g * Czza + Czzb + f * torch.outer(dz, dz)
    Czx = g * Czxa + Czxb + f * torch.outer(dz, dx)
    cxx = g * cxxa + cxxb + f * dx * dx
    return _pack(N, mz, mx, Czz, Czx, cxx, prior.input_dim, prior.A.dtype)


def solve(state, *, ridge: float = 1e-6, min_scale: float = 1e-3):
    """Ridge solution ``(W, b, log sigma)`` from a (centred or legacy) state.

    The intercept is unpenalised, so the ridge problem on ``[pa, 1]`` reduces
    exactly to ``W = (Czz + ridge I)^-1 Czx`` and ``b = mean_x - mean_z^T W``;
    the residual sum of squares is evaluated on centred quantities too.
    """
    device, dtype = state.A.device, state.A.dtype
    # Solve where a LAPACK kernel exists (CPU on MPS, the device elsewhere),
    # in that device's widest dtype; the system is only ``[D_pa, D_pa]``.
    ld = linalg_device(device)
    acc = accum_dtype(ld)
    A, B, c, N = _hop((state.A, state.B, state.c, state.N), ld)
    N, mz, mx, Czz, Czx, cxx = unpack(
        NormalEquationState(A, B, c, N, state.input_dim, state.centred), acc,
    )
    d_x = Czx.shape[1]
    p = Czz.shape[0]
    if state.input_dim == 0 or p == 0:
        W = torch.zeros(0, d_x, device=ld, dtype=acc)
        sse = cxx
    else:
        reg = ridge * torch.eye(p, device=ld, dtype=acc)
        W = torch.linalg.solve(Czz + reg, Czx)
        # Exact ridge RSS: cxx - 2 W^T Czx + W^T Czz W (per output column).
        sse = cxx - 2.0 * (W * Czx).sum(0) + (W * (Czz @ W)).sum(0)
    b = mx - mz @ W
    n = N.clamp_min(1.0)
    std = (sse.clamp_min(0.0) / n).clamp_min(min_scale * min_scale).sqrt()
    std = std.clamp_min(min_scale)
    # Cast before the hop: the solve dtype (float64) may not exist on device.
    return _hop((W.to(dtype), b.to(dtype), torch.log(std).to(dtype)), device)


def _hop(tensors, device):
    """Move several tensors with ONE transfer (each ``.to`` across devices is
    a separate copy and, from MPS/CUDA to the CPU, a separate sync)."""
    tensors = tuple(tensors)
    if all(t.device == torch.device(device) for t in tensors):
        return tensors
    dt = tensors[0].dtype
    flat = torch.cat([t.reshape(-1).to(dt) for t in tensors]).to(device)
    out, i = [], 0
    for t in tensors:
        out.append(flat[i:i + t.numel()].reshape(t.shape))
        i += t.numel()
    return tuple(out)
