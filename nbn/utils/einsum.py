from __future__ import annotations

import opt_einsum
import torch


def log_einsum_exp(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    """Numerically stable log-domain einsum.

    Equivalent to ``torch.log(einsum(torch.exp(operands)))`` but avoids
    overflow and most underflow by subtracting each operand's maximum before
    exponentiation.

    Limits
    ------
    * An operand that is entirely ``-inf`` (an impossible configuration) has
      no finite maximum; its shift is taken as 0 so the result is an exact
      ``-inf`` rather than ``NaN`` (``-inf - (-inf)``).
    * The shift is per *operand*, not per output cell, so a product of terms
      that are each far below their operand's maximum can still underflow
      after exponentiation.  Such cells come back as ``-inf`` — never as a
      finite floor, which would be a confidently wrong number.

    Parameters
    ----------
    equation:
        Standard einsum equation string.
    *operands:
        Log-domain tensors.

    Returns
    -------
    torch.Tensor
        Log of the einsum result.
    """
    # Shift each operand by its max to prevent underflow; a non-finite max
    # (all -inf, or +inf) would turn the shift itself into NaN, so use 0.
    shifts = []
    for op in operands:
        m = op.amax().detach()
        shifts.append(torch.where(torch.isfinite(m), m, torch.zeros_like(m)))
    linear = [torch.exp(op - s) for op, s in zip(operands, shifts)]
    result = opt_einsum.contract(equation, *linear)
    # No clamp: log(0) = -inf is the honest answer for an impossible cell.
    return torch.log(result) + sum(shifts)
