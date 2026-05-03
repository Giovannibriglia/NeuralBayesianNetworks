from __future__ import annotations

import opt_einsum
import torch


def log_einsum_exp(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    """Numerically stable log-domain einsum.

    Equivalent to ``torch.log(einsum(torch.exp(operands)))`` but avoids
    underflow/overflow by subtracting per-operand maximum values before
    exponentiation.

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
    # Shift each operand by its max to prevent underflow.
    shifts = [op.amax().detach() for op in operands]
    shifted = [op - s for op, s in zip(operands, shifts)]
    linear = [torch.exp(s) for s in shifted]
    result = opt_einsum.contract(equation, *linear)
    # Re-add all shifts as a scalar offset.
    log_result = torch.log(result.clamp_min(1e-40))
    total_shift = sum(shifts)
    return log_result + total_shift
