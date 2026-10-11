from __future__ import annotations

from typing import Dict, List

import torch


class LogFactor:
    """A factor in log-space over a set of discrete variables.

    Storing values in log-space lets us use ``log_einsum_exp`` (equivalent to
    sum-product in probability space) without underflow.

    Parameters
    ----------
    log_values:
        Tensor of log-probabilities.  Axes correspond to ``variables`` in order.
    variables:
        Names of the variables spanning each axis of ``log_values``.
    cardinalities:
        Number of states for each variable in ``variables``.
    """

    def __init__(
        self,
        log_values: torch.Tensor,
        variables: List[str],
        cardinalities: Dict[str, int],
    ) -> None:
        assert log_values.dim() == len(variables), (
            f"log_values has {log_values.dim()} dims but {len(variables)} variables"
        )
        self.log_values = log_values
        self.variables = list(variables)
        self.cardinalities = {v: cardinalities[v] for v in variables}

    def condition(self, evidence: Dict[str, int]) -> LogFactor:
        """Slice out evidence variables, returning a reduced factor."""
        result = self.log_values
        remaining_vars = list(self.variables)
        for var, val in evidence.items():
            if var not in remaining_vars:
                continue
            idx = int(val)
            card = int(result.shape[remaining_vars.index(var)])
            if not 0 <= idx < card:
                raise ValueError(
                    f"Evidence {var}={idx} is outside its {card} states."
                )
            # remaining_vars tracks the axes of ``result`` exactly (the
            # selected axis is removed below), so its index *is* the axis;
            # an extra running offset double-counts and slices the wrong axis.
            result = result.select(remaining_vars.index(var), idx)
            remaining_vars.remove(var)
        return LogFactor(result, remaining_vars, self.cardinalities)

    def marginalise(self, var: str) -> LogFactor:
        """Sum out a single variable using log-sum-exp."""
        if var not in self.variables:
            raise ValueError(f"Variable '{var}' not in factor scope {self.variables}")
        dim = self.variables.index(var)
        result = torch.logsumexp(self.log_values, dim=dim)
        remaining = [v for v in self.variables if v != var]
        return LogFactor(result, remaining, self.cardinalities)

    def normalise(self) -> LogFactor:
        """Normalise so values sum to 1 in probability space (subtract log-sum)."""
        log_z = torch.logsumexp(self.log_values.reshape(-1), dim=0)
        return LogFactor(self.log_values - log_z, self.variables, self.cardinalities)

    def to_probs(self) -> torch.Tensor:
        return torch.exp(self.normalise().log_values)

    def __repr__(self) -> str:
        return f"LogFactor(vars={self.variables}, shape={tuple(self.log_values.shape)})"
