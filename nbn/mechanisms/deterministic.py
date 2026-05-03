from __future__ import annotations

import torch
from torch.distributions import Distribution

from nbn.mechanisms.base import Mechanism
from nbn.utils.batching import ensure_2d


class _DeltaDistribution(Distribution):
    """A Dirac delta distribution at a fixed value."""

    has_rsample = True

    def __init__(self, value: torch.Tensor) -> None:
        self._value = value
        super().__init__(batch_shape=value.shape[:-1], event_shape=value.shape[-1:], validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        return self._value.expand(*sample_shape, *self._value.shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.sample(sample_shape)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)


class DeterministicMechanism(Mechanism):
    """Deterministic CPD: delta distribution at a fixed value.

    Useful for Pearl's do-operator interventions: replace the CPD of a node
    with ``DeterministicMechanism(value)`` to implement ``do(X = value)``.

    Parameters
    ----------
    value:
        Fixed output value, shape ``[D_x]`` or ``[B, D_x]``.
    """

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_fixed_value", value)

    def forward(self, parents: torch.Tensor | None) -> _DeltaDistribution:
        if parents is not None and self._fixed_value.dim() == 1:
            b = ensure_2d(parents).shape[0]
            val = self._fixed_value.unsqueeze(0).expand(b, -1)
        else:
            val = self._fixed_value
        return _DeltaDistribution(val)

    def sample(self, parents: torch.Tensor | None, n: int = 1) -> torch.Tensor:
        b = 1 if parents is None else ensure_2d(parents).shape[0]
        val = self._fixed_value
        if val.dim() == 1:
            return val.view(1, 1, -1).expand(b, n, -1)
        return val.unsqueeze(1).expand(-1, n, -1)

    def log_prob(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        return torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)

    def fit_local(self, x: torch.Tensor, parents: torch.Tensor | None, **kwargs) -> dict:
        return {}
