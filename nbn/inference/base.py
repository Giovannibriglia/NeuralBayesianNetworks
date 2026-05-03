from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List

import torch

if TYPE_CHECKING:
    from nbn.core.network import NeuralBayesianNetwork


class InferenceEngine(ABC):
    """Abstract base class for inference engines.

    All engines must implement:
    * ``query(model, targets, evidence)`` — single query, returns a
      ``torch.distributions.Distribution`` or a tensor of probabilities.
    * ``query_batch(model, targets, evidence)`` — batched query returning
      ``[B, K]`` tensor where K is the number of target states.
    """

    @abstractmethod
    def query(
        self,
        model: NeuralBayesianNetwork,
        targets: List[str],
        evidence: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return posterior probabilities or samples for ``targets`` given ``evidence``.

        Parameters
        ----------
        model: NeuralBayesianNetwork
        targets: list of target node names.
        evidence: dict mapping node name → observed value tensor.

        Returns
        -------
        torch.Tensor
            For discrete targets: normalised probability vector ``[K]`` or
            ``[B, K]``.
            For continuous: weighted samples ``(weights, samples)`` tuple.
        """

    def query_batch(
        self,
        model: NeuralBayesianNetwork,
        targets: List[str],
        evidence: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Batched version of ``query``.  Default implementation loops; subclasses override."""
        b = next(iter(evidence.values())).shape[0]
        results = []
        for i in range(b):
            ev_i = {k: v[i:i+1] for k, v in evidence.items()}
            results.append(self.query(model, targets, ev_i, **kwargs))
        return torch.cat(results, dim=0)
