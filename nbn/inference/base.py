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
        """Batched version of ``query``.  Default implementation loops; subclasses override.

        Every engine shipped with nbn overrides this with a vectorised path,
        so it is the reference implementation for third-party engines rather
        than a hot path.

        The per-row results are *stacked*, not concatenated: ``query``
        returns a ``[K]`` probability vector for a single discrete target, and
        concatenating B of those yields ``[B*K]`` — a silently wrong shape
        that still indexes and still sums, so it fails far from its cause.
        Engines whose ``query`` already returns ``[1, K]`` are concatenated as
        before.
        """
        b = next(iter(evidence.values())).shape[0]
        results = []
        for i in range(b):
            ev_i = {k: v[i:i+1] for k, v in evidence.items()}
            results.append(self.query(model, targets, ev_i, **kwargs))
        if not all(isinstance(r, torch.Tensor) for r in results):
            raise NotImplementedError(
                f"{type(self).__name__}.query returned a non-tensor result "
                f"(continuous targets yield a (weights, samples) tuple); "
                f"override query_batch to define how those batch together."
            )
        if results[0].dim() == 1:
            return torch.stack(results, dim=0)
        return torch.cat(results, dim=0)
