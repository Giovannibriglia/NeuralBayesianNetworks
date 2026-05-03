from __future__ import annotations

import logging
from typing import Dict, List

import torch

from nbn.inference.base import InferenceEngine
from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
from nbn.inference.tensor_ve import TensorVariableElimination

logger = logging.getLogger(__name__)


class HybridRouter(InferenceEngine):
    """Auto-selecting inference engine.

    Routing logic:
    1. All mechanisms discrete **and** induced treewidth ≤ τ → ``TensorVariableElimination``.
    2. Otherwise → ``LikelihoodWeightingEngine``.

    Parameters
    ----------
    treewidth_threshold: int
        Treewidth limit above which we fall back to LW.
    n_samples: int
        Number of samples for the likelihood-weighting fallback.
    """

    def __init__(
        self,
        treewidth_threshold: int = 25,
        n_samples: int = 2048,
    ) -> None:
        self.treewidth_threshold = treewidth_threshold
        self._ve = TensorVariableElimination(treewidth_threshold)
        self._lw = LikelihoodWeightingEngine(n_samples)
        self._last_engine: str | None = None

    def _select(self, model) -> InferenceEngine:
        all_discrete = all(
            m.is_discrete for m in model.mechanisms.values()
        )
        if not all_discrete:
            self._last_engine = "likelihood_weighting"
            return self._lw
        try:
            tw = model.dag.induced_width()
        except Exception:
            tw = self.treewidth_threshold + 1

        if tw <= self.treewidth_threshold:
            self._last_engine = "tensor_ve"
            return self._ve
        logger.debug(
            "Treewidth %d > threshold %d; using likelihood weighting.", tw, self.treewidth_threshold
        )
        self._last_engine = "likelihood_weighting"
        return self._lw

    def query(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._select(model).query(model, targets, evidence, **kwargs)

    def query_batch(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        return self._select(model).query_batch(model, targets, evidence, **kwargs)
