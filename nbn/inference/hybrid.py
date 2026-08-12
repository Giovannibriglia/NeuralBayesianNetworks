from __future__ import annotations

import logging
from typing import Dict, List

import networkx as nx
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
        except (nx.NetworkXError, ValueError):
            # Treewidth could not be computed for this graph (a malformed DAG
            # surfaces as nx.NetworkXError; a degenerate greedy min-fill step as
            # ValueError) -> fall back to LW. Narrowed from a bare `except`
            # (#232): the bare catch silently masked the induced_width bugs for
            # months (#226/#231), degrading exact VE to stochastic LW. Other
            # exceptions (AttributeError / TypeError / RuntimeError) signal real
            # bugs and must propagate, not silently downgrade the engine.
            tw = self.treewidth_threshold + 1

        if tw <= self.treewidth_threshold:
            self._last_engine = "tensor_ve"
            return self._ve
        logger.debug(
            "Treewidth %d > threshold %d; using likelihood weighting.", tw, self.treewidth_threshold
        )
        self._last_engine = "likelihood_weighting"
        return self._lw

    def _dispatch(self, method: str, model, targets, evidence, kwargs):
        """Route to the selected engine with a VE→LW out-of-memory safety net.

        VE's memory guard (and CUDA itself) surfaces an over-budget plan as
        ``torch.cuda.OutOfMemoryError``. For the router that must not be
        terminal: LW answers the same query in bounded memory, so degrade to
        it (with a warning) instead of failing the query. Only OOM is caught
        — any other exception signals a real bug and propagates (#232
        precedent: broad catches here silently masked engine bugs).
        """
        engine = self._select(model)
        try:
            return getattr(engine, method)(model, targets, evidence, **kwargs)
        except torch.cuda.OutOfMemoryError:
            if engine is self._lw:
                raise  # LW itself OOMed — nothing further to degrade to.
            logger.warning(
                "TensorVE exceeded the memory budget; retrying with "
                "likelihood weighting."
            )
            self._last_engine = "likelihood_weighting"
            return getattr(self._lw, method)(model, targets, evidence, **kwargs)

    def query(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._dispatch("query", model, targets, evidence, kwargs)

    def query_batch(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        return self._dispatch("query_batch", model, targets, evidence, kwargs)
