from __future__ import annotations

import logging
from typing import Dict, List

import networkx as nx
import torch

from nbn.inference._memory import is_oom_error
from nbn.inference.base import InferenceEngine
from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
from nbn.inference.tensor_ve import TensorVariableElimination

logger = logging.getLogger(__name__)


class HybridRouter(InferenceEngine):
    """Auto-selecting inference engine.

    Routing logic:

    1. More than one target → ``LikelihoodWeightingEngine`` (VE answers
       exactly one target; a multi-target query used to crash, B6).
    2. Any continuous mechanism → LW.
    3. Induced treewidth > τ → LW (a cheap structural pre-filter).
    4. Otherwise price the *actual* VE plan of this query: its estimated
       peak bytes for one row against the device memory budget (VE splits
       larger batches into rows-chunks that fit).  Fits → exact
       ``TensorVariableElimination``; does not → LW.  The unweighted
       treewidth alone ignored cardinalities: a 40-node, in-degree-5,
       card-4 network had width 21 (→ VE) and an estimated 3.3 TB peak (B5).
    5. If VE still runs out of memory on any device (CUDA OOM, MPS
       "out of memory", CPU allocator failure), the query is retried with LW.

    Parameters
    ----------
    treewidth_threshold: int
        Treewidth limit above which we fall back to LW.
    n_samples: int
        Number of samples for the likelihood-weighting fallback.
    memory_budget: float, optional
        Hard cap in bytes on VE's estimated peak; combined with the device
        headroom (see ``TensorVariableElimination.memory_budget_bytes``).
    """

    def __init__(
        self,
        treewidth_threshold: int = 25,
        n_samples: int = 2048,
        memory_budget: float | None = None,
    ) -> None:
        self.treewidth_threshold = treewidth_threshold
        self._ve = TensorVariableElimination(
            treewidth_threshold, memory_budget=memory_budget,
        )
        self._lw = LikelihoodWeightingEngine(n_samples)
        self._last_engine: str | None = None

    def _use_lw(self, reason: str, *args) -> InferenceEngine:
        logger.debug(reason, *args)
        self._last_engine = "likelihood_weighting"
        return self._lw

    def _select(
        self,
        model,
        targets: List[str] | None = None,
        evidence: Dict[str, torch.Tensor] | None = None,
        do: Dict[str, torch.Tensor] | None = None,
        order: str = "min_fill",
    ) -> InferenceEngine:
        if targets is not None and len(targets) != 1:
            return self._use_lw("%d targets; VE answers one, using LW.", len(targets))
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

        if tw > self.treewidth_threshold:
            return self._use_lw(
                "Treewidth %d > threshold %d; using likelihood weighting.",
                tw, self.treewidth_threshold,
            )
        if targets is not None and not self._ve_fits(
            model, targets[0], evidence, do, order,
        ):
            return self._use_lw(
                "VE plan for %r exceeds the memory budget; using likelihood "
                "weighting.", targets[0],
            )
        self._last_engine = "tensor_ve"
        return self._ve

    def _ve_fits(self, model, target, evidence, do, order) -> bool:
        """Whether one row of VE's plan for this query fits the memory budget."""
        evidence, do = evidence or {}, do or {}
        if target in evidence or target in do:
            return True  # answered as a point mass, no elimination
        return self._ve.fits_memory(
            model, target, tuple(evidence), tuple(do), order=order,
        )

    def _dispatch(self, method: str, model, targets, evidence, kwargs):
        """Route to the selected engine with a VE→LW out-of-memory safety net.

        VE's memory guard (and the allocators themselves) surface an
        over-budget plan as an out-of-memory error — ``torch.OutOfMemoryError``
        on CUDA and from the guard on every device, a ``RuntimeError`` with
        an "out of memory" / "can't allocate memory" message on MPS and CPU.
        For the router that must not be terminal: LW answers the same query
        in bounded memory, so degrade to it (with a warning) instead of
        failing the query.  Only OOM is caught — any other exception signals
        a real bug and propagates (#232 precedent: broad catches here
        silently masked engine bugs).
        """
        engine = self._select(
            model, targets, evidence, kwargs.get("do"),
            kwargs.get("order", "min_fill"),
        )
        try:
            return getattr(engine, method)(model, targets, evidence, **kwargs)
        except Exception as exc:
            if engine is self._lw or not is_oom_error(exc):
                raise  # LW itself OOMed, or not an OOM at all.
            logger.warning(
                "TensorVE exceeded the memory budget (%s); retrying with "
                "likelihood weighting.", type(exc).__name__,
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
