"""Recognition network for Engine A (#181).

Maps ``[B, 2n]`` (evidence values concat observed-mask) to per-node
proposal parameters via a single flat MLP.  Per-node output heads
construct mechanism-family distributions so the proposal reuses the same
``torch.distributions`` classes the model's mechanisms use — sampling and
``log_prob`` are then directly compatible with the importance weights.

Stage (a) heads: ``cat`` / ``neuralcat`` (Categorical) and ``mdn``
(MixtureSameFamily).  Stage (b) adds ``lg`` (Normal) and ``flow``.

See docs/v0.14-batched-inference-engines-research.md (Engine A section)
for the research context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    MixtureSameFamily,
    Normal,
)

from nbn.mechanisms.categorical_table import CategoricalTableMechanism
from nbn.mechanisms.linear_gaussian import LinearGaussianMechanism
from nbn.mechanisms.mdn import MDNMechanism, _build_mlp
from nbn.mechanisms.neural_categorical import NeuralCategoricalMechanism

# Numerical guards mirrored from the mechanisms (issue #95 / Bug C lineage):
# bound log-scale before exp() to keep float32 finite on pathological inputs.
_LOG_SCALE_CLAMP = 7.0
_MIN_SCALE = 1e-3


@dataclass
class _Head:
    """Per-node proposal-head metadata.

    ``kind``: one of ``"discrete"``, ``"mdn"``, ``"lg"``, ``"flow"``.
    ``param_size``: number of MLP outputs allocated to this node.
    ``k``: number of classes (discrete) or mixture components (mdn).
    ``d_x``: event dimension (continuous); 1 for discrete (scalar event).
    """

    kind: str
    param_size: int
    k: int
    d_x: int


def _head_for(mech, var) -> _Head:
    """Classify a fitted mechanism into a proposal head.

    Reads dimensions from the *fitted* mechanism (the recognition net is
    built after ``model.fit()``), so ``K`` / ``d_x`` always match the
    model's own parameterisation — no cardinality drift against the
    importance weights.
    """
    if isinstance(mech, CategoricalTableMechanism):
        k = int(mech._n_classes) or int(getattr(var, "cardinality", 0) or 2)
        return _Head("discrete", param_size=k, k=k, d_x=1)
    if isinstance(mech, NeuralCategoricalMechanism):
        k = int(mech.n_classes)
        return _Head("discrete", param_size=k, k=k, d_x=1)
    if isinstance(mech, MDNMechanism):
        k = int(mech.num_components)
        d_x = int(mech.output_dim)
        # logits[k] + loc[k*d_x] + log_scale[k*d_x]
        return _Head("mdn", param_size=k + 2 * k * d_x, k=k, d_x=d_x)
    if isinstance(mech, LinearGaussianMechanism):
        d_x = int(mech.output_dim)
        # mean[d_x] + log_var[d_x]
        return _Head("lg", param_size=2 * d_x, k=1, d_x=d_x)
    # flow is added in Stage (b); see _flow_head.
    flow_head = _flow_head(mech)
    if flow_head is not None:
        return flow_head
    raise NotImplementedError(
        f"RecognitionNetwork has no proposal head for mechanism type "
        f"{type(mech).__name__!r}. Supported: CategoricalTable, "
        f"NeuralCategorical, MDN, LinearGaussian, NormalizingFlow."
    )


def _flow_head(mech) -> _Head | None:
    """Stage (b): NormalizingFlow proposal head.

    A flow's exact density is intractable as a cheap closed form to emit
    from the MLP, so the proposal approximates the flow node with a
    diagonal-Gaussian surrogate (mean + log-var per output dim).  This
    keeps Engine A's asymptotic-correctness guarantee intact: the
    importance weight ``log p_flow(x|pa) - log q_gauss(x|e)`` uses the
    *true* flow density for ``p`` and the Gaussian only as the proposal,
    which merely needs matching support (all of R^{d_x}).
    """
    try:
        from nbn.mechanisms.normalizing_flow import NormalizingFlowMechanism
    except Exception:  # pragma: no cover - zuko optional
        return None
    if isinstance(mech, NormalizingFlowMechanism):
        d_x = int(getattr(mech, "output_dim", 1) or 1)
        return _Head("flow", param_size=2 * d_x, k=1, d_x=d_x)
    return None


class RecognitionNetwork(nn.Module):
    """Evidence-conditioned proposal network (flat MLP) for Engine A.

    Parameters
    ----------
    model:
        A *fitted* ``NeuralBayesianNetwork``.  Node order, per-node head
        type, and parameter sizes are read from the model's mechanisms.
    hidden:
        MLP hidden-layer widths.
    """

    def __init__(self, model, hidden=(128, 128), activation: str = "relu") -> None:
        super().__init__()
        self.node_order: List[str] = list(model.dag.topological_order())
        self.node_index: Dict[str, int] = {n: i for i, n in enumerate(self.node_order)}

        self.heads: Dict[str, _Head] = {}
        self.param_slices: Dict[str, slice] = {}
        offset = 0
        for node in self.node_order:
            head = _head_for(model.mechanisms[node], model.variables.get(node))
            self.heads[node] = head
            self.param_slices[node] = slice(offset, offset + head.param_size)
            offset += head.param_size
        self.total_param = offset

        n = len(self.node_order)
        # Input: [evidence values | observed mask] → [B, 2n]
        self.mlp = _build_mlp(2 * n, tuple(hidden), self.total_param, activation)

    # ------------------------------------------------------------------
    # Forward / slicing
    # ------------------------------------------------------------------

    def forward(self, evidence_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``[B, n] , [B, n] → [B, total_param]``."""
        return self.mlp(torch.cat([evidence_values, mask], dim=-1))

    def node_param_slice(self, params: torch.Tensor, node: str) -> torch.Tensor:
        """Extract a node's parameter block from the flat output ``[..., total_param]``."""
        return params[..., self.param_slices[node]]

    # ------------------------------------------------------------------
    # Per-node proposal distribution construction
    # ------------------------------------------------------------------

    def make_dist(self, node: str, params: torch.Tensor) -> Distribution:
        """Build the proposal distribution ``q(node | evidence)`` from its params.

        ``params`` has shape ``[..., param_size]``; the returned distribution
        has matching batch shape.
        """
        head = self.heads[node]
        if head.kind == "discrete":
            return Categorical(logits=params)
        if head.kind == "mdn":
            k, d_x = head.k, head.d_x
            logits = params[..., :k]
            rest = params[..., k:].reshape(*params.shape[:-1], k, 2 * d_x)
            loc = rest[..., :d_x]
            log_scale = rest[..., d_x:].clamp(max=_LOG_SCALE_CLAMP)
            scale = log_scale.exp().clamp_min(_MIN_SCALE)
            mix = Categorical(logits=logits)
            comp = Independent(Normal(loc, scale), 1)
            return MixtureSameFamily(mix, comp)
        if head.kind in ("lg", "flow"):
            d_x = head.d_x
            mean = params[..., :d_x]
            log_var = params[..., d_x:].clamp(max=2 * _LOG_SCALE_CLAMP)
            scale = (0.5 * log_var).exp().clamp_min(_MIN_SCALE)
            return Independent(Normal(mean, scale), 1)
        raise NotImplementedError(head.kind)  # pragma: no cover

    def is_discrete(self, node: str) -> bool:
        return self.heads[node].kind == "discrete"

    def log_prob_target(
        self, node: str, params: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Log q(target | evidence) for a *given* value column (training NLL).

        ``target`` is the raw value column ``[N]``; shape/dtype are coerced
        to what the per-node distribution expects.
        """
        dist = self.make_dist(node, params)
        if self.is_discrete(node):
            return dist.log_prob(target.long())
        # Continuous heads have event_shape [d_x]; a scalar column needs a
        # trailing dim.  Multi-dim continuous nodes already arrive [N, d_x].
        t = target if target.dim() >= 2 else target.unsqueeze(-1)
        return dist.log_prob(t)
