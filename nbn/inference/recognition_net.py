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
import torch.nn.functional as F
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


def _enc_width(head: _Head) -> int:
    """Width this node contributes when it appears as a *parent* of another.

    Discrete parents are one-hot encoded (``k`` columns); continuous parents
    pass their raw value(s) (``d_x`` columns).
    """
    return head.k if head.kind == "discrete" else head.d_x


class ParentConditionedProposal(nn.Module):
    """Parent-conditioned proposal for Engine A (β / P2).

    Unlike :class:`RecognitionNetwork` (an evidence-only marginal proposal
    ``q(xᵢ | e)`` emitted in a single forward pass — kept for Engine B / AVI),
    this proposal is ``q(xᵢ | e, paᵢ)``: a shared trunk encodes the evidence
    into a context vector, and **independent per-node heads** map
    ``(context, encoded parents)`` to that node's distribution parameters.  The
    engine calls the heads *inside* the ancestral-sampling loop with the actual
    sampled parent values, so the importance weight
    ``log p(xᵢ | pa_sampled) − log q(xᵢ | e, pa_sampled)`` no longer carries the
    factorization mismatch that collapses ESS for the marginal proposal (the
    discrete-network catastrophe; see the P2 investigation, α-positive on
    chain/child/alarm).

    Heads are sized to each node's own structural parent set — no padding.
    Input width = ``d_ctx + Σ_{p∈parents} enc_width(p)``; discrete parents are
    one-hot by their own cardinality, continuous parents pass raw values.

    Scope (v1, β): discrete / lg / mdn heads with **scalar** parents
    (``d_x == 1``); flow keeps the Gaussian surrogate (P3 / #185-L1).  The
    public surface mirrors what ``_estimate_ess_fraction`` (P1) needs —
    ``node_order``, ``make_dist`` — and adds ``context`` / ``node_params`` for
    the in-loop call.
    """

    def __init__(
        self,
        model,
        d_ctx: int = 128,
        trunk_hidden=(128, 128),
        head_hidden: int = 64,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.node_order: List[str] = list(model.dag.topological_order())
        self.node_index: Dict[str, int] = {n: i for i, n in enumerate(self.node_order)}
        n = len(self.node_order)

        self.heads: Dict[str, _Head] = {
            node: _head_for(model.mechanisms[node], model.variables.get(node))
            for node in self.node_order
        }
        # Parent positions (into node_order) per node, in DAG-parent order.
        self.parents_idx: List[List[int]] = [
            [self.node_index[p] for p in model.dag.parents(node)]
            for node in self.node_order
        ]
        # v1 guard: scalar parents only (multi-dim continuous parents need a
        # column-layout encoder — deferred with flow to P3).
        for i, node in enumerate(self.node_order):
            for p in self.parents_idx[i]:
                ph = self.heads[self.node_order[p]]
                if ph.kind != "discrete" and ph.d_x != 1:
                    raise NotImplementedError(
                        "ParentConditionedProposal v1 supports scalar parents "
                        f"only; parent {self.node_order[p]!r} of {node!r} has "
                        f"d_x={ph.d_x}. Multi-dim continuous parents are deferred."
                    )

        self.d_ctx = d_ctx
        # Shared evidence trunk: [evidence values | mask] → context.
        self.trunk = _build_mlp(2 * n, tuple(trunk_hidden), d_ctx, activation)
        # Independent per-node heads, indexed by topo position.  A ModuleList
        # (not ModuleDict) sidesteps the dotted-node-name ModuleDict bug (#200).
        self.head_mlps = nn.ModuleList()
        self.parent_enc_dim: List[int] = []
        for i, node in enumerate(self.node_order):
            enc = sum(_enc_width(self.heads[self.node_order[p]]) for p in self.parents_idx[i])
            self.parent_enc_dim.append(enc)
            self.head_mlps.append(
                _build_mlp(d_ctx + enc, (head_hidden,), self.heads[node].param_size, activation)
            )

    # ------------------------------------------------------------------
    # Forward pieces — trunk once per query, heads in the sampling loop
    # ------------------------------------------------------------------

    def context(self, evidence_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``[B, n] , [B, n] → [B, d_ctx]`` evidence context (computed once)."""
        return self.trunk(torch.cat([evidence_values, mask], dim=-1))

    def encode_parents(self, node_i: int, parent_values: torch.Tensor | None) -> torch.Tensor | None:
        """Encode a node's parents: one-hot discrete (by cardinality), raw continuous.

        ``parent_values``: ``[M, n_parents]`` scalar columns in ``parents_idx``
        order.  Returns ``[M, parent_enc_dim[node_i]]`` (or ``None`` for roots).
        """
        pidx = self.parents_idx[node_i]
        if not pidx:
            return None
        cols = []
        for j, p in enumerate(pidx):
            ph = self.heads[self.node_order[p]]
            col = parent_values[:, j]
            if ph.kind == "discrete":
                cols.append(F.one_hot(col.long(), ph.k).float())
            else:
                cols.append(col.unsqueeze(-1).float())  # scalar continuous parent
        return torch.cat(cols, dim=-1)

    def node_params(
        self, node_i: int, context: torch.Tensor, parent_values: torch.Tensor | None
    ) -> torch.Tensor:
        """``q(x_{node_i} | context, parents)`` parameters → ``[M, param_size]``.

        ``context`` is ``[M, d_ctx]`` (already broadcast to the M=B·S particles
        by the caller); ``parent_values`` is ``[M, n_parents]`` or ``None``.
        """
        enc = self.encode_parents(node_i, parent_values)
        inp = context if enc is None else torch.cat([context, enc], dim=-1)
        return self.head_mlps[node_i](inp)

    # ------------------------------------------------------------------
    # Per-node distribution construction (mirrors RecognitionNetwork)
    # ------------------------------------------------------------------

    def make_dist(self, node: str, params: torch.Tensor) -> Distribution:
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
            return MixtureSameFamily(Categorical(logits=logits),
                                     Independent(Normal(loc, scale), 1))
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
        """Log q(target | …) for a given value column (training NLL)."""
        dist = self.make_dist(node, params)
        if self.is_discrete(node):
            return dist.log_prob(target.long())
        t = target if target.dim() >= 2 else target.unsqueeze(-1)
        return dist.log_prob(t)
