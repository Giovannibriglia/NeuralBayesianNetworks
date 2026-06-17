from __future__ import annotations

import warnings
from typing import List

import torch
import torch.nn as nn
from torch.distributions import Categorical

from nbn.mechanisms.base import Mechanism
from nbn.utils.batching import ensure_2d, flatten_samples

_CARDINALITY_WARN_THRESHOLD = 1_000_000


class CategoricalTableMechanism(Mechanism):
    """Tabular categorical CPD stored as a CPT in logit space.

    The conditional probability table (CPT) has shape
    ``[*parent_cardinalities, K]`` where ``K`` is the self cardinality.
    Indexing into the CPT uses stride arithmetic so the lookup is a single
    vectorized gather on GPU.

    Parameters
    ----------
    alpha:
        Dirichlet pseudo-count for Laplace smoothing (default 1.0 = uniform
        Dirichlet prior with 1 pseudo-observation per class).

    Notes
    -----
    Memory footprint grows as ``prod(parent_cardinalities) * K``.  When this
    product exceeds ``_CARDINALITY_WARN_THRESHOLD`` a ``UserWarning`` is
    emitted and you should switch to ``NeuralCategoricalMechanism`` instead.
    """

    is_discrete: bool = True

    def __init__(self, alpha: float = 0.0) -> None:
        # ``alpha`` is the per-class Dirichlet pseudo-count for Laplace
        # smoothing.  Default 0.0 matches pgmpy's ``MaximumLikelihoodEstimator``
        # exactly (no smoothing) so accuracy on small networks is on par.
        # Pass ``alpha=1.0`` to recover the v0.2 behaviour (uniform prior).
        super().__init__()
        self.alpha = float(alpha)
        # Populated by fit_local:
        self._logits: nn.Parameter | None = None  # shape [prod(pa_cards), K]
        self._n_classes: int = 0
        self._parent_cards: List[int] = []
        self._parent_strides: List[int] = []
        self._class_values: torch.Tensor | None = None  # [K]
        self.output_dim = 1

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        parent_cards: List[int] | None = None,
        n_classes: int | None = None,
        **kwargs,
    ) -> dict:
        """Vectorized count-based MLE with Dirichlet(alpha) smoothing.

        Parameters
        ----------
        x: shape ``[N]`` or ``[N, 1]`` — integer class labels.
        parents: shape ``[N, P]`` — integer parent class labels, or ``None``.
        parent_cards: declared cardinalities of each parent; inferred from
            observed data (max + 1) if None.
        n_classes: the variable's declared cardinality (Bug 1a, #127). When
            provided, the CPT spans the full declared range rather than only
            the classes observed in training — observed-distinct truncation
            was the source of the query-time factor-axis mismatches and
            ``select() index out of range`` errors. Falls back to
            observed-max + 1 when None (legacy behaviour for callers that
            don't have the declared cardinality).
        """
        x = x.long().reshape(-1)
        n = x.shape[0]
        device = x.device

        # Bug 1a (#127): span the *declared* cardinality, defending against
        # observed values that exceed it (shouldn't happen, but a truncated
        # CPT is far worse than an over-wide one).
        observed_k = int(x.max().item()) + 1 if n > 0 else 1
        k = observed_k if n_classes is None else max(int(n_classes), observed_k)
        self._n_classes = k
        # Categorical data is label-encoded to contiguous indices 0..K-1, so
        # the class values are simply ``arange(K)``. Storing the full declared
        # range (not just observed values) keeps the CPT axis aligned with the
        # cardinality the oracle and evidence indices assume.
        class_values = torch.arange(k, device=device)
        self._class_values = class_values

        if parents is None or parents.shape[-1] == 0:
            # Root node
            self._parent_cards = []
            self._parent_strides = []
            parent_idx = torch.zeros(n, dtype=torch.long, device=device)
            n_parent_states = 1
        else:
            parents = parents.long().reshape(n, -1)
            p = parents.shape[1]
            observed_pc = [int(parents[:, d].max()) + 1 for d in range(p)]
            if parent_cards is None:
                parent_cards = observed_pc
            else:
                # Honor declared parent cardinality (Bug 1a); never below what
                # we actually observed in this cell's training data.
                parent_cards = [
                    max(int(parent_cards[d]), observed_pc[d]) for d in range(p)
                ]
            self._parent_cards = list(parent_cards)
            strides = []
            stride = 1
            for c in reversed(parent_cards):
                strides.append(stride)
                stride *= c
            self._parent_strides = list(reversed(strides))
            parent_idx = sum(
                parents[:, d] * self._parent_strides[d] for d in range(p)
            )  # [N]
            n_parent_states = int(stride)

        total_states = n_parent_states * k
        if total_states > _CARDINALITY_WARN_THRESHOLD:
            warnings.warn(
                f"CategoricalTableMechanism: CPT has {total_states:,} entries "
                f"({n_parent_states} parent states × {k} classes). "
                "Consider NeuralCategoricalMechanism for large parent spaces.",
                UserWarning,
                stacklevel=2,
            )

        # class_values is the contiguous range 0..K-1, so labels are already
        # their own indices.
        x_idx = x  # [N]

        flat_idx = parent_idx * k + x_idx  # [N]
        counts = torch.zeros(n_parent_states * k, device=device, dtype=torch.float)
        counts.scatter_add_(0, flat_idx, torch.ones(n, device=device))
        counts = counts.reshape(n_parent_states, k)

        # Bug 1a (#127): a declared class that never appears anywhere in
        # training would otherwise be a structural zero (probability 0 for
        # every parent configuration). Give such classes a Laplace
        # pseudo-count of 1.0 so every declared class keeps nonzero
        # probability. This is a no-op when every declared class is observed,
        # preserving the pgmpy-MLE parity that ``alpha=0`` provides on
        # fully-observed networks.
        unobserved_classes = counts.sum(dim=0) == 0  # [k]
        if bool(unobserved_classes.any()):
            counts[:, unobserved_classes] += 1.0

        # Dirichlet smoothing
        counts = counts + self.alpha

        # Store as logits in log space (log of normalised probs)
        probs = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        logits_data = torch.log(probs.clamp_min(1e-12))
        self._logits = nn.Parameter(logits_data)
        return {"n_classes": k, "n_parent_states": n_parent_states}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parent_to_row_idx(self, parents: torch.Tensor) -> torch.Tensor:
        """Map ``[*, P]`` integer parent tensor to CPT row indices ``[*]``."""
        if not self._parent_strides:
            return torch.zeros(
                parents.shape[:-1], dtype=torch.long, device=parents.device
            )
        idx = torch.zeros(
            parents.shape[:-1], dtype=torch.long, device=parents.device
        )
        for d, stride in enumerate(self._parent_strides):
            idx = idx + parents[..., d].long() * stride
        return idx

    def _logits_for_parents(self, parents: torch.Tensor | None, batch_shape) -> torch.Tensor:
        """Return logits of shape ``[*batch_shape, K]``."""
        assert self._logits is not None, "Call fit_local before using this mechanism."
        if parents is None or (parents.shape[-1] == 0):
            # Root: all rows are identical
            logits = self._logits[0].expand(*batch_shape, -1)
        else:
            row_idx = self._parent_to_row_idx(parents)  # [*batch_shape]
            logits = self._logits[row_idx]  # [*batch_shape, K]
        return logits

    # ------------------------------------------------------------------
    # Distribution interface
    # ------------------------------------------------------------------

    def forward(self, parents: torch.Tensor | None) -> Categorical:
        """Return Categorical distribution conditioned on parents.

        Parameters
        ----------
        parents: ``[B, D_pa]`` integer tensor, or ``None`` for root nodes.

        Returns
        -------
        Categorical with batch_shape ``[B]`` and event_shape ``[]``.
        """
        assert self._logits is not None, "Call fit_local before forward()."
        if parents is None:
            b = 1
            logits = self._logits[0].unsqueeze(0)  # [1, K]
        else:
            parents_2d = ensure_2d(parents)
            b = parents_2d.shape[0]
            logits = self._logits_for_parents(parents_2d, (b,))  # [B, K]
        return Categorical(logits=logits)

    def sample(self, parents: torch.Tensor | None, n: int = 1) -> torch.Tensor:
        """Sample integer class indices → float values using class_values map.

        Returns
        -------
        torch.Tensor of shape ``[B, n, 1]``.
        """
        assert self._logits is not None
        if parents is None:
            b = 1
            logits = self._logits[0].unsqueeze(0).expand(b, -1)
        else:
            parents_2d = ensure_2d(parents).long()
            b = parents_2d.shape[0]
            parents_exp = parents_2d.unsqueeze(1).expand(-1, n, -1)  # [B, n, P]
            flat, _, _ = flatten_samples(parents_exp)
            row_idx = self._parent_to_row_idx(flat)  # [B*n]
            logits_flat = self._logits[row_idx]  # [B*n, K]
            idx = Categorical(logits=logits_flat).sample()  # [B*n]
            vals = self._class_values.to(device=idx.device, dtype=torch.float)
            out = vals[idx].reshape(b, n, 1)
            return out

        idx = Categorical(logits=logits.unsqueeze(1).expand(-1, n, -1).reshape(b * n, -1)).sample()
        vals = self._class_values.to(device=idx.device, dtype=torch.float)
        return vals[idx].reshape(b, n, 1)

    def log_prob(
        self, x: torch.Tensor, parents: torch.Tensor | None
    ) -> torch.Tensor:
        """Log-probability of integer class labels x.

        Parameters
        ----------
        x: shape ``[B]``, ``[B, 1]``, or ``[B, S, 1]``.
        parents: shape ``[B, D_pa]`` or ``[B, S, D_pa]``, or ``None``.

        Returns
        -------
        torch.Tensor of shape ``[B]`` or ``[B, S]``.
        """
        assert self._logits is not None
        # Normalise x shape
        squeeze_s = False
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [B, 1]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, 1]
            squeeze_s = True
        # x: [B, S, 1]
        b, s, _ = x.shape
        x_idx = x.long().squeeze(-1)  # [B, S]

        # Map class values → contiguous indices
        if self._class_values is not None:
            cv = self._class_values.to(device=x.device)
            cls_to_idx = torch.zeros(
                int(cv.max()) + 1, dtype=torch.long, device=x.device
            )
            cls_to_idx[cv.long()] = torch.arange(len(cv), device=x.device)
            x_idx = cls_to_idx[x_idx]

        if parents is None:
            logits = self._logits[0].unsqueeze(0).unsqueeze(0).expand(b, s, -1)
        else:
            if parents.dim() == 2:
                parents = parents.unsqueeze(1).expand(-1, s, -1)
            flat, _, _ = flatten_samples(parents.long())
            row_idx = self._parent_to_row_idx(flat)  # [B*S]
            logits_flat = self._logits[row_idx]  # [B*S, K]
            logits = logits_flat.reshape(b, s, -1)

        log_probs = torch.log_softmax(logits, dim=-1)  # [B, S, K]
        lp = log_probs.gather(-1, x_idx.unsqueeze(-1)).squeeze(-1)  # [B, S]
        if squeeze_s:
            lp = lp.squeeze(1)  # [B]
        return lp

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True iff ``fit_local`` populated the ``_logits`` tabulation."""
        return self._logits is not None

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def cpt(self) -> torch.Tensor:
        """CPT probabilities of shape ``[n_parent_states, K]``."""
        assert self._logits is not None
        return torch.softmax(self._logits, dim=-1).detach()

    def tabulate(
        self, parent_cards: list[int] | None = None
    ) -> torch.Tensor:
        """Return tabulated CPD in logit space (shape ``[*parent_cards, K]``).

        ``parent_cards`` is ignored when the mechanism's own
        ``_parent_cards`` (set by ``fit_local``) is non-empty — the
        stored tabulation already encodes parent cardinality.  Passed
        in for API uniformity with enumeration-based mechanisms (see
        ``Mechanism.tabulate``).
        """
        assert self._logits is not None, "Call fit_local before tabulate()."
        if not self._parent_cards:
            return self._logits[0]
        return self._logits.reshape(*self._parent_cards, self._n_classes)
