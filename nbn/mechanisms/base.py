from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.distributions import Distribution


class Mechanism(nn.Module, ABC):
    """Abstract base class for a learnable conditional distribution P(X | pa(X)).

    Design contract
    ---------------
    * ``forward(parents)`` returns a ``torch.distributions.Distribution`` whose
      ``batch_shape`` starts with ``[B]`` (or ``[B, S]`` when parents is
      ``[B, S, D_pa]``).
    * ``log_prob(x, parents)`` → ``[B]`` or ``[B, S]``.
    * ``sample(parents, n)`` → ``[B, S, D_x]``.
    * Every parameter is an ``nn.Parameter`` — autograd, ``.to(device)``,
      ``torch.compile`` all work for free.

    Shape conventions
    -----------------
    parents: ``[B, D_pa]`` (2-D) or ``[B, S, D_pa]`` (3-D with particle dim).
    x:       ``[B, D_x]``  (2-D) or ``[B, S, D_x]`` (3-D).
    """

    is_discrete: bool = False
    output_dim: int = 1

    @abstractmethod
    def forward(
        self, parents: torch.Tensor | None
    ) -> Distribution:
        """Return the conditional distribution P(X | parents).

        Parameters
        ----------
        parents:
            Conditioning context of shape ``[B, D_pa]`` or ``[B, S, D_pa]``.
            ``None`` for root nodes (no parents).

        Returns
        -------
        Distribution
            A batched distribution with batch_shape starting with ``[B]``.
        """

    def log_prob(
        self, x: torch.Tensor, parents: torch.Tensor | None
    ) -> torch.Tensor:
        """Log conditional probability log P(x | parents).

        Parameters
        ----------
        x: shape ``[B, D_x]`` or ``[B, S, D_x]``.
        parents: shape ``[B, D_pa]`` or ``[B, S, D_pa]``, or ``None``.

        Returns
        -------
        torch.Tensor of shape ``[B]`` or ``[B, S]``.
        """
        return self.forward(parents).log_prob(x)

    def sample(
        self, parents: torch.Tensor | None, n: int = 1
    ) -> torch.Tensor:
        """Draw ``n`` samples per row.

        Parameters
        ----------
        parents: shape ``[B, D_pa]``, or ``None`` for root nodes.
        n: number of samples per batch element.

        Returns
        -------
        torch.Tensor of shape ``[B, n, D_x]``.
        """
        dist = self.forward(parents)
        # dist.batch_shape = [B]; sample n → [n, B, D_x] → permute to [B, n, D_x]
        s = dist.sample((n,))
        if s.dim() == 2:
            # [n, B] → [B, n, 1]
            return s.T.unsqueeze(-1)
        if s.dim() == 3:
            # [n, B, D_x] → [B, n, D_x]
            return s.permute(1, 0, 2)
        return s

    def rsample(
        self, parents: torch.Tensor | None, n: int = 1
    ) -> torch.Tensor:
        """Reparameterized sample (differentiable).  Raises if not supported."""
        dist = self.forward(parents)
        if not dist.has_rsample:
            raise RuntimeError(
                f"{type(self).__name__} does not support reparameterized sampling"
            )
        s = dist.rsample((n,))
        if s.dim() == 2:
            return s.T.unsqueeze(-1)
        if s.dim() == 3:
            return s.permute(1, 0, 2)
        return s

    @abstractmethod
    def fit_local(
        self, x: torch.Tensor, parents: torch.Tensor | None, **kwargs
    ) -> dict:
        """Closed-form or small-loop local MLE.  Returns a dict of metrics."""

    def tabulate(
        self, parent_cards: list[int] | None = None
    ) -> torch.Tensor:
        """Return the tabulated CPD as a tensor.

        Shape ``[*parent_cards, K]`` for non-root discrete mechanisms;
        ``[K]`` for root.  Returned values are in logit space — apply
        ``softmax`` along the last axis to get a normalised CPD.  Some
        mechanisms (e.g. ``CategoricalTableMechanism``) store log-probs
        internally; others (``NeuralCategoricalMechanism``) return
        unnormalised logits.  Both are equivalent under softmax
        shift-invariance.

        ``parent_cards`` is the cardinality of each parent in DAG
        order; required for mechanisms that enumerate via
        ``forward()`` (e.g. neural categorical), and may be ignored
        by mechanisms that already store the tabulation
        (e.g. categorical table).

        Default raises ``NotImplementedError`` — only meaningful for
        discrete mechanisms.  Used by:

        * The benchmarking metric site
          (``benchmarking/crash_test_runner.py:_avg_tv_per_node``)
        * The variable-elimination engine
          (``nbn/inference/tensor_ve.py:_extract_factors``)

        Both previously read ``mech._logits`` directly, which broke
        on mechanisms whose CPD is computed per-call rather than
        stored.  See v0.8 issues #59 and #26.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.tabulate() is only defined for "
            f"discrete mechanisms."
        )
