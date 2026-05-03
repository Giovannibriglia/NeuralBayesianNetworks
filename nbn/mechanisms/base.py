from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

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
        self, parents: Optional[torch.Tensor]
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
        self, x: torch.Tensor, parents: Optional[torch.Tensor]
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
        self, parents: Optional[torch.Tensor], n: int = 1
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
        self, parents: Optional[torch.Tensor], n: int = 1
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
        self, x: torch.Tensor, parents: Optional[torch.Tensor], **kwargs
    ) -> dict:
        """Closed-form or small-loop local MLE.  Returns a dict of metrics."""
