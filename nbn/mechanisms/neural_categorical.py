from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from nbn.mechanisms.base import Mechanism
from nbn.mechanisms.mdn import _build_mlp
from nbn.utils.batching import ensure_2d, flatten_samples


class NeuralCategoricalMechanism(Mechanism):
    """MLP-based categorical CPD: logits = MLP(embed(pa)).

    Use this instead of ``CategoricalTableMechanism`` when the product of
    parent cardinalities is large (> ~1 million states), since this scales
    linearly rather than exponentially in parent cardinality.

    Parameters
    ----------
    n_classes: int
        Number of output classes K.
    hidden: tuple of int
        MLP hidden layer widths.
    embedding_dim: int | None
        If not None, discrete parent values are embedded (requires
        ``parent_cards`` to be passed at fit time).
    activation: str
        Activation function.
    """

    is_discrete: bool = True

    def __init__(
        self,
        n_classes: int = 2,
        hidden: Tuple[int, ...] = (64, 64),
        embedding_dim: int | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.hidden = tuple(hidden)
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.output_dim = 1
        self.net: nn.Module | None = None
        self.embeddings: nn.ModuleList | None = None
        self._d_pa = 0

    def fit_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        parent_cards: list | None = None,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 512,
        **kwargs,
    ) -> dict:
        x = x.long().reshape(-1)
        k = self.n_classes
        device = x.device

        if parents is None or parents.shape[-1] == 0:
            self._d_pa = 0
            self._root_logits = nn.Parameter(torch.zeros(k, device=device))
            return {"n_classes": k}

        parents = ensure_2d(parents).to(device=device)
        d_pa = parents.shape[1]
        self._d_pa = d_pa

        if self.embedding_dim is not None and parent_cards is not None:
            self.embeddings = nn.ModuleList([
                nn.Embedding(c, self.embedding_dim) for c in parent_cards
            ])
            in_dim = self.embedding_dim * d_pa
        else:
            in_dim = d_pa

        self.net = _build_mlp(in_dim, self.hidden, k, self.activation).to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(parents, x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.train()
        for _ in range(epochs):
            for bp, bx in loader:
                logits = self._logits_from_parents(bp.float())
                loss = nn.CrossEntropyLoss()(logits, bx)
                opt.zero_grad(); loss.backward()
                opt.step()
        self.eval()
        return {"n_classes": k, "d_pa": d_pa}

    def _logits_from_parents(self, parents: torch.Tensor) -> torch.Tensor:
        if self.embeddings is not None:
            parts = [self.embeddings[i](parents[:, i].long()) for i in range(parents.shape[1])]
            inp = torch.cat(parts, dim=-1)
        else:
            inp = parents.float()
        return self.net(inp)

    def forward(self, parents: torch.Tensor | None) -> Categorical:
        if self._d_pa == 0 or parents is None:
            b = 1 if parents is None else ensure_2d(parents).shape[0]
            return Categorical(logits=self._root_logits.unsqueeze(0).expand(b, -1))
        parents_2d = ensure_2d(parents)
        return Categorical(logits=self._logits_from_parents(parents_2d))

    def log_prob(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        squeeze_s = False
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.dim() == 2:
            x = x.unsqueeze(1); squeeze_s = True
        b, s, _ = x.shape
        x_idx = x.long().squeeze(-1)  # [B, S]

        if self._d_pa == 0 or parents is None:
            logits = self._root_logits.view(1, 1, -1).expand(b, s, -1)
        else:
            if parents.dim() == 2:
                parents = parents.unsqueeze(1).expand(-1, s, -1)
            flat, _, _ = flatten_samples(parents)
            logits = self._logits_from_parents(flat.float()).reshape(b, s, -1)

        lp = torch.log_softmax(logits, dim=-1).gather(-1, x_idx.unsqueeze(-1)).squeeze(-1)
        return lp.squeeze(1) if squeeze_s else lp
