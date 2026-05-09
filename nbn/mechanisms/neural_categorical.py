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
            # v0.8-#52: pre-fix initialised ``_root_logits`` to zeros
            # (uniform distribution over K classes) and returned
            # immediately, ignoring training data — surfaced by the
            # v0.7-#43 audit.  Closed-form MLE for a marginal categorical
            # is the empirical log-frequency; ``+1e-8`` smoothing avoids
            # ``log(0)`` when a class doesn't appear in train_data.
            counts = torch.bincount(x, minlength=k).float() + 1e-8
            log_freq = torch.log(counts / counts.sum())
            self._root_logits = nn.Parameter(log_freq.to(device))
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

    @property
    def is_fitted(self) -> bool:
        """True iff ``fit_local`` produced a usable CPD.

        Root nodes (``_d_pa == 0``): fitted iff ``_root_logits``
        parameter is registered.  Non-root: fitted iff the MLP
        ``net`` is built.  Pre-``fit_local``: ``_d_pa = 0`` (default)
        and ``_root_logits`` is unregistered, so this returns
        ``False``.
        """
        if self._d_pa == 0:
            return getattr(self, "_root_logits", None) is not None
        return self.net is not None

    def tabulate(
        self, parent_cards: list[int] | None = None
    ) -> torch.Tensor:
        """Return tabulated CPD in logit space.

        Root nodes (``self._d_pa == 0``): returns ``self._root_logits``
        of shape ``[K]``.

        Non-root nodes: enumerates the cartesian product of parent
        values once, runs a single batched ``_logits_from_parents``
        forward pass, and reshapes to ``[*parent_cards, K]``.
        ``parent_cards`` is required (the mechanism does not store
        them; embeddings store them implicitly via ``num_embeddings``
        but the no-embedding path would have no signal).

        At v0.6c-d cardinality / max_in_degree caps (4 / 4),
        ``prod(parent_cards) <= 256`` so enumeration is a single
        small MLP forward — well under the 10ms budget surfaced in
        Phase 0.
        """
        if self._d_pa == 0:
            assert self._root_logits is not None, "Call fit_local first."
            return self._root_logits
        if parent_cards is None:
            raise ValueError(
                "NeuralCategoricalMechanism.tabulate(): parent_cards is "
                "required for non-root nodes — the mechanism does not "
                "store parent cardinalities, and the MLP cannot be "
                "enumerated without them."
            )
        if len(parent_cards) != self._d_pa:
            raise ValueError(
                f"parent_cards has {len(parent_cards)} entries but "
                f"mechanism was fit with d_pa={self._d_pa}."
            )
        device = next(self.net.parameters()).device
        grids = torch.meshgrid(
            *[torch.arange(c, device=device) for c in parent_cards],
            indexing="ij",
        )
        pa_flat = torch.stack(
            [g.reshape(-1) for g in grids], dim=-1,
        ).float()  # [prod(pa_cards), d_pa]
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                logits_flat = self._logits_from_parents(pa_flat)
        finally:
            if was_training:
                self.train()
        return logits_flat.reshape(*parent_cards, self.n_classes)

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
