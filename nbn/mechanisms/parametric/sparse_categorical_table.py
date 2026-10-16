"""Count-based categorical CPD that stores only the parent rows seen in data.

A dense table ``P(X | pa)`` has ``K * prod(parent cards)`` entries: exponential
in the in-degree.  With ``N`` training rows at most ``N`` parent
configurations carry data, and the maximum-likelihood table is *uniform* on
every other row (``CategoricalTableMechanism`` gives rows without data the
uniform fallback, for any ``alpha``: see its ``_smooth``).  This mechanism
stores the observed rows (sorted keys + their counts and logits) plus one
shared uniform row, so it fits in ``O(N)`` memory whatever the in-degree, and
answers exactly what the dense table answers:

* ``log_prob``, ``sample``, ``forward`` -- identical laws (same lookups);
* ``tabulate()`` -- rebuilds the dense table on demand (exact inference needs
  dense factors anyway: a node with ``p`` parents forces a factor over
  ``p + 1`` variables);
* smoothing -- the same ``_smooth`` on the compact count table, which is the
  dense table restricted to rows with data (the pseudo-count of a never-seen
  class is shared in proportion to ``n_r / N``, zero on empty rows).

Not supported: incremental ``update_local`` and ``fit_from_counts`` (they
assume the dense layout); they raise.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

from nbn.learning.weighting import validate_weights
from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism
from nbn.utils.device import accum_dtype


class SparseCategoricalTableMechanism(CategoricalTableMechanism):
    """``CategoricalTableMechanism`` with rows stored only where data exist."""

    supports_update: bool = False

    def __init__(self, alpha: float = 0.0) -> None:
        super().__init__(alpha=alpha)
        # Sorted dense row keys of the stored rows ([R_obs]); the last row of
        # ``_logits`` (index R_obs) is the shared uniform row.
        self.register_buffer("_keys", None)

    # ------------------------------------------------------------------ #
    def fit_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        parent_cards: List[int] | None = None,
        n_classes: int | None = None,
        weights: torch.Tensor | None = None,
        warm_start: bool = False,
        **kwargs,
    ) -> dict:
        x = x.long().reshape(-1)
        n = x.shape[0]
        device = x.device
        w = validate_weights(weights, n, where="SparseCategoricalTableMechanism.fit_local",
                             device=device)
        if w is not None and bool((w == 0).any()):
            keep = w > 0
            x, w = x[keep], w[keep]
            if parents is not None and parents.shape[-1] > 0:
                parents = parents.reshape(n, -1)[keep]
            n = x.shape[0]
        observed_k = int(x.max().item()) + 1 if n > 0 else 1
        k = observed_k if n_classes is None else max(int(n_classes), observed_k)
        self._n_classes = k
        self._class_values = torch.arange(k, device=device)

        if parents is None or parents.shape[-1] == 0:
            self._parent_cards = []
            self._parent_strides = []
            key = torch.zeros(n, dtype=torch.long, device=device)
            n_parent_states = 1
        else:
            parents = parents.long().reshape(n, -1)
            p = parents.shape[1]
            observed_pc = [int(parents[:, d].max()) + 1 for d in range(p)]
            if parent_cards is None:
                parent_cards = observed_pc
            else:
                parent_cards = [max(int(parent_cards[d]), observed_pc[d]) for d in range(p)]
            if sum(math.log2(max(c, 1)) for c in parent_cards) >= 62:
                raise ValueError("SparseCategoricalTableMechanism: parent space exceeds 2^62 "
                                 "configurations; row keys would overflow int64.")
            n_parent_states = self._set_parent_layout(parent_cards)
            key = CategoricalTableMechanism._parent_to_row_idx(self, parents)
        keys, inv = torch.unique(key, sorted=True, return_inverse=True)
        acc = accum_dtype(device)
        contrib = torch.ones(n, device=device, dtype=acc) if w is None else w
        counts = torch.zeros(keys.numel() * k, device=device, dtype=acc)
        counts.scatter_add_(0, inv * k + x, contrib)
        self._keys = keys
        self._set_counts(counts.reshape(keys.numel(), k))
        uniform = torch.full((1, k), -math.log(k), device=device,
                             dtype=self._logits.dtype)
        self._logits = nn.Parameter(torch.cat([self._logits.detach(), uniform], 0))
        return {"n_classes": k, "n_parent_states": n_parent_states,
                "n_stored_rows": int(keys.numel()), "warm_started": False}

    # ------------------------------------------------------------------ #
    def _parent_to_row_idx(self, parents: torch.Tensor) -> torch.Tensor:
        """Stored row of each parent configuration; the uniform row if unseen."""
        key = CategoricalTableMechanism._parent_to_row_idx(self, parents)
        keys = self._keys.to(key.device)
        r = keys.numel()
        pos = torch.searchsorted(keys, key.reshape(-1)).clamp(max=r - 1)
        found = keys[pos] == key.reshape(-1)
        return torch.where(found, pos, torch.full_like(pos, r)).reshape(key.shape)

    @property
    def is_fitted(self) -> bool:
        return self._logits is not None and self._keys is not None

    @property
    def n_stored_rows(self) -> int:
        return 0 if self._keys is None else int(self._keys.numel())

    def tabulate(self, parent_cards: list[int] | None = None) -> torch.Tensor:
        """Dense logits ``[*parent_cards, K]`` (rows without data: uniform)."""
        assert self.is_fitted, "Call fit_local before tabulate()."
        k = self._n_classes
        rows = math.prod(self._parent_cards) if self._parent_cards else 1
        dense = torch.full((rows, k), -math.log(k), device=self._logits.device,
                           dtype=self._logits.dtype)
        dense[self._keys] = self._logits.detach()[:-1]
        if not self._parent_cards:
            return dense[0]
        return dense.reshape(*self._parent_cards, k)

    tabulate_returns_view = False

    def tabulate_working_bytes(self, parent_cards: list[int] | None = None) -> int:
        rows = math.prod(self._parent_cards) if self._parent_cards else 1
        return int(rows * self._n_classes * self._logits.element_size())

    @property
    def cpt(self) -> torch.Tensor:
        k = self._n_classes
        return torch.softmax(self.tabulate().reshape(-1, k), dim=-1).detach()

    def update_local(self, *a, **k):
        raise NotImplementedError(
            "SparseCategoricalTableMechanism does not support update_local; refit, or "
            "use CategoricalTableMechanism when the dense table fits in memory.")

    def fit_from_counts(self, *a, **k):
        raise NotImplementedError(
            "SparseCategoricalTableMechanism.fit_from_counts is not supported "
            "(dense count tables are what it avoids); use fit_local.")
