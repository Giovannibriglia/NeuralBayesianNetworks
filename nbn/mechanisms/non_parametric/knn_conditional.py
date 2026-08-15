"""k-Nearest-Neighbour conditional density mechanism (non-parametric).

For a query ``x`` we find its ``k`` nearest neighbours among the training
parents and build a *local* conditional from their child values:

* **Continuous child** — a ``k``-component Gaussian mixture (local KDE) with
  uniform weights ``1/k`` centred at the neighbours' child values, using an
  *adaptive* per-query bandwidth derived from the local child spread.  This is
  the bandwidth-adaptive complement to :class:`ConditionalKDEMechanism`
  (Loftsgaarden--Quesenberry 1965 in spirit: resolution follows local
  density).
* **Discrete child** — the Laplace/Lidstone-smoothed empirical distribution of
  the neighbours' class labels, returned as a ``torch.distributions.Categorical``.

A root node (no parents) has no parent space to search, so the estimator
degrades gracefully to the marginal: a full-sample KDE for a continuous child,
or smoothed global class frequencies for a discrete child.

Memory: the neighbour search tiles over the **query** dimension, so peak
memory is ``O(query_chunk * N)`` rather than ``O(M * N)``.  (A KeOps
``argKmin`` backend would make this linear-memory and FAISS-competitive; kept
optional.)
"""
from __future__ import annotations

import math

import torch
from torch.distributions import Categorical, Distribution

from nbn.mechanisms.base import Mechanism
from nbn.utils.batching import _sanitise_parents, ensure_2d, flatten_samples

_LOG_2PI = math.log(2.0 * math.pi)


class _KNNContinuousDistribution(Distribution):
    """Distribution-API wrapper for the continuous-child kNN estimator."""

    has_rsample = False
    arg_constraints: dict = {}

    def __init__(self, mech: KNNConditionalMechanism, parents: torch.Tensor | None, b: int) -> None:
        self._mech = mech
        self._parents = parents
        super().__init__(
            batch_shape=torch.Size([b]),
            event_shape=torch.Size([mech.output_dim]),
            validate_args=False,
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._mech.log_prob(value, self._parents)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:  # type: ignore[override]
        n = 1
        for s in sample_shape:
            n *= int(s)
        out = self._mech.sample(self._parents, n=n).permute(1, 0, 2)  # [n, B, D_x]
        return out.reshape(*sample_shape, *out.shape[1:]) if sample_shape else out[0]


class KNNConditionalMechanism(Mechanism):
    """k-NN conditional density / distribution estimator.

    Parameters
    ----------
    k:
        Neighbour count.  ``None`` → ``round(sqrt(N))`` chosen at fit time
        (the classic consistency-preserving rule ``k → ∞, k/N → 0``).
    discrete_child:
        If True the child is categorical and ``forward`` returns a
        ``Categorical`` from smoothed neighbour label frequencies.
    alpha:
        Lidstone pseudo-count for the discrete child (``1.0`` = Laplace).
    bw_factor:
        Multiplier on the adaptive continuous bandwidth.
    min_bandwidth:
        Floor on the continuous bandwidth.
    query_chunk:
        Query rows per neighbour-search tile.
    """

    def __init__(
        self,
        k: int | None = None,
        discrete_child: bool = False,
        alpha: float = 1.0,
        bw_factor: float = 1.0,
        min_bandwidth: float = 1e-3,
        query_chunk: int = 1024,
    ) -> None:
        super().__init__()
        self.k = None if k is None else int(k)
        self.discrete_child = bool(discrete_child)
        self.is_discrete = bool(discrete_child)
        self.alpha = float(alpha)
        self.bw_factor = float(bw_factor)
        self.min_bandwidth = float(min_bandwidth)
        self.query_chunk = int(query_chunk)
        self.output_dim = 1
        self._d_pa = 0
        self._n_classes = 0
        self._k_eff = 0
        self.register_buffer("_train_y", None)    # [N, D_x] (float) or [N] long for discrete
        self.register_buffer("_train_pa", None)    # [N, D_pa] standardised
        self.register_buffer("_pa_mean", None)
        self.register_buffer("_pa_std", None)
        self.register_buffer("_b_global", None)    # [D_x] global child bandwidth (continuous)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    # Deliberately False.  A k-NN estimator selects its k nearest neighbours
    # by distance alone; there is no principled reading of a *fractional*
    # multiplicity in that selection ("the 3.7th nearest neighbour" is not a
    # thing), and inventing a weighted-vote variant would silently change
    # which estimator the caller believes they fitted.  ``nbn.learning.fit``
    # checks this flag before fitting any node, so a network containing one of
    # these fails fast and names the node to swap.
    supports_weights: bool = False

    def fit_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        n_classes: int | None = None,
        weights: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        if weights is not None:
            raise NotImplementedError(
                "KNNConditionalMechanism does not support per-sample weights: "
                "k-nearest-neighbour selection is by distance alone, and a "
                "fractional multiplicity has no principled reading in it.  Use "
                "ConditionalKDEMechanism (weighted Nadaraya-Watson) or a "
                "parametric mechanism for weighted fitting."
            )
        device = x.device
        if self.discrete_child:
            y = x.long().reshape(-1)
            observed_k = int(y.max().item()) + 1 if y.numel() else 1
            self._n_classes = observed_k if n_classes is None else max(int(n_classes), observed_k)
            self._train_y = y
            self.output_dim = 1
            n = y.shape[0]
        else:
            y = ensure_2d(x).float()
            n, d_x = y.shape
            self.output_dim = d_x
            self._train_y = y
            std = y.std(0, unbiased=True).clamp_min(self.min_bandwidth)
            self._b_global = (std * n ** (-1.0 / (d_x + 4.0)) * self.bw_factor).clamp_min(self.min_bandwidth)

        if parents is None or parents.shape[-1] == 0:
            self._d_pa = 0
            self._pa_mean = torch.zeros(1, 0, device=device)
            self._pa_std = torch.ones(1, 0, device=device)
            self._train_pa = torch.zeros(n, 0, device=device)
        else:
            pa = ensure_2d(parents).float().to(device)
            self._d_pa = pa.shape[1]
            self._pa_mean = pa.mean(0, keepdim=True)
            self._pa_std = pa.std(0, keepdim=True).clamp_min(self.min_bandwidth)
            self._train_pa = (pa - self._pa_mean) / self._pa_std

        self._k_eff = self.k if self.k is not None else max(1, int(round(math.sqrt(max(n, 1)))))
        self._k_eff = min(self._k_eff, n)
        return {"n_train": int(n), "k": int(self._k_eff), "d_pa": int(self._d_pa)}

    @property
    def is_fitted(self) -> bool:
        return self._train_y is not None

    # ------------------------------------------------------------------
    # Neighbour search (tiled over query)
    # ------------------------------------------------------------------
    def _knn_indices(self, xs: torch.Tensor) -> torch.Tensor:
        """Return neighbour indices ``[M, k]`` for standardised queries ``xs`` ``[M, D_pa]``."""
        tpa = self._train_pa
        m = xs.shape[0]
        k = self._k_eff
        out = torch.empty(m, k, dtype=torch.long, device=xs.device)
        for start in range(0, m, self.query_chunk):
            sl = slice(start, start + self.query_chunk)
            d = torch.cdist(xs[sl], tpa)              # [q, N]
            out[sl] = torch.topk(d, k, largest=False, dim=1).indices
        return out

    def _std_parents(self, parents: torch.Tensor) -> torch.Tensor:
        parents = _sanitise_parents(parents.float(), mech_name="KNNConditional")
        return (parents - self._pa_mean) / self._pa_std

    # ------------------------------------------------------------------
    # Discrete child
    # ------------------------------------------------------------------
    def _categorical_probs(self, parents: torch.Tensor | None, m: int) -> torch.Tensor:
        """Return ``[m, K]`` smoothed class probabilities."""
        ty = self._train_y                            # [N] long
        kcls = self._n_classes
        if self._d_pa == 0 or parents is None:
            counts = torch.bincount(ty, minlength=kcls).float().to(ty.device)
            probs = (counts + self.alpha)
            probs = probs / probs.sum()
            return probs.unsqueeze(0).expand(m, -1)
        xs = self._std_parents(ensure_2d(parents))
        idx = self._knn_indices(xs)                   # [m, k]
        nbr = ty[idx]                                 # [m, k]
        counts = torch.zeros(m, kcls, device=ty.device)
        counts.scatter_add_(1, nbr, torch.ones_like(nbr, dtype=torch.float))
        probs = counts + self.alpha
        return probs / probs.sum(-1, keepdim=True)

    # ------------------------------------------------------------------
    # Continuous child
    # ------------------------------------------------------------------
    def _adaptive_bandwidth(self, nbr_y: torch.Tensor) -> torch.Tensor:
        """Per-query local bandwidth ``[m, D_x]`` from neighbour child values ``[m, k, D_x]``."""
        k = nbr_y.shape[1]
        local_std = nbr_y.std(1, unbiased=True) if k > 1 else self._b_global.expand(nbr_y.shape[0], -1)
        bw = local_std * (k ** (-1.0 / (self.output_dim + 4.0))) * self.bw_factor
        # Fall back to the global bandwidth where local spread collapses.
        bw = torch.where(bw > self.min_bandwidth, bw, self._b_global.to(bw.device))
        return bw.clamp_min(self.min_bandwidth)

    def _cont_logp_2d(self, y: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        ty = self._train_y
        device = ty.device
        y = y.to(device).float()
        m = y.shape[0]
        if self._d_pa == 0 or parents is None:
            # marginal full-sample KDE (uniform weights)
            b = self._b_global.to(device)
            log_b_norm = (torch.log(b) + 0.5 * _LOG_2PI).sum()
            dy = (y.unsqueeze(1) - ty.unsqueeze(0)) / b        # [M, N, D_x]
            logk = (-0.5 * dy.pow(2)).sum(-1) - log_b_norm     # [M, N]
            return torch.logsumexp(logk, dim=1) - math.log(ty.shape[0])
        xs = self._std_parents(ensure_2d(parents)).to(device)
        idx = self._knn_indices(xs)                            # [M, k]
        nbr_y = ty[idx]                                        # [M, k, D_x]
        b = self._adaptive_bandwidth(nbr_y)                    # [M, D_x]
        log_b_norm = (torch.log(b) + 0.5 * _LOG_2PI).sum(-1)   # [M]
        dy = (y.unsqueeze(1) - nbr_y) / b.unsqueeze(1)         # [M, k, D_x]
        logk = (-0.5 * dy.pow(2)).sum(-1)                      # [M, k]
        return torch.logsumexp(logk, dim=1) - log_b_norm - math.log(self._k_eff)

    def _cont_sample_2d(self, parents: torch.Tensor | None, n_samples: int) -> torch.Tensor:
        ty = self._train_y
        device = ty.device
        d_x = ty.shape[1]
        if self._d_pa == 0 or parents is None:
            m = 1 if parents is None else ensure_2d(parents).shape[0]
            b = self._b_global.to(device)
            idx = torch.randint(0, ty.shape[0], (m, n_samples), device=device)
            base = ty[idx]
            return base + b * torch.randn_like(base)
        xs = self._std_parents(ensure_2d(parents)).to(device)
        m = xs.shape[0]
        idx = self._knn_indices(xs)                            # [m, k]
        nbr_y = ty[idx]                                       # [m, k, D_x]
        b = self._adaptive_bandwidth(nbr_y)                   # [m, D_x]
        pick = torch.randint(0, self._k_eff, (m, n_samples), device=device)
        base = torch.gather(nbr_y, 1, pick.unsqueeze(-1).expand(-1, -1, d_x))  # [m, n, D_x]
        return base + b.unsqueeze(1) * torch.randn_like(base)

    # ------------------------------------------------------------------
    # Distribution interface
    # ------------------------------------------------------------------
    def forward(self, parents: torch.Tensor | None) -> Distribution:
        b = 1 if parents is None else ensure_2d(parents).shape[0]
        if self.discrete_child:
            return Categorical(probs=self._categorical_probs(parents, b))
        return _KNNContinuousDistribution(self, parents, b)

    def log_prob(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        if self.discrete_child:
            squeeze_s = False
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            if x.dim() == 2:
                x = x.unsqueeze(1)
                squeeze_s = True
            b, s, _ = x.shape
            if parents is not None and parents.dim() == 2:
                parents = parents.unsqueeze(1).expand(-1, s, -1)
            pa_flat = None if parents is None else flatten_samples(parents)[0]
            probs = self._categorical_probs(pa_flat, b * s)            # [B*S, K]
            xi = x.reshape(b * s).long()
            lp = torch.log(probs.gather(1, xi.unsqueeze(-1)).squeeze(-1).clamp_min(1e-38)).reshape(b, s)
            return lp.squeeze(1) if squeeze_s else lp
        # continuous
        squeeze_s = False
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_s = True
        b, s, d_x = x.shape
        if parents is not None and parents.dim() == 2:
            parents = parents.unsqueeze(1).expand(-1, s, -1)
        pa_flat = None if parents is None else flatten_samples(parents)[0]
        lp = self._cont_logp_2d(x.reshape(b * s, d_x), pa_flat).reshape(b, s)
        return lp.squeeze(1) if squeeze_s else lp

    def sample(self, parents: torch.Tensor | None, n: int = 1) -> torch.Tensor:
        if self.discrete_child:
            b = 1 if parents is None else ensure_2d(parents).shape[0]
            probs = self._categorical_probs(parents, b)               # [B, K]
            idx = Categorical(probs=probs.unsqueeze(1).expand(-1, n, -1)).sample()  # [B, n]
            return idx.float().unsqueeze(-1)
        return self._cont_sample_2d(parents, n)

    def tabulate(self, parent_cards: list[int] | None = None) -> torch.Tensor:
        """Discrete-child tabulation by enumerating discrete parent configs.

        Only defined when ``discrete_child`` and every parent is discrete with
        a declared cardinality (so the parent space is finite and enumerable).
        Returns logits of shape ``[*parent_cards, K]`` (or ``[K]`` for root).
        """
        if not self.discrete_child:
            raise NotImplementedError("tabulate() is only defined for the discrete-child kNN mechanism.")
        if self._d_pa == 0:
            probs = self._categorical_probs(None, 1)[0]
            return torch.log(probs.clamp_min(1e-38))
        if parent_cards is None:
            raise ValueError("parent_cards required to tabulate a conditional kNN CPD.")
        grids = torch.cartesian_prod(*[torch.arange(c) for c in parent_cards])
        if grids.dim() == 1:
            grids = grids.unsqueeze(-1)
        probs = self._categorical_probs(grids.float().to(self._train_pa.device), grids.shape[0])
        return torch.log(probs.clamp_min(1e-38)).reshape(*parent_cards, self._n_classes)
