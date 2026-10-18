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
memory is ``O(query_chunk * N)`` rather than ``O(M * N)``; the root
continuous marginal streams over training chunks the same way.  (A KeOps
``argKmin`` backend would make this linear-memory and FAISS-competitive; kept
optional.)

Neighbour memoisation: the ``[M, k]`` neighbour indices of the last few
parent batches are cached (exact match by ``torch.equal``), so ``log_prob``
and ``sample`` on the same parent rows -- ``forward(pa)`` then both, or a
scored ancestral sample -- run the ``cdist`` + ``topk`` search once.  The
cache is cleared whenever the stored sample changes (fit / update / a state
dict load).

Streaming: the stored sample is the sufficient statistic, so ``update_local``
appends the new rows and re-resolves the standardisation and ``k`` from the
pooled sample; ``fit_local(A)`` then ``update_local(B)`` equals
``fit_local(A | B)``.  Forgetting is refused (like weights): the neighbour
selection has no notion of a fractional row.
"""
from __future__ import annotations

import math

import torch
from torch.distributions import Categorical, Distribution

from nbn.mechanisms.base import Mechanism
from nbn.utils.batching import _sanitise_parents, ensure_2d, flatten_samples

_LOG_2PI = math.log(2.0 * math.pi)


def _select_lowest_index(d: torch.Tensor, k: int) -> torch.Tensor:
    """Indices ``[q, k]`` of the k smallest entries per row of ``d`` ``[q, N]``,
    ties at the k-th value broken towards the lowest column index (ascending).

    Fast path: the ``k + 1`` smallest by ``topk``; when the k-th and (k+1)-th
    values differ, the first ``k`` are the unique answer.  Only rows where
    they tie go through the exact mask rule (rows closer than the k-th value,
    then the lowest-indexed rows at it).
    """
    q, n = d.shape
    if k >= n:
        return torch.arange(n, device=d.device).expand(q, n).clone()
    vals, idx = torch.topk(d, k + 1, largest=False, dim=1, sorted=True)
    out = idx[:, :k].sort(dim=1).values
    tied = vals[:, k] == vals[:, k - 1]
    if bool(tied.any()):
        rows = tied.nonzero().squeeze(1)
        dt = d[rows]
        t = vals[rows, k - 1:k]
        lt = dt < t
        eq = dt == t
        need = k - lt.sum(dim=1, keepdim=True)
        rank = eq.long().cumsum(dim=1)
        sel = lt | (eq & (rank <= need))
        out[rows] = sel.nonzero()[:, 1].reshape(rows.shape[0], k)
    return out


def _clear_neighbour_cache_hook(module, incompatible_keys) -> None:
    """``load_state_dict`` post-hook: a module-level function so the module
    stays picklable (``torch.save(model)`` is how the benchmark's fit-once
    cache persists a fitted network)."""
    module._clear_neighbour_cache()


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
        Query rows per neighbour-search tile (and training rows per tile of
        the root continuous marginal).
    neighbour_cache_size:
        How many recent parent batches keep their neighbour indices
        memoised (``0`` disables the cache; see the module notes).
    neighbour_cache_max_bytes:
        Largest neighbour-index block that is memoised (default 4 MiB).  The
        cache serves ``forward`` → ``log_prob`` + ``sample`` on the *same*
        rows; particle blocks of likelihood weighting are never reused, and
        caching them held ``4 x particles x k`` indices per node (3 GB on a
        100-node network).  A pure memo: answers do not depend on it.
    tie_break:
        Which training rows count as the ``k`` nearest when several lie at
        exactly the k-th distance (common with discrete parents).
        ``"lowest_index"`` (default): distances in float64 (float32 on MPS,
        which has no float64) and ties at the k-th distance go to the lowest
        training-row index, so the neighbour set is a function of the data
        alone, whatever search backend runs.  ``"any"``: the historical
        behaviour, whatever ``torch.topk`` returns on float32 distances
        (unspecified among ties).
    search:
        Neighbour-search backend; every backend returns the same neighbour
        set under ``tie_break="lowest_index"``.  ``"brute"``: tiled
        ``cdist`` + top-k (GPU friendly).  ``"kdtree"``: SciPy's ``cKDTree``
        on the CPU, ``O(log N + k)`` per query instead of ``O(N)``; a query
        whose k-th and (k+1)-th distances tie is re-resolved by the brute
        rule.  ``"auto"`` (default): the k-d tree on CPU when the parents
        have at most 4 dimensions, there are at least 4096 training rows and
        SciPy is installed, brute force otherwise.  In every mode, identical
        query rows are searched once.
    """

    def __init__(
        self,
        k: int | None = None,
        discrete_child: bool = False,
        alpha: float = 1.0,
        bw_factor: float = 1.0,
        min_bandwidth: float = 1e-3,
        query_chunk: int = 1024,
        neighbour_cache_size: int = 4,
        tie_break: str = "lowest_index",
        search: str = "auto",
        neighbour_cache_max_bytes: int = 4 * 2**20,
    ) -> None:
        super().__init__()
        if tie_break not in ("lowest_index", "any"):
            raise ValueError("tie_break must be 'lowest_index' or 'any'.")
        if search not in ("auto", "brute", "kdtree"):
            raise ValueError("search must be 'auto', 'brute' or 'kdtree'.")
        self.tie_break = tie_break
        self.search = search
        self.neighbour_cache_max_bytes = int(neighbour_cache_max_bytes)
        self.k = None if k is None else int(k)
        self.discrete_child = bool(discrete_child)
        self.is_discrete = bool(discrete_child)
        self.alpha = float(alpha)
        self.bw_factor = float(bw_factor)
        self.min_bandwidth = float(min_bandwidth)
        self.query_chunk = int(query_chunk)
        self.neighbour_cache_size = int(neighbour_cache_size)
        self.output_dim = 1
        self._d_pa = 0
        self._n_classes = 0
        self._k_eff = 0
        # Memoised neighbour searches: list of (standardised query batch,
        # indices), most recent last.  Plain attribute, never persisted.
        self._nbr_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._kdtree = None
        # A loaded state dict swaps the stored sample under the cache.
        self.register_load_state_dict_post_hook(_clear_neighbour_cache_hook)
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
    supports_update: bool = True

    def fit_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        n_classes: int | None = None,
        weights: torch.Tensor | None = None,
        warm_start: bool = False,
        **kwargs,
    ) -> dict:
        """Store the training sample and resolve the neighbourhood size.

        ``warm_start`` is accepted and ignored: the stored sample, the
        standardisation statistics and ``k`` are all recomputed from this
        call's data with no dependence on the previous state, so recomputing
        *is* the continuation.  Reported as ``warm_started: False``.  Note the
        asymmetry with ``weights``, which this mechanism *refuses* -- an
        unsupported weight vector would silently change the answer, whereas an
        ineffective ``warm_start`` cannot.  See
        :mod:`nbn.learning.warm_start`.
        """
        if weights is not None:
            raise NotImplementedError(
                "KNNConditionalMechanism does not support per-sample weights: "
                "k-nearest-neighbour selection is by distance alone, and a "
                "fractional multiplicity has no principled reading in it.  Use "
                "ConditionalKDEMechanism (weighted Nadaraya-Watson) or a "
                "parametric mechanism for weighted fitting."
            )
        self._clear_neighbour_cache()   # the stored sample is about to change
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
            # nan_to_num: the unbiased std of a single row is NaN, which
            # clamp_min passes through (NaN compares false) into every
            # log_prob.  0 is the population value, so the floor applies.
            std = y.std(0, unbiased=True).nan_to_num(0.0).clamp_min(self.min_bandwidth)
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
            self._pa_std = (
                pa.std(0, keepdim=True).nan_to_num(0.0).clamp_min(self.min_bandwidth)
            )
            self._train_pa = (pa - self._pa_mean) / self._pa_std

        self._k_eff = self.k if self.k is not None else max(1, int(round(math.sqrt(max(n, 1)))))
        self._k_eff = min(self._k_eff, n)
        return {
            "n_train": int(n), "k": int(self._k_eff), "d_pa": int(self._d_pa),
            "warm_started": False,
        }

    def update_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        n_classes: int | None = None,
        *,
        forgetting: float = 1.0,
        weights: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        """Append new rows to the stored sample and refit on the pooled sample.

        Exact: the estimator is its sample, so ``fit_local(A)`` followed by
        ``update_local(B)`` is ``fit_local(A | B)`` -- standardisation, the
        global child bandwidth and ``k = round(sqrt(N))`` (when ``k`` was not
        fixed) are all re-resolved from the pooled ``N``.  For a discrete
        child the class count can only grow (declared cardinality, classes
        observed so far, classes in the new rows).  ``forgetting < 1`` and
        ``weights`` are refused for the reason given at ``supports_weights``:
        k-nearest-neighbour selection is by distance alone and has no reading
        of a fractional row; a fading kNN would be a different estimator.  Use
        :class:`ConditionalKDEMechanism` when forgetting is needed.
        """
        assert self._train_y is not None, "call fit_local before update_local"
        if weights is not None:
            raise NotImplementedError(
                "KNNConditionalMechanism does not support per-sample weights "
                "(see fit_local)."
            )
        if float(forgetting) != 1.0:
            raise NotImplementedError(
                "KNNConditionalMechanism.update_local only supports forgetting=1.0: "
                "neighbour selection is by distance alone, so a faded (fractional) "
                "row has no principled reading.  Use ConditionalKDEMechanism for "
                "exponential forgetting."
            )
        device = self._train_y.device
        if self.discrete_child:
            y_new = x.long().reshape(-1).to(device)
            n_classes = max(self._n_classes, 0 if n_classes is None else int(n_classes))
        else:
            y_new = ensure_2d(x).float().to(device)
            if y_new.shape[1] != self._train_y.shape[1]:
                raise ValueError(
                    f"update child dim {y_new.shape[1]} differs from the fitted "
                    f"{self._train_y.shape[1]}"
                )
        has_pa = parents is not None and parents.shape[-1] > 0
        if has_pa != (self._d_pa > 0) or (has_pa and parents.shape[-1] != self._d_pa):
            raise ValueError(
                f"update parents dim {0 if not has_pa else parents.shape[-1]} "
                f"differs from the fitted {self._d_pa}"
            )
        if has_pa:
            pa_old = self._train_pa * self._pa_std + self._pa_mean   # undo standardisation
            pa_pool = torch.cat([pa_old, ensure_2d(parents).float().to(device)], dim=0)
        else:
            pa_pool = None
        n_new = y_new.shape[0]
        y_pool = torch.cat([self._train_y, y_new], dim=0)
        info = self.fit_local(y_pool, pa_pool, n_classes=n_classes if self.discrete_child else None)
        info.update({"method": "knn_append", "n_new": int(n_new)})
        return info

    @property
    def is_fitted(self) -> bool:
        return self._train_y is not None

    # ------------------------------------------------------------------
    # Neighbour search (tiled over query)
    # ------------------------------------------------------------------
    def _clear_neighbour_cache(self) -> None:
        self._nbr_cache = []
        self._kdtree = None   # built lazily from the stored sample

    def __getstate__(self):
        state = super().__getstate__() if hasattr(super(), "__getstate__") else self.__dict__.copy()
        state = dict(state)
        state["_kdtree"] = None          # rebuilt on demand, never persisted
        state["_nbr_cache"] = []
        return state

    def _knn_indices(self, xs: torch.Tensor) -> torch.Tensor:
        """Return neighbour indices ``[M, k]`` for standardised queries ``xs`` ``[M, D_pa]``.

        Memoised on the exact content of ``xs`` (``torch.equal``, an
        ``O(M·D_pa)`` compare against the ``O(M·N·D_pa)`` search) for the last
        ``neighbour_cache_size`` batches, so ``log_prob`` then ``sample`` on
        the same parent rows search once.
        """
        if self.neighbour_cache_size > 0:
            for key, idx in reversed(self._nbr_cache):
                if (key.shape == xs.shape and key.device == xs.device
                        and key.dtype == xs.dtype and torch.equal(key, xs)):
                    return idx
        out = self._knn_search(xs)
        cap = getattr(self, "neighbour_cache_max_bytes", 4 * 2**20)
        if self.neighbour_cache_size > 0 and out.numel() * out.element_size() <= cap:
            self._nbr_cache.append((xs.detach().clone(), out))
            del self._nbr_cache[:-self.neighbour_cache_size]
        return out

    def _knn_search(self, xs: torch.Tensor) -> torch.Tensor:
        """The uncached neighbour search behind ``_knn_indices``.

        Identical query rows (discrete parents, shared particles) are searched
        once; the backend is chosen by ``search``; ties follow ``tie_break``.
        """
        m = xs.shape[0]
        if m > 1:
            uniq, inv = torch.unique(xs, dim=0, return_inverse=True)
            if uniq.shape[0] < m:
                return self._knn_search_unique(uniq)[inv]
        return self._knn_search_unique(xs)

    def _knn_search_unique(self, xs: torch.Tensor) -> torch.Tensor:
        tie = getattr(self, "tie_break", "lowest_index")
        if tie == "any":
            return self._brute_any(xs)
        mode = getattr(self, "search", "auto")
        if mode == "kdtree" or (mode == "auto" and self._kdtree_ok(xs)):
            try:
                return self._kdtree_search(xs)
            except ImportError:
                if mode == "kdtree":
                    raise
        return self._brute_lowest_index(xs)

    def _brute_any(self, xs: torch.Tensor) -> torch.Tensor:
        """Historical search: float32 ``cdist`` + ``topk`` (ties unspecified)."""
        tpa = self._train_pa
        m = xs.shape[0]
        k = self._k_eff
        out = torch.empty(m, k, dtype=torch.long, device=xs.device)
        for start in range(0, m, self.query_chunk):
            sl = slice(start, start + self.query_chunk)
            d = torch.cdist(xs[sl], tpa)              # [q, N]
            out[sl] = torch.topk(d, k, largest=False, dim=1).indices
        return out

    def _exact_dtype(self, device) -> torch.dtype:
        return torch.float32 if torch.device(device).type == "mps" else torch.float64

    def _brute_lowest_index(self, xs: torch.Tensor) -> torch.Tensor:
        """Exact k nearest, ties at the k-th distance to the lowest index.

        Two stages per query tile, both exact in result:

        1. Fast float32 squared distances (matmul form) give the ``k + m``
           best candidates.
        2. Exact distances (float64; float32 by differences on MPS) are
           computed for those candidates only, and the ``k`` nearest are taken
           with the lowest-index rule.  The set is accepted when its largest
           exact distance is below what the (k+m)-th fast distance guarantees
           for every excluded row, after a rigorous bound on the float32
           rounding of the fast form (with a 10x safety factor): then no
           training row outside the candidates can be as close.  Rows that
           fail the check are re-solved by :meth:`_exact_rows` over all
           training rows.
        """
        tpa = self._train_pa.float()
        n = tpa.shape[0]
        k = self._k_eff
        if k >= n:
            return torch.arange(n, device=xs.device).expand(xs.shape[0], n).clone()
        dt = self._exact_dtype(xs.device)
        extra = min(n - k, max(8, k // 8))
        kk = k + extra
        out = torch.empty(xs.shape[0], k, dtype=torch.long, device=xs.device)
        for start in range(0, xs.shape[0], self.query_chunk):
            sl = slice(start, start + self.query_chunk)
            xq = xs[sl].float()
            q_norm = xq.pow(2).sum(1, keepdim=True)
            # cdist's fast (matmul-form) path; its squared values carry the
            # rounding error bounded below.
            dist = torch.cdist(xq, tpa)                                  # [q, N]
            vals, cand = torch.topk(dist, kk, largest=False, dim=1, sorted=False)
            v_last = vals.amax(dim=1).pow(2)             # the (k+m)-th fast squared distance
            diff = xq.to(dt).unsqueeze(1) - self._train_pa.to(dt)[cand]  # [q, kk, D]
            e2 = diff.pow(2).sum(-1)                                     # exact, [q, kk]
            # order candidates by (exact distance, index): stable sorts
            o1 = cand.sort(dim=1, stable=True)
            e2_by_idx = e2.gather(1, o1.indices)
            o2 = e2_by_idx.sort(dim=1, stable=True)
            chosen = o1.values.gather(1, o2.indices[:, :k])
            kth = o2.values[:, k - 1]
            if kk < n:
                # Rounding of the fast form: |err| <= 2 (D + 2) u (|x|^2 + |y|^2)
                # and |y|^2 <= 2|x|^2 + 2|x - y|^2, so every excluded row has a
                # true squared distance >= (v - 3 c |x|^2) / (1 + 2 c), with
                # c = 2 (D + 2) u (here x10 for safety); the chosen k-th must
                # be below that, allowing the same relative slack on it.
                c = 20.0 * (tpa.shape[1] + 2) * 6e-8
                qn = q_norm.squeeze(1).to(dt)
                bound = (v_last.to(dt) - 3.0 * c * qn) / (1.0 + 2.0 * c)
                ok = kth * (1.0 + c) + 1e-30 < bound
            else:
                ok = torch.ones_like(kth, dtype=torch.bool)
            res = chosen.sort(dim=1).values
            if not bool(ok.all()):
                bad = (~ok).nonzero().squeeze(1)
                res[bad] = self._exact_rows(xq[bad])
            out[sl] = res
        return out

    def _exact_rows(self, xq: torch.Tensor) -> torch.Tensor:
        """The lowest-index rule over all training rows, by direct differences."""
        dt = self._exact_dtype(xq.device)
        d = torch.cdist(xq.to(dt), self._train_pa.to(dt),
                        compute_mode="donot_use_mm_for_euclid_dist")
        return _select_lowest_index(d, self._k_eff)

    def _kdtree_ok(self, xs: torch.Tensor) -> bool:
        if xs.device.type != "cpu" or self._d_pa == 0 or self._d_pa > 4:
            return False
        if self._train_pa is None or self._train_pa.shape[0] < 4096:
            return False
        try:
            import scipy.spatial  # noqa: F401
        except ImportError:
            return False
        return True

    def _kdtree_search(self, xs: torch.Tensor) -> torch.Tensor:
        """k-d tree search; rows with a tie at the k-th distance go to the brute rule."""
        from scipy.spatial import cKDTree

        tree = getattr(self, "_kdtree", None)
        if tree is None:
            tree = cKDTree(self._train_pa.detach().cpu().double().numpy())
            self._kdtree = tree
        k = self._k_eff
        n = self._train_pa.shape[0]
        q = xs.detach().cpu().double().numpy()
        kk = min(k + 1, n)
        # Rows are independent, so SciPy may split them over torch's CPU threads;
        # the answer does not depend on the number of workers.
        workers = max(1, torch.get_num_threads()) if q.shape[0] >= 2048 else 1
        dist, idx = tree.query(q, k=kk, workers=workers)
        dist = torch.from_numpy(dist).reshape(q.shape[0], kk)
        idx = torch.from_numpy(idx).reshape(q.shape[0], kk).long()
        out = idx[:, :k].sort(dim=1).values
        if kk > k:
            # The set is ambiguous only if the (k+1)-th row ties the k-th; the
            # tree's own order inside a tie is not the documented rule.
            tied = dist[:, k] <= dist[:, k - 1]
            # A tie anywhere at the k-th distance can also hide rows *inside*
            # the first k that share it with outside rows; the brute rule
            # settles both cases.
            if bool(tied.any()):
                rows = tied.nonzero().squeeze(1)
                out[rows] = self._brute_lowest_index(xs[rows.to(xs.device)]).cpu()
        return out.to(xs.device)

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
            # marginal full-sample KDE (uniform weights), streamed over
            # training chunks: peak memory O(M * query_chunk), not [M, N, D_x].
            b = self._b_global.to(device)
            log_b_norm = (torch.log(b) + 0.5 * _LOG_2PI).sum()
            n = ty.shape[0]
            acc = torch.full((m,), float("-inf"), device=device)
            for start in range(0, n, self.query_chunk):
                ty_c = ty[start:start + self.query_chunk]
                dy = (y.unsqueeze(1) - ty_c.unsqueeze(0)) / b   # [M, c, D_x]
                logk = (-0.5 * dy.pow(2)).sum(-1)              # [M, c]
                acc = torch.logaddexp(acc, torch.logsumexp(logk, dim=1))
            return acc - log_b_norm - math.log(n)
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

    def tabulate_working_bytes(self, parent_cards: list[int] | None = None) -> int | list[int]:
        """Enumerating P parent configurations, buffer by buffer: the int64
        grid and its standardised float copies (and their cached copy), one
        ``query_chunk x N`` distance block with its top-k, the neighbour
        indices / labels, and the count / probability tables; with
        ``tie_break="lowest_index"`` also the row de-duplication, the
        exact-dtype distance block and the tie-break masks."""
        if not self.discrete_child:
            return super().tabulate_working_bytes(parent_cards)
        es = self._train_pa.element_size() if self._train_pa is not None else 4
        if self._d_pa == 0 or not parent_cards:
            return [self._n_classes * es] * 4
        p, d = math.prod(parent_cards), len(parent_cards)
        n = int(self._train_y.shape[0])
        k, q, kc = int(self._k_eff), min(p, int(self.query_chunk)), self._n_classes
        base = ([p * d * 8, p * d] + [p * d * es] * 6       # grid, mask, float copies
                + [q * n * es] * 2 + [q * k * es, q * k * 8]  # distances, top-k
                + [p * k * 8, p * k * 8, p * k * es]          # indices, labels, ones
                + [p * kc * es] * 5)                          # counts, probabilities
        if getattr(self, "tie_break", "lowest_index") == "any":
            return base
        ed = torch.empty((), dtype=self._exact_dtype(self._train_pa.device)).element_size()
        return (base
                + [p * d * es, p * 8, p * d * 8, p * 8]       # unique rows, inverse, sort
                + [n * d * ed, q * d * ed, q * n * ed]        # exact-dtype copies, distances
                + [q * k * ed, q * k * 8]                     # k-th value top-k
                + [q * n] * 3 + [q * n * 8]                   # lt / eq / sel masks, rank
                + [q * k * 16, q * k * 8])                    # nonzero, selected
