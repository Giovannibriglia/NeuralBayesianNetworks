"""Conditional Kernel Density Estimation mechanism (non-parametric).

Implements the Nadaraya--Watson conditional density estimator (Rosenblatt
1969; Hyndman, Bashtannyk & Grunwald 1996):

    p(y | x) = sum_i K_h(x - x_i) K_b(y - y_i) / sum_i K_h(x - x_i)

with product Gaussian kernels.  Equivalently this is a Gaussian mixture with
one component per training point, whose mixing weights depend on the parents
``x`` through the parent-kernel and whose component means are the training
child values ``y_i`` with a fixed child bandwidth ``b``.

Design notes
------------
* **Memory.** The naive estimator materialises an ``[M, N]`` kernel matrix
  (M queries, N train points).  To respect the 8 GB-VRAM target we never
  materialise it for ``log_prob``: the numerator ``sum_i K_h K_b`` and the
  denominator ``sum_i K_h`` are accumulated online in log-space over training
  **chunks** via :func:`torch.logaddexp`, so peak memory is ``O(M * chunk)``.
  Sampling tiles over the **query** dimension instead.
* **Optional acceleration.** A KeOps ``LazyTensor`` backend (linear-memory
  online map-reduce) would give a further 10--100x on large N, but is kept out
  of the hard dependency set; the chunked torch path below is the portable
  default.
* **Neighbour truncation (opt-in).** ``n_neighbors=k`` restricts both sums to
  the ``k`` training rows nearest to the query in bandwidth-scaled parent
  space.  Gaussian parent kernels decay as ``exp(-d²/2)``, so the rows beyond
  the ``k``-th neighbour contribute a vanishing share of either sum once ``k``
  covers the kernel's support; the query cost drops from ``Θ(M·N)`` to a
  ``Θ(M·k)`` gather plus one chunked neighbour search.  The neighbour search
  streams over the training rows with a running top-``k`` (never a whole
  ``[M, N]`` distance matrix).  ``None`` (the default) is the exact estimator,
  as is any ``k >= N``; roots have no parent space and are always exact.
* **Bandwidth.** Scott's and Silverman's rules-of-thumb are provided, computed
  on **standardised** parents (so the per-dimension scale is O(1)) and on the
  raw child.  ``torch.quantile``-based robust scale (IQR) guards against
  heavy tails, mirroring the project's existing use of ``torch.quantile``.
  Under per-sample weights the rule is replication-exact: the sample size is
  the total weight, the scale is the frequency-weighted std, and the IQR is
  the weighted quantile (:func:`nbn.learning.weighting.weighted_quantile`),
  so integer weights give the bandwidth of the replicated rows.
* **Streaming.** The sufficient statistic of a KDE is its (weighted) sample,
  so ``update_local`` is exact: it appends the new rows to the stored sample
  and re-derives the standardisation and the rule-of-thumb bandwidths from
  the pooled sample.  ``fit_local(A)`` then ``update_local(B)`` reproduces
  ``fit_local(A | B)``; ``forgetting < 1`` fades the weight of the rows that
  were already stored (the kernel analogue of fading Dirichlet counts).

Shape contracts (see :class:`nbn.mechanisms.base.Mechanism`)
------------------------------------------------------------
parents: ``[B, D_pa]`` / ``[B, S, D_pa]`` / ``None`` (root).
y:       ``[B]`` / ``[B, D_x]`` / ``[B, S, D_x]``.
log_prob → ``[B]`` or ``[B, S]``;   sample → ``[B, n, D_x]``.
"""
from __future__ import annotations

import math

import torch
from torch.distributions import Distribution

from nbn.learning.weighting import validate_weights, weighted_moments, weighted_quantile
from nbn.mechanisms.base import Mechanism
from nbn.utils.batching import _sanitise_parents, ensure_2d, flatten_samples

_LOG_2PI = math.log(2.0 * math.pi)


class _KDEConditionalDistribution(Distribution):
    """Lightweight wrapper so ``forward(parents)`` honours the Distribution API.

    All heavy lifting delegates to the owning mechanism's chunked kernels;
    this object just carries the conditioning context.
    """

    has_rsample = False
    arg_constraints: dict = {}

    def __init__(self, mech: ConditionalKDEMechanism, parents: torch.Tensor | None, b: int) -> None:
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
        out = self._mech.sample(self._parents, n=n)  # [B, n, D_x]
        out = out.permute(1, 0, 2)  # [n, B, D_x]
        return out.reshape(*sample_shape, *out.shape[1:]) if sample_shape else out[0]


class ConditionalKDEMechanism(Mechanism):
    """Nadaraya--Watson conditional density estimator with Gaussian kernels.

    Parameters
    ----------
    bandwidth:
        Rule for the rule-of-thumb bandwidth: ``"scott"`` or ``"silverman"``.
    bw_factor:
        Scalar multiplier applied to every bandwidth (tune for over/under
        smoothing). Pass ``"auto"`` to select it by held-out negative
        log-likelihood at fit time (resolves to a float on the first ``fit_local``).
    min_bandwidth:
        Floor on every bandwidth, preventing degenerate zero-width kernels.
    train_chunk:
        Number of training points processed per tile in ``log_prob`` (caps
        peak memory at ``O(M * train_chunk)``).
    query_chunk:
        Number of query rows processed per tile in ``sample`` (and in the
        truncated neighbour search).
    n_neighbors:
        ``None`` (default) evaluates the exact Nadaraya--Watson sums over all
        ``N`` training rows.  An integer ``k`` truncates both sums to the
        ``k`` nearest training rows in bandwidth-scaled parent space (see the
        module notes); ``k >= N`` and root nodes take the exact path.  The
        truncation is an approximation whose error is set by how far, in
        bandwidths, the ``k``-th neighbour lies: pick ``k`` so that it is
        ~3 bandwidths out (``k`` grows with ``N·h^D_pa``).  Measured at
        ``N=10000, D_pa=2`` (Scott ``h≈0.22`` std) on in-distribution
        queries: ``k=2048`` → median ``|Δ log p|`` 3e-5, ``k=256`` → 6e-2.
    """

    is_discrete: bool = False
    supports_weights: bool = True
    supports_update: bool = True

    def __init__(
        self,
        bandwidth: str = "scott",
        bw_factor: float | str = 1.0,
        min_bandwidth: float = 1e-3,
        train_chunk: int = 8192,
        query_chunk: int = 1024,
        n_neighbors: int | None = None,
    ) -> None:
        super().__init__()
        if bandwidth not in ("scott", "silverman"):
            raise ValueError(f"bandwidth must be 'scott' or 'silverman', got {bandwidth!r}")
        if isinstance(bw_factor, str) and bw_factor != "auto":
            raise ValueError(f"bw_factor must be a float or 'auto', got {bw_factor!r}")
        if n_neighbors is not None and int(n_neighbors) < 1:
            raise ValueError(f"n_neighbors must be a positive int or None, got {n_neighbors!r}")
        self.bandwidth = bandwidth
        self.bw_factor = bw_factor if bw_factor == "auto" else float(bw_factor)
        self.min_bandwidth = float(min_bandwidth)
        self.train_chunk = int(train_chunk)
        self.query_chunk = int(query_chunk)
        self.n_neighbors = None if n_neighbors is None else int(n_neighbors)
        self.output_dim = 1
        self._d_pa = 0
        # Buffers (populated by fit_local) — carried by .to(device)/state_dict.
        self.register_buffer("_train_y", None)   # [N, D_x]
        # log of each training point's weight, or None when unweighted.  A
        # weighted conditional KDE is the natural estimator here: the
        # Nadaraya-Watson ratio simply carries w_i on every kernel, in both
        # the numerator and the denominator, so the weights cancel out of a
        # constant and a weight of 0 removes the point exactly (log 0 = -inf
        # drops it from both logsumexps).
        self.register_buffer("_train_logw", None)  # [N]
        self.register_buffer("_train_pa", None)   # [N, D_pa] standardised, or empty
        self.register_buffer("_h", None)          # [D_pa] parent bandwidths (std space)
        self.register_buffer("_b", None)          # [D_x] child bandwidths
        self.register_buffer("_pa_mean", None)    # [1, D_pa]
        self.register_buffer("_pa_std", None)     # [1, D_pa]

    # ------------------------------------------------------------------
    # Bandwidth rules
    # ------------------------------------------------------------------
    def _rule_bandwidth(
        self, data: torch.Tensor, weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-dimension Scott/Silverman bandwidth for ``data`` ``[N, D]``.

        ``weights`` are per-row multiplicities (see
        :mod:`nbn.learning.weighting`); the rule is then evaluated on the
        replicated sample: ``n`` is the total weight, the std is
        frequency-weighted (``sum(w) - 1`` denominator) and the IQR comes
        from the weighted quantile.  Unweighted, every quantity is the plain
        one, so ``weights=None`` is byte-identical to the historical rule.
        """
        d = data.shape[1]
        if weights is None:
            n = float(data.shape[0])
        else:
            n = float(weights.sum())
        _, std = weighted_moments(data, weights, unbiased=True)
        # Robust scale: min(std, IQR/1.349) when IQR is informative.
        q = weighted_quantile(
            data, torch.tensor([0.25, 0.75], device=data.device, dtype=torch.float64), weights,
        )
        iqr = (q[1] - q[0]) / 1.349
        scale = torch.where((iqr > 0) & (iqr < std), iqr, std).clamp_min(self.min_bandwidth)
        if self.bandwidth == "silverman":
            factor = (4.0 / ((d + 2.0) * n)) ** (1.0 / (d + 4.0))
        else:  # scott
            factor = n ** (-1.0 / (d + 4.0))
        bw = (scale * factor * self.bw_factor).clamp_min(self.min_bandwidth)
        return bw

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        weights: torch.Tensor | None = None,
        warm_start: bool = False,
        **kwargs,
    ) -> dict:
        """Store the (weighted) training sample and its bandwidths.

        ``warm_start`` is accepted and ignored.  A KDE's "parameters" are the
        training sample itself plus rule-of-thumb bandwidths derived from it;
        both are recomputed exactly from this call's data and weights, with no
        dependence on what was there before, so recomputing *is* the
        continuation.  Reported as ``warm_started: False``.  See
        :mod:`nbn.learning.warm_start`.
        """
        w = validate_weights(
            weights, ensure_2d(x).shape[0],
            where="ConditionalKDEMechanism.fit_local",
        )
        if self.bw_factor == "auto":
            self.bw_factor = self._select_bw_factor(x, parents)
        info = self._fit_core(x, parents, _w_vec=w, **kwargs)
        self._train_logw = (
            None if w is None
            else torch.log(w.to(device=self._train_y.device)).to(self._train_y.dtype)
        )
        info["warm_started"] = False
        return info

    def update_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        *,
        forgetting: float = 1.0,
        weights: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        """Fold new rows into the stored sample (recursive, no rehearsal).

        The sufficient statistic of a Nadaraya--Watson estimator is its
        weighted sample, so the posterior-as-prior update is to append the new
        rows and re-derive, from the pooled sample, exactly what ``fit_local``
        derives: the parent standardisation and the rule-of-thumb bandwidths.
        With ``forgetting == 1.0`` a chunked ``fit_local(A)`` →
        ``update_local(B)`` therefore reproduces ``fit_local(A | B)`` (same
        buffers, same ``log_prob``).  ``forgetting < 1.0`` multiplies the
        weight of every row already stored by the factor before appending, so
        a row's weight decays geometrically with its age in update calls; rows
        are never dropped.  ``weights`` gives the multiplicity of the *new*
        rows, with the same contract as in ``fit_local``.  The resolved
        ``bw_factor`` is kept (``"auto"`` is selected once, at fit time).
        """
        assert self._train_y is not None, "call fit_local before update_local"
        forgetting = float(forgetting)
        if not (0.0 < forgetting <= 1.0):
            raise ValueError(f"forgetting factor must be in (0, 1], got {forgetting!r}")
        device = self._train_y.device
        y_new = ensure_2d(x).float().to(device)
        n_old, n_new = self._train_y.shape[0], y_new.shape[0]
        if y_new.shape[1] != self._train_y.shape[1]:
            raise ValueError(
                f"update child dim {y_new.shape[1]} differs from the fitted "
                f"{self._train_y.shape[1]}"
            )
        w_new = validate_weights(
            weights, n_new, where="ConditionalKDEMechanism.update_local",
        )
        has_pa = parents is not None and parents.shape[-1] > 0
        if has_pa != (self._d_pa > 0) or (has_pa and parents.shape[-1] != self._d_pa):
            raise ValueError(
                f"update parents dim {0 if not has_pa else parents.shape[-1]} "
                f"differs from the fitted {self._d_pa}"
            )
        if has_pa:
            # Stored parents are standardised; undo that so the pooled sample is
            # standardised afresh, as a pooled fit would.
            pa_old = self._train_pa * self._pa_std + self._pa_mean
            pa_pool = torch.cat([pa_old, ensure_2d(parents).float().to(device)], dim=0)
        else:
            pa_pool = None
        y_pool = torch.cat([self._train_y, y_new], dim=0)

        unweighted = forgetting == 1.0 and w_new is None and self._train_logw is None
        if unweighted:
            info = self._fit_core(y_pool, pa_pool)
            self._train_logw = None
        else:
            logw_old = (
                torch.zeros(n_old, dtype=torch.float64, device=device)
                if self._train_logw is None else self._train_logw.double()
            ) + math.log(forgetting)
            logw_new = (
                torch.zeros(n_new, dtype=torch.float64, device=device)
                if w_new is None else torch.log(w_new.to(device))
            )
            logw_pool = torch.cat([logw_old, logw_new])
            info = self._fit_core(y_pool, pa_pool, _w_vec=logw_pool.exp())
            self._train_logw = logw_pool.to(self._train_y.dtype)
        info.update({"method": "kde_append", "n_new": int(n_new), "forgetting": forgetting})
        return info

    def _select_bw_factor(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        candidates: tuple[float, ...] = (0.2, 0.3, 0.45, 0.6, 0.8, 1.0),
        val_frac: float = 0.25,
    ) -> float:
        """Pick ``bw_factor`` by held-out negative log-likelihood (lower better)."""
        y = ensure_2d(x).float()
        n = y.shape[0]
        gen = torch.Generator(device="cpu").manual_seed(0)
        perm = torch.randperm(n, generator=gen)
        n_val = max(8, int(round(val_frac * n)))
        if n_val >= n:  # too few points to split — fall back to the rule-of-thumb
            return 1.0
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
        has_pa = parents is not None and parents.shape[-1] > 0
        pa = ensure_2d(parents).float() if has_pa else None
        best_c, best_nll = 1.0, float("inf")
        for c in candidates:
            self.bw_factor = float(c)
            self._fit_core(y[fit_idx], pa[fit_idx] if has_pa else None)
            with torch.no_grad():
                nll = float((-self.log_prob(
                    y[val_idx], pa[val_idx] if has_pa else None)).mean())
            if nll == nll and nll < best_nll:  # nll == nll rejects NaN
                best_nll, best_c = nll, float(c)
        self.bw_factor = "auto"  # restore the request; caller resolves to best_c
        return best_c

    def _fit_core(self, x: torch.Tensor, parents: torch.Tensor | None, **kwargs) -> dict:
        y = ensure_2d(x).float()  # [N, D_x]
        n, d_x = y.shape
        device = y.device
        self.output_dim = d_x
        w_vec = kwargs.get("_w_vec")
        self._b = self._rule_bandwidth(y, w_vec)

        if parents is None or parents.shape[-1] == 0:
            self._d_pa = 0
            self._pa_mean = torch.zeros(1, 0, device=device)
            self._pa_std = torch.ones(1, 0, device=device)
            self._train_pa = torch.zeros(n, 0, device=device)
            self._h = torch.zeros(0, device=device)
        else:
            pa = ensure_2d(parents).float().to(device)
            self._d_pa = pa.shape[1]
            pa_m, pa_s = weighted_moments(pa, w_vec, unbiased=True)
            self._pa_mean = pa_m.unsqueeze(0)
            self._pa_std = pa_s.unsqueeze(0).clamp_min(self.min_bandwidth)
            pa_std_space = (pa - self._pa_mean) / self._pa_std
            self._train_pa = pa_std_space
            self._h = self._rule_bandwidth(pa_std_space, w_vec)

        self._train_y = y
        return {
            "n_train": int(n),
            "d_pa": int(self._d_pa),
            "d_x": int(d_x),
            "child_bandwidth": self._b.detach().cpu().tolist(),
            "parent_bandwidth": self._h.detach().cpu().tolist(),
        }

    @property
    def is_fitted(self) -> bool:
        return self._train_y is not None

    # ------------------------------------------------------------------
    # Core chunked kernels (operate on flattened 2-D inputs)
    # ------------------------------------------------------------------
    def _std_parents(self, parents: torch.Tensor) -> torch.Tensor:
        parents = _sanitise_parents(parents.float(), mech_name="ConditionalKDE")
        return (parents - self._pa_mean) / self._pa_std

    def _truncation_k(self) -> int | None:
        """Effective neighbour count, or ``None`` when the exact path applies.

        Truncation needs a parent space to search and only pays (and only
        changes the answer) when ``k < N``; ``k >= N`` sums the same rows as
        the exact estimator, so it takes the exact code path and stays
        byte-identical.
        """
        k = self.n_neighbors
        if k is None or self._d_pa == 0 or self._train_y is None:
            return None
        n = self._train_y.shape[0]
        return k if k < n else None

    def _topk_neighbours(self, xs: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``k`` nearest training rows to each standardised query row.

        Distances are measured in bandwidth-scaled parent space, so the
        parent log-kernel of a selected row is exactly ``-0.5 * d2``.
        Streams over training chunks with a running top-``k`` and tiles over
        query rows: peak memory is ``O(query_chunk * (train_chunk + k))``,
        never ``[M, N]``.

        Returns ``(idx [M, k] long, d2 [M, k])``, each row sorted ascending
        by distance.
        """
        tpa = self._train_pa                                   # [N, D_pa]
        h = self._h.to(tpa.device)
        n = tpa.shape[0]
        tpa_h = tpa / h                                        # [N, D_pa]
        xs_h = xs / h                                          # [M, D_pa]
        m = xs_h.shape[0]
        idx_out = torch.empty(m, k, dtype=torch.long, device=tpa.device)
        for qs in range(0, m, self.query_chunk):
            qsl = slice(qs, qs + self.query_chunk)
            xq = xs_h[qsl]                                     # [q, D_pa]
            best_d = None                                      # [q, <=k]
            best_idx = None
            for ts in range(0, n, self.train_chunk):
                tsl = slice(ts, min(ts + self.train_chunk, n))
                # Selection only: cdist's matmul form is far cheaper than the
                # [q, c, D_pa] difference tensor; its float32 cancellation
                # error can at most permute near-ties, and the kernel weights
                # are recomputed exactly on the selected rows below.
                d = torch.cdist(xq, tpa_h[tsl])                # [q, c]
                cand_idx = torch.arange(tsl.start, tsl.stop, device=tpa.device)
                cand_idx = cand_idx.unsqueeze(0).expand(d.shape[0], -1)
                if best_d is not None:
                    d = torch.cat([best_d, d], dim=1)
                    cand_idx = torch.cat([best_idx, cand_idx], dim=1)
                kk = min(k, d.shape[1])
                best_d, pos = torch.topk(d, kk, largest=False, dim=1, sorted=True)
                best_idx = torch.gather(cand_idx, 1, pos)
            idx_out[qsl] = best_idx
        # Exact bandwidth-scaled squared distances of the selected rows —
        # the same arithmetic as the exact path's ``dx``.
        diff = xs_h.unsqueeze(1) - tpa_h[idx_out]              # [M, k, D_pa]
        return idx_out, diff.pow(2).sum(-1)

    def _logp_2d_truncated(
        self, y: torch.Tensor, xs: torch.Tensor, k: int,
    ) -> torch.Tensor:
        """Truncated Nadaraya--Watson: sums over each query's ``k`` neighbours.

        ``y`` ``[M, D_x]`` on the training device, ``xs`` ``[M, D_pa]``
        standardised parents.  Same log-space arithmetic as the exact
        ``_logp_2d`` restricted to the gathered rows.
        """
        ty = self._train_y
        device = ty.device
        b = self._b.to(device)
        log_b_norm = (torch.log(b) + 0.5 * _LOG_2PI).sum()
        idx, d2 = self._topk_neighbours(xs, k)                # [M, k]
        ty_k = ty[idx]                                         # [M, k, D_x]
        dy = (y.unsqueeze(1) - ty_k) / b                       # [M, k, D_x]
        logKy = (-0.5 * dy.pow(2)).sum(-1) - log_b_norm        # [M, k]
        logKpa = -0.5 * d2                                     # [M, k]
        if self._train_logw is not None:
            logKpa = logKpa + self._train_logw.to(device)[idx]
        return torch.logsumexp(logKpa + logKy, dim=1) - torch.logsumexp(logKpa, dim=1)

    def _logp_2d(self, y: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        """log p(y | parents) for ``y`` ``[M, D_x]``, ``parents`` ``[M, D_pa]``/None → ``[M]``."""
        assert self._train_y is not None, "Call fit_local before log_prob."
        ty = self._train_y                       # [N, D_x]
        n = ty.shape[0]
        m = y.shape[0]
        device = ty.device
        y = y.to(device).float()
        b = self._b.to(device)
        log_b_norm = (torch.log(b) + 0.5 * _LOG_2PI).sum()  # scalar child-kernel norm

        if self._d_pa == 0 or parents is None:
            xs = None
        else:
            xs = self._std_parents(parents).to(device)        # [M, D_pa]
            k_trunc = self._truncation_k()
            if k_trunc is not None:
                return self._logp_2d_truncated(y, xs, k_trunc)
            tpa = self._train_pa                              # [N, D_pa]
            h = self._h.to(device)

        logw = self._train_logw
        neg_inf = torch.full((m,), float("-inf"), device=device)
        log_num = neg_inf.clone()   # logsumexp_i [log w_i + logK_pa + logK_y]
        log_den = neg_inf.clone()   # logsumexp_i [log w_i + logK_pa]
        for start in range(0, n, self.train_chunk):
            sl = slice(start, start + self.train_chunk)
            ty_c = ty[sl]                                     # [c, D_x]
            # child log-kernel: [M, c]
            dy = (y.unsqueeze(1) - ty_c.unsqueeze(0)) / b      # [M, c, D_x]
            logKy = (-0.5 * dy.pow(2)).sum(-1) - log_b_norm    # [M, c]
            if xs is None:
                logKpa = torch.zeros(m, ty_c.shape[0], device=device)
            else:
                tpa_c = tpa[sl]                                # [c, D_pa]
                dx = (xs.unsqueeze(1) - tpa_c.unsqueeze(0)) / h  # [M, c, D_pa]
                logKpa = (-0.5 * dx.pow(2)).sum(-1)            # [M, c] (norm const drops out)
            if logw is not None:
                # Broadcast this chunk's log-weights across the query axis.
                logKpa = logKpa + logw[sl].to(device).unsqueeze(0)
            log_num = torch.logaddexp(log_num, torch.logsumexp(logKpa + logKy, dim=1))
            log_den = torch.logaddexp(log_den, torch.logsumexp(logKpa, dim=1))
        return log_num - log_den

    def _sample_2d(self, parents: torch.Tensor | None, n_samples: int) -> torch.Tensor:
        """Return ``[M, n_samples, D_x]`` conditional samples."""
        assert self._train_y is not None, "Call fit_local before sample."
        ty = self._train_y
        n, d_x = ty.shape
        device = ty.device
        b = self._b.to(device)

        if self._d_pa == 0 or parents is None:
            m = 1 if parents is None else ensure_2d(parents).shape[0]
            idx = torch.randint(0, n, (m, n_samples), device=device)
            base = ty[idx]                                    # [m, n, D_x]
            return base + b * torch.randn_like(base)

        xs = self._std_parents(ensure_2d(parents)).to(device)  # [M, D_pa]
        m = xs.shape[0]
        k_trunc = self._truncation_k()
        if k_trunc is not None:
            # Mixture over each query's k neighbours only: the component
            # logits are the parent log-kernels (+ log-weights) of those rows.
            idx, d2 = self._topk_neighbours(xs, k_trunc)       # [M, k]
            logits = -0.5 * d2
            if self._train_logw is not None:
                logits = logits + self._train_logw.to(device)[idx]
            pick = torch.distributions.Categorical(logits=logits).sample((n_samples,))  # [n, M]
            comp = torch.gather(idx, 1, pick.transpose(0, 1))  # [M, n] training rows
            base = ty[comp]                                    # [M, n, D_x]
            return base + b * torch.randn_like(base)
        h = self._h.to(device)
        tpa = self._train_pa
        out = torch.empty(m, n_samples, d_x, device=device)
        for start in range(0, m, self.query_chunk):
            sl = slice(start, start + self.query_chunk)
            xq = xs[sl]                                        # [q, D_pa]
            dx = (xq.unsqueeze(1) - tpa.unsqueeze(0)) / h       # [q, N, D_pa]
            logw = (-0.5 * dx.pow(2)).sum(-1)                  # [q, N]
            idx = torch.distributions.Categorical(logits=logw).sample((n_samples,))  # [n, q]
            idx = idx.transpose(0, 1)                          # [q, n]
            base = ty[idx]                                     # [q, n, D_x]
            out[sl] = base + b * torch.randn_like(base)
        return out

    # ------------------------------------------------------------------
    # Distribution interface (shape normalisation, then delegate)
    # ------------------------------------------------------------------
    def forward(self, parents: torch.Tensor | None) -> _KDEConditionalDistribution:
        b = 1 if parents is None else ensure_2d(parents).shape[0]
        return _KDEConditionalDistribution(self, parents, b)

    def log_prob(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        squeeze_s = False
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_s = True
        b, s, d_x = x.shape
        if parents is None:
            pa_flat = None
        else:
            if parents.dim() == 2:
                parents = parents.unsqueeze(1).expand(-1, s, -1)
            pa_flat, _, _ = flatten_samples(parents)
        y_flat = x.reshape(b * s, d_x)
        lp = self._logp_2d(y_flat, pa_flat).reshape(b, s)
        return lp.squeeze(1) if squeeze_s else lp

    def sample(self, parents: torch.Tensor | None, n: int = 1) -> torch.Tensor:
        out = self._sample_2d(parents, n)
        if parents is None:
            return out  # [1, n, D_x]
        return out  # [B, n, D_x]
