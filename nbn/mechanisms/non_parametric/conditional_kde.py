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
* **Bandwidth.** Scott's and Silverman's rules-of-thumb are provided, computed
  on **standardised** parents (so the per-dimension scale is O(1)) and on the
  raw child.  ``torch.quantile``-based robust scale (IQR) guards against
  heavy tails, mirroring the project's existing use of ``torch.quantile``.

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

from nbn.learning.weighting import validate_weights, weighted_moments
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
        Number of query rows processed per tile in ``sample``.
    """

    is_discrete: bool = False
    supports_weights: bool = True

    def __init__(
        self,
        bandwidth: str = "scott",
        bw_factor: float | str = 1.0,
        min_bandwidth: float = 1e-3,
        train_chunk: int = 8192,
        query_chunk: int = 1024,
    ) -> None:
        super().__init__()
        if bandwidth not in ("scott", "silverman"):
            raise ValueError(f"bandwidth must be 'scott' or 'silverman', got {bandwidth!r}")
        if isinstance(bw_factor, str) and bw_factor != "auto":
            raise ValueError(f"bw_factor must be a float or 'auto', got {bw_factor!r}")
        self.bandwidth = bandwidth
        self.bw_factor = bw_factor if bw_factor == "auto" else float(bw_factor)
        self.min_bandwidth = float(min_bandwidth)
        self.train_chunk = int(train_chunk)
        self.query_chunk = int(query_chunk)
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
    def _rule_bandwidth(self, data: torch.Tensor) -> torch.Tensor:
        """Per-dimension Scott/Silverman bandwidth for ``data`` ``[N, D]``."""
        n, d = data.shape
        std = data.std(0, unbiased=True)
        # Robust scale: min(std, IQR/1.349) when IQR is informative.
        q = torch.quantile(
            data, torch.tensor([0.25, 0.75], device=data.device, dtype=data.dtype), dim=0
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
        self._b = self._rule_bandwidth(y)

        if parents is None or parents.shape[-1] == 0:
            self._d_pa = 0
            self._pa_mean = torch.zeros(1, 0, device=device)
            self._pa_std = torch.ones(1, 0, device=device)
            self._train_pa = torch.zeros(n, 0, device=device)
            self._h = torch.zeros(0, device=device)
        else:
            pa = ensure_2d(parents).float().to(device)
            self._d_pa = pa.shape[1]
            pa_m, pa_s = weighted_moments(pa, kwargs.get("_w_vec"), unbiased=True)
            self._pa_mean = pa_m.unsqueeze(0)
            self._pa_std = pa_s.unsqueeze(0).clamp_min(self.min_bandwidth)
            pa_std_space = (pa - self._pa_mean) / self._pa_std
            self._train_pa = pa_std_space
            self._h = self._rule_bandwidth(pa_std_space)

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
