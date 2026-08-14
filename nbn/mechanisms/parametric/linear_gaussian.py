from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from nbn.mechanisms.base import Mechanism
from nbn.update import recursive_gaussian
from nbn.utils.batching import ensure_2d, flatten_samples


class LinearGaussianMechanism(Mechanism):
    """Linear Gaussian CPD: P(X | pa) = N(W·pa + b, diag(σ²)).

    Parameters
    ----------
    ridge:
        L2 regularisation strength for closed-form ridge regression.
    min_scale:
        Minimum allowed standard deviation (prevents degenerate variance).
    learnable:
        If True, also exposes W, b, log_scale as ``nn.Parameter`` so that
        gradient-based joint training can fine-tune after the closed-form fit.
    """

    is_discrete: bool = False
    supports_update: bool = True

    def __init__(
        self,
        ridge: float = 1e-6,
        min_scale: float = 1e-3,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.ridge = float(ridge)
        self.min_scale = float(min_scale)
        self.learnable = learnable
        # Populated by fit_local:
        self._weight: nn.Parameter | None = None   # [D_pa, D_x]
        self._bias: nn.Parameter | None = None     # [D_x]
        self._log_scale: nn.Parameter | None = None  # [D_x]
        self.output_dim = 1
        self._input_dim = 0
        # Persisted normal-equation sufficient statistics for incremental
        # update (A = Z^T Z, B = Z^T x, c = sum x^2, N).  Buffers so .to(),
        # state_dict, and intervene()'s deepcopy carry them.  None until fit.
        self.register_buffer("_neq_A", None)
        self.register_buffer("_neq_B", None)
        self.register_buffer("_neq_c", None)
        self.register_buffer("_neq_N", None)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        **kwargs,
    ) -> dict:
        """Closed-form ridge regression.

        Parameters
        ----------
        x: shape ``[N, D_x]``.
        parents: shape ``[N, D_pa]``, or ``None`` for root nodes.
        """
        x = ensure_2d(x)
        n, d_x = x.shape
        device = x.device
        self.output_dim = d_x

        if parents is None or (parents.numel() > 0 and parents.shape[-1] == 0):
            # Root node: fit a simple Gaussian
            mean = x.mean(0)
            std = x.std(0, unbiased=False).clamp_min(1e-6)
            w = torch.zeros(0, d_x, device=device, dtype=x.dtype)
            b = mean
            log_s = torch.log(std)
            self._input_dim = 0
        else:
            parents = ensure_2d(parents).to(device=device, dtype=x.dtype)
            d_pa = parents.shape[1]
            self._input_dim = d_pa
            ones = torch.ones(n, 1, device=device, dtype=x.dtype)
            x_aug = torch.cat([parents, ones], dim=1)  # [N, D_pa+1]

            if self.ridge > 0:
                reg_pa = math.sqrt(self.ridge) * torch.eye(
                    d_pa, device=device, dtype=x.dtype
                )
                reg_row = torch.zeros(d_pa, 1, device=device, dtype=x.dtype)
                reg_x = torch.cat([reg_pa, reg_row], dim=1)
                x_aug_r = torch.cat([x_aug, reg_x], dim=0)
                y_r = torch.cat([x, torch.zeros(d_pa, d_x, device=device, dtype=x.dtype)], dim=0)
            else:
                x_aug_r, y_r = x_aug, x

            theta = torch.linalg.lstsq(x_aug_r, y_r).solution  # [D_pa+1, D_x]
            w = theta[:-1]       # [D_pa, D_x]
            b = theta[-1]        # [D_x]
            resid = x - x_aug @ theta
            std = resid.std(0, unbiased=False).clamp_min(self.min_scale)
            log_s = torch.log(std)

        self._set_params(w, b, log_s)

        # Persist normal-equation sufficient statistics so update_local can
        # accumulate new data without rehearsal (posterior-as-prior).  Uses the
        # same design matrix Z = [parents, 1] the closed-form fit solves over.
        st = recursive_gaussian.batch_statistics(parents, x)
        self._neq_A = st.A.detach().clone()
        self._neq_B = st.B.detach().clone()
        self._neq_c = st.c.detach().clone()
        self._neq_N = st.N.detach().clone()

        return {"n_params": (w.numel() + b.numel() + log_s.numel())}

    def _set_params(
        self, w: torch.Tensor, b: torch.Tensor, log_s: torch.Tensor
    ) -> None:
        """Write W/b/log_scale back through the learnable-vs-buffer branch.

        Shared by ``fit_local`` and ``update_local`` so both honour the
        ``learnable`` flag identically.
        """
        if self.learnable:
            self._weight = nn.Parameter(w)
            self._bias = nn.Parameter(b)
            self._log_scale = nn.Parameter(log_s)
        else:
            if "_weight_buf" in self._buffers:
                self._weight_buf = w
                self._bias_buf = b
                self._log_scale_buf = log_s
            else:
                self.register_buffer("_weight_buf", w)
                self.register_buffer("_bias_buf", b)
                self.register_buffer("_log_scale_buf", log_s)
            self._weight = self._weight_buf      # type: ignore[assignment]
            self._bias = self._bias_buf          # type: ignore[assignment]
            self._log_scale = self._log_scale_buf  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def update_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        *,
        forgetting: float = 1.0,
        **kwargs,
    ) -> dict:
        """Fold new data into the linear-Gaussian CPD via recursive least sq.

        Accumulates the new-data normal-equation statistics onto the persisted
        prior and re-solves.  For ``forgetting == 1.0`` a chunked
        ``fit``→``update`` reproduces a single pooled ridge fit (to solver
        precision); ``forgetting < 1.0`` fades the older statistics.
        """
        assert self._neq_A is not None, "call fit_local before update_local"
        x = ensure_2d(x)
        prior = recursive_gaussian.NormalEquationState(
            self._neq_A, self._neq_B, self._neq_c, self._neq_N, self._input_dim
        )
        batch = recursive_gaussian.batch_statistics(parents, x)
        state = recursive_gaussian.accumulate(prior, batch, forgetting=forgetting)
        w, b, log_s = recursive_gaussian.solve(
            state, ridge=self.ridge, min_scale=self.min_scale
        )
        self.output_dim = x.shape[1]
        self._set_params(w, b, log_s)
        self._neq_A = state.A.detach().clone()
        self._neq_B = state.B.detach().clone()
        self._neq_c = state.c.detach().clone()
        self._neq_N = state.N.detach().clone()
        return {"method": "recursive_gaussian"}

    def _scale(self) -> torch.Tensor:
        return torch.exp(self._log_scale).clamp_min(self.min_scale)

    def _cast_parents(
        self, parents: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Cast ``parents`` to the fitted parameter dtype.

        ``fit_local`` already casts its parent block to ``x.dtype`` before
        solving, but the inference path did not: a discrete parent arrives as
        a ``long`` column straight out of ``pack_parents`` (which concatenates
        raw data dtypes, because tabular discrete children need their parent
        values as *indices*).  ``parents_2d @ self._weight`` is a matmul, and
        matmul does not type-promote, so a discrete-parent → continuous-child
        edge — exactly the shape of a discrete action driving a continuous
        reward — died with ``expected m1 and m2 to have the same dtype``.
        The first caller to hit it is ``learning.fit``'s post-fit LL pass, so
        the crash surfaced from ``model.fit()`` even though the ridge solve
        itself had succeeded.  MDN and flow never had the bug: they
        standardise parents (``(pa - mean) / std``) before the first matmul,
        and that division type-promotes.
        """
        if parents is None:
            return None
        ref = self._weight if self._weight is not None else self._bias
        if ref is not None and parents.dtype != ref.dtype:
            parents = parents.to(ref.dtype)
        return parents

    # ------------------------------------------------------------------
    # Distribution interface
    # ------------------------------------------------------------------

    def forward(self, parents: torch.Tensor | None) -> Independent:
        assert self._bias is not None, "Call fit_local before forward()."
        scale = self._scale()  # [D_x]
        parents = self._cast_parents(parents)
        if self._input_dim == 0 or parents is None:
            b = 1 if parents is None else ensure_2d(parents).shape[0]
            loc = self._bias.unsqueeze(0).expand(b, -1)  # [B, D_x]
            scale_e = scale.unsqueeze(0).expand(b, -1)
        else:
            parents_2d = ensure_2d(parents)  # [B, D_pa]
            loc = parents_2d @ self._weight + self._bias  # [B, D_x]
            scale_e = scale.unsqueeze(0).expand(parents_2d.shape[0], -1)
        return Independent(Normal(loc, scale_e), 1)

    def sample(self, parents: torch.Tensor | None, n: int = 1) -> torch.Tensor:
        """Return ``[B, n, D_x]``."""
        assert self._bias is not None
        scale = self._scale()
        parents = self._cast_parents(parents)
        if self._input_dim == 0 or parents is None:
            b = 1 if parents is None else ensure_2d(parents).shape[0]
            loc = self._bias.view(1, 1, -1).expand(b, n, -1)
            sc = scale.view(1, 1, -1).expand(b, n, -1)
        else:
            parents_2d = ensure_2d(parents)
            b = parents_2d.shape[0]
            loc_2d = parents_2d @ self._weight + self._bias  # [B, D_x]
            loc = loc_2d.unsqueeze(1).expand(-1, n, -1)
            sc = scale.view(1, 1, -1).expand(b, n, -1)
        return loc + sc * torch.randn_like(loc)

    @property
    def is_fitted(self) -> bool:
        """True iff ``fit_local`` (or ``update_local``) solved for W/b/sigma.

        Without this override the mechanism inherited ``Mechanism.is_fitted``'s
        ``False`` default and stayed ``False`` after a perfectly successful
        fit, so any caller gating on it saw a fitted model as unfitted.
        """
        return self._bias is not None

    def log_prob(
        self, x: torch.Tensor, parents: torch.Tensor | None
    ) -> torch.Tensor:
        assert self._bias is not None
        parents = self._cast_parents(parents)
        x = ensure_2d(x)  # [B, D_x] or keep [B, S, D_x]
        squeeze_s = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_s = True
        b, s, d_x = x.shape
        scale = self._scale()
        if self._input_dim == 0 or parents is None:
            loc = self._bias.view(1, 1, -1).expand(b, s, -1)
            sc = scale.view(1, 1, -1).expand(b, s, -1)
        else:
            if parents.dim() == 2:
                parents = parents.unsqueeze(1).expand(-1, s, -1)
            flat, _, _ = flatten_samples(parents)
            loc_flat = flat @ self._weight + self._bias  # [B*S, D_x]
            loc = loc_flat.reshape(b, s, d_x)
            sc = scale.view(1, 1, -1).expand(b, s, -1)
        var = sc.pow(2)
        lp = -0.5 * (((x - loc) / sc).pow(2) + 2 * torch.log(sc) + math.log(2 * math.pi))
        lp = lp.sum(-1)  # [B, S]
        if squeeze_s:
            lp = lp.squeeze(1)
        return lp
