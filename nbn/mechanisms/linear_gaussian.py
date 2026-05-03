from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from nbn.mechanisms.base import Mechanism
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

        if self.learnable:
            self._weight = nn.Parameter(w)
            self._bias = nn.Parameter(b)
            self._log_scale = nn.Parameter(log_s)
        else:
            self.register_buffer("_weight_buf", w)
            self.register_buffer("_bias_buf", b)
            self.register_buffer("_log_scale_buf", log_s)
            self._weight = self._weight_buf      # type: ignore[assignment]
            self._bias = self._bias_buf          # type: ignore[assignment]
            self._log_scale = self._log_scale_buf  # type: ignore[assignment]

        return {"n_params": (w.numel() + b.numel() + log_s.numel())}

    def _scale(self) -> torch.Tensor:
        return torch.exp(self._log_scale).clamp_min(self.min_scale)

    # ------------------------------------------------------------------
    # Distribution interface
    # ------------------------------------------------------------------

    def forward(self, parents: torch.Tensor | None) -> Independent:
        assert self._bias is not None, "Call fit_local before forward()."
        scale = self._scale()  # [D_x]
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

    def log_prob(
        self, x: torch.Tensor, parents: torch.Tensor | None
    ) -> torch.Tensor:
        assert self._bias is not None
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
