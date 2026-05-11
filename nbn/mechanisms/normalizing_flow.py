from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Distribution

from nbn.mechanisms.base import Mechanism
from nbn.utils.batching import ensure_2d, flatten_samples


class _FlowDistribution(Distribution):
    """Wraps a zuko flow as a torch.distributions.Distribution."""

    has_rsample = True

    def __init__(self, flow_dist) -> None:
        self._flow_dist = flow_dist
        super().__init__(
            batch_shape=flow_dist.batch_shape,
            event_shape=flow_dist.event_shape,
            validate_args=False,
        )

    def rsample(self, sample_shape=torch.Size()):
        return self._flow_dist.rsample(sample_shape)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._flow_dist.log_prob(x)


class NormalizingFlowMechanism(Mechanism):
    """Conditional Normalizing Flow CPD using ``zuko``.

    Each conditional distribution P(X | pa) is modelled by a Neural Spline
    Flow (NSF) conditioned on pa.  The flow is invertible and differentiable,
    so both ``log_prob`` and ``rsample`` are available for autograd.

    Parameters
    ----------
    d_x: int
        Dimensionality of the output variable.
    num_transforms: int
        Number of spline transforms.
    hidden: tuple of int
        Hidden widths of the conditioner network.
    bins: int
        Number of spline bins (rational-quadratic).

    Notes
    -----
    Requires ``zuko>=1.2`` (``pip install zuko``).
    """

    is_discrete: bool = False

    def __init__(
        self,
        d_x: int = 1,
        num_transforms: int = 5,
        hidden: Tuple[int, ...] = (64, 64),
        bins: int = 8,
    ) -> None:
        super().__init__()
        try:
            import zuko
        except ImportError as e:
            raise ImportError(
                "NormalizingFlowMechanism requires zuko. "
                "Install with: pip install zuko"
            ) from e
        self._zuko = zuko
        self.d_x = d_x
        self.output_dim = d_x
        self.num_transforms = num_transforms
        self.hidden = tuple(hidden)
        self.bins = bins
        self._d_pa: int = 0
        self._flow: nn.Module | None = None

    def _build_flow(self, d_pa: int, device: torch.device) -> None:
        import zuko
        self._d_pa = d_pa
        self._flow = zuko.flows.NSF(
            features=self.d_x,
            context=d_pa,
            transforms=self.num_transforms,
            hidden_features=list(self.hidden),
            bins=self.bins,
        ).to(device)

    def fit_local(
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        epochs: int = 300,
        lr: float = 5e-4,
        batch_size: int = 512,
        **kwargs,
    ) -> dict:
        x = ensure_2d(x)  # [N, D_x]
        n, d_x = x.shape
        device = x.device
        self.d_x = d_x
        self.output_dim = d_x

        if parents is None or parents.shape[-1] == 0:
            d_pa = 0
        else:
            parents = ensure_2d(parents).to(device=device, dtype=x.dtype)
            d_pa = parents.shape[1]

        self._build_flow(d_pa, device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        if d_pa == 0:
            self.train()
            for _ in range(epochs):
                perm = torch.randperm(n, device=device)
                for i in range(0, n, batch_size):
                    bx = x[perm[i:i + batch_size]]
                    loss = -self._flow(None).log_prob(bx).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
        else:
            dataset = torch.utils.data.TensorDataset(parents, x)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.train()
            for _ in range(epochs):
                for bp, bx in loader:
                    loss = -self._flow(bp).log_prob(bx).mean()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                    opt.step()
        self.eval()
        return {"d_pa": d_pa, "d_x": d_x}

    def forward(self, parents: torch.Tensor | None) -> _FlowDistribution:
        assert self._flow is not None, "Call fit_local before forward()."
        if self._d_pa == 0 or parents is None:
            return _FlowDistribution(self._flow(None))
        return _FlowDistribution(self._flow(ensure_2d(parents)))

    def log_prob(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        assert self._flow is not None
        squeeze_s = False
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.dim() == 2:
            x = x.unsqueeze(1); squeeze_s = True
        b, s, d_x = x.shape

        if self._d_pa == 0 or parents is None:
            ctx = None
        else:
            if parents.dim() == 2:
                parents = parents.unsqueeze(1).expand(-1, s, -1)
            flat, _, _ = flatten_samples(parents)
            ctx = flat

        x_flat = x.reshape(b * s, d_x)
        lp = self._flow(ctx).log_prob(x_flat).reshape(b, s)
        return lp.squeeze(1) if squeeze_s else lp

    def sample(self, parents: torch.Tensor | None, n: int = 1) -> torch.Tensor:
        assert self._flow is not None
        b = 1 if parents is None else ensure_2d(parents).shape[0]
        if self._d_pa == 0 or parents is None:
            with torch.no_grad():
                samp = self._flow(None).sample((b * n,))  # [B*n, D_x]
        else:
            ctx = ensure_2d(parents).unsqueeze(1).expand(-1, n, -1)
            flat, _, _ = flatten_samples(ctx)
            with torch.no_grad():
                samp = self._flow(flat).sample()  # [B*n, D_x]
        return samp.reshape(b, n, self.d_x)


class ConditionalFlowMechanism(NormalizingFlowMechanism):
    """Alias for ``NormalizingFlowMechanism`` — emphasises the conditioned use case."""
    pass
