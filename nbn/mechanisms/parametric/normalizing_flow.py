from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Distribution

from nbn.learning.warm_start import check_shapes
from nbn.learning.weighting import (
    select,
    validate_weights,
    weighted_mean,
    weighted_moments,
)
from nbn.mechanisms.base import Mechanism
from nbn.utils.batching import _sanitise_parents, ensure_2d, flatten_samples


class _FlowDistribution(Distribution):
    """Wraps a zuko flow as a torch.distributions.Distribution.

    The flow models the *standardised* child ``z = (x - loc) / scale``; this
    wrapper presents it in raw units: ``x = loc + scale * z`` and
    ``log p(x) = log p_flow(z) - sum(log scale)``.  ``loc``/``scale`` are
    ``None`` (identity) for a flow without standardisation buffers.
    """

    has_rsample = True

    def __init__(self, flow_dist, loc=None, scale=None) -> None:
        self._flow_dist = flow_dist
        self._loc = loc
        self._scale = scale
        super().__init__(
            batch_shape=flow_dist.batch_shape,
            event_shape=flow_dist.event_shape,
            validate_args=False,
        )

    def rsample(self, sample_shape=torch.Size()):
        z = self._flow_dist.rsample(sample_shape)
        return z if self._loc is None else self._loc + self._scale * z

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if self._loc is None:
            return self._flow_dist.log_prob(x)
        z = (x.to(self._scale.dtype) - self._loc) / self._scale
        return self._flow_dist.log_prob(z) - torch.log(self._scale).sum()


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
    supports_weights: bool = True
    warm_start_is_noop: bool = False

    def __init__(
        self,
        d_x: int = 1,
        num_transforms: int = 5,
        hidden: Tuple[int, ...] = (64, 64),
        bins: int = 8,
    ) -> None:
        super().__init__()
        try:
            import zuko  # noqa: F401 — import validates zuko is installed
        except ImportError as e:
            raise ImportError(
                "NormalizingFlowMechanism requires zuko. "
                "Install with: pip install zuko"
            ) from e
        # NB: we deliberately do NOT stash the zuko module on self (it is a
        # module object and cannot be pickled, which broke torch.save of a
        # fitted model — issue #191 Path 2). Every method re-imports zuko
        # locally, so the attribute was dead. See _build_flow().
        self.d_x = d_x
        self.output_dim = d_x
        self.num_transforms = num_transforms
        self.hidden = tuple(hidden)
        self.bins = bins
        self._d_pa: int = 0
        self._flow: nn.Module | None = None
        # Data standardisation (weighted population moments, set by a cold
        # fit_local, frozen under warm_start).  zuko's rational-quadratic
        # splines act on [-5, 5] and are the identity outside it, and the
        # conditioner MLP saturates on large contexts, so a child living at
        # ~100 was fitted with NLL 5003 instead of 1.4 and sampled near 0.
        # The flow now sees z = (x - _x_mean) / _x_std and the clamped
        # context (pa - _pa_mean) / _pa_std; log_prob subtracts
        # sum(log _x_std) (the Jacobian) and sample maps back.  None =
        # identity, which keeps an unfitted or hand-built flow working.
        self.register_buffer("_x_mean", None)    # [D_x]
        self.register_buffer("_x_std", None)     # [D_x]
        self.register_buffer("_pa_mean", None)   # [D_pa]
        self.register_buffer("_pa_std", None)    # [D_pa]
        # Standardised contexts are clamped to +-_pa_clamp (as MDN does): a
        # far-out-of-distribution particle must not drive the conditioner
        # into overflow; at 50 sigma the prediction is noise regardless.
        self._pa_clamp: float = 50.0

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
        weights: torch.Tensor | None = None,
        warm_start: bool = False,
        early_stopping: bool = True,
        patience: int = 5,
        min_delta: float = 1e-3,
        val_fraction: float = 0.1,
        root_lr_multiplier: float = 10.0,
        **kwargs,
    ) -> dict:
        """Fit the flow by minibatch Adam on the negative log-likelihood.

        ``epochs`` is a *cap*, not a budget: with ``early_stopping`` (the
        default) a ``val_fraction`` slice of the rows is held out, its
        (weighted) mean NLL is evaluated after every epoch, and training stops
        once it has failed to improve by ``min_delta`` nats/row for
        ``patience`` consecutive epochs.  The parameters are then restored to
        the best epoch seen.  Measured on the n=50 synthetic problems (20480
        rows, batch 1024): conditional nodes reach their held-out optimum
        within ~60-140 steps and *overfit* afterwards, so the former fixed
        100-300 epochs (2000-6000 steps) were ~20x more work for a slightly
        worse fit.  The split is skipped (full-budget training, as before)
        when the slice would hold fewer than ``_MIN_VAL_ROWS`` rows, and when
        ``epochs == 0`` (the warm-start no-op contract).

        ``root_lr_multiplier`` scales ``lr`` for a *root* node (no parents).
        An unconditional zuko NSF has no conditioner network -- its spline
        parameters are plain biases that move ~lr per Adam step from a poor
        initialisation -- so at lr 5e-4 a root was still descending after
        2000 steps and lost to LinearGaussian on every root tested, while
        the same node at 5e-3 matched LinearGaussian in 300-500 steps.
        Conditional nodes were unchanged across that lr range.  A multiplier
        (rather than an absolute root lr) keeps ``lr=0.0`` a no-op everywhere.
        """
        import copy

        warm = bool(warm_start) and self.is_fitted
        # Train in the parameters' dtype (the default dtype for a fresh
        # build): float64 data reached a float32 zuko net and crashed.
        x = ensure_2d(x).to(self._param_dtype() if warm else torch.get_default_dtype())
        n, d_x = x.shape
        device = x.device
        w_vec = validate_weights(weights, n, where="NormalizingFlowMechanism.fit_local", device=device)

        if parents is None or parents.shape[-1] == 0:
            d_pa = 0
        else:
            parents = ensure_2d(parents).to(device=device, dtype=x.dtype)
            d_pa = parents.shape[1]

        # Validate before mutating self.d_x/output_dim, so a rejected warm
        # start leaves the mechanism describing the shape it actually has.
        # No check_branch here: unlike MDN and neural-categorical, a root flow
        # (context width 0) is still a trained zuko NSF rather than a
        # closed-form branch, so root-ness is fully captured by d_pa.
        if warm:
            check_shapes("NormalizingFlowMechanism.fit_local", {
                "d_x": (self.d_x, d_x), "d_pa": (self._d_pa, d_pa),
            })
        self.d_x = d_x
        self.output_dim = d_x

        # Standardisation buffers are frozen under a warm start: the flow's
        # weights were trained against the map they define, and recomputing
        # them would apply every learned spline to shifted, rescaled data.
        # See nbn/learning/warm_start.py.  Population moments: the scale only
        # fixes the flow's working units (the Jacobian keeps log_prob exact),
        # and the population form is replication-exact and defined for one
        # row or a total weight <= 1.
        if not warm:
            x_m, x_s = weighted_moments(x, w_vec)
            x_s = x_s.nan_to_num(0.0)
            self._x_mean = x_m.detach()
            self._x_std = torch.where(x_s > 0, x_s, torch.ones_like(x_s)).detach()
            if d_pa:
                pa_m, pa_s = weighted_moments(parents, w_vec)
                # Same 1e-3 floor as MDN's parent standardisation.
                self._pa_mean = pa_m.detach()
                self._pa_std = pa_s.nan_to_num(0.0).clamp_min(1e-3).detach()
            else:
                self._pa_mean = self._pa_std = None
            self._build_flow(d_pa, device)
        # Everything below trains on standardised values.
        x = self._std_x(x)
        if d_pa:
            parents = self._std_ctx(parents)
        # Fresh optimiser either way -- Adam's moments are not in
        # state_dict(), so persisting them would survive a caller's
        # load_state_dict revert of a rejected step.
        eff_lr = lr * root_lr_multiplier if d_pa == 0 else lr
        opt = torch.optim.Adam(self.parameters(), lr=eff_lr)

        # Train/validation split for early stopping.
        n_val = int(n * val_fraction) if (early_stopping and epochs > 0) else 0
        if n_val < self._MIN_VAL_ROWS or n - n_val < 1:
            n_val = 0
        if n_val:
            split = torch.randperm(n, device=device)
            val_idx, tr_idx = split[:n_val], split[n_val:]
            x_tr, x_val = x[tr_idx], x[val_idx]
            pa_tr = parents[tr_idx] if d_pa else None
            pa_val = parents[val_idx] if d_pa else None
            w_tr, w_val = select(w_vec, tr_idx), select(w_vec, val_idx)
        else:
            x_tr, pa_tr, w_tr = x, (parents if d_pa else None), w_vec
            x_val = pa_val = w_val = None
        n_tr = x_tr.shape[0]

        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        epochs_run = steps = 0
        early_stopped = False

        self.train()
        for _ in range(epochs):
            perm = torch.randperm(n_tr, device=device)
            for i in range(0, n_tr, batch_size):
                idx = perm[i:i + batch_size]
                bx = x_tr[idx]
                ctx = pa_tr[idx] if d_pa else None
                loss = weighted_mean(
                    -self._flow(ctx).log_prob(bx), select(w_tr, idx),
                )
                opt.zero_grad(); loss.backward()
                if d_pa:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                opt.step()
                steps += 1
            epochs_run += 1
            if not n_val:
                continue
            val = self._val_nll(x_val, pa_val, w_val)
            if val < best_val - min_delta:
                best_val = val
                # state_dict() aliases the live parameters -- snapshot a copy.
                best_state = copy.deepcopy(self._flow.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    early_stopped = True
                    break
        if best_state is not None:
            self._flow.load_state_dict(best_state)
        self.eval()
        return {
            "d_pa": d_pa, "d_x": d_x, "warm_started": warm,
            "epochs_run": epochs_run, "steps": steps,
            "early_stopped": early_stopped, "n_val": n_val,
            "best_val_nll": best_val if n_val else None,
        }

    # ------------------------------------------------------------------
    # Standardisation helpers
    # ------------------------------------------------------------------

    def _param_dtype(self) -> torch.dtype:
        """dtype of the fitted flow (default dtype before any fit)."""
        for p in self.parameters():
            return p.dtype
        return torch.get_default_dtype()

    def _std_x(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self._param_dtype())
        return x if self._x_mean is None else (x - self._x_mean) / self._x_std

    def _std_ctx(self, parents: torch.Tensor) -> torch.Tensor:
        """Cast, standardise and clamp a context block ``[..., D_pa]``.

        The cast also covers discrete (Long) parents, which would otherwise
        hit ``F.linear(Long, Float)``.
        """
        ctx = parents.to(self._param_dtype())
        if self._pa_mean is None:
            return ctx
        ctx = (ctx - self._pa_mean) / self._pa_std
        return ctx.clamp(-self._pa_clamp, self._pa_clamp)

    def _log_det(self) -> torch.Tensor:
        """``sum(log _x_std)``: the raw-vs-standardised log-density offset."""
        return torch.log(self._x_std).sum()

    def _unstd_x(self, z: torch.Tensor) -> torch.Tensor:
        return z if self._x_mean is None else self._x_mean + self._x_std * z

    # Below this many held-out rows the validation NLL is too noisy to gate
    # on, so fit_local trains the full ``epochs`` budget instead.
    _MIN_VAL_ROWS: int = 32

    def _val_nll(
        self,
        x_val: torch.Tensor,
        pa_val: torch.Tensor | None,
        w_val: torch.Tensor | None,
    ) -> float:
        """(Weighted) mean NLL of the held-out slice; ``inf`` if non-finite.

        ``x_val``/``pa_val`` are already standardised; the constant Jacobian
        ``sum(log _x_std)`` is added so the reported ``best_val_nll`` is in
        raw units, like ``log_prob`` (a constant shift, so the early-stopping
        comparisons are unaffected).
        """
        self.eval()
        with torch.no_grad():
            val = weighted_mean(-self._flow(pa_val).log_prob(x_val), w_val)
            if self._x_std is not None:
                val = val + self._log_det()
        self.train()
        v = float(val)
        return v if v == v and v != float("inf") else float("inf")

    @property
    def is_fitted(self) -> bool:
        """True iff ``fit_local`` built the zuko flow.

        Without this override the mechanism inherited
        ``Mechanism.is_fitted``'s ``False`` default and reported unfitted
        after a successful fit.
        """
        return self._flow is not None

    def forward(self, parents: torch.Tensor | None) -> _FlowDistribution:
        assert self._flow is not None, "Call fit_local before forward()."
        if self._d_pa == 0 or parents is None:
            ctx = None
        else:
            # Sanitise NaN/Inf -> 0 (a non-finite context from a deep
            # ancestral chain would trip a zuko NSF device assert; mirrors
            # MDN's guard, #82), then standardise and clamp.
            ctx = self._std_ctx(_sanitise_parents(
                ensure_2d(parents).to(self._param_dtype()), mech_name="Flow.forward",
            ))
        return _FlowDistribution(self._flow(ctx), self._x_mean, self._x_std)

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
            parents = parents.to(self._param_dtype())  # Long parents -> float
            parents = _sanitise_parents(parents, mech_name="Flow.log_prob")
            if parents.dim() == 2:
                parents = parents.unsqueeze(1).expand(-1, s, -1)
            flat, _, _ = flatten_samples(parents)
            ctx = self._std_ctx(flat)

        z_flat = self._std_x(x.reshape(b * s, d_x))
        lp = self._flow(ctx).log_prob(z_flat).reshape(b, s)
        if self._x_std is not None:
            lp = lp - self._log_det()      # change of variables x = m + s z
        return lp.squeeze(1) if squeeze_s else lp

    def sample(self, parents: torch.Tensor | None, n: int = 1) -> torch.Tensor:
        assert self._flow is not None
        b = 1 if parents is None else ensure_2d(parents).shape[0]
        if self._d_pa == 0 or parents is None:
            with torch.no_grad():
                samp = self._flow(None).sample((b * n,))  # [B*n, D_x]
        else:
            ctx = _sanitise_parents(
                ensure_2d(parents).to(self._param_dtype()), mech_name="Flow.sample",
            ).unsqueeze(1).expand(-1, n, -1)
            flat, _, _ = flatten_samples(ctx)
            with torch.no_grad():
                samp = self._flow(self._std_ctx(flat)).sample()  # [B*n, D_x]
        return self._unstd_x(samp).reshape(b, n, self.d_x)


class ConditionalFlowMechanism(NormalizingFlowMechanism):
    """Alias for ``NormalizingFlowMechanism`` — emphasises the conditioned use case."""
    pass
