"""Online EWC / online-Laplace updater for gradient-trained mechanisms.

No-rehearsal incremental update for neural mechanisms (MDN, neural-categorical).
The old task's posterior over weights is approximated as a Gaussian centered at
the post-fit weights ``theta*`` with diagonal precision = empirical Fisher ``F``
(a Laplace approximation).  Training on new data then minimises

    -E_new[log p(x | pa)]  +  (lam / 2) * sum_i F_i * (theta_i - theta*_i)^2,

so the quadratic penalty anchors weights that mattered for the old data
(large ``F``) while letting unimportant ones adapt.  This is the standard EWC
penalty (Kirkpatrick et al. 2017, PNAS); the *online* variant keeps a single
running Gaussian — after each update ``theta*`` is reset to the current weights
and ``F`` is decayed and accumulated (``F <- forgetting * F_old + F_new``),
following Schwarz et al. 2018 / Huszar 2018.

State (``theta*`` and ``F``) is persisted per mechanism as two flat buffers
(``_ewc_mu``, ``_ewc_fisher``) — concatenations over ``mech.parameters()`` in
order — so ``.to()``, ``state_dict`` save/load, and ``intervene()``'s deepcopy
carry them, and the param↔slot mapping needs no dotted-name buffers.

Fisher estimator — what the default computes and why
-----------------------------------------------------
``F`` is the diagonal *empirical* Fisher: the mean over samples of the
per-sample squared NLL gradient, ``F_i = (1/N) sum_n (d(-log p(x_n))/dtheta_i)^2``.
The default ``fisher_batch_size = 1`` computes exactly this — each backward pass
sees one row, so nothing cancels.  ``sample_cap`` bounds the number of rows used,
so consolidation is O(sample_cap) backward passes regardless of fit-set size.

``fisher_batch_size > 1`` instead squares the gradient of the **summed** NLL over
each minibatch and averages over minibatches — a cheaper approximation that is
*biased low*: near the fit optimum the per-sample gradients largely cancel before
squaring, so larger batches underestimate ``F``, slacken the EWC penalty, and
weaken protection of the old task.  Use it only when consolidation cost dominates
and you accept that bias; the default is the faithful per-sample estimate.

At ``fisher_batch_size = 1`` the per-sample gradients are computed in one
vectorised pass (``torch.func.vmap(grad(functional_call(...)))``, chunked to
bound memory) instead of ``sample_cap`` sequential backward passes.  The two
agree to float rounding; the vectorised pass is ~25x faster on CPU and avoids
thousands of tiny kernel launches on cuda / mps, where the loop dominated
``fit()``.  On mps the pass runs on a CPU copy (see
``_per_sample_fisher_vmap``).  A mechanism whose ``log_prob`` cannot be traced by ``vmap`` falls
back to the loop.
"""
from __future__ import annotations

import copy
import logging
from typing import List

import torch
from torch import nn
from torch.func import functional_call, grad, vmap

from nbn.utils.batching import assume_finite_parents

logger = logging.getLogger(__name__)

# Upper bound on per-sample-gradient elements materialised at once by the
# vectorised Fisher (chunk_size * n_params); 2**24 float32s = 64 MiB.
_VMAP_MAX_ELEMENTS = 2 ** 24

# Default EWC penalty strength.  This is the stability<->plasticity knob: too
# large freezes the weights (no adaptation to new data), too small fails to
# protect the old task.  It is scale-coupled to the Fisher magnitude, so the
# right value is problem-dependent — calibrate on validation data.
#
# 50.0 was chosen by a CPU-pinned sweep over the two-regime forgetting benchmark
# (see tests/unit/test_update_neural_ewc.py) at the default per-sample Fisher
# (fisher_batch_size=1).  It sits on a minimum-variance plateau: across five
# seeds the protection margin clustered tightly at ~0.5-0.6 NLL (vs noisy,
# threshold-grazing margins at lam<=20 and high-variance ones at lam>=60) while
# plasticity stayed saturated.  Because the default Fisher is the faithful
# per-sample estimate, lam is no longer fighting an under-scaled Fisher, so a
# single scalar is robust here without per-Fisher normalisation.
DEFAULT_LAM: float = 50.0


def _trainable_params(mech) -> List[torch.Tensor]:
    """Trainable leaf parameters in ``parameters()`` order (mechanism-agnostic)."""
    return [p for p in mech.parameters() if p.requires_grad]


def _flatten(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _estimate_fisher(mech, x, parents, *, fisher_batch_size: int, sample_cap: int):
    """Diagonal empirical Fisher per trainable parameter.

    At the default ``fisher_batch_size = 1`` this is the exact per-sample
    empirical Fisher (mean of per-sample squared gradients); larger batches are
    a biased-low approximation (see the module docstring).  Returns
    ``(params, fisher)`` where ``fisher[i]`` has the shape of ``params[i]``.
    Runs in eval mode (restores the prior mode) and under ``enable_grad`` so it
    works even if called inside a ``no_grad`` context.
    """
    params = _trainable_params(mech)
    n = int(x.shape[0])
    device = x.device
    if n > sample_cap:
        idx = torch.randperm(n, device=device)[:sample_cap]
        x = x.index_select(0, idx)
        if parents is not None:
            parents = parents.index_select(0, idx)
        n = sample_cap

    bs = max(int(fisher_batch_size), 1)
    if bs == 1:
        try:
            return params, _per_sample_fisher_vmap(mech, x, parents)
        except Exception as exc:  # a log_prob vmap cannot trace -> exact loop
            logger.debug(
                "%s: vectorised Fisher unavailable (%s: %s); using the "
                "per-sample loop", type(mech).__name__, type(exc).__name__, exc,
            )

    fisher = [torch.zeros_like(p) for p in params]
    was_training = mech.training
    mech.eval()
    num_batches = 0
    with torch.enable_grad():
        for i in range(0, n, bs):
            xb = x[i:i + bs]
            pb = parents[i:i + bs] if parents is not None else None
            nll = -mech.log_prob(xb, pb).sum()
            grads = torch.autograd.grad(
                nll, params, retain_graph=False, allow_unused=True
            )
            for f, g in zip(fisher, grads):
                if g is not None:  # allow_unused: a param not on the path → skip
                    f.add_(g.detach().pow(2))
            num_batches += 1
    if was_training:
        mech.train()

    inv = 1.0 / max(num_batches, 1)
    return params, [f * inv for f in fisher]


class _LogProb(nn.Module):
    """Expose ``mech.log_prob`` as ``forward`` so ``functional_call`` can drive it."""

    def __init__(self, mech) -> None:
        super().__init__()
        self.mech = mech

    def forward(self, x, parents):
        return self.mech.log_prob(x, parents)


def _per_sample_fisher_vmap(mech, x, parents) -> List[torch.Tensor]:
    """Exact per-sample empirical Fisher in one chunked ``vmap`` pass.

    Same estimator as the ``fisher_batch_size = 1`` loop: the mean over rows
    of the squared gradient of that row's NLL.  Parents are sanitised once,
    outside the traced region, and ``torch.distributions`` argument
    validation is off inside it: both are data-dependent Python branches that
    ``vmap`` cannot trace, and the rows are the fit's own training data.

    On ``mps`` the pass runs on a CPU copy of the mechanism and the result is
    moved back.  The vmapped gradients come out wrong on some Metal devices
    (GitHub's virtualised macOS runners: every element off, up to 200x) while
    matching on others, so Metal is not trusted with it.  At <= ``sample_cap``
    rows the CPU pass takes tens of milliseconds.
    """
    if x.device.type == "mps":
        cpu_mech = copy.deepcopy(mech).to("cpu")
        fisher = _per_sample_fisher_vmap(
            cpu_mech, x.cpu(), None if parents is None else parents.cpu(),
        )
        return [f.to(x.device) for f in fisher]
    if parents is not None:
        from nbn.utils.batching import _sanitise_parents

        parents = _sanitise_parents(parents, mech_name=type(mech).__name__)
    wrapper = _LogProb(mech)
    names = [k for k, p in wrapper.named_parameters() if p.requires_grad]
    params = {k: p.detach() for k, p in wrapper.named_parameters() if p.requires_grad}
    buffers = dict(wrapper.named_buffers())

    def nll(p, xi, pi):
        pa = None if pi is None else pi.unsqueeze(0)
        return -functional_call(wrapper, (p, buffers), (xi.unsqueeze(0), pa)).sum()

    n_params = max(sum(p.numel() for p in params.values()), 1)
    chunk = max(1, min(int(x.shape[0]), _VMAP_MAX_ELEMENTS // n_params))
    was_training = mech.training
    was_validating = torch.distributions.Distribution._validate_args
    mech.eval()
    torch.distributions.Distribution.set_default_validate_args(False)
    try:
        with torch.enable_grad(), assume_finite_parents():
            per_sample = vmap(
                grad(nll), in_dims=(None, 0, None if parents is None else 0),
                chunk_size=chunk,
            )(params, x, parents)
    finally:
        torch.distributions.Distribution.set_default_validate_args(was_validating)
        if was_training:
            mech.train()
    return [per_sample[k].pow(2).mean(0) for k in names]


def _store(mech, mu: torch.Tensor, fisher: torch.Tensor) -> None:
    """Write theta*/F to buffers, registering them on first call (refit-safe)."""
    for name, val in (("_ewc_mu", mu), ("_ewc_fisher", fisher)):
        if name in mech._buffers:
            mech._buffers[name] = val
        else:
            mech.register_buffer(name, val)


def consolidate(
    mech,
    x: torch.Tensor,
    parents: torch.Tensor | None,
    *,
    fisher_batch_size: int = 1,
    sample_cap: int = 4096,
) -> None:
    """Snapshot ``theta*`` and the diagonal Fisher ``F`` after a fit.

    Call at the end of ``fit_local`` (data still in hand).  Mechanism-agnostic:
    works for root and non-root via ``parameters()`` + ``log_prob`` only.
    """
    params, fisher = _estimate_fisher(
        mech, x, parents, fisher_batch_size=fisher_batch_size, sample_cap=sample_cap
    )
    mu = _flatten([p.detach() for p in params])
    fish = _flatten(fisher)
    _store(mech, mu, fish)


def ewc_update(
    mech,
    x: torch.Tensor,
    parents: torch.Tensor | None,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    lam: float,
    forgetting: float,
    fisher_batch_size: int = 1,
    sample_cap: int = 4096,
) -> dict:
    """Train on new data under the EWC penalty, then refresh the running prior.

    Minimises ``-log_prob.mean() + (lam/2) * sum_i F_i (theta_i - theta*_i)^2``
    with Adam, reusing the mechanism's own minibatch structure.  Afterwards
    performs the online-EWC refresh: ``theta* <- current weights`` and
    ``F <- forgetting * F_old + F_new`` (F_new re-estimated on the new data).
    """
    if getattr(mech, "_ewc_mu", None) is None:
        raise RuntimeError(
            "EWC state missing — this mechanism was fitted with "
            "consolidate=False (or never fitted), so no theta*/Fisher "
            "snapshot exists to anchor the update. Refit with "
            "consolidate=True (the default of model.fit) before calling "
            "update()."
        )
    params = _trainable_params(mech)
    mu = mech._ewc_mu
    fisher = mech._ewc_fisher
    total = sum(p.numel() for p in params)
    if total != mu.numel():
        raise RuntimeError(
            f"EWC state size {mu.numel()} != current parameter count {total}; "
            "the parameter set changed since consolidation"
        )

    opt = torch.optim.Adam(params, lr=lr)
    n = int(x.shape[0])
    device = x.device
    bs = max(int(batch_size), 1)
    mech.train()
    for _ in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = x.index_select(0, idx)
            pb = parents.index_select(0, idx) if parents is not None else None
            nll = -mech.log_prob(xb, pb).mean()
            flat = torch.cat([p.reshape(-1) for p in params])
            penalty = 0.5 * lam * (fisher * (flat - mu).pow(2)).sum()
            loss = nll + penalty
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
    mech.eval()

    # Online-EWC refresh: anchor to the new weights, decay-and-accumulate F.
    _, fisher_new = _estimate_fisher(
        mech, x, parents, fisher_batch_size=fisher_batch_size, sample_cap=sample_cap
    )
    mech._ewc_mu = _flatten([p.detach() for p in params])
    mech._ewc_fisher = forgetting * fisher + _flatten(fisher_new)
    return {"method": "online_ewc"}
