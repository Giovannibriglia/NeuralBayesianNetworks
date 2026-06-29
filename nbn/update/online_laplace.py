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

# TODO: a future opt-in could compute the exact per-sample empirical Fisher in a
# single vectorized pass via ``torch.func.vmap(grad(functional_call(...)))`` —
# deferred (would add a torch.func code path; the per-row loop here is correct
# and bounded by sample_cap).
"""
from __future__ import annotations

from typing import List

import torch

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

    fisher = [torch.zeros_like(p) for p in params]
    was_training = mech.training
    mech.eval()
    bs = max(int(fisher_batch_size), 1)
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
        raise AssertionError(
            "EWC state missing — fit_local (which consolidates) must run before "
            "update_local"
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
