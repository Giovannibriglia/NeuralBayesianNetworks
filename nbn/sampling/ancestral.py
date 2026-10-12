from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping

import torch

from nbn.core.variables import validate_discrete_value
from nbn.mechanisms.parametric.deterministic import DeterministicMechanism
from nbn.utils.device import to_device

if TYPE_CHECKING:
    from nbn.core.network import NeuralBayesianNetwork


def _normalise_do_value(model, node: str, val: Any, dev: torch.device) -> torch.Tensor:
    """Turn a user do-value into a float ``[D]`` (or ``[1, D]``) tensor on ``dev``.

    Accepts Python scalars, 0-d tensors and integer tensors (``do={"A": 1}``
    used to crash, and an int64 value produced int64 samples for that node).
    Float tensors keep their dtype and autograd graph, so ``sample(do=...)``
    stays differentiable w.r.t. the intervention value.
    """
    t = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
    t = to_device(t, dev)
    if not t.is_floating_point():
        t = t.to(torch.get_default_dtype())
    if t.dim() == 0:
        t = t.reshape(1)
    # Every node here yields exactly ``n`` rows, so there is no axis for a
    # per-row intervention value to live on.  Without this check a batched
    # do-value died further down in ``expand`` with an opaque shape error.
    if t.dim() > 1 and t.shape[0] != 1:
        raise ValueError(
            f"Batched do-value for '{node}' (shape {tuple(t.shape)}): "
            f"ancestral sampling draws one mutilated model's worth of "
            f"samples and has no batch axis to vary the intervention "
            f"along.  Loop over the values, or use "
            f"query()/query_batch(do=...), which accept a batch."
        )
    var = model.variables.get(node)
    if var is not None and var.is_discrete:
        validate_discrete_value(node, t, var.cardinality)
    return t


def _normalise_evidence(model, node: str, val: Any, n: int, dev: torch.device) -> torch.Tensor:
    """Shape evidence for ``node`` to ``[n, D]``, or raise.

    Accepted: a scalar / 0-d tensor; ``[D]`` (one value for every row);
    ``[n]`` for a one-dimensional node (one value per row); ``[1, D]`` or
    ``[n, D]``.  Anything else raises: a ``[B, D]`` evidence with
    ``B not in (1, n)`` used to be stored as-is, so its rows silently paired
    up with the wrong samples of every other node.
    """
    t = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
    shape = tuple(t.shape)
    t = to_device(t, dev)
    var = model.variables.get(node)
    d = int(var.dim) if var is not None else (t.shape[-1] if t.dim() > 1 else 1)
    if t.dim() == 0:
        t = t.reshape(1, 1)
    elif t.dim() == 1:
        if t.shape[0] == d:
            t = t.reshape(1, d)            # documented [D] form
        elif d == 1 and t.shape[0] == n:
            t = t.reshape(n, 1)            # per-row values of a scalar node
    if t.dim() != 2 or t.shape[-1] != d or t.shape[0] not in (1, n):
        raise ValueError(
            f"Evidence for '{node}' has shape {shape}; expected a scalar, "
            f"[{d}], [1, {d}] or [{n}, {d}] (leading dim 1 or n={n}, last "
            f"dim = the variable's dimension {d})."
        )
    if var is not None and var.is_discrete:
        # A negative index would silently wrap to the last state in a CPT
        # lookup, an out-of-range one crash deep inside a mechanism.
        validate_discrete_value(node, t, var.cardinality, what="Evidence")
    return t.expand(n, d)


def ancestral_sample(
    model: NeuralBayesianNetwork,
    n: int = 1,
    evidence: Dict[str, torch.Tensor] | None = None,
    do: Mapping[str, torch.Tensor] | None = None,
    device: str | None = None,
    return_log_prob: bool = False,
    return_log_weight: bool = False,
):
    """Batched ancestral sampler.

    Traverses the topological order once, sampling each node conditioned on
    its already-sampled parents.

    **Evidence is clamped, not conditioned on.**  An evidence node takes its
    observed value and its descendants are sampled given it, but its
    *ancestors* are drawn from the prior — so the samples follow the clamped
    (do-like for descendants) distribution, **not** the posterior
    ``p(. | evidence)``.  For the posterior, weight each sample by
    ``exp(log_weight)`` (``return_log_weight=True``) and self-normalise: that
    is likelihood weighting.  Or use ``model.query``.

    Do-intervened nodes are clamped without contributing likelihood.

    Parameters
    ----------
    model: NeuralBayesianNetwork
    n: number of samples.
    evidence:
        Dict node → value: a scalar, ``[D]`` (same for every row), ``[1, D]``
        or ``[n, D]`` (one value per row; ``[n]`` for a scalar node).  Other
        shapes raise.
    do:
        Do-intervention values (Python scalar, 0-d or ``[D]`` tensor; one
        value per node).  A ``DeterministicMechanism`` replaces the node's
        CPD on its downstream path (mutilated graph).  Discrete values must
        be integral state indices in range.
    device:
        Override device.
    return_log_prob:
        If True, also return the per-sample ``[n]`` log-probability of the
        whole sample under the mutilated model — sampled *and* evidence
        nodes, do-targets excluded.  This is a joint density, not a
        likelihood-weighting weight.
    return_log_weight:
        If True, also return the per-sample ``[n]`` likelihood-weighting
        log-weight ``sum_{e in evidence} log p(e | pa(e))`` (zeros without
        evidence).

    Returns
    -------
    ``samples`` — a dict node → ``[n, D_x]`` — alone, or a tuple
    ``(samples, log_prob)``, ``(samples, log_weight)`` or
    ``(samples, log_prob, log_weight)`` according to the flags.
    """
    dev = torch.device(device or model.device)
    evidence = dict(evidence or {})
    do = dict(do or {})
    unknown = sorted(k for k in (*evidence, *do) if k not in model.variables)
    if unknown:
        raise ValueError(f"Unknown evidence/do node(s): {unknown}.")

    # Build a potentially mutilated mechanism dict
    mechanisms = dict(model.mechanisms)
    for node, val in do.items():
        mechanisms[node] = DeterministicMechanism(_normalise_do_value(model, node, val, dev))
    ev = {k: _normalise_evidence(model, k, v, n, dev) for k, v in evidence.items()}

    out: Dict[str, torch.Tensor] = {}
    log_probs: Dict[str, torch.Tensor] = {}
    want_lp = return_log_prob or return_log_weight

    for node in model.dag.topological_order():
        parents = model.dag.parents(node)
        if parents:
            pa_parts = [out[p].squeeze(-1) if out[p].dim() == 3 and out[p].shape[-1] == 1
                        else out[p] for p in parents]
            # Concatenate: all [n, D] → [n, sum(D)]
            pa_tensor = torch.cat([p.reshape(n, -1) for p in pa_parts], dim=-1)
        else:
            pa_tensor = None

        if node in ev:
            val = ev[node]
            out[node] = val
            if want_lp:
                log_probs[node] = mechanisms[node].log_prob(val, pa_tensor)
        elif node in do:
            out[node] = mechanisms[node].sample(pa_tensor, n=1).squeeze(1).expand(n, -1)
        else:
            # Two regimes:
            #   * Root node (pa_tensor is None) — mechanism has B=1 internally,
            #     so we ask for n iid samples; output shape ``[1, n, D]``,
            #     squeeze the singleton batch axis to get ``[n, D]``.
            #   * Non-root (pa_tensor has shape ``[n, P]``) — each row already
            #     carries its row's parent values, so we want exactly one
            #     sample per row; output shape ``[n, 1, D]``, squeeze the
            #     singleton sample axis to get ``[n, D]``.
            # The pre-fix path always asked for ``n`` samples even with a
            # batched parent tensor, materialising a wasteful ``[n, n, D]``
            # intermediate (root cause of v0.5 issue #11) and then taking
            # ``samp[0]`` — every observation came out conditioned on
            # parent row 0, so synthetic-BN training data carried no
            # actual parent dependence (LG fitter saw weight ≈ 0 against
            # truth ≈ ±2 at n_train=2000).  This fix closes both issues.
            if pa_tensor is None:
                samp = mechanisms[node].sample(None, n=n)  # [1, n, D]
                samp = samp.squeeze(0)                     # [n, D]
            else:
                samp = mechanisms[node].sample(pa_tensor, n=1)  # [n_rows, 1, D]
                samp = samp.squeeze(1)
            out[node] = samp.reshape(n, -1)
            if return_log_prob:
                log_probs[node] = mechanisms[node].log_prob(out[node], pa_tensor)

    def _total(nodes) -> torch.Tensor:
        total = torch.zeros(n, device=dev)
        for k in nodes:
            total = total + log_probs[k].reshape(n)
        return total

    extras = []
    if return_log_prob:
        extras.append(_total(log_probs))
    if return_log_weight:
        extras.append(_total(ev))
    if extras:
        return (out, *extras)
    return out
