from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Mapping

import torch

from nbn.mechanisms.deterministic import DeterministicMechanism

if TYPE_CHECKING:
    from nbn.core.network import NeuralBayesianNetwork


def ancestral_sample(
    model: NeuralBayesianNetwork,
    n: int = 1,
    evidence: Dict[str, torch.Tensor] | None = None,
    do: Mapping[str, torch.Tensor] | None = None,
    device: str | None = None,
    return_log_prob: bool = False,
) -> Dict[str, torch.Tensor]:
    """Batched ancestral sampler.

    Traverses the topological order once, sampling each node conditioned on
    its already-sampled parents.  Evidence nodes are clamped to their observed
    values; do-intervened nodes are clamped without contributing likelihood.

    Parameters
    ----------
    model: NeuralBayesianNetwork
    n: number of samples.
    evidence:
        Dict mapping node name → tensor of shape ``[D]`` or ``[B, D]``.
        When provided, ``n`` samples are drawn and evidence rows are replicated.
    do:
        Do-intervention values.  A ``DeterministicMechanism`` replaces the
        node's CPD on its downstream path (mutilated graph).
    device:
        Override device.
    return_log_prob:
        If True, also return per-sample log-probabilities under the model.

    Returns
    -------
    Dict[str, torch.Tensor]
        One tensor per node with shape ``[n, D_x]`` (or ``[B, n, D_x]`` if
        evidence has batch dimension B).
    """
    dev = torch.device(device or model.device)
    evidence = evidence or {}
    do = dict(do or {})

    # Build a potentially mutilated mechanism dict
    mechanisms = dict(model.mechanisms)
    for node, val in do.items():
        val_t = val.to(dev)
        mechanisms[node] = DeterministicMechanism(val_t)

    out: Dict[str, torch.Tensor] = {}
    log_probs: Dict[str, torch.Tensor] = {}

    for node in model.dag.topological_order():
        parents = model.dag.parents(node)
        if parents:
            pa_parts = [out[p].squeeze(-1) if out[p].dim() == 3 and out[p].shape[-1] == 1
                        else out[p] for p in parents]
            # Concatenate: all [n, D] → [n, sum(D)]
            pa_tensor = torch.cat([p.reshape(n, -1) for p in pa_parts], dim=-1)
        else:
            pa_tensor = None

        if node in evidence:
            val = evidence[node].to(dev)
            if val.dim() == 1:
                val = val.unsqueeze(0).expand(n, -1)
            elif val.dim() == 0:
                val = val.unsqueeze(0).unsqueeze(0).expand(n, 1)
            out[node] = val
            if return_log_prob:
                lp = mechanisms[node].log_prob(val, pa_tensor)
                log_probs[node] = lp
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
                lp = mechanisms[node].log_prob(out[node], pa_tensor)
                log_probs[node] = lp

    if return_log_prob:
        total_lp = sum(log_probs.values())
        return out, total_lp  # type: ignore[return-value]
    return out
