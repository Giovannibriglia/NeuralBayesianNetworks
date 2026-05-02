from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Mapping, Optional

import torch

from nbn.mechanisms.deterministic import DeterministicMechanism
from nbn.utils.batching import pack_parents

if TYPE_CHECKING:
    from nbn.core.network import NeuralBayesianNetwork


def ancestral_sample(
    model: "NeuralBayesianNetwork",
    n: int = 1,
    evidence: Optional[Dict[str, torch.Tensor]] = None,
    do: Optional[Mapping[str, torch.Tensor]] = None,
    device: Optional[str] = None,
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
            samp = mechanisms[node].sample(pa_tensor, n=n)  # [n, n, D] or [1, n, D]
            if samp.shape[0] == 1:
                samp = samp.squeeze(0)  # [n, D]
            else:
                samp = samp[0]  # first batch row
            out[node] = samp.reshape(n, -1)
            if return_log_prob:
                lp = mechanisms[node].log_prob(out[node], pa_tensor)
                log_probs[node] = lp

    if return_log_prob:
        total_lp = sum(log_probs.values())
        return out, total_lp  # type: ignore[return-value]
    return out
