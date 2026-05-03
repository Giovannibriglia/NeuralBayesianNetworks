from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from nbn.inference.base import InferenceEngine
from nbn.inference.state import get_inference_state
from nbn.utils.batching import ensure_2d


class LikelihoodWeightingEngine(InferenceEngine):
    """Batched likelihood-weighting (self-normalised IS) inference engine.

    Works for any mixture of discrete/continuous mechanisms because it only
    calls ``mechanism.sample()`` and ``mechanism.log_prob()`` — no enumeration.

    Returns
    -------
    For a single discrete target: normalised probability vector ``[K]`` or ``[B, K]``
    (marginalised by weighting the empirical histogram).
    For continuous / multi-target: ``(weights [B, S], samples [B, S, D])`` tuple.

    Parameters
    ----------
    n_samples: int
        Number of ancestral samples (particles) per query.
    """

    def __init__(self, n_samples: int = 2048) -> None:
        self.n_samples = int(n_samples)
        self._cache: Dict = {}

    def _run(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor] | None,
        do: Dict[str, torch.Tensor] | None,
        n_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core IS loop.  Returns ``(log_weights [B, S], samples [B, S, total_dim])``."""
        evidence = evidence or {}
        do = do or {}
        device = model.device
        dtype = torch.float32

        state = get_inference_state(
            model, targets,
            tuple(sorted(evidence.keys())),
            tuple(sorted(do.keys())),
            self._cache,
        )
        # Normalise scalar / 0-D evidence values to 1-D so v.shape[0] works.
        evidence = {
            k: (v if (isinstance(v, torch.Tensor) and v.dim() >= 1)
                else (torch.as_tensor(v).reshape(1) if not isinstance(v, torch.Tensor)
                      else v.reshape(1)))
            for k, v in evidence.items()
        }
        b = max((v.shape[0] for v in evidence.values()), default=1)
        s = n_samples

        buf = torch.zeros(b, s, state.total_dim, device=device, dtype=dtype)
        log_w = torch.zeros(b, s, device=device, dtype=dtype)

        # Pre-load fixed values [B, 1, D] for evidence and do nodes
        fixed: List[torch.Tensor | None] = [None] * len(state.topo_order)
        for node, val in {**do, **evidence}.items():
            idx = state.node_to_idx[node]
            v = ensure_2d(val.to(device=device, dtype=dtype))  # [B, D]
            fixed[idx] = v.unsqueeze(1).expand(b, s, -1)  # [B, S, D]

        for i, node in enumerate(state.topo_order):
            sl = state.node_slices[i]
            mech = model.mechanisms[node]

            # Build parent tensor [B, S, D_pa]
            psl = state.parent_slices[i]
            if psl:
                pa = torch.cat([buf[..., ps] for ps in psl], dim=-1)  # [B, S, D_pa]
            else:
                pa = None

            if fixed[i] is not None:
                buf[..., sl] = fixed[i]
                # Score evidence nodes (not do nodes)
                if state.evidence_mask[i]:
                    obs = fixed[i]  # [B, S, D]
                    if pa is not None:
                        pa_flat = pa.reshape(b * s, -1)
                        obs_flat = obs.reshape(b * s, -1)
                        lp = mech.log_prob(obs_flat, pa_flat)
                        log_w = log_w + lp.reshape(b, s)
                    else:
                        lp = mech.log_prob(obs.reshape(b * s, -1), None)
                        log_w = log_w + lp.reshape(b, s)
            else:
                # Sample from prior
                if pa is not None:
                    pa_flat = pa.reshape(b * s, -1)
                    samp = mech.sample(pa_flat, n=1).squeeze(1)  # [B*S, D]
                else:
                    samp = mech.sample(None, n=b * s).reshape(b * s, -1)
                buf[..., sl] = samp.reshape(b, s, -1)

        return log_w, buf

    def query(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor] | None = None,
        do: Dict[str, torch.Tensor] | None = None,
        n_samples: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        s = n_samples or self.n_samples
        log_w, buf = self._run(model, targets, evidence, do, s)

        # Normalise weights
        weights = torch.softmax(log_w, dim=-1)  # [B, S]

        if len(targets) == 1:
            tgt = targets[0]
            idx = model.dag.topological_order().index(tgt)
            mech = model.mechanisms[tgt]
            sl = slice(sum(model.mechanisms[n].output_dim for n in model.dag.topological_order()[:idx]),
                       sum(model.mechanisms[n].output_dim for n in model.dag.topological_order()[:idx+1]))

            if mech.is_discrete and hasattr(mech, '_class_values') and mech._class_values is not None:
                # Build weighted histogram over class values [K]
                cv = mech._class_values.to(device=model.device)
                k = len(cv)
                state = get_inference_state(model, targets, tuple(sorted((evidence or {}).keys())),
                                             tuple(sorted((do or {}).keys())), self._cache)
                tgt_sl = state.node_slices[state.node_to_idx[tgt]]
                samp_vals = buf[..., tgt_sl].squeeze(-1).long()  # [B, S]
                b = weights.shape[0]
                probs = torch.zeros(b, k, device=model.device)
                for ci in range(k):
                    mask = (samp_vals == int(cv[ci])).float()
                    probs[:, ci] = (weights * mask).sum(dim=-1)
                probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)
                return probs.squeeze(0) if b == 1 else probs

        state = get_inference_state(model, targets, tuple(sorted((evidence or {}).keys())),
                                     tuple(sorted((do or {}).keys())), self._cache)
        tgt_slices = [state.node_slices[state.node_to_idx[t]] for t in targets]
        samps = [buf[..., sl] for sl in tgt_slices]
        return weights, torch.cat(samps, dim=-1)

    def query_batch(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Batched query — all B evidence rows processed in a single GPU launch."""
        return self.query(model, targets, evidence, **kwargs)
