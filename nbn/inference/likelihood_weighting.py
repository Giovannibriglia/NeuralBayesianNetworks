from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from nbn.inference.base import InferenceEngine
from nbn.inference.state import get_inference_state
from nbn.utils.batching import ensure_2d


# ---------------------------------------------------------------------------
# PSIS k-hat diagnostic (X2) — Pareto-smoothed importance sampling tail index.
# ---------------------------------------------------------------------------

def _gpdfit(x: np.ndarray) -> float:
    """Zhang–Stephens (2009) generalized-Pareto shape estimate ``k̂``.

    ``x``: sorted-ascending positive tail exceedances.  Ported from the
    arviz/loo empirical-Bayes estimator (validated against the Gaussian-IS
    regimes in the X2 investigation).
    """
    n = len(x)
    prior_bs, prior_k = 3.0, 10.0
    m = 30 + int(n ** 0.5)
    b = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b /= prior_bs * x[int(n / 4 + 0.5) - 1]
    b += 1.0 / x[-1]
    k = np.log1p(-b[:, None] * x[None, :]).mean(axis=1)              # [m]
    log_lik = n * (np.log(-b / k) - k - 1.0)
    w = 1.0 / np.exp(log_lik[None, :] - log_lik[:, None]).sum(axis=1)
    b_post = np.sum(b * w)
    k_post = np.log1p(-b_post * x).mean()
    return (n * k_post + prior_k * 0.5) / (n + prior_k)             # weak prior → 0.5


def psis_khat(log_weights: torch.Tensor) -> float | None:
    """Pareto-smoothed importance-sampling ``k̂`` for one set of IS weights.

    Fits a generalized Pareto distribution to the upper tail of the importance
    weights and returns its shape parameter.  Interpretation (Vehtari et al.):
    ``k̂ < 0.5`` reliable; ``0.5 ≤ k̂ < 0.7`` usable; ``0.7 ≤ k̂ < 1`` unreliable;
    ``k̂ ≥ 1`` broken.

    Returns ``None`` on degenerate input — near-uniform weights (e.g. proposal
    == target, no heavy tail), too few particles for a meaningful tail
    (``M = min(0.2N, 3√N) < 5``), or any non-finite intermediate — which
    downstream treats as "no reliability concern / not measurable".
    """
    lw = log_weights.detach().reshape(-1).to(torch.float64).cpu().numpy()
    n = lw.shape[0]
    if n < 25 or not np.isfinite(lw).all():
        return None
    M = int(min(0.2 * n, 3.0 * np.sqrt(n)))
    if M < 5:
        return None
    mx = lw.max()
    lw = lw - (mx + np.log(np.exp(lw - mx).sum()))                  # normalize (scale-free)
    order = np.sort(lw)
    cutoff = order[n - M - 1]
    x = np.exp(order[n - M:]) - np.exp(cutoff)                      # tail exceedances
    x = np.sort(x[x > 0])
    if len(x) < 5:
        return None                                                # degenerate tail
    try:
        k = _gpdfit(x)
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return None
    return float(k) if np.isfinite(k) else None


def _psis_khat_rows(log_w: torch.Tensor) -> torch.Tensor:
    """Per-row ``k̂`` for ``log_w`` ``[B, S]`` → ``[B]`` (NaN where degenerate)."""
    ks = [psis_khat(log_w[b]) for b in range(log_w.shape[0])]
    return torch.tensor([float("nan") if v is None else v for v in ks])


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
        # Normalise scalar / 0-D evidence and do values to 1-D so v.shape[0]
        # works.
        def _norm(d):
            return {
                k: (v if (isinstance(v, torch.Tensor) and v.dim() >= 1)
                    else (torch.as_tensor(v).reshape(1) if not isinstance(v, torch.Tensor)
                          else v.reshape(1)))
                for k, v in d.items()
            }
        evidence = _norm(evidence)
        do = _norm(do)
        # The batch axis is the widest of evidence AND do.  Deriving it from
        # evidence alone made a batched intervention (with no evidence to
        # widen B) die in the ``expand`` below with an opaque shape error —
        # ``do`` is a peer of ``evidence`` here, both are just clamped nodes,
        # so a per-row intervention costs no more than a per-row observation.
        b = max(
            (v.shape[0] for v in list(evidence.values()) + list(do.values())),
            default=1,
        )
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

    # Query-time only: mechanisms are eval()'d but their params still require
    # grad, so without this every particle's sample()/log_prob() for MDN /
    # flow / neuralcat builds and retains an autograd graph across the whole
    # topological sweep — pure waste of query-time compute and memory (VE
    # already detaches at factor build, see tensor_ve._extract_factors).
    # ``query_batch`` delegates here and ``AmortizedISEngine`` inherits both,
    # so this covers every public LW/AIS query entry; proposal TRAINING
    # (``AmortizedISEngine.train_proposal``) is not routed through ``query``
    # and keeps its gradients.
    @torch.inference_mode()
    def query(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor] | None = None,
        do: Dict[str, torch.Tensor] | None = None,
        n_samples: int | None = None,
        return_ess: bool = False,
        return_psis_k: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        s = n_samples or self.n_samples
        log_w, buf = self._run(model, targets, evidence, do, s)

        # Normalise weights
        weights = torch.softmax(log_w, dim=-1)  # [B, S]

        # Per-query effective-sample-size fraction (X1).  ESS = (Σw)²/Σw² =
        # 1/Σw² after softmax; the fraction ESS/S ∈ (0, 1] is comparable across
        # configs with different particle counts.  Computed only when requested
        # so default callers are unchanged.
        ess_frac = None
        if return_ess:
            # Detached: a pure diagnostic, never part of any autograd graph.
            ess_frac = (1.0 / (weights.pow(2).sum(dim=-1) * s)).detach()  # [B]

        # Per-query PSIS k̂ tail diagnostic (X2).  NaN per row where degenerate
        # (near-uniform weights); a better quality signal than ESS but used as a
        # diagnostic only — the fit-time gate (P1) is unchanged.
        khat = _psis_khat_rows(log_w.detach()) if return_psis_k else None  # [B]

        # Append requested diagnostics in a stable order (ess, then khat) so
        # ``return_ess`` alone keeps its X1 contract.
        def _wrap(out):
            extras = []
            if return_ess:
                extras.append(ess_frac)
            if return_psis_k:
                extras.append(khat)
            return (out, *extras) if extras else out

        if len(targets) == 1:
            tgt = targets[0]
            mech = model.mechanisms[tgt]

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
                out = probs.squeeze(0) if b == 1 else probs
                return _wrap(out)

        state = get_inference_state(model, targets, tuple(sorted((evidence or {}).keys())),
                                     tuple(sorted((do or {}).keys())), self._cache)
        tgt_slices = [state.node_slices[state.node_to_idx[t]] for t in targets]
        samps = [buf[..., sl] for sl in tgt_slices]
        out = (weights, torch.cat(samps, dim=-1))
        return _wrap(out)

    def query_batch(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor],
        return_ess: bool = False,
        return_psis_k: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Batched query — all B evidence rows processed in a single GPU launch."""
        return self.query(model, targets, evidence, return_ess=return_ess,
                          return_psis_k=return_psis_k, **kwargs)
