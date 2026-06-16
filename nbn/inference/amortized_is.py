"""Amortized neural-proposal importance sampling (Engine A, #181; β / P2).

Self-normalized importance sampling with a learned, **parent-conditioned**
proposal.  A shared trunk encodes the evidence into a context vector once per
query; independent per-node heads then map ``(context, sampled parents)`` to
each latent node's proposal distribution.  The heads are called *inside* the
ancestral-sampling loop with the actual sampled parent values, so the weight
``log p(xᵢ | pa_sampled) − log q(xᵢ | e, pa_sampled)`` no longer carries the
factorization mismatch of the earlier evidence-only proposal ``q(xᵢ | e)``,
which collapsed ESS as the latent count grew (the discrete-network catastrophe;
see the P2 investigation — α-positive on chain/child/alarm).

The engine subclasses ``LikelihoodWeightingEngine`` and overrides only the core
IS loop ``_run``.  All of LW's downstream machinery — weighted histogram for
discrete targets, per-row resampling for continuous targets, ``query`` /
``query_batch``, and the X1 per-query ESS — is reused unchanged.

When the proposal is *not* trained, or the fit-time ESS gate rejects it
(``recognition_net is None``), ``_run`` falls back to the parent LW behaviour
(P1, #204), so the engine always produces results.

Scope (v1): discrete / lg / mdn heads with scalar parents; flow keeps the
Gaussian surrogate (P3 / #185-L1).

References:
- Le, Baydin & Wood (2017) Inference Compilation (sequential/parent-conditioned
  proposals trained on the model joint — the pattern this proposal follows)
- Walecki et al. (2018) Universal Marginalizer Importance Sampling
- Paige & Wood (2016) Inference Networks for SMC

See docs/v0.14-batched-inference-engines-research.md for the research
context.

References:
- Le, Baydin & Wood (2017) Inference Compilation
- Walecki et al. (2018) Universal Marginalizer Importance Sampling
- Paige & Wood (2016) Inference Networks for SMC

See docs/v0.14-batched-inference-engines-research.md for the research
context.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import torch

from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
from nbn.inference.recognition_net import ParentConditionedProposal
from nbn.inference.state import get_inference_state
from nbn.sampling.ancestral import ancestral_sample
from nbn.utils.batching import ensure_2d

logger = logging.getLogger(__name__)

# Fit-time held-out ESS-fraction below which the learned proposal is rejected
# and the engine falls back to its LW path (recognition_net unset).  Shared by
# the diagnostic warning and the fallback action so "warning fires" and "action
# taken" stay consistent (#185 follow-up; addresses the discrete-network
# accuracy catastrophe surfaced in the June 12 bnlearn_complete run).
_ESS_FALLBACK_THRESHOLD = 0.1

# Training-budget scaling law for the parent-conditioned proposal (β / P2).
# Grad steps scale ~linearly with node count: the P2 prototype reached the
# low-per-node-KL regime at ~3200 steps for alarm (37 nodes) ≈ 86·n; rounded to
# 90·n with a floor.  Undertraining is caught by the fit-time ESS gate (→ LW
# fallback), so this is a target, not a hard requirement.  ``time_budget_s`` is
# a safety cap (pre-vectorization, very large nets may hit it and fall back).
_STEPS_PER_NODE = 90
_MIN_TRAIN_STEPS = 500


class AmortizedISEngine(LikelihoodWeightingEngine):
    """Amortized importance-sampling engine with a learned proposal.

    Parameters
    ----------
    n_samples:
        Number of particles per query (importance-sampling sample size).
    """

    def __init__(self, n_samples: int = 1024) -> None:
        super().__init__(n_samples=n_samples)
        self.recognition_net: ParentConditionedProposal | None = None

    # ------------------------------------------------------------------
    # Proposal training (called once, from adapter.fit())
    # ------------------------------------------------------------------

    def train_proposal(
        self,
        model,
        n_training_samples: int = 20000,
        lr: float = 1e-3,
        batch_size: int = 512,
        mask_prob: float = 0.5,
        time_budget_s: float = 600.0,
        steps_per_node: int = _STEPS_PER_NODE,
        device: str | None = None,
    ) -> dict:
        """Train the parent-conditioned proposal ``q(xᵢ | e, paᵢ)`` on prior samples.

        Universal-marginalizer objective with **teacher-forced parents**: draw
        joint samples from the fitted model, randomly mask each entry (observed
        with probability ``mask_prob``), and train each node's head to maximize
        the log-likelihood of the *unobserved* entries given the observed
        evidence context **and that node's true parent values**.

        Budget is step-based and scales with node count (``steps_per_node·n``,
        floored at ``_MIN_TRAIN_STEPS``); ``time_budget_s`` is a safety cap.
        Returns a metrics dict (final loss, grad steps, held-out ESS proxy,
        proposal_used).
        """
        dev = torch.device(device or model.device)
        net = ParentConditionedProposal(model).to(dev)
        self.recognition_net = net
        self.device = dev

        n_nodes = len(net.node_order)
        target_steps = self._target_train_steps(n_nodes, steps_per_node)

        # Ancestral prior samples → [N, n_nodes] value matrix (topo order).
        # Under no_grad: fixed training data; differentiable mechanisms would
        # otherwise tie the model's params into the proposal's autograd graph.
        with torch.no_grad():
            samples = ancestral_sample(model, n=n_training_samples, device=str(dev))
            x = self._stack_values(samples, net.node_order, dev).detach()  # [N, n_nodes]
        n = x.shape[0]

        opt = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        start = time.monotonic()
        last_loss = float("nan")
        steps = 0
        stop = False
        while steps < target_steps and not stop:
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, batch_size):
                xb = x[perm[i:i + batch_size]]                # [Bb, n_nodes]
                mask = (torch.rand_like(xb) < mask_prob).float()  # 1 = observed
                ctx = net.context(xb * mask, mask)            # [Bb, d_ctx]
                loss = self._proposal_nll(net, ctx, xb, mask)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "AmortizedISEngine.train_proposal: non-finite loss "
                        "(NaN/Inf) — proposal training diverged."
                    )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                last_loss = float(loss.detach())
                steps += 1
                if steps >= target_steps or time.monotonic() - start > time_budget_s:
                    stop = True
                    break
            if time.monotonic() - start > time_budget_s:
                logger.info(
                    "AmortizedISEngine: proposal training hit time budget "
                    "(%.0fs) after %d/%d steps.", time_budget_s, steps, target_steps,
                )
        net.eval()

        ess_frac = self._estimate_ess_fraction(model)
        proposal_used = "learned"
        if ess_frac is not None and ess_frac < _ESS_FALLBACK_THRESHOLD:
            # Diagnostic: why the proposal is being rejected.
            logger.warning(
                "AmortizedISEngine proposal appears under-trained "
                "(held-out ESS ≈ %.1f%% of particles, threshold %.0f%%). "
                "Consider more training samples/epochs.",
                100.0 * ess_frac, 100.0 * _ESS_FALLBACK_THRESHOLD,
            )
            # Action: unset the proposal so ``_run`` takes the inherited LW
            # path (recognition_net is None).  A bad learned proposal is worse
            # than LW's prior proposal (June 12: AIS TV/node ≈ 0.53 vs LW
            # ≈ 0.035 on discrete), so falling back is strictly safer.
            logger.warning(
                "AIS proposal rejected (fit-time ESS-fraction %.3f < threshold "
                "%.2f); falling back to LW for this engine.",
                ess_frac, _ESS_FALLBACK_THRESHOLD,
            )
            self.recognition_net = None
            proposal_used = "lw_fallback"
        return {
            "final_loss": last_loss,
            "grad_steps": steps,
            "target_steps": target_steps,
            "ess_fraction": ess_frac,
            "proposal_used": proposal_used,
        }

    @staticmethod
    def _target_train_steps(n_nodes: int, steps_per_node: int = _STEPS_PER_NODE) -> int:
        """Step-based training budget: ``steps_per_node·n`` floored at the min.

        Grad steps scale ~linearly with node count (P2 evidence: ~86·n on alarm).
        Undertraining is caught by the fit-time ESS gate → LW fallback.
        """
        return max(_MIN_TRAIN_STEPS, steps_per_node * int(n_nodes))

    @staticmethod
    def _stack_values(samples, node_order, dev) -> torch.Tensor:
        cols = []
        for node in node_order:
            v = samples[node]
            if v.dim() >= 2:
                v = v[..., 0]
            cols.append(v.reshape(-1).to(device=dev, dtype=torch.float32))
        return torch.stack(cols, dim=1)

    @staticmethod
    def _proposal_nll(net, ctx, xb, mask) -> torch.Tensor:
        """Mean NLL of unobserved entries — grouped batched path (β Phase D, #207).

        Delegates to the proposal's Strategy-F ``grouped_nll`` (one batched
        gather + bmm per signature group), 16–48× faster than the per-node loop
        at scale.  ``_proposal_nll_per_node`` is the equivalent oracle the
        differential test pins this against.
        """
        return net.grouped_nll(ctx, xb, mask)

    @staticmethod
    def _proposal_nll_per_node(net, ctx, xb, mask) -> torch.Tensor:
        """Per-node oracle (pre-Phase-D loop), kept for the differential test.

        ``ctx`` is the evidence context ``[Bb, d_ctx]``; each node's head is fed
        ``(ctx, its TRUE parent values)`` (teacher-forced) and scored on the
        node's own value where it is unobserved.
        """
        total = xb.new_zeros(())
        count = xb.new_zeros(())
        for j, node in enumerate(net.node_order):
            unobs = 1.0 - mask[:, j]
            n_unobs = unobs.sum()
            if float(n_unobs) == 0.0:
                continue
            ppos = net.parents_idx[j]
            pv = xb[:, ppos] if ppos else None          # [Bb, n_parents] scalar cols
            params = net.node_params(j, ctx, pv)        # [Bb, P]
            lp = net.log_prob_target(node, params, xb[:, j])  # [Bb]
            total = total - (lp * unobs).sum()
            count = count + n_unobs
        return total / count.clamp_min(1.0)

    def _estimate_ess_fraction(
        self, model, n_eval: int = 64, diag_particles: int = 256,
    ) -> float | None:
        """Held-out ESS proxy: mean effective-sample-size fraction.

        Observes roughly half of the nodes (from fresh prior samples) as
        evidence, runs the proposal, and returns the mean of
        ``ESS_b / diag_particles`` across the eval batch.  Returns ``None``
        on any failure — a diagnostic must never break ``fit``.
        """
        try:
            dev = self.device or model.device
            with torch.no_grad():
                test = ancestral_sample(model, n=n_eval, device=str(dev))
                node_order = self.recognition_net.node_order
                # Observe the first half of topo order; target the last node.
                n_nodes = len(node_order)
                obs_nodes = node_order[: max(1, n_nodes // 2)]
                tgt = node_order[-1]
                if tgt in obs_nodes:
                    obs_nodes = obs_nodes[:-1] or [node_order[0]]
                evidence: Dict[str, torch.Tensor] = {}
                for node in obs_nodes:
                    v = test[node]
                    if v.dim() >= 2:
                        v = v[..., 0]
                    evidence[node] = v.reshape(-1, 1).to(device=dev, dtype=torch.float32)
                log_w, _ = self._run(model, [tgt], evidence, {}, diag_particles)
                w = torch.softmax(log_w, dim=-1)             # [B, S]
                ess = 1.0 / w.pow(2).sum(dim=-1).clamp_min(1e-12)  # [B]
                return float((ess / diag_particles).mean())
        except Exception as exc:  # pragma: no cover - diagnostic safety
            logger.debug("ESS diagnostic skipped: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Core IS loop — overrides LW._run to use the learned proposal
    # ------------------------------------------------------------------

    def _run(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor] | None,
        do: Dict[str, torch.Tensor] | None,
        n_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.recognition_net is None:
            # No trained proposal → fall back to prior-proposal LW.
            return super()._run(model, targets, evidence, do, n_samples)

        evidence = evidence or {}
        do = do or {}
        device = model.device
        dtype = torch.float32
        net = self.recognition_net

        state = get_inference_state(
            model, targets,
            tuple(sorted(evidence.keys())),
            tuple(sorted(do.keys())),
            self._cache,
        )
        # Normalise scalar / 0-D evidence to 1-D (mirror LW).
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

        # Fixed values [B, S, D] for evidence + do nodes (mirror LW).
        fixed: List[torch.Tensor | None] = [None] * len(state.topo_order)
        for node, val in {**do, **evidence}.items():
            idx = state.node_to_idx[node]
            v = ensure_2d(val.to(device=device, dtype=dtype))   # [B, D]
            fixed[idx] = v.unsqueeze(1).expand(b, s, -1)         # [B, S, D]

        # Evidence context: ONE trunk forward pass per query (evidence is fixed
        # across particles); the per-node heads are called in the loop below
        # with the actual sampled parents.  [B, 2n] → [B, d_ctx] → [B*S, d_ctx].
        ev_values = torch.zeros(b, len(state.topo_order), device=device, dtype=dtype)
        ev_mask = torch.zeros(b, len(state.topo_order), device=device, dtype=dtype)
        for node, val in evidence.items():
            i = net.node_index[node]
            v = ensure_2d(val.to(device=device, dtype=dtype))   # [B, D]
            ev_values[:, i] = v[:, 0]
            ev_mask[:, i] = 1.0
        ctx = net.context(ev_values, ev_mask)                              # [B, d_ctx]
        ctx_bs = ctx.unsqueeze(1).expand(b, s, -1).reshape(b * s, -1)      # [B*S, d_ctx]

        for i, node in enumerate(state.topo_order):
            sl = state.node_slices[i]
            mech = model.mechanisms[node]
            psl = state.parent_slices[i]
            pa = torch.cat([buf[..., ps] for ps in psl], dim=-1) if psl else None

            if fixed[i] is not None:
                # Evidence / do node — clamp; score evidence (not do), as LW.
                buf[..., sl] = fixed[i]
                if state.evidence_mask[i]:
                    obs = fixed[i].reshape(b * s, -1)
                    pa_flat = pa.reshape(b * s, -1) if pa is not None else None
                    lp = mech.log_prob(obs, pa_flat)
                    log_w = log_w + lp.reshape(b, s)
            else:
                # Latent node — sample from q(xᵢ | ctx, SAMPLED parents) and
                # accumulate log p_model(x|pa) - log q(x|ctx,pa).  Gather the
                # proposal's parents by net position (== state position here);
                # scalar parents → one column each (v1 scope).
                ni = net.node_index[node]
                ppos = net.parents_idx[ni]
                if ppos:
                    pv = torch.cat([buf[..., state.node_slices[p]] for p in ppos], dim=-1)
                    parent_values = pv.reshape(b * s, -1)                   # [B*S, n_parents]
                else:
                    parent_values = None
                node_params = net.node_params(ni, ctx_bs, parent_values)   # [B*S, P]
                dist_q = net.make_dist(node, node_params)                  # batch [B*S]
                samp = dist_q.sample()                                     # [B*S] or [B*S, D]
                log_q = dist_q.log_prob(samp)                              # [B*S]
                if samp.dim() == 1:
                    samp = samp.unsqueeze(-1)
                samp = samp.to(dtype)                                      # [B*S, D]
                buf[..., sl] = samp.reshape(b, s, -1)

                pa_flat = pa.reshape(b * s, -1) if pa is not None else None
                log_p = mech.log_prob(samp, pa_flat)                       # [B*S]
                log_w = log_w + (log_p - log_q).reshape(b, s)

        return log_w, buf
