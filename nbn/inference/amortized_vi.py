"""Amortized variational inference (Engine B, #182).

Black-box variational inference with an evidence-conditioned recognition
network.  Different accuracy class from Engine A: a bounded ELBO
approximation rather than asymptotic correctness, and single-forward-pass
query-time inference (fastest at large B).

Subclasses ``InferenceEngine`` directly (not ``LikelihoodWeightingEngine``)
— there is no importance-sampling loop.  The recognition net's forward
pass directly produces the variational posterior parameters; the engine
reads out the relevant marginal:

* discrete target → Categorical probabilities;
* continuous target → particles sampled from the variational ``q`` with
  uniform weights (the adapter resamples them, as for LW).

Training maximizes the mean-field ELBO of the FULL joint, amortized over
random-mask evidence patterns::

    ELBO(e) = Σ_j E_q[log p(x_j | pa_j)]  −  Σ_{i latent} E_q[log q(x_i | e)]

The first sum runs over EVERY node — observed ones included. That is what
carries evidence *downstream* of a latent node back into its posterior
(an observed child's ``log p(e_child | x_i, …)`` is the likelihood term
that distinguishes p(x_i | e) from the prior p(x_i | pa_i)). An earlier
version summed only the latent nodes' own terms with detached parents, so
q learned the prior conditional and diagnostic queries (evidence on
descendants, target upstream) returned the prior.

Gradient estimators: continuous latents are reparameterized (gradients
flow through their value into their own term AND every child term);
discrete latents use the Rao-Blackwellized own term ``Σ_k q(k)·log p(k |
pa)`` plus, for each child, a *local-expectation* surrogate
``Σ_k q_i(k)·log p(x_child | pa_{−i}, x_i = k)`` (Titsias &
Lázaro-Gredilla 2015) — nbn's integer-indexed mechanisms never see a
relaxed sample.

References:
- Ranganath, Gerrish & Blei (2014) Black Box Variational Inference
- Titsias & Lázaro-Gredilla (2015) Local Expectation Gradients
- Cremer, Li & Duvenaud (2018) Inference Suboptimality in VAEs

See docs/v0.14-batched-inference-engines-research.md (Engine B section).
"""
from __future__ import annotations

import copy
import logging
import time
from typing import Dict, List, Tuple

import torch

from torch.distributions import Categorical

from nbn.inference.base import InferenceEngine
from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
from nbn.inference.recognition_net import RecognitionNetwork
from nbn.sampling.ancestral import ancestral_sample

logger = logging.getLogger(__name__)

# Training budget: gradient steps scale with node count (as for the AIS
# proposal, amortized_is._STEPS_PER_NODE), floored, with a wall-clock safety
# cap. The previous fixed 20 epochs × (20000/4096) = 100 steps under a 30 s
# cap reached 25 steps at n=50 (flow, GPU) — badly under-trained.
_STEPS_PER_NODE = 20
_MIN_TRAIN_STEPS = 300

# Fit-time quality gate (#249).  The trained ``q`` is judged on what the
# benchmark actually reads out — the TARGET MARGINAL — against a
# likelihood-weighting reference on held-out evidence patterns (random masks,
# random latent target, so downstream evidence is represented).  Per row,
# ``d_q`` = distance(q(x_t | e), LW estimate) and ``d_prior`` = distance(naive
# clamped-prior marginal — LW's particles with uniform weights, i.e. what an
# evidence-ignoring posterior returns —, LW estimate); TV for discrete
# targets, a sd-normalised quantile (W1) distance for continuous ones.
# ``gain = mean d_prior − mean d_q``: positive means q is closer to the LW
# answer than doing nothing.  Below the threshold q is unset and the engine
# answers every query with likelihood weighting — the analogue of the AIS ESS
# gate.  A joint-KL / joint-ESS criterion is deliberately NOT used: on
# upstream-only evidence the ancestral prior IS the exact joint posterior, so
# a mean-field q can never beat it in joint KL even when its marginals are
# fine.  Needs no enumeration oracle, so unlike ``_estimate_elbo_gap`` it
# runs on every network (continuous, n > 12) — every benchmark-scale cell.
_GAIN_FALLBACK_THRESHOLD = 0.0

# Training early stop on the held-out ELBO (#249): checked on the same FIXED
# evidence patterns every ``check_every`` steps / ``check_every_s`` seconds;
# stop after ``patience`` checks without a ``min_delta`` (nats per latent)
# improvement; the best checkpoint is restored.
_ELBO_PATIENCE = 5
_ELBO_MIN_DELTA = 0.01
_EVAL_PATTERNS = 4
_EVAL_ROWS_PER_PATTERN = 8
_EVAL_MC = 128
# LW reference particles for the gate: at least this many, or the engine's
# own n_samples (the benchmark's 2048) — at n=50 with half the nodes observed
# LW's ESS is ~2.5 %, so 256 particles leave ~6 effective ones and the
# reference is noise for q and naive prior alike.
_EVAL_PARTICLES_MIN = 256


def _weighted_quantiles(x: torch.Tensor, w: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Per-row weighted quantiles: ``x``, ``w`` ``[B, S]`` (w normalised), ``u`` ``[Q]`` → ``[B, Q]``."""
    xs, order = torch.sort(x, dim=-1)
    ws = torch.gather(w, -1, order)
    cdf = ws.cumsum(dim=-1)
    idx = torch.searchsorted(cdf, u.unsqueeze(0).expand(x.shape[0], -1).contiguous())
    idx = idx.clamp_max(x.shape[-1] - 1)
    return torch.gather(xs, -1, idx)


class AmortizedVIEngine(InferenceEngine):
    """Amortized variational-inference engine (#182)."""

    def __init__(self, n_samples: int = 1024, gumbel_tau: float = 0.5) -> None:
        self.n_samples = int(n_samples)
        self.gumbel_tau = float(gumbel_tau)
        self.recognition_net: RecognitionNetwork | None = None
        self.device: torch.device | None = None
        self._parent_cols: Dict[str, List[int]] = {}
        # Fit-time gate outcome (#249): "learned" | "lw_fallback" | None (untrained).
        self.proposal_used: str | None = None
        # LW engine that answers queries after a fallback (recognition_net unset).
        self._fallback: LikelihoodWeightingEngine | None = None

    # ------------------------------------------------------------------
    # Training (called once, from adapter.fit())
    # ------------------------------------------------------------------

    def train_proposal(
        self,
        model,
        n_training_samples: int = 20000,
        n_epochs: int | None = None,
        lr: float = 1e-3,
        batch_size: int = 1024,
        mask_prob: float = 0.5,
        time_budget_s: float = 120.0,
        steps_per_node: int = _STEPS_PER_NODE,
        min_steps: int = _MIN_TRAIN_STEPS,
        device: str | None = None,
        early_stop: bool = True,
        check_every: int | None = None,
        check_every_s: float | None = None,
        patience: int = _ELBO_PATIENCE,
        min_delta: float = _ELBO_MIN_DELTA,
        quality_gate: bool = True,
    ) -> dict:
        """Train the variational recognition network by ELBO maximization.

        Budget is step-based — ``max(min_steps, steps_per_node · n_nodes)``
        gradient steps — with ``time_budget_s`` as a wall-clock safety cap.
        ``n_epochs`` (explicit passes over the prior-sample set) overrides the
        step target when given; it is kept for callers that want a tiny,
        deterministic budget (tests).

        With ``early_stop`` (#249) the held-out ELBO (nats per latent) is
        measured on a fixed evidence set every ``check_every`` steps (default
        ``max(25, target_steps // 20)``) or ``check_every_s`` seconds (default
        ``time_budget_s / 12``); training stops after ``patience`` checks
        without a ``min_delta`` improvement and the best checkpoint is
        restored.  With ``quality_gate`` the trained ``q`` is then compared to
        the prior on the same held-out set (``elbo_gain``, see module
        constants); a ``q`` that is no better than the prior is unset and the
        engine answers queries with likelihood weighting (``proposal_used =
        "lw_fallback"``).
        """
        dev = torch.device(device or model.device)
        net = RecognitionNetwork(model).to(dev)
        self.recognition_net = net
        self.device = dev
        self._fallback = None
        self.proposal_used = None
        self._parent_cols = {
            node: [net.node_index[p] for p in model.dag.parents(node)]
            for node in net.node_order
        }
        self._parent_names = {
            node: list(model.dag.parents(node)) for node in net.node_order
        }

        with torch.no_grad():
            samples = ancestral_sample(model, n=n_training_samples, device=str(dev))
            x = self._stack_values(samples, net.node_order, dev).detach()  # [N, n]
        n = x.shape[0]
        batches_per_epoch = max(1, -(-n // batch_size))
        if n_epochs is not None:
            target_steps = int(n_epochs) * batches_per_epoch
        else:
            target_steps = max(int(min_steps),
                               int(steps_per_node) * len(net.node_order))
        if check_every is None:
            check_every = max(25, target_steps // 20)
        if check_every_s is None:
            check_every_s = float(time_budget_s) / 12.0

        # Fixed held-out evidence set shared by the early-stop checkpoints and
        # the quality gate (same rows every time → comparable values).
        eval_set = (self._make_eval_set(model)
                    if (early_stop or quality_gate) else None)
        if eval_set is None:
            early_stop = False
            quality_gate = False

        opt = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        start = time.monotonic()
        last_check = start
        last_loss = float("nan")
        steps = 0
        stop = False
        early_stopped = False
        best_elbo: float | None = None
        best_state = None
        best_step = 0
        bad_checks = 0
        elbo_history: List[Tuple[int, float]] = []
        while steps < target_steps and not stop:
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = x[idx]
                mask = (torch.rand_like(xb) < mask_prob).float()  # 1 = observed
                loss = self._elbo_loss(model, net, xb, mask)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "AmortizedVIEngine.train_proposal: non-finite ELBO "
                        "loss (NaN/Inf) — variational training diverged."
                    )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                last_loss = float(loss.detach())
                steps += 1
                if steps >= target_steps:
                    stop = True
                    break
                now = time.monotonic()
                if now - start > time_budget_s:
                    logger.info(
                        "AmortizedVIEngine: ELBO training hit time budget "
                        "(%.0fs) after %d/%d steps.",
                        time_budget_s, steps, target_steps,
                    )
                    stop = True
                    break
                if early_stop and (
                        steps % check_every == 0
                        or now - last_check > check_every_s):
                    last_check = now
                    net.eval()
                    elbo_now = self._heldout_elbo_per_latent(model, eval_set)
                    net.train()
                    if elbo_now is None:
                        early_stop = False
                        continue
                    elbo_history.append((steps, float(elbo_now)))
                    if best_elbo is None or elbo_now > best_elbo + min_delta:
                        best_elbo, best_step, bad_checks = float(elbo_now), steps, 0
                        # state_dict() aliases the live params — snapshot by copy.
                        best_state = copy.deepcopy(net.state_dict())
                    else:
                        bad_checks += 1
                        if bad_checks >= patience:
                            early_stopped = True
                            stop = True
                            logger.info(
                                "AmortizedVIEngine: held-out ELBO plateaued at "
                                "%.4f nats/latent (best at step %d); stopping "
                                "after %d/%d steps.",
                                best_elbo, best_step, steps, target_steps,
                            )
                            break
        net.eval()
        train_time_s = time.monotonic() - start

        if best_state is not None:
            final = self._heldout_elbo_per_latent(model, eval_set)
            if final is None or best_elbo > final:
                net.load_state_dict(best_state)

        gap = self._estimate_elbo_gap(model)
        if gap is not None and gap > 5.0:
            logger.warning(
                "AmortizedVIEngine variational gap looks large "
                "(held-out log p − ELBO ≈ %.2f nats). The posterior "
                "approximation may be loose (KL gap); inference still "
                "runs. Consider more training samples/epochs.", gap,
            )

        quality = self._estimate_quality(model, eval_set) if quality_gate else None
        proposal_used = "learned"
        if quality is not None and quality["gain"] < _GAIN_FALLBACK_THRESHOLD:
            # Diagnostic ("why") and action ("what we did") as distinct records,
            # mirroring the AIS gate.
            logger.warning(
                "AmortizedVIEngine variational posterior is no closer to the LW "
                "reference than the naive clamped prior on held-out evidence "
                "(d_q ≈ %.3f vs d_prior ≈ %.3f; LW ESS ≈ %.1f%%). Consider more "
                "training samples/steps.", quality["d_q"], quality["d_prior"],
                100.0 * quality["lw_ess"],
            )
            logger.warning(
                "AVI posterior rejected (fit-time gain %.3f < threshold %.2f); "
                "falling back to LW for this engine.",
                quality["gain"], _GAIN_FALLBACK_THRESHOLD,
            )
            self.recognition_net = None
            self._fallback = LikelihoodWeightingEngine(n_samples=self.n_samples)
            proposal_used = "lw_fallback"
        self.proposal_used = proposal_used
        return {
            "final_loss": last_loss, "elbo_gap": gap,
            "grad_steps": steps, "target_steps": target_steps,
            "proposal_used": proposal_used,
            "gain": None if quality is None else quality["gain"],
            "d_q": None if quality is None else quality["d_q"],
            "d_prior": None if quality is None else quality["d_prior"],
            "lw_ess": None if quality is None else quality["lw_ess"],
            "early_stopped": early_stopped, "best_step": best_step,
            "elbo_history": elbo_history, "train_time_s": train_time_s,
        }

    # ------------------------------------------------------------------
    # Held-out evidence set, ELBO-per-latent checkpoints, quality gate (#249)
    # ------------------------------------------------------------------

    def _make_eval_set(
        self, model, n_patterns: int = _EVAL_PATTERNS,
        rows_per_pattern: int = _EVAL_ROWS_PER_PATTERN, mask_prob: float = 0.5,
        seed: int = 0,
    ) -> List[Dict[str, object]] | None:
        """Fixed held-out evidence patterns from fresh prior samples.

        Each pattern observes a random subset of nodes (``mask_prob``, at
        least one observed and one latent — so downstream evidence occurs,
        unlike a "first half of topo order" split) and names one random
        latent target; ``rows_per_pattern`` rows share the pattern so LW can
        batch them.  Returns ``[{"evidence": {node: [R, 1]}, "target": node,
        "n_latent": int}, ...]`` or ``None`` on any failure — never breaks fit.
        """
        try:
            net = self.recognition_net
            dev = self.device or model.device
            nodes = list(net.node_order)
            n_nodes = len(nodes)
            g = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                test = ancestral_sample(
                    model, n=n_patterns * rows_per_pattern, device=str(dev))
            patterns = []
            for k in range(n_patterns):
                obs = (torch.rand(n_nodes, generator=g) < mask_prob)
                if bool(obs.all()):
                    obs[int(torch.randint(n_nodes, (1,), generator=g))] = False
                if not bool(obs.any()):
                    obs[int(torch.randint(n_nodes, (1,), generator=g))] = True
                latent = [nd for nd, o in zip(nodes, obs.tolist()) if not o]
                tgt = latent[int(torch.randint(len(latent), (1,), generator=g))]
                rows = slice(k * rows_per_pattern, (k + 1) * rows_per_pattern)
                ev = {}
                for nd, o in zip(nodes, obs.tolist()):
                    if not o:
                        continue
                    v = test[nd]
                    if v.dim() >= 2:
                        v = v[..., 0]
                    ev[nd] = v.reshape(-1)[rows].reshape(-1, 1).to(
                        device=dev, dtype=torch.float32)
                patterns.append({"evidence": ev, "target": tgt, "n_latent": len(latent)})
            return patterns
        except Exception as exc:  # pragma: no cover - diagnostic safety
            logger.warning(
                "AmortizedVIEngine: could not build the held-out evidence set "
                "(%r); early stop and quality gate disabled.", exc,
            )
            return None

    def _log_weights(self, model, evidence, n_mc: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """``log p(z, e) − log q(z | e)`` for ``n_mc`` joint draws ``z ~ q(·|e)``.

        Returns ``(log_w [S, B], n_latent [B])``.  ``mean_S`` of ``log_w`` is
        the ELBO; the softmax over S gives self-normalised IS weights.
        """
        net = self.recognition_net
        dev = self.device or model.device
        with torch.no_grad():
            B, ev_values, mask = self._build_evidence(net, evidence, dev)
            params = net(ev_values, mask)                     # [B, total]
            S = n_mc
            n_nodes = len(net.node_order)
            assign = ev_values.unsqueeze(0).expand(S, B, n_nodes).clone()
            log_q = torch.zeros(S, B, device=dev)
            for node in net.node_order:
                j = net.node_index[node]
                pe = net.node_param_slice(params, node).unsqueeze(0).expand(S, B, -1)
                d = net.make_dist(node, pe)                   # batch [S, B]
                s = d.sample()
                val = s[..., 0] if (not net.is_discrete(node) and s.dim() > 2) else s
                m = mask[:, j].unsqueeze(0)                   # [1, B]
                latent = 1.0 - m
                assign[..., j] = m * ev_values[:, j].unsqueeze(0) + latent * val.float()
                log_q = log_q + d.log_prob(s) * latent
            log_p = torch.zeros(S, B, device=dev)
            for node in net.node_order:
                j = net.node_index[node]
                cols = self._parent_cols[node]
                mech = model.mechanisms[node]
                x = assign[..., j].reshape(S * B, 1)
                pa = assign[..., cols].reshape(S * B, -1) if cols else None
                log_p = log_p + mech.log_prob(x, pa).reshape(S, B)
            n_latent = (1.0 - mask).sum(dim=1)                # [B]
            return log_p - log_q, n_latent

    def _heldout_elbo_per_latent(self, model, eval_set, n_mc: int = _EVAL_MC) -> float | None:
        """Mean held-out ELBO over the eval patterns, per latent node (nats)."""
        try:
            vals = []
            for pat in eval_set:
                log_w, n_latent = self._log_weights(model, pat["evidence"], n_mc)
                vals.append(float(log_w.mean() / n_latent.mean().clamp_min(1.0)))
            return sum(vals) / len(vals)
        except Exception as exc:  # pragma: no cover - diagnostic safety
            logger.warning("AmortizedVIEngine: held-out ELBO check skipped (%r).", exc)
            return None

    def _target_marginal(self, model, net, tgt, evidence, n_draws: int):
        """q(x_t | e) for one pattern: ``probs [B, K]`` (discrete) or samples ``[B, S]``."""
        dev = self.device or model.device
        with torch.no_grad():
            B, ev_values, mask = self._build_evidence(net, evidence, dev)
            params = net(ev_values, mask)
            dist = net.make_dist(tgt, net.node_param_slice(params, tgt))
            if net.is_discrete(tgt):
                return dist.probs                              # [B, K]
            s = dist.sample((n_draws,))                        # [S, B, d_x]
            return s[..., 0].permute(1, 0)                     # [B, S]

    def _estimate_quality(
        self, model, eval_set, n_particles: int | None = None,
    ) -> Dict[str, float] | None:
        """Target-marginal agreement with likelihood weighting on the held-out patterns.

        Per pattern: an LW run (``n_particles``) gives weighted particles of
        the target; ``d_q`` = distance(q's marginal, LW weighted) and
        ``d_prior`` = distance(LW particles with UNIFORM weights — the naive
        clamped-prior marginal —, LW weighted).  TV for discrete targets, a
        sd-normalised mean absolute quantile difference (W1) for continuous.
        ``gain = d_prior − d_q`` averaged over rows and patterns; ``lw_ess`` is
        the reference's own mean ESS fraction.  ``None`` on any failure.
        """
        try:
            net = self.recognition_net
            from nbn.inference.state import get_inference_state
            if n_particles is None:
                n_particles = max(_EVAL_PARTICLES_MIN, int(self.n_samples))
            probe = LikelihoodWeightingEngine(n_samples=n_particles)
            d_q_all, d_p_all, ess_all = [], [], []
            u = torch.linspace(0.02, 0.98, 49)
            for pat in eval_set:
                tgt, ev = pat["target"], pat["evidence"]
                with torch.no_grad():
                    log_w, buf = probe._run(model, [tgt], ev, {}, n_particles)   # [B, S]
                    w = torch.softmax(log_w, dim=-1)
                    ess_all.append(float((1.0 / (w.pow(2).sum(-1) * n_particles)).mean()))
                    state = get_inference_state(
                        model, [tgt], tuple(sorted(ev.keys())), (), probe._cache)
                    x = buf[..., state.node_slices[state.node_to_idx[tgt]]][..., 0]  # [B, S]
                    q_marg = self._target_marginal(model, net, tgt, ev, n_particles)
                    B = x.shape[0]
                    if net.is_discrete(tgt):
                        K = q_marg.shape[-1]
                        cls = x.long().clamp(0, K - 1)
                        lw = torch.zeros(B, K, device=x.device).scatter_add_(1, cls, w)
                        naive = torch.zeros(B, K, device=x.device).scatter_add_(
                            1, cls, torch.full_like(w, 1.0 / n_particles))
                        d_q = 0.5 * (q_marg.to(lw.device) - lw).abs().sum(-1)
                        d_p = 0.5 * (naive - lw).abs().sum(-1)
                    else:
                        uu = u.to(x.device)
                        q_lw = _weighted_quantiles(x, w, uu)
                        q_naive = _weighted_quantiles(x, torch.full_like(w, 1.0 / n_particles), uu)
                        q_q = _weighted_quantiles(
                            q_marg.to(x.device),
                            torch.full_like(q_marg, 1.0 / q_marg.shape[-1]).to(x.device), uu)
                        mean = (w * x).sum(-1, keepdim=True)
                        sd = (w * (x - mean).pow(2)).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
                        d_q = ((q_q - q_lw).abs() / sd).mean(-1)
                        d_p = ((q_naive - q_lw).abs() / sd).mean(-1)
                    d_q_all.append(float(d_q.mean()))
                    d_p_all.append(float(d_p.mean()))
            d_q_m = sum(d_q_all) / len(d_q_all)
            d_p_m = sum(d_p_all) / len(d_p_all)
            return {"d_q": d_q_m, "d_prior": d_p_m, "gain": d_p_m - d_q_m,
                    "lw_ess": sum(ess_all) / len(ess_all)}
        except Exception as exc:  # pragma: no cover - diagnostic safety
            logger.warning(
                "AmortizedVIEngine: fit-time quality gate SKIPPED (diagnostic "
                "raised %r); the variational posterior is used untested.", exc,
            )
            return None

    @staticmethod
    def _stack_values(samples, node_order, dev) -> torch.Tensor:
        cols = []
        for node in node_order:
            v = samples[node]
            if v.dim() >= 2:
                v = v[..., 0]
            cols.append(v.reshape(-1).to(device=dev, dtype=torch.float32))
        return torch.stack(cols, dim=1)

    # ------------------------------------------------------------------
    # ELBO (differentiable loss)
    # ------------------------------------------------------------------

    def _rsample_continuous(self, dist) -> torch.Tensor:
        """Reparameterized sample for a continuous variational head.

        Uses ``rsample`` when available (Normal/Independent-Normal for
        lg/flow heads). For MixtureSameFamily (mdn head), which has no
        ``rsample``, a *straight-through* Gumbel-max sample: the value is a
        genuine draw from ``q`` (Gumbel-max picks the component exactly, the
        Normal within it is reparameterized) while the gradient flows to the
        mixture logits through the softmax relaxation.

        The value being a real draw matters: a plain Gumbel-*softmax* convex
        mix of component draws is NOT distributed as ``q``, and evaluating
        ``log q`` at it let the optimizer game the entropy term — push
        components far apart with tiny scales so the blended point sits
        where ``log q`` is hugely negative (observed: ELBO → +8e7, a
        posterior mean of 3.4 on a problem whose truth is 0.25).
        """
        if dist.has_rsample:
            return dist.rsample()
        # MixtureSameFamily mdn head.
        mix = dist.mixture_distribution            # Categorical, logits [..., K]
        comp = dist.component_distribution         # Independent(Normal, 1)
        normal = comp.base_dist                    # Normal, loc/scale [..., K, d_x]
        g = torch.rand_like(mix.logits).clamp_min(1e-12)
        g = -torch.log(-torch.log(g))
        perturbed = mix.logits + g
        soft = torch.softmax(perturbed / self.gumbel_tau, dim=-1)         # [..., K]
        hard = torch.nn.functional.one_hot(
            perturbed.argmax(dim=-1), soft.shape[-1]).to(soft.dtype)
        w = hard + soft - soft.detach()                                   # ST estimator
        eps = torch.randn_like(normal.loc)
        comp_samples = normal.loc + normal.scale * eps                    # [..., K, d_x]
        return (w.unsqueeze(-1) * comp_samples).sum(-2)                   # [..., d_x]

    @staticmethod
    def _log_table(mech, pa, rows: int) -> torch.Tensor:
        """``log p(k | pa)`` for every class k of a discrete mechanism, ``[rows, K]``."""
        t = mech.forward(pa).probs.clamp_min(1e-12).log()
        if t.shape[0] == 1 and rows > 1:
            t = t.expand(rows, -1)
        return t

    def _elbo_loss(self, model, net, xb, mask) -> torch.Tensor:
        """Mean negative full-joint ELBO over a training minibatch (differentiable).

        ``xb`` ``[Bb, n]`` are prior samples, ``mask`` ``[Bb, n]`` marks the
        entries treated as observed (1) vs latent (0). One joint draw
        ``z ~ q(· | e)`` is taken per row; latent entries of the assignment
        are the draw, observed entries the data. Then

        * entropy: ``−log q(z_i)`` for continuous latents (reparameterized),
          the analytic Categorical entropy for discrete latents;
        * likelihood: ``log p(x_j | pa_j)`` for EVERY node ``j`` (observed
          or latent), with a latent discrete node's own term
          Rao-Blackwellized over ``q_j``;
        * for each *discrete latent parent* ``i`` of ``j``, a zero-valued
          surrogate whose gradient is the local expectation
          ``Σ_k q_i(k) · log p(x_j | pa_{−i}, x_i = k)`` — the path by which
          a child's likelihood reaches a discrete parent's posterior.

        Normalized by the number of latent entries in the batch.
        """
        params = net(xb * mask, mask)                     # [Bb, total_param]
        Bb = xb.shape[0]
        val: Dict[str, torch.Tensor] = {}                 # assignment column per node [Bb]
        q_logits: Dict[str, torch.Tensor] = {}            # discrete: log q(k) [Bb, K]
        q_logp_at: Dict[str, torch.Tensor] = {}           # continuous: log q(z) [Bb]

        # --- one joint draw from q ---------------------------------------
        for node in net.node_order:
            j = net.node_index[node]
            obs = mask[:, j]
            p = net.node_param_slice(params, node)
            if net.is_discrete(node):
                lq = torch.log_softmax(p, dim=-1)
                q_logits[node] = lq
                s = Categorical(logits=lq.detach()).sample().to(xb.dtype)
            else:
                qd = net.make_dist(node, p)
                z = self._rsample_continuous(qd)          # [Bb, d_x], reparameterized
                q_logp_at[node] = qd.log_prob(z)          # [Bb]
                s = z[..., 0]
            val[node] = obs * xb[:, j] + (1.0 - obs) * s

        elbo = xb.new_zeros(())

        # --- entropy of q over the latent entries ---------------------------
        for node in net.node_order:
            lat = 1.0 - mask[:, net.node_index[node]]
            if net.is_discrete(node):
                lq = q_logits[node]
                elbo = elbo - (lat * (lq.exp() * lq).sum(-1)).sum()
            else:
                elbo = elbo - (lat * q_logp_at[node]).sum()

        # --- log p(x_j | pa_j) for every node + local-expectation surrogates
        for node in net.node_order:
            j = net.node_index[node]
            lat = 1.0 - mask[:, j]
            parents = self._parent_names[node]
            pa = torch.stack([val[p] for p in parents], dim=1) if parents else None
            x_j = val[node]
            mech = model.mechanisms[node]
            discrete = net.is_discrete(node)
            if discrete:
                q_j = q_logits[node].exp()
                idx = x_j.long().unsqueeze(1)
                own = self._discrete_term(mech, pa, Bb, lat, q_j, idx)
            else:
                own = mech.log_prob(x_j.unsqueeze(-1), pa)    # [Bb]
            elbo = elbo + own.sum()

            for pi, pnode in enumerate(parents):
                if not net.is_discrete(pnode):
                    continue                              # continuous parent: reparameterized
                lat_i = 1.0 - mask[:, net.node_index[pnode]]
                k_count = q_logits[pnode].shape[-1]
                with torch.no_grad():
                    f_k = []
                    for k in range(k_count):
                        pa_k = pa.clone()
                        pa_k[:, pi] = float(k)
                        if discrete:
                            f_k.append(self._discrete_term(
                                mech, pa_k, Bb, lat, q_j.detach(), idx))
                        else:
                            f_k.append(mech.log_prob(x_j.detach().unsqueeze(-1), pa_k))
                    f_k = torch.stack(f_k, dim=1)          # [Bb, K], constants
                acc = (q_logits[pnode].exp() * f_k).sum(-1)  # d/dθ = local expectation
                elbo = elbo + (lat_i * (acc - acc.detach())).sum()

        n_latent = (1.0 - mask).sum().clamp_min(1.0)
        return -elbo / n_latent

    def _discrete_term(self, mech, pa, rows, lat, q_j, idx) -> torch.Tensor:
        """Own likelihood term of a discrete node: ``Σ_k q_j(k) log p(k | pa)``
        on rows where it is latent, ``log p(x_j | pa)`` where observed."""
        lp_tab = self._log_table(mech, pa, rows)          # [Bb, K]
        own_lat = (q_j * lp_tab).sum(-1)
        own_obs = lp_tab.gather(1, idx).squeeze(1)
        return lat * own_lat + (1.0 - lat) * own_obs

    # ------------------------------------------------------------------
    # ELBO value (no-grad MC estimate) — for the lower-bound test + diag
    # ------------------------------------------------------------------

    def elbo(self, model, evidence: Dict[str, torch.Tensor], n_mc: int = 512) -> float:
        """Monte-Carlo ELBO estimate for a single evidence pattern (nats).

        ``E_q[log p(x_latent, evidence) − log q(x_latent | evidence)]``,
        averaged over ``n_mc`` reparameterization-free samples and over the
        evidence batch.  A valid lower bound on ``log p(evidence)``.
        """
        assert self.recognition_net is not None, "Call train_proposal before elbo()."
        log_w, _ = self._log_weights(model, evidence, n_mc)
        return float(log_w.mean())

    def _estimate_elbo_gap(self, model, n_eval: int = 32) -> float | None:
        """Held-out (log p(evidence) − ELBO) proxy via brute-force marginal.

        Only computed for small discrete nets (where log p(evidence) is
        cheaply enumerable); returns ``None`` otherwise or on any failure —
        a diagnostic must never break ``fit``.
        """
        try:
            net = self.recognition_net
            dev = self.device or model.device
            discrete = all(net.is_discrete(nd) for nd in net.node_order)
            n_nodes = len(net.node_order)
            if not discrete or n_nodes > 12:
                return None
            with torch.no_grad():
                test = ancestral_sample(model, n=n_eval, device=str(dev))
                obs = net.node_order[: max(1, n_nodes // 2)]
                evidence = {
                    nd: test[nd].reshape(-1, 1).to(device=dev, dtype=torch.float32)
                    for nd in obs
                }
                log_pe = self._log_marginal_discrete(model, net, evidence, dev)  # [B]
                elbo = self.elbo(model, evidence, n_mc=256)
                return float((log_pe.mean()).item() - elbo)
        except Exception as exc:  # pragma: no cover - diagnostic safety
            logger.warning("AmortizedVIEngine: ELBO-gap diagnostic skipped (%r).", exc)
            return None

    @staticmethod
    def _log_marginal_discrete(model, net, evidence, dev) -> torch.Tensor:
        """Exact log p(evidence) by enumerating all discrete joint states."""
        nodes = net.node_order
        cards = [int(model.variables[nd].cardinality) for nd in nodes]
        import itertools
        configs = list(itertools.product(*[range(c) for c in cards]))
        full = torch.tensor(configs, dtype=torch.float32, device=dev)  # [C, n]
        C = full.shape[0]
        # joint log p over all configs
        log_joint = torch.zeros(C, device=dev)
        for node in nodes:
            j = net.node_index[node]
            cols = [net.node_index[p] for p in model.dag.parents(node)]
            mech = model.mechanisms[node]
            x = full[:, j].reshape(C, 1)
            pa = full[:, cols] if cols else None
            log_joint = log_joint + mech.log_prob(x, pa)
        # For each evidence row, sum joint over configs matching evidence.
        B = next(iter(evidence.values())).shape[0]
        out = torch.empty(B, device=dev)
        for b in range(B):
            keep = torch.ones(C, dtype=torch.bool, device=dev)
            for k, v in evidence.items():
                j = net.node_index[k]
                keep &= (full[:, j] == float(v[b, 0]))
            out[b] = torch.logsumexp(log_joint[keep], dim=0)
        return out

    # ------------------------------------------------------------------
    # Query-time inference — single forward pass
    # ------------------------------------------------------------------

    def _build_evidence(self, net, evidence, dev):
        norm: Dict[str, torch.Tensor] = {}
        B = 1
        for k, v in (evidence or {}).items():
            if v is None:
                continue
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            t = t.to(device=dev, dtype=torch.float32)
            if t.dim() == 0:
                t = t.reshape(1)
            if t.dim() == 1:
                t = t.reshape(-1, 1)
            norm[k] = t
            B = max(B, t.shape[0])
        n = len(net.node_order)
        ev_values = torch.zeros(B, n, device=dev)
        mask = torch.zeros(B, n, device=dev)
        for k, t in norm.items():
            if k not in net.node_index:
                continue
            i = net.node_index[k]
            col = t[:, 0]
            if col.shape[0] == 1 and B > 1:
                col = col.expand(B)
            ev_values[:, i] = col
            mask[:, i] = 1.0
        return B, ev_values, mask

    def _infer(self, model, targets, evidence):
        net = self.recognition_net
        assert net is not None, "Call train_proposal before querying."
        assert self._fallback is None
        dev = self.device or model.device
        B, ev_values, mask = self._build_evidence(net, evidence, dev)
        with torch.no_grad():
            params = net(ev_values, mask)              # [B, total]
            tgt = targets[0]
            p = net.node_param_slice(params, tgt)
            dist = net.make_dist(tgt, p)
            if net.is_discrete(tgt):
                return dist.probs                      # [B, K]
            S = self.n_samples
            s = dist.sample((S,))                      # [S, B, d_x]
            s = s.permute(1, 0, 2)                      # [B, S, d_x]
            w = torch.softmax(torch.zeros(B, S, device=dev), dim=-1)
            return w, s

    def query(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if self.recognition_net is None and self._fallback is not None:
            # Fit-time gate rejected q (#249): plain likelihood weighting.
            return self._fallback.query(model, targets, evidence, **kwargs)
        res = self._infer(model, targets, evidence)
        if isinstance(res, tuple):
            return res
        # Discrete: match LW/VE single-query contract ([K] for B=1).
        return res.squeeze(0) if (res.dim() > 1 and res.shape[0] == 1) else res

    def query_batch(
        self,
        model,
        targets: List[str],
        evidence: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if self.recognition_net is None and self._fallback is not None:
            return self._fallback.query_batch(model, targets, evidence, **kwargs)
        return self._infer(model, targets, evidence)
