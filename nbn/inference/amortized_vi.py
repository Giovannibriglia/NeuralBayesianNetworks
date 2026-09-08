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

import logging
import time
from typing import Dict, List, Tuple

import torch

from torch.distributions import Categorical

from nbn.inference.base import InferenceEngine
from nbn.inference.recognition_net import RecognitionNetwork
from nbn.sampling.ancestral import ancestral_sample

logger = logging.getLogger(__name__)

# Training budget: gradient steps scale with node count (as for the AIS
# proposal, amortized_is._STEPS_PER_NODE), floored, with a wall-clock safety
# cap. The previous fixed 20 epochs × (20000/4096) = 100 steps under a 30 s
# cap reached 25 steps at n=50 (flow, GPU) — badly under-trained.
_STEPS_PER_NODE = 20
_MIN_TRAIN_STEPS = 300


class AmortizedVIEngine(InferenceEngine):
    """Amortized variational-inference engine (#182)."""

    def __init__(self, n_samples: int = 1024, gumbel_tau: float = 0.5) -> None:
        self.n_samples = int(n_samples)
        self.gumbel_tau = float(gumbel_tau)
        self.recognition_net: RecognitionNetwork | None = None
        self.device: torch.device | None = None
        self._parent_cols: Dict[str, List[int]] = {}

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
    ) -> dict:
        """Train the variational recognition network by ELBO maximization.

        Budget is step-based — ``max(min_steps, steps_per_node · n_nodes)``
        gradient steps — with ``time_budget_s`` as a wall-clock safety cap.
        ``n_epochs`` (explicit passes over the prior-sample set) overrides the
        step target when given; it is kept for callers that want a tiny,
        deterministic budget (tests).
        """
        dev = torch.device(device or model.device)
        net = RecognitionNetwork(model).to(dev)
        self.recognition_net = net
        self.device = dev
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

        opt = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        start = time.monotonic()
        last_loss = float("nan")
        steps = 0
        stop = False
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
                if time.monotonic() - start > time_budget_s:
                    logger.info(
                        "AmortizedVIEngine: ELBO training hit time budget "
                        "(%.0fs) after %d/%d steps.",
                        time_budget_s, steps, target_steps,
                    )
                    stop = True
                    break
        net.eval()

        gap = self._estimate_elbo_gap(model)
        if gap is not None and gap > 5.0:
            logger.warning(
                "AmortizedVIEngine variational gap looks large "
                "(held-out log p − ELBO ≈ %.2f nats). The posterior "
                "approximation may be loose (KL gap); inference still "
                "runs. Consider more training samples/epochs.", gap,
            )
        return {
            "final_loss": last_loss, "elbo_gap": gap,
            "grad_steps": steps, "target_steps": target_steps,
        }

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
        net = self.recognition_net
        assert net is not None, "Call train_proposal before elbo()."
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
            # log p(joint) over all nodes at the assignment
            log_p = torch.zeros(S, B, device=dev)
            for node in net.node_order:
                j = net.node_index[node]
                cols = self._parent_cols[node]
                mech = model.mechanisms[node]
                x = assign[..., j].reshape(S * B, 1)
                pa = assign[..., cols].reshape(S * B, -1) if cols else None
                log_p = log_p + mech.log_prob(x, pa).reshape(S, B)
            elbo = (log_p - log_q).mean()
            return float(elbo)

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
        return self._infer(model, targets, evidence)
