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

Training maximizes a mean-field ELBO amortized over random-mask evidence
patterns.  Continuous nodes use reparameterized gradients; discrete nodes
use a Rao-Blackwellized self term ``Σ_k q(k)·log p(k | pa)`` plus the
entropy, so gradients reach the recognition net without pushing a relaxed
sample through nbn's integer-indexed mechanisms.

References:
- Ranganath, Gerrish & Blei (2014) Black Box Variational Inference
- Cremer, Li & Duvenaud (2018) Inference Suboptimality in VAEs

See docs/v0.14-batched-inference-engines-research.md (Engine B section).
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import torch

from nbn.inference.base import InferenceEngine
from nbn.inference.recognition_net import RecognitionNetwork
from nbn.sampling.ancestral import ancestral_sample

logger = logging.getLogger(__name__)


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
        n_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 4096,
        mask_prob: float = 0.5,
        time_budget_s: float = 30.0,
        device: str | None = None,
    ) -> dict:
        """Train the variational recognition network by ELBO maximization."""
        dev = torch.device(device or model.device)
        net = RecognitionNetwork(model).to(dev)
        self.recognition_net = net
        self.device = dev
        self._parent_cols = {
            node: [net.node_index[p] for p in model.dag.parents(node)]
            for node in net.node_order
        }

        with torch.no_grad():
            samples = ancestral_sample(model, n=n_training_samples, device=str(dev))
            x = self._stack_values(samples, net.node_order, dev).detach()  # [N, n]
        n = x.shape[0]

        opt = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        start = time.monotonic()
        last_loss = float("nan")
        for _epoch in range(n_epochs):
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
            if time.monotonic() - start > time_budget_s:
                logger.info(
                    "AmortizedVIEngine: ELBO training hit time budget "
                    "(%.0fs) after epoch %d.", time_budget_s, _epoch,
                )
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
        return {"final_loss": last_loss, "elbo_gap": gap}

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
        lg/flow heads); falls back to a Gumbel-softmax-over-components +
        reparameterized-Normal pathwise sample for MixtureSameFamily
        (mdn head), which has no ``rsample``.
        """
        if dist.has_rsample:
            return dist.rsample()
        # MixtureSameFamily mdn head.
        mix = dist.mixture_distribution            # Categorical, logits [..., K]
        comp = dist.component_distribution         # Independent(Normal, 1)
        normal = comp.base_dist                    # Normal, loc/scale [..., K, d_x]
        g = torch.rand_like(mix.logits).clamp_min(1e-12)
        g = -torch.log(-torch.log(g))
        soft = torch.softmax((mix.logits + g) / self.gumbel_tau, dim=-1)  # [..., K]
        eps = torch.randn_like(normal.loc)
        comp_samples = normal.loc + normal.scale * eps                    # [..., K, d_x]
        return (soft.unsqueeze(-1) * comp_samples).sum(-2)                # [..., d_x]

    def _elbo_loss(self, model, net, xb, mask) -> torch.Tensor:
        """Mean negative ELBO over a training minibatch (differentiable)."""
        params = net(xb * mask, mask)  # [Bb, total_param]

        # Parent context: hard, detached samples (mean-field — gradient
        # through the parent path is dropped; each node's q is fit to its
        # local conditional given sampled parents + the entropy term).
        with torch.no_grad():
            assign = xb.clone()
            for node in net.node_order:
                j = net.node_index[node]
                d = net.make_dist(node, net.node_param_slice(params, node))
                s = d.sample()
                if s.dim() > 1:
                    s = s[..., 0]
                col = mask[:, j]
                assign[:, j] = col * xb[:, j] + (1.0 - col) * s.float()
        assign = assign.detach()

        total = xb.new_zeros(())
        count = xb.new_zeros(())
        for node in net.node_order:
            j = net.node_index[node]
            weight = 1.0 - mask[:, j]            # rows where node is latent
            if float(weight.sum()) == 0.0:
                continue
            cols = self._parent_cols[node]
            pa = assign[:, cols] if cols else None
            mech = model.mechanisms[node]
            p = net.node_param_slice(params, node)

            if net.is_discrete(node):
                q_logprob = torch.log_softmax(p, dim=-1)      # [Bb, K]
                q_probs = q_logprob.exp()
                m_probs = mech.forward(pa).probs              # [Bb or 1, K]
                if m_probs.shape[0] == 1 and q_probs.shape[0] > 1:
                    m_probs = m_probs.expand(q_probs.shape[0], -1)
                m_logp = m_probs.clamp_min(1e-12).log()
                e_logp = (q_probs * m_logp).sum(-1)           # [Bb]
                e_logq = (q_probs * q_logprob).sum(-1)        # [Bb]
                elbo_i = e_logp - e_logq
            else:
                qd = net.make_dist(node, p)
                x_i = self._rsample_continuous(qd)            # [Bb, d_x]
                log_q = qd.log_prob(x_i)                      # [Bb]
                log_p = mech.log_prob(x_i, pa)                # [Bb]
                elbo_i = log_p - log_q

            total = total - (elbo_i * weight).sum()
            count = count + weight.sum()
        return total / count.clamp_min(1.0)

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
            logger.debug("ELBO-gap diagnostic skipped: %s", exc)
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
