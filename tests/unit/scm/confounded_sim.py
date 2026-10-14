"""A confounded MDP with several logging policies (port of the scm_to_mdp reference simulator).

``W_t`` is a fresh hidden variable (uniform on ``m`` values) drawn at every step; it
enters both the logging policy ``pi_k(a | s, w)`` and the environment
``P(s' | s, a, w)``.  The interventional kernel is ``P(s' | s, do(a)) = mean_w P(s' | s, a, w)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ConfoundedMDP:
    env: np.ndarray        # [S, A, m, S]
    policies: np.ndarray   # [K, S, m, A]
    s0: int = 0

    @property
    def n_states(self) -> int:
        return self.env.shape[0]

    @property
    def n_actions(self) -> int:
        return self.env.shape[1]

    def interventional_kernel(self) -> np.ndarray:
        return self.env.mean(axis=2)

    def observed(self, k: int) -> np.ndarray:
        """Population ``P_k(a, s' | s)`` as ``[S, A, S]``."""
        return np.einsum("swa,sawt->sat", self.policies[k], self.env) / self.env.shape[2]

    def rollout(self, k: int, n_episodes: int, horizon: int, rng: np.random.Generator):
        S, A, m, _ = self.env.shape
        ss, aa, nn = [], [], []
        for _ in range(n_episodes):
            s = self.s0
            for _ in range(horizon):
                w = rng.integers(m)
                a = rng.choice(A, p=self.policies[k, s, w])
                s2 = rng.choice(S, p=self.env[s, a, w])
                ss.append(s)
                aa.append(a)
                nn.append(s2)
                s = s2
        return np.array(ss), np.array(aa), np.array(nn)


def random_confounded_mdp(rng, n_states=3, n_actions=2, n_regimes=2, m=4, strength=3.0, eps=0.1):
    S, A = n_states, n_actions
    logits = rng.normal(size=(S, A, 1, S)) + strength * rng.normal(size=(S, A, m, S))
    env = np.exp(logits - logits.max(-1, keepdims=True))
    env /= env.sum(-1, keepdims=True)
    pol = np.full((n_regimes, S, m, A), eps / A)
    for k in range(n_regimes):
        for s in range(S):
            cuts = np.sort(rng.integers(0, m + 1, size=A - 1))
            perm = rng.permutation(A)
            a_of_w = perm[np.searchsorted(cuts, np.arange(m), side="right")]
            pol[k, s, np.arange(m), a_of_w] += 1 - eps
    return ConfoundedMDP(env, pol)


def pooled(mdp, rng, n_episodes=150, horizon=20):
    parts = [mdp.rollout(k, n_episodes, horizon, rng) + (np.full(n_episodes * horizon, k),)
             for k in range(mdp.policies.shape[0])]
    return [np.concatenate(x) for x in zip(*parts)]


def as_data(s, a, s2, k, device="cpu"):
    return {"s": torch.tensor(s).float().to(device), "a": torch.tensor(a).float().to(device),
            "s'": torch.tensor(s2).float().to(device), "k": torch.tensor(k).to(device)}


def model_for(n_states, n_actions, device="cpu", action_first=False):
    from nbn import NeuralBayesianNetwork

    edges = [("a", "s'"), ("s", "s'")] if action_first else [("s", "s'"), ("a", "s'")]
    return NeuralBayesianNetwork(edges, {"s": ("discrete", n_states), "a": ("discrete", n_actions),
                                         "s'": ("discrete", n_states)}, device=device)


def rows_as_sas(res, n_states, n_actions):
    """``[S, A, S']`` lower / upper from a ConfoundedSet whose parents are (s, a) or (a, s)."""
    lo, hi = res.set.lower.cpu().numpy(), res.set.upper.cpu().numpy()
    if res.parents == ("s", "a"):
        return lo.reshape(n_states, n_actions, -1), hi.reshape(n_states, n_actions, -1)
    return (lo.reshape(n_actions, n_states, -1).transpose(1, 0, 2),
            hi.reshape(n_actions, n_states, -1).transpose(1, 0, 2))


def contains(res, kernel, n_states, n_actions, tol=1e-7):
    lo, hi = rows_as_sas(res, n_states, n_actions)
    return bool(np.all(lo - tol <= kernel) and np.all(kernel <= hi + tol))
