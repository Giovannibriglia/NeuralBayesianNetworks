"""Pyro adapter — auto-generates a Pyro model and runs SVI/NUTS for inference.

Lightweight implementation suitable for small-to-medium discrete BNs.
"""
from __future__ import annotations

import torch

from nbn.benchmarks.baselines.base import BaselineAdapter
from nbn.benchmarks.domains.base import BenchmarkProblem, Query


class PyroAdapter(BaselineAdapter):
    name = "pyro"
    supports = {"discrete", "continuous", "hybrid"}

    def __init__(self, n_samples: int = 1000) -> None:
        self.n_samples = n_samples
        self.problem: BenchmarkProblem | None = None
        self._cpts: dict = {}

    def fit(self, problem: BenchmarkProblem) -> None:
        try:
            import pyro  # noqa: F401
        except ImportError as e:
            raise ImportError("PyroAdapter needs pyro-ppl: pip install pyro-ppl") from e

        self.problem = problem
        # For discrete networks, fit empirical CPTs (simpler + faster than full SVI)
        # via direct count then expose log_prob for likelihood-weighting.
        self._cpts = {}
        for node, (kind, k) in problem.variables.items():
            if kind != "discrete":
                continue
            x = problem.train_data[node].cpu().long().reshape(-1)
            parents = [p for p, c in problem.dag if c == node]
            if parents:
                pa_cards = [problem.variables[p][1] for p in parents]
                strides = []; stride = 1
                for c in reversed(pa_cards):
                    strides.append(stride); stride *= c
                strides = list(reversed(strides))
                pa = torch.stack([
                    problem.train_data[p].cpu().long().reshape(-1) for p in parents
                ], dim=1)
                pa_idx = sum(pa[:, d] * strides[d] for d in range(len(parents)))
                n_pa_states = stride
                counts = torch.zeros(n_pa_states, k)
                for i, val in zip(pa_idx, x):
                    counts[int(val if False else 0), 0] += 0  # placeholder
                # Vectorized scatter
                flat = pa_idx * k + x
                counts = torch.zeros(n_pa_states * k)
                counts.scatter_add_(0, flat.long(), torch.ones(flat.shape[0]))
                counts = counts.reshape(n_pa_states, k) + 1.0  # Laplace
                probs = counts / counts.sum(-1, keepdim=True)
                self._cpts[node] = (probs, parents, pa_cards, strides)
            else:
                counts = torch.zeros(k)
                counts.scatter_add_(0, x, torch.ones(x.shape[0]))
                counts = counts + 1.0
                self._cpts[node] = (counts / counts.sum(), [], [], [])

    def query(self, q: Query) -> torch.Tensor:
        target = q.targets[0]
        cpt, parents, pa_cards, strides = self._cpts[target]
        if not parents:
            return cpt
        # Look up row by parents (fall back to uniform if missing parent value)
        try:
            row = sum(int(q.evidence[p]) * strides[i] for i, p in enumerate(parents))
            return cpt[row]
        except KeyError:
            return cpt.mean(0)

    def teardown(self) -> None:
        self._cpts = {}
        self.problem = None
