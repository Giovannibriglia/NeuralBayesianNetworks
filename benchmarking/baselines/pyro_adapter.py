"""Pyro adapter — real importance-sampling inference.

For discrete BNs we build a Pyro model that ancestrally samples each node
from a Categorical parameterised by the empirical CPT of the training data,
then run inference via ``pyro.infer.Importance`` (the canonical Pyro path
for unconditional models) and read the empirical posterior marginal of the
query target.

Continuous nodes are modelled as Gaussian leaves whose mean/std are fitted
from data.

Notes
-----
This is intentionally not a full SVI / NUTS pipeline — Pyro's machinery is
modular enough that an Importance-sampling baseline already exercises:

    - Pyro model definition (`pyro.sample` + `poutine.condition`)
    - posterior inference via `pyro.infer.Importance`
    - empirical marginal extraction for a chosen target

Upgrading to NUTS / amortised SVI for hybrid networks is tracked in v0.3.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from benchmarking.baselines.base import BaselineAdapter
from benchmarking.domains.base import BenchmarkProblem, Query


class PyroAdapter(BaselineAdapter):
    """Pyro Importance-sampling adapter for discrete/mixed BNs."""

    name = "pyro"
    supports = {"discrete", "continuous", "hybrid"}

    def __init__(self, n_samples: int = 200) -> None:
        self.n_samples = int(n_samples)
        self.problem: BenchmarkProblem | None = None
        self._cpts: Dict[str, torch.Tensor] = {}
        self._gaussian: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._parents: Dict[str, List[str]] = {}
        self._cards: Dict[str, int] = {}
        self._topo: List[str] = []

    # ------------------------------------------------------------------
    # Fitting — empirical CPTs / Gaussian leaves
    # ------------------------------------------------------------------

    def fit(self, problem: BenchmarkProblem) -> None:
        try:
            import pyro  # noqa: F401
        except ImportError as e:
            raise ImportError("PyroAdapter needs pyro-ppl: pip install pyro-ppl") from e

        self.problem = problem
        import networkx as nx
        g = nx.DiGraph(); g.add_nodes_from(problem.variables); g.add_edges_from(problem.dag)
        self._topo = list(nx.topological_sort(g))
        self._parents = {n: list(g.predecessors(n)) for n in self._topo}

        for node, (kind, k) in problem.variables.items():
            if kind == "discrete":
                self._cards[node] = k
                self._fit_discrete_cpt(node, k, problem)
            else:
                self._fit_gaussian_leaf(node, problem)

    def _fit_discrete_cpt(self, node, k, problem):
        x = problem.train_data[node].cpu().long().reshape(-1)
        parents = self._parents[node]
        if not parents:
            counts = torch.zeros(k)
            counts.scatter_add_(0, x, torch.ones_like(x, dtype=torch.float))
            self._cpts[node] = ((counts + 1.0) / (counts.sum() + k)).unsqueeze(0)
            return

        pa_cards = [self._cards.get(p, problem.variables[p][1]) for p in parents]
        strides, stride = [], 1
        for c in reversed(pa_cards):
            strides.append(stride); stride *= c
        strides = list(reversed(strides))
        n_pa = stride
        pa_idx = torch.zeros_like(x)
        for d, p in enumerate(parents):
            pa_idx = pa_idx + problem.train_data[p].cpu().long().reshape(-1) * strides[d]
        flat = pa_idx * k + x
        cnt = torch.zeros(n_pa * k)
        cnt.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        cnt = cnt.reshape(n_pa, k) + 1.0
        self._cpts[node] = cnt / cnt.sum(-1, keepdim=True)

    def _fit_gaussian_leaf(self, node, problem):
        x = problem.train_data[node].cpu().float().reshape(-1)
        self._gaussian[node] = (x.mean(), x.std().clamp_min(1e-3))

    # ------------------------------------------------------------------
    # Pyro model + Importance sampler
    # ------------------------------------------------------------------

    def _pyro_model(self):
        """Generative Pyro model: ancestral sampling node by node."""
        import pyro
        import pyro.distributions as dist

        cpts = self._cpts
        gauss = self._gaussian
        parents = self._parents
        cards = self._cards
        topo = self._topo

        def model():
            s = {}
            for node in topo:
                pa = parents[node]
                if node in cpts:
                    if pa:
                        row, stride = 0, 1
                        for p in reversed(pa):
                            v = s[p].long() if isinstance(s[p], torch.Tensor) else int(s[p])
                            row = row + v * stride
                            stride *= cards.get(p, 2)
                        probs = cpts[node][row]
                    else:
                        probs = cpts[node][0]
                    s[node] = pyro.sample(node, dist.Categorical(probs))
                else:
                    mu, sigma = gauss[node]
                    s[node] = pyro.sample(node, dist.Normal(mu, sigma))
            return s
        return model

    def query(self, q: Query) -> torch.Tensor:
        target = q.targets[0]
        marg = self._posterior_samples(q, n_samples=self.n_samples)
        if target in self._cards:
            k = self._cards[target]
            counts = torch.zeros(k)
            for v in marg.long().reshape(-1).tolist():
                if 0 <= v < k:
                    counts[v] += 1
            return counts / counts.sum().clamp_min(1e-12)
        return marg.mean(0).reshape(1).float()

    # ------------------------------------------------------------------
    # Batched samples (v0.4)
    # ------------------------------------------------------------------

    def query_batch_samples(
        self, q: Query, n_samples: int = 2000,
    ) -> torch.Tensor:
        """Per-row Importance posterior sampling.  Returns ``[B, n_samples, 1]``.

        Pyro has no native batched-evidence dispatch, so we loop over
        rows.  This is intentionally slow on large B; the inference
        crash test caps Pyro to a small B-subsample with the timeout
        guard from ``_crash_test_utils``.
        """
        b = self._batch_size(q)
        target = q.targets[0]
        out = torch.empty((b, n_samples, 1), dtype=torch.float)
        for i in range(b):
            row_q = self._row_query(q, i)
            samples = self._posterior_samples(row_q, n_samples=n_samples)
            out[i] = samples.float().reshape(n_samples, 1)
        return out

    def _posterior_samples(
        self, q: Query, *, n_samples: int,
    ) -> torch.Tensor:
        """Draw ``n_samples`` posterior samples of ``q.targets[0]``."""
        try:
            import pyro.poutine as poutine
            from pyro.infer import EmpiricalMarginal, Importance
        except ImportError as e:  # pragma: no cover
            raise NotImplementedError("pyro-ppl not installed") from e

        target = q.targets[0]
        evidence = {}
        for k, v in q.evidence.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            evidence[k] = (
                torch.tensor(int(val)) if k in self._cards
                else torch.tensor(float(val))
            )
        model = self._pyro_model()
        conditioned = poutine.condition(model, data=evidence)
        posterior = Importance(conditioned, num_samples=n_samples).run()
        marg = EmpiricalMarginal(posterior, sites=target)
        out = torch.stack([marg() for _ in range(n_samples)]).float().reshape(n_samples)
        return out

    def _row_query(self, q: Query, i: int) -> Query:
        ev = {}
        for k, v in q.evidence.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                ev[k] = v[i]
            else:
                ev[k] = v
        return Query(targets=q.targets, evidence=ev, kind=q.kind)

    def _batch_size(self, q: Query) -> int:
        for v in q.evidence.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                return int(v.shape[0])
        return 1

    def teardown(self) -> None:
        self._cpts = {}
        self._gaussian = {}
        self.problem = None
