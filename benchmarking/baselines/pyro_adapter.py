"""Pyro adapter — real importance-sampling inference.

For discrete BNs we build a Pyro model that ancestrally samples each node
from a Categorical parameterised by the empirical CPT of the training data,
then run inference via ``pyro.infer.Importance`` (the canonical Pyro path
for unconditional models) and read the empirical posterior marginal of the
query target.

For ``continuous_lg`` networks each continuous node is fit with a
linear-Gaussian conditional ``Normal(beta_0 + sum_i beta_i * pa_i, sigma)``
via least-squares regression on its continuous parents (see
``_fit_lg_leaf``); ``_pyro_model`` then samples the conditional Normal
ancestrally.

.. note::
    ``continuous_nongauss`` and ``hybrid`` are still out of scope.
    ``continuous_nongauss`` is excluded because the sampling
    distribution itself is the structural mismatch — the lstsq path
    would still yield a linear mean fit, but ``Normal(mu, sigma)``
    is the wrong family when residuals aren't Gaussian; correcting
    this needs SVI with a parameterised guide, not a different
    fit-path.  ``hybrid`` is excluded because the LG-conditional
    path assumes continuous parents only — a continuous node with
    discrete parents currently falls back to a marginal Gaussian
    fit, which is structurally wrong.  ``_BASELINE_APPLICABILITY``
    accordingly excludes both families for ``pyro-empirical`` and
    ``pyro-empirical-importance``.  The mixed-parent (discrete-parent
    / continuous-child) gap is tracked as a v0.8 follow-up.

Notes
-----
This is intentionally not a full SVI / NUTS pipeline — Pyro's machinery is
modular enough that an Importance-sampling baseline already exercises:

    - Pyro model definition (`pyro.sample` + `poutine.condition`)
    - posterior inference via `pyro.infer.Importance`
    - empirical marginal extraction for a chosen target

Upgrading to NUTS / amortised SVI for hybrid networks is tracked in v0.7
(separate issue from the continuous-correctness one).
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

    def __init__(self, n_samples: int = 50) -> None:
        # n_samples=50 default: paper-config (n_nodes=10, B=1024) hits a
        # 600s per-cell timeout — at the prior 200 default, the per-row
        # ``[marg() for _ in range(n_samples)]`` loop in
        # ``_posterior_samples`` extrapolates to ~1273s (200 × 1024 calls
        # to ``EmpiricalMarginal()``).  Pyro's ``EmpiricalMarginal``
        # exposes no batched-sample API; the only native path is the
        # per-call loop, so we cut n_samples ~4× to fit the budget
        # (projected ~318s, ~282s margin).  MC standard error rises
        # ~2× (sqrt(1/200)≈0.07 → sqrt(1/50)≈0.14), acceptable for
        # this baseline's role as a noisy importance-sampling reference.
        self.n_samples = int(n_samples)
        self.problem: BenchmarkProblem | None = None
        self._cpts: Dict[str, torch.Tensor] = {}
        # _gaussian[node] = (beta, sigma, cont_parents) where beta is a
        # 1-D tensor of length len(cont_parents)+1 (intercept first), and
        # cont_parents is the ordered list of continuous parent names
        # used to build the design matrix.  Empty cont_parents → beta is
        # length-1 (just the marginal mean), recovering the prior
        # marginal-Gaussian fit for continuous root nodes.
        self._gaussian: Dict[
            str, tuple[torch.Tensor, torch.Tensor, List[str]]
        ] = {}
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
                self._fit_lg_leaf(node, problem)

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

    def _fit_lg_leaf(self, node, problem):
        """Linear-Gaussian conditional fit for a continuous node.

        Fits ``y = beta_0 + sum_i beta_i * pa_i + eps`` via least-squares
        on the continuous parents.  Discrete parents (only present in
        the hybrid family, which is gated out in
        ``_BASELINE_APPLICABILITY``) are ignored — a continuous node
        with no continuous parents falls back to the marginal-Gaussian
        fit ``Normal(mean, std)``.
        """
        y = problem.train_data[node].cpu().float().reshape(-1)
        cont_parents = [
            p for p in self._parents[node]
            if problem.variables[p][0] == "continuous"
        ]
        if not cont_parents:
            beta = y.mean().reshape(1)
            sigma = y.std().clamp_min(1e-3)
            self._gaussian[node] = (beta, sigma, [])
            return

        cols = [torch.ones_like(y)]
        for p in cont_parents:
            cols.append(problem.train_data[p].cpu().float().reshape(-1))
        X = torch.stack(cols, dim=1)
        # gelsd handles rank-deficient X (e.g. n_train < num_parents+1).
        beta = torch.linalg.lstsq(X, y, driver="gelsd").solution.reshape(-1)
        residuals = y - X @ beta
        sigma = residuals.std().clamp_min(1e-3)
        self._gaussian[node] = (beta, sigma, cont_parents)

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
                    beta, sigma, cont_pa = gauss[node]
                    mu = beta[0]
                    for i, p in enumerate(cont_pa):
                        mu = mu + beta[i + 1] * s[p]
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
