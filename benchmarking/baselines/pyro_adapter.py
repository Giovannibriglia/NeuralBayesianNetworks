"""Pyro adapter — real importance-sampling inference.

For discrete BNs we build a Pyro model that ancestrally samples each node
from a Categorical parameterised by the empirical CPT of the training data,
then run inference via ``pyro.infer.Importance`` (the canonical Pyro path
for unconditional models) and read the empirical posterior marginal of the
query target.

For ``continuous_lg`` and ``hybrid`` networks each continuous node is fit
with a linear-Gaussian conditional
``Normal(beta_0 + sum_i beta_i * pa_i + sum_j gamma_j * oh_j, sigma)``
via least-squares regression.  Continuous parents contribute directly;
discrete parents are one-hot encoded (cardinality - 1 columns, dropping the
last category to avoid collinearity with the intercept) and concatenated to
the design matrix.  A single ``torch.linalg.lstsq`` call fits both
contributions (``_fit_lg_leaf``); ``_pyro_model`` reconstructs ``mu`` from
both the continuous and the one-hot discrete contributions.

Edge cases handled:

* Continuous-only parents — same as the previous ``continuous_lg`` path.
* Discrete-only parents — one-hot regression only (no continuous columns).
* No parents at all — marginal ``Normal(mean, std)`` (root node).
* Mixed (continuous + discrete) parents — canonical hybrid case.

.. note::
    ``continuous_nongauss`` is still out of scope.  The sampling
    distribution itself is the structural mismatch — the lstsq path
    yields a linear mean fit, but ``Normal(mu, sigma)`` is the wrong
    family when residuals aren't Gaussian; correcting this needs SVI
    with a parameterised guide, not a different fit-path.
    ``_BASELINE_APPLICABILITY`` accordingly excludes ``continuous_nongauss``
    for ``pyro-empirical`` and ``pyro-empirical-importance``.

Notes
-----
This is intentionally not a full SVI / NUTS pipeline — Pyro's machinery is
modular enough that an Importance-sampling baseline already exercises:

    - Pyro model definition (`pyro.sample` + `poutine.condition`)
    - posterior inference via `pyro.infer.Importance`
    - empirical marginal extraction for a chosen target

Upgrading to NUTS / amortised SVI is tracked in v0.7
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

    def __init__(self, n_samples: int = 50, device: str = "cpu") -> None:
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
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.problem: BenchmarkProblem | None = None
        self._cpts: Dict[str, torch.Tensor] = {}
        # _gaussian[node] = (beta, sigma, cont_parents, disc_parents, disc_cards)
        #
        # beta   — 1-D tensor of length 1 + len(cont_parents) + sum(k-1 for k in disc_cards)
        #          layout: [intercept | cont coefficients | one-hot coefficients (card-1 per disc pa)]
        # sigma  — scalar residual std, clamped ≥ 1e-3
        # cont_parents — ordered list of continuous parent names
        # disc_parents — ordered list of discrete parent names (may be empty)
        # disc_cards   — list of int cardinalities matching disc_parents
        #
        # Root nodes (no parents) → beta length-1 (marginal mean), disc_parents=[].
        self._gaussian: Dict[
            str, tuple[torch.Tensor, torch.Tensor, List[str], List[str], List[int]]
        ] = {}
        # _cpt_parents[node] — the discrete-only parent list used to build
        # the CPT for that node.  In hybrid networks a discrete node may
        # have continuous parents; those are ignored in the CPT (we model
        # P(node | discrete_parents) marginalising over continuous ones).
        # _pyro_model uses this list instead of self._parents[node] for
        # CPT row-index construction, ensuring fit and inference use the
        # same cardinalities.
        self._cpt_parents: Dict[str, List[str]] = {}
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
        x = problem.train_data[node].to(self.device).long().reshape(-1)
        # In hybrid networks a discrete node may have continuous parents.
        # The CPT approach only handles discrete parents — filter to those.
        all_parents = self._parents[node]
        disc_parents = [
            p for p in all_parents
            if problem.variables[p][0] == "discrete"
        ]
        self._cpt_parents[node] = disc_parents

        if not disc_parents:
            counts = torch.zeros(k, device=self.device)
            counts.scatter_add_(0, x, torch.ones_like(x, dtype=torch.float))
            self._cpts[node] = ((counts + 1.0) / (counts.sum() + k)).unsqueeze(0)
            return

        pa_cards = [self._cards[p] for p in disc_parents]
        strides, stride = [], 1
        for c in reversed(pa_cards):
            strides.append(stride); stride *= c
        strides = list(reversed(strides))
        n_pa = stride
        pa_idx = torch.zeros_like(x)
        for d, p in enumerate(disc_parents):
            pa_idx = pa_idx + problem.train_data[p].to(self.device).long().reshape(-1) * strides[d]
        flat = pa_idx * k + x
        cnt = torch.zeros(n_pa * k, device=self.device)
        cnt.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        cnt = cnt.reshape(n_pa, k) + 1.0
        self._cpts[node] = cnt / cnt.sum(-1, keepdim=True)

    def _fit_lg_leaf(self, node, problem):
        """Linear-Gaussian conditional fit for a continuous node.

        Fits ``y = beta_0 + Σ beta_i * cont_pa_i + Σ gamma_j * oh_j + eps``
        via least-squares.  Discrete parents are one-hot encoded using
        cardinality - 1 columns (last category dropped to avoid collinearity
        with the intercept).  Continuous parents enter directly.

        Root nodes (no parents of any kind) → marginal Normal(mean, std).
        """
        y = problem.train_data[node].to(self.device).float().reshape(-1)
        cont_parents = [
            p for p in self._parents[node]
            if problem.variables[p][0] == "continuous"
        ]
        disc_parents = [
            p for p in self._parents[node]
            if problem.variables[p][0] == "discrete"
        ]
        disc_cards = [problem.variables[p][1] for p in disc_parents]

        if not cont_parents and not disc_parents:
            beta = y.mean().reshape(1)
            sigma = y.std().clamp_min(1e-3)
            self._gaussian[node] = (beta, sigma, [], [], [])
            return

        cols = [torch.ones_like(y)]
        for p in cont_parents:
            cols.append(problem.train_data[p].to(self.device).float().reshape(-1))
        for p, k in zip(disc_parents, disc_cards):
            x_disc = problem.train_data[p].to(self.device).long().reshape(-1).clamp(0, k - 1)
            # one-hot, drop last column (k-1 indicator columns)
            oh = torch.zeros(len(y), k, device=self.device)
            oh.scatter_(1, x_disc.unsqueeze(1), 1.0)
            cols.append(oh[:, :-1])  # [N, k-1]
        X = torch.cat([c.reshape(len(y), -1) for c in cols], dim=1)
        # gelsd handles rank-deficient X on CPU; CUDA only supports gels
        # (full-rank QR path), acceptable for paper-scale data.
        driver = "gels" if self.device.startswith("cuda") else "gelsd"
        beta = torch.linalg.lstsq(X, y, driver=driver).solution.reshape(-1)
        residuals = y - X @ beta
        sigma = residuals.std().clamp_min(1e-3)
        self._gaussian[node] = (beta, sigma, cont_parents, disc_parents, disc_cards)

    # ------------------------------------------------------------------
    # Pyro model + Importance sampler
    # ------------------------------------------------------------------

    def _pyro_model(self):
        """Generative Pyro model: ancestral sampling node by node."""
        import pyro
        import pyro.distributions as dist

        cpts = self._cpts
        gauss = self._gaussian
        cpt_parents = self._cpt_parents  # discrete-only parent list per CPT node
        cards = self._cards
        topo = self._topo

        def model():
            s = {}
            for node in topo:
                if node in cpts:
                    # Use the discrete-only parent list that was used during
                    # CPT fitting — in hybrid networks a discrete node's
                    # continuous parents are not in the CPT table.
                    cpa = cpt_parents.get(node, [])
                    if cpa:
                        row, stride = 0, 1
                        for p in reversed(cpa):
                            v = s[p].long() if isinstance(s[p], torch.Tensor) else int(s[p])
                            row = row + v * stride
                            stride *= cards[p]
                        probs = cpts[node][row]
                    else:
                        probs = cpts[node][0]
                    s[node] = pyro.sample(node, dist.Categorical(probs))
                else:
                    beta, sigma, cont_pa, disc_pa, disc_cards = gauss[node]
                    mu = beta[0]
                    for i, p in enumerate(cont_pa):
                        mu = mu + beta[i + 1] * s[p]
                    # Discrete-parent one-hot contribution.
                    # beta layout after the cont block:
                    #   positions [1 + len(cont_pa) : 1 + len(cont_pa) + k-1]
                    #   are the one-hot coefficients for each discrete parent.
                    offset = 1 + len(cont_pa)
                    for p, k in zip(disc_pa, disc_cards):
                        v = s[p].long() if isinstance(s[p], torch.Tensor) else int(s[p])
                        v = max(0, min(int(v), k - 1))
                        # last category → all indicators 0 (absorbed into intercept)
                        if v < k - 1:
                            mu = mu + beta[offset + v]
                        offset += k - 1
                    s[node] = pyro.sample(node, dist.Normal(mu, sigma))
            return s
        return model

    def query(self, q: Query) -> torch.Tensor:
        target = q.targets[0]
        marg = self._posterior_samples(q, n_samples=self.n_samples)
        if target in self._cards:
            k = self._cards[target]
            counts = torch.zeros(k, device=self.device)
            for v in marg.long().reshape(-1).tolist():
                if 0 <= v < k:
                    counts[v] += 1
            return (counts / counts.sum().clamp_min(1e-12)).cpu()
        return marg.mean(0).reshape(1).float().cpu()

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
        out = torch.empty((b, n_samples, 1), dtype=torch.float, device=self.device)
        for i in range(b):
            row_q = self._row_query(q, i)
            samples = self._posterior_samples(row_q, n_samples=n_samples)
            out[i] = samples.float().reshape(n_samples, 1)
        return out.cpu()

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
                torch.tensor(int(val), device=self.device) if k in self._cards
                else torch.tensor(float(val), device=self.device)
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
        self._cpt_parents = {}
        self._gaussian = {}
        self.problem = None
