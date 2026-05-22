"""pgmpy adapter — VariableElimination for discrete BNs and (v0.4)
LinearGaussianBayesianNetwork for the ``continuous_lg`` family.

Discrete networks use ``DiscreteBayesianNetwork`` + maximum-likelihood
estimation + ``VariableElimination``.  Continuous Linear-Gaussian
networks use pgmpy's ``LinearGaussianBayesianNetwork`` whose joint is a
single multivariate Gaussian; conditioning is closed-form.

For ``continuous_nongauss`` and ``hybrid``, pgmpy is structurally
inapplicable; ``query_batch_samples`` returns a NaN-shape sentinel and
the runner records ``not_supported='non_gaussian_continuous'``.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List

import torch

from benchmarking.baselines.base import BaselineAdapter
from benchmarking.domains.base import BenchmarkProblem, Query


class PgmpyAdapter(BaselineAdapter):
    """Wraps pgmpy ``VariableElimination`` (discrete) or LG closed-form (continuous_lg)."""

    name = "pgmpy"
    supports = {"discrete", "continuous"}

    def __init__(self, *, param_method: str = "mle") -> None:
        # v0.6c-C-1b: ``param_method`` selects between MLE and Bayesian
        # estimators on discrete networks (BDeu prior, equivalent_sample_size=5
        # — the standard pgmpy default for the Bayesian path).  The
        # continuous_lg path uses MLE only (closed-form least-squares;
        # no Bayesian variant in pgmpy as of 0.1.25).
        self.param_method = param_method
        self.model: Any = None            # discrete VE model
        self.infer: Any = None            # discrete inference engine
        self.lg_model: Any = None         # LinearGaussianBayesianNetwork
        self.lg_topo: List[str] | None = None
        self.problem: BenchmarkProblem | None = None
        self.kind: str = "unfit"          # "discrete" | "continuous_lg" | "unsupported"

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, problem: BenchmarkProblem, epochs: int = 20) -> None:
        try:
            from pgmpy.inference import VariableElimination
            from pgmpy.models import DiscreteBayesianNetwork
        except ImportError as e:
            raise ImportError("PgmpyAdapter needs pgmpy: pip install pgmpy") from e
        self.problem = problem

        var_kinds = {kind for (kind, _) in problem.variables.values()}
        name = problem.name if isinstance(problem.name, str) else ""
        if var_kinds == {"discrete"}:
            self._fit_discrete(problem, DiscreteBayesianNetwork, VariableElimination)
        elif var_kinds == {"continuous"}:
            # Synthetic-BN names encode the family.  pgmpy's continuous
            # path is Linear-Gaussian-only, so silently fitting an LG to
            # a non-Gaussian SCM would produce a biased baseline.  We
            # refuse ``continuous_nongauss`` explicitly; ``continuous_lg``
            # proceeds.
            if name.startswith("continuous_nongauss"):
                self.kind = "unsupported"
            else:
                self._fit_lg(problem)
        else:
            self.kind = "unsupported"

    def _fit_discrete(self, problem, DiscreteBayesianNetwork, VariableElimination) -> None:  # noqa: N803
        import pandas as pd
        df = pd.DataFrame({
            k: v.cpu().long().reshape(-1).numpy()
            for k, v in problem.train_data.items()
        })
        bn = DiscreteBayesianNetwork(problem.dag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            if self.param_method == "bayes":
                # v0.6c-C-1b: BayesianEstimator with the standard pgmpy
                # default prior (BDeu, equivalent_sample_size=5).
                #
                # v0.8-#53: pgmpy 1.x dropped the `prior_type` and
                # `equivalent_sample_size` kwargs from `bn.fit`;
                # they must be passed to `BayesianEstimator.get_parameters`
                # instead.  Pre-fix path was structurally dead — pgmpy 1.x
                # raises `TypeError` on the legacy kwarg form — but the
                # bayes branch never fired through `_fit_and_score_pgmpy`
                # (the runner constructed `PgmpyAdapter()` with no kwargs),
                # so the API regression was hidden until #53's plumbing
                # made the branch reachable.
                from pgmpy.estimators import BayesianEstimator
                estimator = BayesianEstimator(model=bn, data=df)
                cpds = estimator.get_parameters(
                    prior_type="BDeu", equivalent_sample_size=5,
                )
                bn.add_cpds(*cpds)
            else:
                try:
                    from pgmpy.parameter_estimator import DiscreteMLE
                    bn.fit(df, estimator=DiscreteMLE())
                except (ImportError, TypeError):
                    from pgmpy.estimators import MaximumLikelihoodEstimator
                    bn.fit(df, estimator=MaximumLikelihoodEstimator)
        self.model = bn
        self.infer = VariableElimination(bn)
        self.kind = "discrete"

    def _fit_lg(self, problem: BenchmarkProblem) -> None:
        """Fit a LinearGaussianBayesianNetwork via closed-form MLE per node."""
        try:
            from pgmpy.models import LinearGaussianBayesianNetwork
        except ImportError:
            self.kind = "unsupported"
            return
        try:
            from pgmpy.factors.continuous import LinearGaussianCPD
        except ImportError:
            self.kind = "unsupported"
            return

        import networkx as nx
        g = nx.DiGraph()
        g.add_nodes_from(problem.variables)
        g.add_edges_from(problem.dag)
        topo = list(nx.topological_sort(g))

        bn = LinearGaussianBayesianNetwork(problem.dag)
        # pgmpy's LGBN constructor seeds nodes from edges only; add isolated nodes.
        for node in topo:
            if node not in bn.nodes():
                bn.add_node(node)

        for node in topo:
            parents = list(g.predecessors(node))
            x = problem.train_data[node].cpu().float().reshape(-1, 1)
            n = x.shape[0]
            if not parents:
                mean = float(x.mean().item())
                var = float(x.var(unbiased=False).clamp_min(1e-6).item())
                cpd = LinearGaussianCPD(
                    variable=node, beta=[mean], std=var ** 0.5, evidence=[],
                )
            else:
                pa = torch.cat(
                    [problem.train_data[p].cpu().float().reshape(-1, 1)
                     for p in parents], dim=-1,
                )
                ones = torch.ones(n, 1, dtype=pa.dtype)
                x_aug = torch.cat([ones, pa], dim=-1)  # [N, 1+P]: bias term first
                theta = torch.linalg.lstsq(x_aug, x).solution.reshape(-1)  # [1+P]
                resid = x.reshape(-1) - (x_aug @ theta)
                var = float(resid.var(unbiased=False).clamp_min(1e-6).item())
                # pgmpy >=0.1.25: beta[0] is intercept, beta[1:] are slopes
                # in the same order as ``evidence``.
                cpd = LinearGaussianCPD(
                    variable=node,
                    beta=[float(theta[0].item()), *[float(b.item()) for b in theta[1:]]],
                    std=var ** 0.5,
                    evidence=parents,
                )
            bn.add_cpds(cpd)
        self.lg_model = bn
        self.lg_topo = topo
        self.kind = "continuous_lg"

    # ------------------------------------------------------------------
    # Single-row query (kept for back-compat)
    # ------------------------------------------------------------------

    def query(self, q: Query) -> torch.Tensor:
        if self.kind == "discrete":
            assert self.infer is not None
            ev = {
                k: int(v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in q.evidence.items()
            }
            result = self.infer.query(
                variables=list(q.targets), evidence=ev, show_progress=False,
            )
            return torch.from_numpy(result.values).float()
        if self.kind == "continuous_lg":
            mean, std = self._lg_posterior_moments(q)
            return mean.reshape(-1).cpu()
        raise NotImplementedError(
            f"PgmpyAdapter cannot handle this problem (kind={self.kind!r}).",
        )

    # ------------------------------------------------------------------
    # Batched samples (v0.4)
    # ------------------------------------------------------------------

    def query_batch_samples(
        self, q: Query, n_samples: int = 2000,
    ) -> torch.Tensor:
        if self.kind == "discrete":
            return self._discrete_samples(q, n_samples)
        if self.kind == "continuous_lg":
            return self._lg_samples(q, n_samples)
        # Unsupported: NaN sentinel
        b = self._batch_size(q)
        return torch.full(
            (b, n_samples, len(q.targets)), float("nan"),
        )

    def _discrete_samples(self, q: Query, n_samples: int) -> torch.Tensor:
        """Sample ``n_samples`` integer values per row from VE marginals."""
        assert self.infer is not None
        b = self._batch_size(q)
        ev_per_row = self._unpack_batch_evidence(q, b)
        target = q.targets[0]
        out_rows = []
        for ev in ev_per_row:
            result = self.infer.query(
                variables=[target], evidence=ev, show_progress=False,
            )
            probs = torch.from_numpy(result.values).float().clamp_min(1e-12)
            probs = probs / probs.sum()
            idx = torch.multinomial(
                probs, n_samples, replacement=True,
            ).float().reshape(n_samples, 1)
            out_rows.append(idx)
        return torch.stack(out_rows, dim=0)  # [B, n_samples, 1]

    def _lg_samples(self, q: Query, n_samples: int) -> torch.Tensor:
        """Closed-form Gaussian posterior → reparam samples."""
        b = self._batch_size(q)
        ev_per_row = self._unpack_batch_evidence(q, b)
        out = torch.empty((b, n_samples, len(q.targets)), dtype=torch.float)
        for i, ev in enumerate(ev_per_row):
            mean, std = self._lg_posterior_moments(
                Query(targets=q.targets, evidence=ev, kind=q.kind),
            )
            noise = torch.randn(n_samples, len(q.targets))
            out[i] = mean.reshape(1, -1) + std.reshape(1, -1) * noise
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _batch_size(self, q: Query) -> int:
        for v in q.evidence.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                return int(v.shape[0])
        return 1

    def _unpack_batch_evidence(
        self, q: Query, b: int,
    ) -> List[Dict[str, Any]]:
        """Turn a batched evidence dict into ``b`` per-row dicts."""
        out: List[Dict[str, Any]] = []
        for i in range(b):
            row: Dict[str, Any] = {}
            for k, v in q.evidence.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    val = v[i]
                else:
                    val = v
                if isinstance(val, torch.Tensor):
                    val = val.item()
                # pgmpy expects Python ints for discrete evidence,
                # floats for continuous.
                if (self.kind == "discrete"
                        or (self.kind == "continuous_lg" and k in self._lg_topo_disc())):
                    row[k] = int(val)
                else:
                    row[k] = float(val)
            out.append(row)
        return out

    def _lg_topo_disc(self) -> set[str]:
        """Discrete-evidence node names for LG path (currently always empty)."""
        return set()

    def _lg_posterior_moments(self, q: Query) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form mean + std of the LG posterior over ``q.targets[0]``.

        We assemble the joint Gaussian (μ, Σ) over all nodes from the LG
        CPDs in topological order, then condition on ``q.evidence``.
        """
        assert self.lg_model is not None and self.lg_topo is not None
        topo = self.lg_topo
        node_idx = {n: i for i, n in enumerate(topo)}
        n_total = len(topo)

        # Build the linear-Gaussian SEM:  X = A·X + b + ε, where ε ~ N(0, D)
        A = torch.zeros(n_total, n_total)
        b = torch.zeros(n_total)
        d = torch.zeros(n_total)
        for cpd in self.lg_model.get_cpds():
            j = node_idx[cpd.variable]
            betas = cpd.beta
            evidence = list(cpd.evidence) if hasattr(cpd, "evidence") else []
            b[j] = float(betas[0])
            for k, ev_node in enumerate(evidence):
                A[j, node_idx[ev_node]] = float(betas[k + 1])
            d[j] = float(cpd.std) ** 2

        # Joint moments: μ = (I-A)^-1 b ; Σ = (I-A)^-1 D (I-A)^-T
        eye = torch.eye(n_total)
        IA_inv = torch.linalg.inv(eye - A)
        mu = IA_inv @ b
        Sigma = IA_inv @ torch.diag(d) @ IA_inv.T

        target_idx = node_idx[q.targets[0]]
        ev_idx = [node_idx[k] for k in q.evidence]
        ev_vals = torch.tensor([float(v) for v in q.evidence.values()])
        if not ev_idx:
            return mu[target_idx:target_idx + 1], Sigma[target_idx, target_idx].clamp_min(0).sqrt().reshape(1)
        mu_t = mu[target_idx]
        mu_e = mu[ev_idx]
        Sigma_te = Sigma[target_idx, ev_idx]
        Sigma_ee = Sigma[ev_idx][:, ev_idx]
        delta = ev_vals - mu_e
        beta = torch.linalg.solve(Sigma_ee, delta)
        post_mean = mu_t + Sigma_te @ beta
        post_var = Sigma[target_idx, target_idx] - Sigma_te @ torch.linalg.solve(
            Sigma_ee, Sigma_te,
        )
        post_std = post_var.clamp_min(1e-12).sqrt()
        return post_mean.reshape(1), post_std.reshape(1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        self.model = None
        self.infer = None
        self.lg_model = None
        self.lg_topo = None
        self.kind = "unfit"
