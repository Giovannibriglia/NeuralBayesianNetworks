"""pgmpy adapter for v0.13 BaselineAdapter protocol.

Stateful fit-then-query contract. Covers three inference labels:

  - pgmpy-mle-ve     (discrete, MLE + Variable Elimination)
  - pgmpy-bayes-ve   (discrete, Bayesian estimator + VE)
  - pgmpy-lg-predict (continuous_lg, closed-form LG posterior samples)

For pgmpy-lg-predict, query() returns Posterior(samples=...) drawn from the
analytical Gaussian posterior N(post_mean, post_std²).  post_mean and
post_std are computed via Schur complement from the fitted CPD parameters
(exact closed-form; no Monte Carlo approximation).  Default n_samples=1024,
matching NBNAdapter for cross-adapter consistency.

This is a behaviour change from the old adapter's query(), which returned
only the posterior mean as a scalar tensor.  The change is justified by the
v0.13 Posterior contract requiring a samples tensor for continuous targets
so that W1 accuracy can be computed honestly without treating a single point
as a degenerate distribution.  The old query_batch_samples() already had the
correct reparameterised sampling logic; this adapter promotes it to the
primary query() path.

The old adapter at benchmarking/baselines/pgmpy_adapter.py is untouched
and continues to serve the v0.12 runner.

Reference: docs/v0.13-benchmark-redesign.md §4.1
Old (functional) adapter: benchmarking/baselines/pgmpy_adapter.py
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import torch

from benchmarking.core._device import resolve_device
from benchmarking.core.applicability import BASELINE_FAMILY_APPLICABILITY as _BASELINE_APPLICABILITY
from benchmarking.core.interfaces import default_query_batch
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.domains.posterior import Posterior

logger = logging.getLogger(__name__)

_VALID_PARAM_METHODS = frozenset({"mle", "bayes", "lg"})
_VALID_INFERENCE_METHODS = frozenset({"ve", "predict"})


class PgmpyAdapter:
    """Stateful pgmpy adapter implementing the v0.13 BaselineAdapter protocol.

    Handles three pgmpy inference labels via param_method + inference_method:

        pgmpy-mle-ve       param_method="mle",   inference_method="ve"
        pgmpy-bayes-ve     param_method="bayes",  inference_method="ve"
        pgmpy-lg-predict   param_method="lg",    inference_method="predict"

    Construction::

        adapter = PgmpyAdapter(param_method="mle",  inference_method="ve")
        adapter = PgmpyAdapter(param_method="bayes", inference_method="ve")
        adapter = PgmpyAdapter(param_method="lg",   inference_method="predict")
        adapter = PgmpyAdapter(param_method="lg",   inference_method="predict",
                               n_samples=2048)

    The ``name`` attribute is derived: ``"pgmpy-{param_method}-{inference_method}"``.
    """

    # v0.14 (#148) §5.6: sequential-only (query_batch is the default
    # helper) — the speed-benchmark sweep runs this once at batch_size=1.
    supports_batched_queries: bool = False

    def __init__(
        self,
        param_method: str,
        inference_method: str,
        n_samples: int = 1024,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        if param_method not in _VALID_PARAM_METHODS:
            raise ValueError(
                f"Unknown param_method {param_method!r}. "
                f"Valid: {sorted(_VALID_PARAM_METHODS)}"
            )
        if inference_method not in _VALID_INFERENCE_METHODS:
            raise ValueError(
                f"Unknown inference_method {inference_method!r}. "
                f"Valid: {sorted(_VALID_INFERENCE_METHODS)}"
            )
        self.param_method = param_method
        self.inference_method = inference_method
        self.n_samples = int(n_samples)
        # pgmpy is a CPU-only library: its inference and estimators run on
        # numpy/pandas, not torch tensors.  We accept the device arg for a
        # uniform adapter contract but always run on CPU; if a GPU device
        # was explicitly requested, log the override so it isn't silently
        # surprising in the device column.
        resolved = resolve_device(device)
        if resolved.startswith("cuda"):
            logger.info(
                "PgmpyAdapter: device=%r requested but pgmpy is CPU-only; "
                "running on cpu.", resolved,
            )
        self.device = "cpu"
        self.name = f"pgmpy-{param_method}-{inference_method}"

        # State populated by fit()
        self._kind: str = "unfit"               # "discrete" | "continuous_lg"
        self._model: Any | None = None          # DiscreteBayesianNetwork | LGBN
        self._infer: Any | None = None          # VariableElimination (discrete)
        self._lg_topo: list[str] | None = None  # topological order (LG path)
        self.problem: BenchmarkProblem | None = None

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------

    def fit(self, problem: BenchmarkProblem, **kwargs: Any) -> None:
        """Fit on problem.train_data. State stored on self.

        kwargs:
            epochs (int): accepted for runner API compatibility; not used —
                pgmpy fitting is closed-form / analytic, not epoch-based.
        """
        try:
            from pgmpy.inference import VariableElimination
            from pgmpy.models import DiscreteBayesianNetwork
        except ImportError as exc:
            raise ImportError(
                "PgmpyAdapter needs pgmpy: pip install pgmpy"
            ) from exc

        self.problem = problem

        if self.param_method in {"mle", "bayes"}:
            self._fit_discrete(problem, DiscreteBayesianNetwork, VariableElimination)
        else:
            # param_method == "lg"
            self._fit_lg(problem)

    def _fit_discrete(
        self,
        problem: BenchmarkProblem,
        DiscreteBayesianNetwork: Any,  # noqa: N803
        VariableElimination: Any,      # noqa: N803
    ) -> None:
        import pandas as pd

        df = pd.DataFrame({
            k: v.cpu().long().reshape(-1).numpy()
            for k, v in problem.train_data.items()
        })
        bn = DiscreteBayesianNetwork(problem.dag)
        # pgmpy seeds nodes from edges only, so isolated nodes (no parents or
        # children) are dropped from the model — querying them later raises
        # "node not in graph". Add them explicitly so MLE/Bayes estimate their
        # marginal CPDs. Mirrors the LG path below (_fit_lg).
        for node in problem.variables:
            if node not in bn.nodes():
                bn.add_node(node)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            if self.param_method == "bayes":
                # BDeu prior, equivalent_sample_size=5 — standard pgmpy default.
                # v0.8-#53: pgmpy 1.x requires passing prior kwargs to
                # BayesianEstimator.get_parameters(), not to bn.fit().
                from pgmpy.estimators import BayesianEstimator

                estimator = BayesianEstimator(model=bn, data=df)
                cpds = estimator.get_parameters(
                    prior_type="BDeu", equivalent_sample_size=5,
                )
                bn.add_cpds(*cpds)
            else:
                # MLE path: try new pgmpy 1.x DiscreteMLE first; fall back to
                # MaximumLikelihoodEstimator for older installs.
                try:
                    from pgmpy.parameter_estimator import DiscreteMLE

                    bn.fit(df, estimator=DiscreteMLE())
                except (ImportError, TypeError):
                    from pgmpy.estimators import MaximumLikelihoodEstimator

                    bn.fit(df, estimator=MaximumLikelihoodEstimator)

        self._model = bn
        self._infer = VariableElimination(bn)
        self._kind = "discrete"

    def _fit_lg(self, problem: BenchmarkProblem) -> None:
        """Fit LinearGaussianBayesianNetwork via closed-form MLE per node.

        Ported faithfully from benchmarking/baselines/pgmpy_adapter.py::_fit_lg().
        Each node's CPD is fit by lstsq over (intercept | parent columns);
        residual variance is stored in LinearGaussianCPD.std.

        pgmpy >=0.1.25: LinearGaussianCPD expects beta[0]=intercept,
        beta[1:]=slopes in the same order as ``evidence``.
        """
        try:
            from pgmpy.models import LinearGaussianBayesianNetwork
        except ImportError as exc:
            raise ImportError(
                "PgmpyAdapter needs pgmpy: pip install pgmpy"
            ) from exc
        try:
            from pgmpy.factors.continuous import LinearGaussianCPD
        except ImportError as exc:
            raise ImportError(
                "PgmpyAdapter requires pgmpy with LinearGaussianCPD support."
            ) from exc

        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from(problem.variables)
        g.add_edges_from(problem.dag)
        topo = list(nx.topological_sort(g))

        bn = LinearGaussianBayesianNetwork(problem.dag)
        # pgmpy LGBN seeds nodes from edges only; add isolated nodes explicitly.
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
                     for p in parents],
                    dim=-1,
                )
                ones = torch.ones(n, 1, dtype=pa.dtype)
                x_aug = torch.cat([ones, pa], dim=-1)           # [N, 1+P]
                theta = torch.linalg.lstsq(x_aug, x).solution.reshape(-1)  # [1+P]
                resid = x.reshape(-1) - (x_aug @ theta)
                var = float(resid.var(unbiased=False).clamp_min(1e-6).item())
                cpd = LinearGaussianCPD(
                    variable=node,
                    beta=[float(theta[0].item()),
                          *[float(b.item()) for b in theta[1:]]],
                    std=var ** 0.5,
                    evidence=parents,
                )
            bn.add_cpds(cpd)

        self._model = bn
        self._lg_topo = topo
        self._kind = "continuous_lg"

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    def query(self, q: Query) -> Posterior:
        """Query the fitted model.

        Returns:
            Posterior(probs=...)   for discrete targets — normalised probs
                                   vector of shape (K,) for the target node.
            Posterior(samples=...) for continuous_lg targets — n_samples
                                   draws from N(post_mean, post_std²) of
                                   shape (n_samples,).

        Raises:
            RuntimeError: if called before fit().
        """
        if self._kind == "unfit":
            raise RuntimeError(
                "Adapter not fitted. Call fit() before query()."
            )
        if self._kind == "discrete":
            return self._query_discrete(q)
        return self._query_lg(q)

    def _query_discrete(self, q: Query) -> Posterior:
        assert self._infer is not None
        # Skip None-valued evidence (Phase 3 empty mode): pgmpy auto-
        # marginalizes any variable absent from the evidence dict.
        ev = {
            k: int(v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in q.evidence.items()
            if v is not None
        }
        result = self._infer.query(
            variables=list(q.targets), evidence=ev, show_progress=False,
        )
        probs = torch.from_numpy(result.values).float().clamp_min(0.0)
        probs = probs / probs.sum()
        return Posterior(probs=probs)

    def _query_lg(self, q: Query) -> Posterior:
        post_mean, post_std = self._lg_posterior_moments(q)
        samples = torch.randn(self.n_samples) * post_std.item() + post_mean.item()
        return Posterior(samples=samples.cpu())

    # -------------------------------------------------------------------------
    # LG posterior moments (Schur complement, ported from old adapter)
    # -------------------------------------------------------------------------

    def _lg_posterior_moments(
        self, q: Query,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form mean + std of the LG posterior over q.targets[0].

        Assembles the full joint Gaussian (μ, Σ) over all nodes from the LG
        CPDs in topological order, then conditions on q.evidence via the
        Schur-complement formula.

        Ported faithfully from
        benchmarking/baselines/pgmpy_adapter.py::_lg_posterior_moments().
        The variance clamp_min(1e-12) before sqrt() guards the degenerate
        case where the target is fully determined by evidence.

        Returns:
            (post_mean, post_std), each a 1-element float tensor.
        """
        assert self._model is not None and self._lg_topo is not None
        topo = self._lg_topo
        node_idx = {n: i for i, n in enumerate(topo)}
        n_total = len(topo)

        # Assemble SEM:  X = A·X + b + ε,  ε ~ N(0, D)
        A = torch.zeros(n_total, n_total)
        b_vec = torch.zeros(n_total)
        d = torch.zeros(n_total)
        for cpd in self._model.get_cpds():
            j = node_idx[cpd.variable]
            betas = cpd.beta
            evidence = list(cpd.evidence) if hasattr(cpd, "evidence") else []
            b_vec[j] = float(betas[0])
            for k, ev_node in enumerate(evidence):
                A[j, node_idx[ev_node]] = float(betas[k + 1])
            d[j] = float(cpd.std) ** 2

        # Joint moments: μ = (I-A)⁻¹ b ;  Σ = (I-A)⁻¹ D (I-A)⁻ᵀ
        eye = torch.eye(n_total)
        IA_inv = torch.linalg.inv(eye - A)
        mu = IA_inv @ b_vec
        Sigma = IA_inv @ torch.diag(d) @ IA_inv.T

        target_idx = node_idx[q.targets[0]]
        # Skip None-valued evidence (Phase 3 empty mode): an empty ev_idx
        # falls through to the unconditional marginal branch below, which is
        # exactly P(target) with the evidence variables marginalized out.
        observed = [
            (node_idx[k], float(v))
            for k, v in q.evidence.items()
            if v is not None
        ]
        ev_idx = [i for i, _ in observed]
        ev_vals = torch.tensor([val for _, val in observed])

        if not ev_idx:
            return (
                mu[target_idx : target_idx + 1],
                Sigma[target_idx, target_idx].clamp_min(0.0).sqrt().reshape(1),
            )

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

    def query_batch(self, queries: list[Query]) -> list[Posterior]:
        """Sequential default (PR 1, #148). Pgmpy stays sequential
        (design doc §3.3); no override planned."""
        return default_query_batch(self, queries)

    # -------------------------------------------------------------------------
    # Applicability
    # -------------------------------------------------------------------------

    def is_applicable(self, problem: BenchmarkProblem) -> bool:
        """Return True if this adapter can handle problem's family.

        Delegates to _BASELINE_APPLICABILITY using self.name as the key.
        Family is inferred from variable kinds (same logic as NBNAdapter):
          all discrete        → "discrete"
          all continuous      → "continuous_lg"
          mixed               → "hybrid"
        """
        kinds = {kind for kind, _ in problem.variables.values()}
        if kinds == {"discrete"}:
            family = "discrete"
        elif kinds == {"continuous"}:
            family = "continuous_lg"
        else:
            family = "hybrid"

        entry = _BASELINE_APPLICABILITY.get(self.name)
        if entry is None:
            return False
        return family in entry.families
