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


def _state_axis_index(var: str, states: list, card: int) -> list[int]:
    """Map declared class values ``0..card-1`` to their pgmpy axis positions.

    ``states`` is ``cpd.state_names[var]`` (the axis order). Returns a list
    ``idx`` such that ``idx[v]`` is the axis position of declared value ``v``.
    Fails loudly (the failure mode that would silently corrupt every pgmpy
    recovery number) if a state is not an integer covering ``[0, card)``.
    """
    import numpy as np

    pos: dict[int, int] = {}
    for axis_p, s in enumerate(states):
        if not isinstance(s, (int, np.integer)) or not (0 <= int(s) < card):
            raise ValueError(
                f"node {var!r}: state {s!r} is not an integer in [0, {card})"
            )
        pos[int(s)] = axis_p
    if set(pos) != set(range(card)):
        raise ValueError(
            f"node {var!r}: learned states {sorted(pos)} do not cover "
            f"[0, {card}) — expected the full declared grid"
        )
    return [pos[v] for v in range(card)]


def _tabular_cpd_to_canonical(cpd, child: str, k: int, variables: dict) -> torch.Tensor:
    """Reshape+permute a pgmpy TabularCPD to the canonical [n_configs, K] layout.

    ``cpd.values`` is multi-dim ``[K, *parent_cards]`` with axis 0 = child and
    axes 1.. = parents in ``cpd.variables[1:]`` order (each axis ordered by
    ``cpd.state_names``). We transpose the parent axes into canonical lex order
    (class axis last), reindex every axis from state position to declared value
    order, then C-order reshape so the first canonical parent varies SLOWEST.
    """
    import numpy as np

    parents_cpd = list(cpd.variables[1:])
    canon = sorted(parents_cpd)
    values = np.asarray(cpd.values, dtype=np.float64)

    # Transpose: canonical parent axes first, class axis last.
    perm = [1 + parents_cpd.index(p) for p in canon] + [0]
    m = np.transpose(values, perm)  # [*canon_parent_cards(axis order), K(axis order)]

    # Reindex each canonical parent axis to declared 0..card-1 order...
    for i, p in enumerate(canon):
        pcard = int(variables[p][1])
        m = np.take(m, _state_axis_index(p, cpd.state_names[p], pcard), axis=i)
    # ...and the (last) class axis to declared 0..K-1 order.
    m = np.take(m, _state_axis_index(child, cpd.state_names[child], k), axis=m.ndim - 1)

    n_configs = 1
    for p in canon:
        n_configs *= int(variables[p][1])
    flat = np.ascontiguousarray(m).reshape(n_configs, k)
    return torch.from_numpy(flat).to(torch.float32)


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

    # Parameter-recovery capability (#109 PR 3): the mle/bayes paths expose
    # their learned discrete CPTs via extract_learned_cpts. Class flag = True
    # means this adapter class implements CPT extraction. Per-cell applicability
    # (the continuous lg path returns {} → not_applicable via the measurement's
    # family gate) is independent of the flag. Precedent for PR 4/5: use
    # flag=False ONLY when the adapter literally cannot extract; otherwise use
    # flag=True and let the cell-level gate decide.
    supports_param_recovery: bool = True

    def __init__(
        self,
        param_method: str,
        inference_method: str | None,
        n_samples: int = 1024,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        if param_method not in _VALID_PARAM_METHODS:
            raise ValueError(
                f"Unknown param_method {param_method!r}. "
                f"Valid: {sorted(_VALID_PARAM_METHODS)}"
            )
        # inference_method=None is the fit-only / parameter-learning
        # construction (#109): the adapter is fit and scored, never queried,
        # so it carries no inference engine and an engine-less name
        # (e.g. "pgmpy-mle"). fit() is independent of inference_method.
        if (inference_method is not None
                and inference_method not in _VALID_INFERENCE_METHODS):
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
        self.name = (
            f"pgmpy-{param_method}-{inference_method}"
            if inference_method is not None else f"pgmpy-{param_method}"
        )

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
                # Use pgmpy's library defaults verbatim (no adapter-side prior
                # override): prior_type="BDeu", equivalent_sample_size=5 in
                # pgmpy 1.1.2 — i.e. identical numbers to the old explicit
                # kwargs, but no longer disguising the default as our choice
                # (v0.14 methodology fidelity).
                # v0.8-#53: pgmpy 1.x requires the estimator path
                # (BayesianEstimator.get_parameters), not bn.fit().
                from pgmpy.estimators import BayesianEstimator

                estimator = BayesianEstimator(model=bn, data=df)
                cpds = estimator.get_parameters()
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

    def extract_learned_cpts(self) -> dict[str, torch.Tensor]:
        """Learned discrete CPTs in the canonical layout (param-recovery, #109).

        Returns ``{node: probs[n_parent_configs, K]}`` for each discrete node
        with all-discrete parents, in the canonical layout the recovery metric
        compares cell-by-cell against the true model (parents sorted lex,
        configs row-major with the first parent slowest, classes ``0..K-1``,
        each row a distribution). The lg (continuous) path returns ``{}`` — its
        cells are ``not_applicable`` via the measurement's family gate.

        Why RE-ESTIMATE instead of reading the stored CPDs? pgmpy infers
        ``state_names`` from the observed data, so a globally-unobserved state
        is dropped from the fitted grid. Padding zeros for such states would
        CORRUPT the bayes path (BDeu assigns prior mass to unseen states by
        design, not zero). Re-estimating with the DECLARED ``state_names``
        yields the full declared grid identically for mle (unseen class → 0,
        unseen parent config → uniform) and bayes (every cell gets prior mass).
        The re-estimation is deterministic (closed-form counting, no RNG) and
        agrees with ``self._model`` on every observed state — the stored model
        and the inference path are left untouched.

        The estimator family matches the original fit (BayesianEstimator for
        ``bayes``, MaximumLikelihoodEstimator otherwise); the extraction logic
        below does NOT branch on the method — both produce TabularCPDs read
        identically. Only mle can yield ``+inf`` recovery KL (hard zeros);
        bayes never zeros.

        Raises ValueError (→ measurement ``status="error"``) if any learned
        state name is not an integer covering ``[0, K)`` — silent mis-alignment
        is the failure mode that would corrupt every pgmpy recovery number.
        """
        if self._kind != "discrete":
            return {}
        if self.problem is None:
            raise RuntimeError(
                "Adapter not fitted. Call fit() before extract_learned_cpts()."
            )
        import pandas as pd
        from pgmpy.models import DiscreteBayesianNetwork

        variables = self.problem.variables
        df = pd.DataFrame({
            k: v.cpu().long().reshape(-1).numpy()
            for k, v in self.problem.train_data.items()
        })
        bn = DiscreteBayesianNetwork(self.problem.dag)
        # Re-seed isolated nodes exactly as _fit_discrete does, so every
        # declared node gets a (marginal) CPD.
        for node in variables:
            if node not in bn.nodes():
                bn.add_node(node)

        # Declared full grid: forces pgmpy to span 0..card-1 for every discrete
        # node regardless of what the training data happened to observe.
        state_names = {
            n: list(range(c)) for n, (kind, c) in variables.items()
            if kind == "discrete"
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            if self.param_method == "bayes":
                from pgmpy.estimators import BayesianEstimator

                cpds = BayesianEstimator(
                    model=bn, data=df, state_names=state_names,
                ).get_parameters()
            else:
                from pgmpy.estimators import MaximumLikelihoodEstimator

                cpds = MaximumLikelihoodEstimator(
                    model=bn, data=df, state_names=state_names,
                ).get_parameters()

        cpd_by_node = {cpd.variable: cpd for cpd in cpds}
        out: dict[str, torch.Tensor] = {}
        for node, (kind, card) in variables.items():
            if kind != "discrete":
                continue
            cpd = cpd_by_node[node]
            parents = list(cpd.variables[1:])
            if any(variables[p][0] != "discrete" for p in parents):
                continue  # discrete node with a continuous parent — omit
            out[node] = _tabular_cpd_to_canonical(cpd, node, int(card), variables)
        return out
