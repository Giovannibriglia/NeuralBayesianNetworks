"""NBN adapter for v0.13 BaselineAdapter protocol.

Stateful fit-then-query contract. State stored on self:
  - self.model:      the fitted NeuralBayesianNetwork
  - self._engine_obj: the fitted inference engine
  - self.problem:    the BenchmarkProblem used for fitting

Mirrors the logic of nbn/bench/baselines/nbn_adapter.py with three
structural changes:
  1. Stateful: fit() stores on self instead of returning.
  2. Constructor uses unified ``mechanism`` + ``engine`` args (instead of
     discrete_mech / continuous_mech + engine).
  3. query() returns Posterior instead of a raw torch.Tensor.

The old adapter at nbn/bench/baselines/nbn_adapter.py is untouched
and continues to serve the v0.12 runner.

Reference: docs/v0.13-benchmark-redesign.md §4.1
Old (functional) adapter: nbn/bench/baselines/nbn_adapter.py
"""
from __future__ import annotations

from typing import Any

import torch

from nbn.bench.core._device import resolve_device
from nbn.bench.core.applicability import BASELINE_FAMILY_APPLICABILITY as _BASELINE_APPLICABILITY
from nbn.bench.core.interfaces import default_query_batch
from nbn.bench.domains.base import BenchmarkProblem, Query
from nbn.bench.domains.posterior import Posterior

# ---- mechanism → internal specifiers -----------------------------------------
# Maps the unified `mechanism` arg to the old adapter's internal fields.

# Which discrete mechanism to use for nodes with kind="discrete"
_DISCRETE_MECH: dict[str, str] = {
    "cat":       "categorical_table",
    # Laplace-smoothed (Dirichlet alpha=1) empirical CPTs. Same table
    # machinery as "cat" but the smoothed estimator instead of MLE-parity
    # alpha=0: at paper scale pyro's alpha=1 estimator beat nbn-cat on BOTH
    # discrete recovery metrics (TV 0.0646 vs 0.0699, KL 0.0222 vs 0.2467,
    # param_learning_complete 20260701) — this label closes that gap with a
    # mechanism the library already shipped.
    "cat-bayes": "smoothed_empirical_categorical",
    "neuralcat": "neural_categorical",
    "lg":        "categorical_table",   # lg is continuous-only; discrete fallback
    "mdn":       "categorical_table",
    "flow":      "categorical_table",
    "hybrid":    "categorical_table",   # HybridRouter default
    # Non-parametric continuous mechanisms (#223 / PR 8) — continuous-only, so
    # categorical_table is the discrete-node fallback (mirrors lg/mdn/flow). The
    # keys here are ALSO the unified valid-mechanism set for __init__ validation.
    "kde":       "categorical_table",
    "knn":       "categorical_table",
    "flexcode":  "categorical_table",
}

# Which continuous mechanism to use for nodes with kind="continuous"
_CONTINUOUS_MECH: dict[str, str] = {
    "cat":       "mdn",                 # cat is discrete-only; continuous fallback
    "cat-bayes": "mdn",                 # discrete-only; continuous fallback like cat
    "neuralcat": "mdn",
    "lg":        "linear_gaussian",
    "mdn":       "mdn",
    "flow":      "flow",
    "hybrid":    "mdn",                 # HybridRouter default
    # Non-parametric continuous mechanisms (#223 / PR 8).
    "kde":       "conditional_kde",
    "knn":       "knn_conditional",
    "flexcode":  "flexcode",
}

# Maps the `engine` arg to the engine_spec string used internally
_ENGINE_SPEC: dict[str, str] = {
    "lw":     "lw",
    "ve":     "ve",
    "ais":    "ais",     # amortized neural-proposal IS (v0.14, #181)
    "avi":    "avi",     # amortized variational inference (v0.14, #182)
    "router": "hybrid",  # HybridRouter
}


def _nan_to_none(v) -> float | None:
    """Per-query PSIS k̂ is NaN where degenerate (near-uniform weights); the
    parquet stores that as None (X2)."""
    f = float(v)
    return None if f != f else f   # NaN != NaN


class NBNAdapter:
    """Stateful NBN adapter implementing the v0.13 BaselineAdapter protocol.

    Handles all 8 NBN inference labels via mechanism + engine arguments:

        nbn-cat-lw         mechanism="cat",       engine="lw"
        nbn-cat-ve         mechanism="cat",       engine="ve"
        nbn-neuralcat-lw   mechanism="neuralcat", engine="lw"
        nbn-neuralcat-ve   mechanism="neuralcat", engine="ve"
        nbn-lg-lw          mechanism="lg",        engine="lw"
        nbn-mdn-lw         mechanism="mdn",       engine="lw"
        nbn-flow-lw        mechanism="flow",      engine="lw"
        nbn-hybrid-router  mechanism="hybrid",    engine="router"

    Construction::

        adapter = NBNAdapter(mechanism="cat", engine="lw")  # device auto-detects

    The ``name`` attribute is derived: ``"nbn-{mechanism}-{engine}"``.
    """

    # v0.14 (#148) §5.6: this adapter has a real library-batched
    # query_batch override (PR 2), so the speed-benchmark sweep runs it
    # at every batch_sizes value. (An explicit class flag rather than
    # the design doc's __func__-identity check, because PR 1 gave every
    # adapter an explicit query_batch method — identity can't tell a
    # real override from a sequential opt-in wrapper.)
    supports_batched_queries: bool = True

    # Parameter-learning capability (#109): this adapter implements score_data
    # (held-out joint log-likelihood over self.model), so ParamLearningMeasurement
    # scores it instead of emitting status="not_supported". Concrete-class flag,
    # getattr-gated by the measurement — mirrors supports_batched_queries above
    # and the opt-in contract documented on the BaselineAdapter protocol.
    supports_scoring: bool = True

    # Parameter-recovery capability (#109 PR 2): this adapter exposes its
    # learned discrete CPTs via extract_learned_cpts, so ParamLearningMeasurement
    # compares them against the true CPDs (param_recovery_tv / _kl) instead of
    # not_supported. Same concrete-class getattr-gated flag precedent; declared
    # alongside its method so the two never drift out of sync.
    supports_param_recovery: bool = True

    # Calibration capability (#109 PR 7). True if the adapter implements
    # ``predictive_samples(test_data) -> dict[node, Tensor[N, S]]``, returning S
    # predictive samples per test row per CONTINUOUS node. ParamLearningMeasurement
    # compares these against the held-out test values (PIT-KS) and against oracle
    # true-conditional samples from problem.true_model (sd_ratio). Discrete-only
    # adapters do not set this flag → their calibration rows are not_supported;
    # continuous-capable adapters that have not implemented predictive_samples yet
    # (currently pgmpy-lg, pyro-empirical on continuous) also leave it unset. The
    # caller (the measurement) is responsible for seeding the predictive draws for
    # reproducibility — the method itself is intentionally stochastic.
    supports_calibration: bool = True

    def __init__(
        self,
        mechanism: str,
        engine: str | None,
        device: str | None = None,
        n_samples: int = 1024,
        **kwargs: Any,
    ) -> None:
        if mechanism not in _DISCRETE_MECH:
            raise ValueError(
                f"Unknown mechanism {mechanism!r}. "
                f"Valid: {sorted(_DISCRETE_MECH)}"
            )
        # engine=None is the fit-only / parameter-learning construction (#109):
        # the adapter is fit and scored via score_data, never queried, so it
        # carries no inference engine and an engine-less name (e.g. "nbn-cat").
        if engine is not None and engine not in _ENGINE_SPEC:
            raise ValueError(
                f"Unknown engine {engine!r}. Valid: {sorted(_ENGINE_SPEC)}"
            )
        self.mechanism = mechanism
        self.engine = engine
        # None / "auto" -> cuda-if-available-else-cpu; concrete passes through.
        self.device = torch.device(resolve_device(device))
        self.n_samples = int(n_samples)
        self.name = (
            f"nbn-{mechanism}-{engine}" if engine is not None
            else f"nbn-{mechanism}"
        )

        # State populated by fit()
        self.model: Any | None = None
        self._engine_obj: Any | None = None
        self.problem: BenchmarkProblem | None = None
        # Per-cell methodology flag (#185 follow-up): which proposal the AIS
        # engine actually used — "learned" or "lw_fallback" (low fit-time ESS).
        # None for engines without a learned proposal (ve / lw) and until fit.
        self.proposal_used: str | None = None

    # -------------------------------------------------------------------------
    # Internal mechanism factory — mirrors old NBNAdapter._make_mech()
    # -------------------------------------------------------------------------

    def _make_mech(
        self,
        kind: str,
        k: int | None,
        parent_kinds: list[str],
    ) -> Any:
        """Return the appropriate mechanism for a single node.

        Mirrors nbn/bench/baselines/nbn_adapter.py::NBNAdapter._make_mech()
        and its fit() special-case for discrete-with-continuous-parents.
        """
        from nbn.mechanisms import (
            CategoricalTableMechanism,
            LinearGaussianMechanism,
            MDNMechanism,
            NeuralCategoricalMechanism,
        )

        if kind == "discrete":
            # Special case (mirrored from old adapter):
            # Discrete child with continuous parents → CategoricalTableMechanism
            # uses integer parent indexing and crashes on float inputs;
            # NeuralCategoricalMechanism accepts float parents via its MLP.
            if any(pk == "continuous" for pk in parent_kinds):
                return NeuralCategoricalMechanism(n_classes=k or 2)

            discrete_mech = _DISCRETE_MECH[self.mechanism]
            if discrete_mech == "neural_categorical":
                return NeuralCategoricalMechanism(n_classes=k or 2)
            if discrete_mech == "smoothed_empirical_categorical":
                # Laplace default (alpha=1.0) — the point of the label.
                from nbn.mechanisms import SmoothedEmpiricalCategoricalMechanism
                return SmoothedEmpiricalCategoricalMechanism()
            return CategoricalTableMechanism()

        # Continuous node
        continuous_mech = _CONTINUOUS_MECH[self.mechanism]
        if continuous_mech == "linear_gaussian":
            return LinearGaussianMechanism()
        if continuous_mech == "flow":
            # Lazy import — only fails if zuko is not installed and "flow" is
            # actually requested.  Mirrors old adapter v0.6c-C-1b comment.
            from nbn.mechanisms import NormalizingFlowMechanism
            return NormalizingFlowMechanism()
        # Non-parametric mechanisms (#223 / PR 8) — lazy imports, defaults only
        # (bw_factor=1.0 / k=auto / n_basis=31 + sharpen=1.0; the "auto" tuning
        # flags are opt-in and deferred).
        if continuous_mech == "conditional_kde":
            from nbn.mechanisms.non_parametric.conditional_kde import (
                ConditionalKDEMechanism,
            )
            return ConditionalKDEMechanism()
        if continuous_mech == "knn_conditional":
            from nbn.mechanisms.non_parametric.knn_conditional import (
                KNNConditionalMechanism,
            )
            return KNNConditionalMechanism()
        if continuous_mech == "flexcode":
            from nbn.mechanisms.non_parametric.flexcode import FlexCodeMechanism
            return FlexCodeMechanism()
        # Default: MDN
        return MDNMechanism(num_components=3, hidden=(32,))

    # -------------------------------------------------------------------------
    # BaselineAdapter protocol
    # -------------------------------------------------------------------------

    def fit(self, problem: BenchmarkProblem, **kwargs: Any) -> None:
        """Fit the NBN model on problem.train_data. State stored on self.

        kwargs:
            epochs / batch_size / lr: optional global training-budget
                overrides passed through from the runner config. When absent
                (the default), each mechanism trains with its own designed
                budget (flow 300 epochs @ lr 5e-4, MDN 200, neural-categorical
                100, ...) — the former hardcoded epochs=20 / batch_size=1024 /
                lr=1e-3 silently starved every neural mechanism.

        Mirrors nbn/bench/baselines/nbn_adapter.py::NBNAdapter.fit()
        exactly, including:
          - Per-node mechanism selection via _make_mech()
          - Discrete-with-continuous-parents special case
          - Engine instantiation after model fitting
        """
        import networkx as nx
        from nbn import NeuralBayesianNetwork

        # None = no override: the mechanism keeps its designed budget.
        epochs = kwargs.get("epochs")
        epochs = int(epochs) if epochs is not None else None
        batch_size = kwargs.get("batch_size")
        batch_size = int(batch_size) if batch_size is not None else None
        lr = kwargs.get("lr")
        lr = float(lr) if lr is not None else None

        model = NeuralBayesianNetwork(
            problem.dag,
            variables=problem.variables,
            device=str(self.device),
        )

        # Build a NetworkX graph for parent lookup (same as old adapter)
        dag_nx = nx.DiGraph()
        dag_nx.add_nodes_from(problem.variables)
        dag_nx.add_edges_from(problem.dag)

        for node, (kind, k) in problem.variables.items():
            parent_kinds = [
                problem.variables[p][0] for p in dag_nx.predecessors(node)
            ]
            mech = self._make_mech(kind, k, parent_kinds)
            model.set_mechanism(node, mech)

        # consolidate=False: benchmarks never call model.update(), so the
        # post-fit EWC Fisher pass (up to 4096 sequential per-sample backward
        # passes per neural node) is pure overhead here.
        model.fit(
            problem.train_data,
            epochs=epochs, batch_size=batch_size, lr=lr,
            consolidate=False,
        )
        self.model = model
        self.problem = problem
        self._attach_engine()

    def load_base_and_attach(
        self, model_path: str, problem: BenchmarkProblem | None = None,
        **kwargs: Any,
    ) -> None:
        """Reload a base model fitted by a sibling baseline, then attach this
        baseline's engine — the fit-once-save-reload path (#191, Path 2).

        Baselines sharing a fit-identity (same library/mechanism/epochs/fit-data,
        differing only in ``inference_method``) fit the base ``model.fit()``
        ONCE; the first writes it with ``torch.save``, the rest call this to
        reuse it instead of re-fitting. The reloaded model is bitwise-identical
        to a fresh fit (Stage-1 verified, CPU+CUDA, all mechanisms).

        ``_attach_engine`` is the SAME engine-construction step ``fit`` runs
        after ``model.fit()``, so a reloaded adapter is indistinguishable from a
        freshly-fit one for query purposes. Per Decision A, ve/lw attach with no
        training; ais/avi run their OWN ``train_proposal`` fresh on the reloaded
        base (recognition nets are per-inference-method, never shared).

        ``map_location=self.device`` relocates the saved tensors onto this
        baseline's device (the fit-identity key excludes device for exactly
        this reason). ``weights_only=False`` is required: the payload is a full
        ``nn.Module`` object, not a bare state-dict.

        ``**kwargs`` (e.g. ``epochs``) are accepted for call-site parity with
        ``fit`` but unused here — no base fitting happens on reload.
        """
        self.model = torch.load(
            model_path, map_location=self.device, weights_only=False,
        )
        self.problem = problem
        self._attach_engine()

    def _attach_engine(self) -> None:
        """Construct ``self._engine_obj`` for ``self.engine`` on ``self.model``.

        The per-inference-method half of fitting: everything ``fit`` does AFTER
        ``model.fit()``. Shared verbatim by ``fit`` (fresh) and
        ``load_base_and_attach`` (reloaded base) so the two paths are identical
        downstream. NEVER shared across inference methods — each baseline builds
        its own engine (and, for ais/avi, trains its own proposal).
        """
        # Fit-only / parameter-learning construction (#109): no inference
        # engine was requested (engine=None), so there is nothing to attach.
        # fit() and load_base_and_attach() both reach here; score_data needs
        # only self.model, and query()/query_batch() are never called.
        if self.engine is None:
            self._engine_obj = None
            self.proposal_used = None
            return

        from nbn.inference.amortized_is import AmortizedISEngine
        from nbn.inference.amortized_vi import AmortizedVIEngine
        from nbn.inference.hybrid import HybridRouter
        from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
        from nbn.inference.tensor_ve import TensorVariableElimination

        engine_spec = _ENGINE_SPEC[self.engine]
        if engine_spec == "lw":
            self._engine_obj = LikelihoodWeightingEngine(n_samples=self.n_samples)
        elif engine_spec == "ve":
            self._engine_obj = TensorVariableElimination()
        elif engine_spec == "ais":
            # Amortized IS (#181): train the evidence-conditioned proposal
            # once, here, so it is reused across all query/query_batch calls
            # within the fit-once-query-many cell (compatible with PR #176).
            self._engine_obj = AmortizedISEngine(n_samples=self.n_samples)
            metrics = self._engine_obj.train_proposal(
                self.model, device=str(self.device))
            # Record which proposal survived the fit-time ESS gate so the
            # parquet can report "learned proposal used on N of M cells".
            self.proposal_used = metrics.get("proposal_used")
        elif engine_spec == "avi":
            # Amortized VI (#182): train the variational posterior network
            # once, here, so it is reused across all query/query_batch calls
            # within the fit-once-query-many cell (compatible with PR #176).
            self._engine_obj = AmortizedVIEngine(n_samples=self.n_samples)
            self._engine_obj.train_proposal(self.model, device=str(self.device))
        else:
            self._engine_obj = HybridRouter()

    def _prep_evidence(self, ev: dict) -> dict:
        """Normalise evidence values to [B, D] tensors on self.device.

        Identical to nbn/bench/baselines/nbn_adapter.py::_prep_evidence().
        """
        out = {}
        for k, v in ev.items():
            if v is None:
                # Phase 3 empty mode: drop unobserved evidence so the engine
                # marginalizes over it.
                continue
            t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            if t.dim() == 0:
                t = t.unsqueeze(0)
            out[k] = t.to(self.device)
        return out

    def query(self, q: Query) -> Posterior:
        """Query the fitted model. This call is what gets timed.

        Returns:
            Posterior(probs=...)   for discrete targets — probs vector [K]
            Posterior(samples=...) for continuous targets — weighted-
                resampled LW particles, shape [n_samples]

        The dispatch is on the engine result type (tuple → continuous LW
        path; tensor → discrete / VE path), which mirrors the old adapter's
        ``isinstance(result, tuple)`` check.
        """
        if self.model is None or self._engine_obj is None:
            raise RuntimeError(
                "Adapter not fitted. Call fit() before query()."
            )

        ev = self._prep_evidence(q.evidence)
        # IS engines (lw / ais) report per-query ESS (X1) and PSIS k̂ (X2).
        want_diag = _ENGINE_SPEC[self.engine] in ("lw", "ais")
        result = self._engine_obj.query(
            self.model, list(q.targets), ev,
            **({"return_ess": True, "return_psis_k": True} if want_diag else {}),
        )
        ess: float | None = None
        khat: float | None = None
        if want_diag:
            result, ess_t, khat_t = result    # (payload, ess_frac [B], khat [B])
            ess = float(ess_t.reshape(-1)[0])  # single query → B=1
            khat = _nan_to_none(khat_t.reshape(-1)[0])

        if isinstance(result, tuple):
            # LW continuous path: (weights[1, S], samples[1, S, D])
            # Resample proportional to weights → unweighted posterior samples.
            w, s = result
            w = w.squeeze(0) if w.dim() > 1 else w  # [S]
            s = s.squeeze(0) if s.dim() > 2 else s  # [S, D]
            idx = torch.multinomial(w, num_samples=w.shape[0], replacement=True)
            # Single target → column 0 of the sample tensor
            samples = s[idx, 0].detach().cpu()       # [S]
            return Posterior(samples=samples, ess=ess, khat=khat)

        # Discrete / VE path: probability vector [1, K] or [K]
        probs = result.squeeze(0) if result.dim() > 1 else result
        return Posterior(probs=probs.detach().cpu(), ess=ess, khat=khat)

    def query_batch(self, queries: list[Query]) -> list[Posterior]:
        """Library-level batching via the nbn engine's ``query_batch`` API.

        Assumes all queries share (targets, frozenset(evidence_keys)) —
        the selector contract per design doc §1.2.  Heterogeneous batches
        (different targets or evidence keys, or mixed concrete/None
        evidence values) fall back to the sequential default helper.

        Returns Posteriors in input order:
            Posterior(probs=...)   for discrete targets — [K] per query,
                unpacked from the engine's [B, K] output
            Posterior(samples=...) for continuous targets — per-row
                multinomial resample of the engine's weighted particles
                (the 2-D generalisation of the B=1 path in ``query()``)

        See docs/v0.14-batched-queries-design.md §3.1.
        """
        if not queries:
            return []
        # B=1: single-query path — identical semantics, no stacking overhead.
        if len(queries) == 1:
            return [self.query(queries[0])]
        if self.model is None or self._engine_obj is None:
            raise RuntimeError(
                "Adapter not fitted. Call fit() before query_batch()."
            )

        # Verify the shared (targets, evidence_keys) contract; fall back to
        # the sequential helper on any heterogeneity.
        first = queries[0]
        targets = first.targets
        evidence_keys = frozenset(first.evidence.keys())
        for q in queries[1:]:
            if q.targets != targets:
                return default_query_batch(self, queries)
            if frozenset(q.evidence.keys()) != evidence_keys:
                return default_query_batch(self, queries)

        # Stack evidence per variable into [B, D] tensors on self.device
        # (D=1 for the scalar evidence the selectors emit).  A key whose
        # value is None in every query is dropped — Phase 3 empty mode,
        # same as _prep_evidence(); mixed None/concrete is heterogeneous.
        b = len(queries)
        stacked_evidence: dict[str, torch.Tensor] = {}
        for key in evidence_keys:
            values = [q.evidence[key] for q in queries]
            none_count = sum(v is None for v in values)
            if none_count == len(values):
                continue  # marginalize, as _prep_evidence does for None
            if none_count > 0:
                return default_query_batch(self, queries)  # mixed modes
            stacked_evidence[key] = torch.stack(
                [torch.as_tensor(v).reshape(-1) for v in values]
            ).to(self.device)  # [B, D]

        # All-empty-mode batch (every evidence value None — e.g. the
        # heaviest selector's V2 queries): the engines infer B from the
        # evidence tensors, so with {} they'd answer a single marginal
        # ([1, K]) for a B-query batch. No batched library path exists
        # for evidence-free queries — fall back to sequential. (Rows
        # still stamp batch_size=B; filter on evidence_mode when
        # analyzing batched-speedup figures.)
        if not stacked_evidence:
            return default_query_batch(self, queries)

        want_diag = _ENGINE_SPEC[self.engine] in ("lw", "ais")
        result = self._engine_obj.query_batch(
            self.model, list(targets), stacked_evidence,
            **({"return_ess": True, "return_psis_k": True} if want_diag else {}),
        )
        ess_b: torch.Tensor | None = None
        khat_b: torch.Tensor | None = None
        if want_diag:
            result, ess_b, khat_b = result   # (payload, ess_frac [B], khat [B])
            ess_b = ess_b.reshape(-1)
            khat_b = khat_b.reshape(-1)

        if isinstance(result, tuple):
            # LW continuous path: (weights [B, S], samples [B, S, D])
            return self._unpack_lw_continuous_batch(*result, ess_b=ess_b, khat_b=khat_b)

        # Discrete / VE path: probability matrix [B, K]
        if result.dim() != 2 or result.shape[0] != b:
            raise RuntimeError(
                f"Unexpected engine.query_batch output for {self.name}: "
                f"shape {tuple(result.shape)}, expected [{b}, K]."
            )
        return [
            Posterior(
                probs=result[i].detach().cpu(),
                ess=(float(ess_b[i]) if ess_b is not None else None),
                khat=(_nan_to_none(khat_b[i]) if khat_b is not None else None),
            )
            for i in range(b)
        ]

    def _unpack_lw_continuous_batch(
        self,
        weights: torch.Tensor,   # [B, S], softmax-normalised per row
        samples: torch.Tensor,   # [B, S, D]
        ess_b: torch.Tensor | None = None,   # [B] per-query ESS fraction, or None
        khat_b: torch.Tensor | None = None,  # [B] per-query PSIS k̂ (NaN→None)
    ) -> list[Posterior]:
        """Per-row multinomial resampling for batched LW continuous output.

        The engine returns per-row weighted particle pools (V1 finding:
        B×S independent draws, per-row weights).  ``torch.multinomial``
        accepts 2-D input and resamples each row independently — this is
        the [B, S] generalisation of the B=1 resample in ``query()``.
        """
        n_particles = weights.shape[1]
        idx = torch.multinomial(
            weights, num_samples=n_particles, replacement=True
        )  # [B, S]
        # Single target → column 0 of the sample tensor (as in query()).
        resampled = torch.gather(samples[..., 0], 1, idx)  # [B, S]
        return [
            Posterior(
                samples=resampled[i].detach().cpu(),
                ess=(float(ess_b[i]) if ess_b is not None else None),
                khat=(_nan_to_none(khat_b[i]) if khat_b is not None else None),
            )
            for i in range(resampled.shape[0])
        ]

    def is_applicable(self, problem: BenchmarkProblem) -> bool:
        """Return True if this adapter can handle problem.family.

        Delegates to the existing _BASELINE_APPLICABILITY table using
        self.name as the lookup key.

        Family derivation: BenchmarkProblem has no .family attribute in
        Phase 1b; family is inferred from variable kinds:
          - all discrete         → "discrete"
          - all continuous       → "continuous_lg" (approximation; cannot
                                   distinguish lg from nongauss from kinds
                                   alone — but all continuous adapters that
                                   apply to continuous_lg also apply to
                                   continuous_nongauss, so no false negatives)
          - mixed                → "hybrid"

        Phase 1c will move applicability fully per-adapter, removing the
        dependency on the static table.
        """
        # Derive family from variable kinds
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

    def score_data(self, test_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-row joint log-prob ``[B]`` of held-out rows (param-learning, #109).

        Assembles the joint log-likelihood the same way ``nbn.learning.fit``
        forms its training loss: walk the DAG in topological order and sum each
        node's conditional ``mechanism.log_prob(x, parents)`` (the 2-arg
        ``(x, parents)`` overload — NOT the 1-arg ``log_prob(value)`` some
        mechanisms also expose). Per the locked convention (#109): the per-row
        joint is the SUM over nodes; ``ParamLearningMeasurement`` then takes the
        MEAN over rows via ``metrics.log_likelihood``. Zero-probability handling
        is left to each mechanism's source-of-truth floor (e.g. the categorical
        CPT clamps at ``log(1e-12)``); no second clamp here.

        ``test_data`` columns are moved onto the model's device first (mirrors
        ``fit``). ``pack_parents`` gathers each node's parent columns into a
        ``[B, D_pa]`` tensor (``None`` for roots), exactly as the fit loop does.

        Discrete node values are guarded against the declared cardinality before
        scoring, so an out-of-support test row raises a clean ``ValueError``
        (classified ``status="error"`` by the measurement) rather than an opaque
        index error from the categorical CPT lookup. Topological order
        guarantees a node is validated before any child uses it as a parent, so
        out-of-range discrete parents are caught too.

        Returns
        -------
        torch.Tensor
            Shape ``[B]`` — one joint log-probability per test row.
        """
        if self.model is None:
            raise RuntimeError(
                "Adapter not fitted. Call fit() before score_data()."
            )
        from nbn.utils.batching import pack_parents

        # Move held-out columns onto the model's device (mirrors fit()).
        data = {
            k: torch.as_tensor(v).to(self.device) for k, v in test_data.items()
        }

        node_lps: list[torch.Tensor] = []
        for node in self.model.dag.topological_order():
            mech = self.model.mechanisms[node]
            x = data[node]

            # Guard discrete values against the declared cardinality.
            var = self.model.variables.get(node)
            if var is not None and var.is_discrete and var.cardinality is not None:
                k = int(var.cardinality)
                idx = x.long()
                if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= k):
                    raise ValueError(
                        f"score_data: discrete node {node!r} has a test value "
                        f"out of range [0, {k}); cannot score out-of-support rows."
                    )

            parents = self.model.dag.parents(node)
            pa_tensor = pack_parents(data, parents)   # [B, D_pa] or None
            lp = mech.log_prob(x, pa_tensor)          # [B]
            node_lps.append(lp.reshape(-1))

        # Per-row joint = sum over nodes -> [B]. Detached + on CPU to match the
        # query() path's return convention (the metric only needs the values).
        return torch.stack(node_lps, dim=0).sum(dim=0).detach().cpu()

    def extract_learned_cpts(self) -> dict[str, torch.Tensor]:
        """Learned discrete CPTs in the canonical layout (param-recovery, #109).

        Delegates to the SHARED extractor (``nbn.bench.core.cpt_extraction``)
        so the learned tables use the exact same enumeration and column-order
        handling the measurement applies to ``problem.true_model`` — that is
        what lets true and learned CPTs compare cell-by-cell. The extractor
        builds each ``[n_parent_configs, K]`` table via batched
        ``mech.forward(configs).probs`` (works for both ``cat`` table CPTs and
        ``neuralcat`` MLPs), and OMITS continuous nodes and discrete nodes with
        any continuous parent (recovery is a fully-discrete-network metric).

        ``self.problem.variables`` (stored at fit) supplies node kinds and
        cardinalities — the adapter already retains the fitting problem, so no
        extra plumbing from the measurement is needed.
        """
        if self.model is None or self.problem is None:
            raise RuntimeError(
                "Adapter not fitted. Call fit() before extract_learned_cpts()."
            )
        from nbn.bench.core.cpt_extraction import extract_discrete_cpts

        return extract_discrete_cpts(self.model, self.problem.variables)

    # Predictive samples per continuous node for calibration (#109 PR 7). S
    # matches the calibration_diagnostic harness default.
    N_CALIBRATION_SAMPLES: int = 400

    def predictive_samples(
        self, test_data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Predictive samples per CONTINUOUS node for calibration (#109 PR 7).

        Returns ``{node: samples[N, S]}`` for each continuous node, where
        ``samples[i]`` are ``S = N_CALIBRATION_SAMPLES`` draws from the fitted
        mechanism's predictive distribution conditioned on test row ``i``'s
        parent values. Discrete nodes are OMITTED — calibration is a
        continuous-node metric (PIT is defined on continuous predictive CDFs).

        Root continuous nodes (no parents) have one unconditional predictive
        distribution; the returned ``[N_test, S]`` is the same S samples expanded
        across all test rows (``mech.sample(parents=None, n=S)`` returns
        ``[1, S, D]``). Non-root nodes get N_test independent predictive
        distributions, one per test row's parent assignment. This is the correct
        semantics for both — root predictives are inherently row-invariant;
        non-root predictives are row-conditional — and matters because PIT-KS at
        a root node then asks "do the y values look like draws from this one
        unconditional predictive?" rather than "are the y values each calibrated
        against their own predictive."

        Delegates to ``continuous_predictive_samples`` — the SAME helper the
        measurement uses to draw oracle samples from ``true_model`` — so the
        fitted (PIT-KS / sharpness numerator) and oracle (sd_ratio denominator)
        draws are sampled by identical logic. INTENTIONALLY STOCHASTIC: the
        caller (ParamLearningMeasurement) wraps this in a seeded ``fork_rng``
        scope for reproducible parquet values; calling it bare gives fresh draws.
        """
        if self.model is None or self.problem is None:
            raise RuntimeError(
                "Adapter not fitted. Call fit() before predictive_samples()."
            )
        from nbn.bench.core.predictive_sampling import (
            continuous_predictive_samples,
        )

        return continuous_predictive_samples(
            self.model, self.problem.variables, test_data,
            self.N_CALIBRATION_SAMPLES, self.device,
        )
