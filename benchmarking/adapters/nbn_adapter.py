"""NBN adapter for v0.13 BaselineAdapter protocol.

Stateful fit-then-query contract. State stored on self:
  - self.model:      the fitted NeuralBayesianNetwork
  - self._engine_obj: the fitted inference engine
  - self.problem:    the BenchmarkProblem used for fitting

Mirrors the logic of benchmarking/baselines/nbn_adapter.py with three
structural changes:
  1. Stateful: fit() stores on self instead of returning.
  2. Constructor uses unified ``mechanism`` + ``engine`` args (instead of
     discrete_mech / continuous_mech + engine).
  3. query() returns Posterior instead of a raw torch.Tensor.

The old adapter at benchmarking/baselines/nbn_adapter.py is untouched
and continues to serve the v0.12 runner.

Reference: docs/v0.13-benchmark-redesign.md §4.1
Old (functional) adapter: benchmarking/baselines/nbn_adapter.py
"""
from __future__ import annotations

from typing import Any

import torch

from benchmarking.core._device import resolve_device
from benchmarking.core.applicability import BASELINE_FAMILY_APPLICABILITY as _BASELINE_APPLICABILITY
from benchmarking.core.interfaces import default_query_batch
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.domains.posterior import Posterior

# ---- mechanism → internal specifiers -----------------------------------------
# Maps the unified `mechanism` arg to the old adapter's internal fields.

# Which discrete mechanism to use for nodes with kind="discrete"
_DISCRETE_MECH: dict[str, str] = {
    "cat":       "categorical_table",
    "neuralcat": "neural_categorical",
    "lg":        "categorical_table",   # lg is continuous-only; discrete fallback
    "mdn":       "categorical_table",
    "flow":      "categorical_table",
    "hybrid":    "categorical_table",   # HybridRouter default
}

# Which continuous mechanism to use for nodes with kind="continuous"
_CONTINUOUS_MECH: dict[str, str] = {
    "cat":       "mdn",                 # cat is discrete-only; continuous fallback
    "neuralcat": "mdn",
    "lg":        "linear_gaussian",
    "mdn":       "mdn",
    "flow":      "flow",
    "hybrid":    "mdn",                 # HybridRouter default
}

# Maps the `engine` arg to the engine_spec string used internally
_ENGINE_SPEC: dict[str, str] = {
    "lw":     "lw",
    "ve":     "ve",
    "router": "hybrid",  # HybridRouter
}


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

    def __init__(
        self,
        mechanism: str,
        engine: str,
        device: str | None = None,
        n_samples: int = 1024,
        **kwargs: Any,
    ) -> None:
        if mechanism not in _DISCRETE_MECH:
            raise ValueError(
                f"Unknown mechanism {mechanism!r}. "
                f"Valid: {sorted(_DISCRETE_MECH)}"
            )
        if engine not in _ENGINE_SPEC:
            raise ValueError(
                f"Unknown engine {engine!r}. Valid: {sorted(_ENGINE_SPEC)}"
            )
        self.mechanism = mechanism
        self.engine = engine
        # None / "auto" -> cuda-if-available-else-cpu; concrete passes through.
        self.device = torch.device(resolve_device(device))
        self.n_samples = int(n_samples)
        self.name = f"nbn-{mechanism}-{engine}"

        # State populated by fit()
        self.model: Any | None = None
        self._engine_obj: Any | None = None
        self.problem: BenchmarkProblem | None = None

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

        Mirrors benchmarking/baselines/nbn_adapter.py::NBNAdapter._make_mech()
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
        # Default: MDN
        return MDNMechanism(num_components=3, hidden=(32,))

    # -------------------------------------------------------------------------
    # BaselineAdapter protocol
    # -------------------------------------------------------------------------

    def fit(self, problem: BenchmarkProblem, **kwargs: Any) -> None:
        """Fit the NBN model on problem.train_data. State stored on self.

        kwargs:
            epochs (int): training epochs (default 20). Passed through from
                the runner; other hyperparameters (batch_size=512, lr=1e-3)
                are hardcoded to match the v0.12 paper run configuration.

        Mirrors benchmarking/baselines/nbn_adapter.py::NBNAdapter.fit()
        exactly, including:
          - Per-node mechanism selection via _make_mech()
          - Discrete-with-continuous-parents special case
          - Engine instantiation after model fitting
        """
        import networkx as nx
        from nbn import NeuralBayesianNetwork
        from nbn.inference.hybrid import HybridRouter
        from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
        from nbn.inference.tensor_ve import TensorVariableElimination

        epochs = int(kwargs.get("epochs", 20))

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

        model.fit(problem.train_data, epochs=epochs, batch_size=512, lr=1e-3)
        self.model = model
        self.problem = problem

        engine_spec = _ENGINE_SPEC[self.engine]
        if engine_spec == "lw":
            self._engine_obj = LikelihoodWeightingEngine(n_samples=self.n_samples)
        elif engine_spec == "ve":
            self._engine_obj = TensorVariableElimination()
        else:
            self._engine_obj = HybridRouter()

    def _prep_evidence(self, ev: dict) -> dict:
        """Normalise evidence values to [B, D] tensors on self.device.

        Identical to benchmarking/baselines/nbn_adapter.py::_prep_evidence().
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
        result = self._engine_obj.query(self.model, list(q.targets), ev)

        if isinstance(result, tuple):
            # LW continuous path: (weights[1, S], samples[1, S, D])
            # Resample proportional to weights → unweighted posterior samples.
            w, s = result
            w = w.squeeze(0) if w.dim() > 1 else w  # [S]
            s = s.squeeze(0) if s.dim() > 2 else s  # [S, D]
            idx = torch.multinomial(w, num_samples=w.shape[0], replacement=True)
            # Single target → column 0 of the sample tensor
            samples = s[idx, 0].detach().cpu()       # [S]
            return Posterior(samples=samples)

        # Discrete / VE path: probability vector [1, K] or [K]
        probs = result.squeeze(0) if result.dim() > 1 else result
        return Posterior(probs=probs.detach().cpu())

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

        result = self._engine_obj.query_batch(
            self.model, list(targets), stacked_evidence
        )

        if isinstance(result, tuple):
            # LW continuous path: (weights [B, S], samples [B, S, D])
            return self._unpack_lw_continuous_batch(*result)

        # Discrete / VE path: probability matrix [B, K]
        if result.dim() != 2 or result.shape[0] != b:
            raise RuntimeError(
                f"Unexpected engine.query_batch output for {self.name}: "
                f"shape {tuple(result.shape)}, expected [{b}, K]."
            )
        return [Posterior(probs=result[i].detach().cpu()) for i in range(b)]

    def _unpack_lw_continuous_batch(
        self,
        weights: torch.Tensor,   # [B, S], softmax-normalised per row
        samples: torch.Tensor,   # [B, S, D]
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
            Posterior(samples=resampled[i].detach().cpu())
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
