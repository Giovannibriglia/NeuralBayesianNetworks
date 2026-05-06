"""NBN's own adapter — wraps every public NBN inference engine + mechanism.

This is intentionally configurable so that ``benchmarking`` and
``examples/crash_test.py`` can compare *all* NBN methods side-by-side
instead of only the default routing.  Concrete preset variants are also
exported and registered in ``benchmarking/baselines/__init__.py``:

    nbn                    — auto-routing default (HybridRouter)
    nbn_ve                 — TensorVariableElimination
    nbn_lw                 — LikelihoodWeightingEngine
    nbn_hybrid             — HybridRouter (alias of nbn for clarity)
    nbn_neural_categorical — discrete: NeuralCategoricalMechanism
    nbn_linear_gaussian    — continuous: LinearGaussianMechanism
"""
from __future__ import annotations

from typing import Literal

import torch

from benchmarking.baselines.base import BaselineAdapter
from benchmarking.domains.base import BenchmarkProblem, Query
from nbn import NeuralBayesianNetwork
from nbn.inference.hybrid import HybridRouter
from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
from nbn.inference.tensor_ve import TensorVariableElimination
from nbn.mechanisms import (
    CategoricalTableMechanism,
    LinearGaussianMechanism,
    MDNMechanism,
    NeuralCategoricalMechanism,
)


DiscreteMech = Literal["categorical_table", "neural_categorical"]
# v0.6c-C-1b: ``flow`` added for ``nbn-flow-lw`` (NormalizingFlow
# mechanism, requires ``zuko``).  Kept conditional so the adapter
# doesn't hard-fail on environments where zuko isn't installed.
ContinuousMech = Literal["mdn", "linear_gaussian", "flow"]


class NBNAdapter(BaselineAdapter):
    """Configurable NBN adapter.

    Parameters
    ----------
    device:
        Where to construct the model.
    engine:
        ``"auto"`` / ``"hybrid"`` (HybridRouter — default), ``"ve"``
        (TensorVariableElimination), ``"lw"`` (LikelihoodWeightingEngine).
    discrete_mech:
        ``"categorical_table"`` (default) or ``"neural_categorical"``.
    continuous_mech:
        ``"mdn"`` (default) or ``"linear_gaussian"``.
    n_samples:
        Forwarded to ``LikelihoodWeightingEngine`` when ``engine='lw'``.
    """

    supports = {"discrete", "continuous", "hybrid", "do", "batched"}

    def __init__(
        self,
        device: str = "cpu",
        engine: str = "auto",
        discrete_mech: DiscreteMech = "categorical_table",
        continuous_mech: ContinuousMech = "mdn",
        n_samples: int = 1024,
    ) -> None:
        self.device = torch.device(device)
        self.engine_spec = engine
        self.discrete_mech = discrete_mech
        self.continuous_mech = continuous_mech
        self.n_samples = int(n_samples)
        self.name = "nbn" if engine in {"auto", "hybrid"} else f"nbn_{engine}"
        self.model: NeuralBayesianNetwork | None = None
        self._engine = None

    # --------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------

    def _make_mech(self, kind: str, k: int):
        if kind == "discrete":
            if self.discrete_mech == "neural_categorical":
                return NeuralCategoricalMechanism(n_classes=k)
            return CategoricalTableMechanism()
        if self.continuous_mech == "linear_gaussian":
            return LinearGaussianMechanism()
        if self.continuous_mech == "flow":
            # v0.6c-C-1b: NormalizingFlow mechanism (requires zuko).
            # Imported lazily so this branch only fails when ``flow`` is
            # actually requested.
            from nbn.mechanisms import NormalizingFlowMechanism
            return NormalizingFlowMechanism()
        return MDNMechanism(num_components=3, hidden=(32,))

    def fit(self, problem: BenchmarkProblem) -> None:
        model = NeuralBayesianNetwork(
            problem.dag, variables=problem.variables, device=str(self.device),
        )
        for node, (kind, k) in problem.variables.items():
            model.set_mechanism(node, self._make_mech(kind, k))
        model.fit(problem.train_data, epochs=20, batch_size=512, lr=1e-3)
        self.model = model

        if self.engine_spec == "lw":
            self._engine = LikelihoodWeightingEngine(n_samples=self.n_samples)
        elif self.engine_spec == "ve":
            self._engine = TensorVariableElimination()
        else:
            self._engine = HybridRouter()

    def _prep_evidence(self, ev) -> dict:
        """Normalise evidence values to ``[B, D]`` tensors on ``self.device``."""
        out = {}
        for k, v in ev.items():
            t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            if t.dim() == 0:
                t = t.unsqueeze(0)
            out[k] = t.to(self.device)
        return out

    def query(self, q: Query) -> torch.Tensor:
        assert self.model is not None and self._engine is not None
        ev = self._prep_evidence(q.evidence)
        result = self._engine.query(self.model, list(q.targets), ev)
        if isinstance(result, tuple):
            # (weights, samples) — reduce to weighted mean
            w, s = result
            return (w.unsqueeze(-1) * s).sum(dim=-2).squeeze(0).detach().cpu()
        return result.detach().cpu()

    def query_batch(self, q: Query) -> torch.Tensor:
        """Batched query — exercises the engine's native batched path.

        Evidence may be ``[B]`` (1-D scalar batch) or ``[B, D]``. We reshape
        scalar-per-batch into ``[B, 1]`` because the inference engines
        operate on per-node ``[B, D_node]`` tensors.
        """
        assert self.model is not None and self._engine is not None
        ev = {}
        for k, v in q.evidence.items():
            t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            if t.dim() == 1:
                t = t.unsqueeze(-1)  # [B] -> [B, 1]
            elif t.dim() == 0:
                t = t.view(1, 1)
            ev[k] = t.to(self.device)
        result = self._engine.query_batch(self.model, list(q.targets), ev)
        if isinstance(result, tuple):
            w, s = result
            return (w.unsqueeze(-1) * s).sum(dim=-2).detach().cpu()
        return result.detach().cpu()

    def query_batch_samples(
        self, q: Query, n_samples: int = 2000,
    ) -> torch.Tensor:
        """v0.7-#35: batched posterior sampling via the engine's native
        batched path.  Returns ``[B, n_samples, n_targets]``.

        Single ``engine.query_batch(...)`` call over the full ``[B, ...]``
        evidence tensor; the per-row sampling step is a GPU-vectorised
        ``torch.distributions.Categorical`` (discrete) or
        ``torch.multinomial`` resample (LW continuous), so the wall-clock
        is dominated by the engine's batched pass — which is the
        architectural advantage we want the paper figure to measure.

        Note on LW continuous resampling: when ``n_samples >
        engine.n_samples``, resampling is with-replacement from a finite
        proposal pool of size ``engine.n_samples``.  For unbiased MC at
        higher sample counts, increase ``nbn_lw_n_samples`` at engine-init
        time (config knob).  In the runner's call site
        (``_time_nbn_inference``), ``n_samples == engine.n_samples`` by
        construction, so the resampling reduces to permutation-with-
        replacement.
        """
        assert self.model is not None and self._engine is not None
        ev = {}
        for k, v in q.evidence.items():
            t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            elif t.dim() == 0:
                t = t.view(1, 1)
            ev[k] = t.to(self.device)
        result = self._engine.query_batch(self.model, list(q.targets), ev)

        if isinstance(result, tuple):
            # LW continuous: (weights[B, S], samples[B, S, D]).
            weights, samples = result
            idx = torch.multinomial(
                weights, num_samples=n_samples, replacement=True,
            )                                                   # [B, n_samples]
            gathered = samples.gather(
                1, idx.unsqueeze(-1).expand(-1, -1, samples.size(-1)),
            )                                                   # [B, n_samples, D]
            return gathered.detach().cpu()

        # Discrete: result is [B, K] probability vector.
        probs = result if result.dim() == 2 else result.unsqueeze(0)
        cat = torch.distributions.Categorical(probs=probs)
        drawn = cat.sample((n_samples,))                        # [n_samples, B]
        return drawn.T.unsqueeze(-1).float().detach().cpu()      # [B, n_samples, 1]

    def teardown(self) -> None:
        self.model = None
        self._engine = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Concrete preset variants
# ---------------------------------------------------------------------

class NBNTensorVEAdapter(NBNAdapter):
    """NBN with `TensorVariableElimination` (exact for discrete networks)."""
    def __init__(self, device: str = "cpu", **kw):
        kw.setdefault("discrete_mech", "categorical_table")
        kw.setdefault("continuous_mech", "mdn")
        super().__init__(device=device, engine="ve", **kw)
        self.name = "nbn_ve"


class NBNLikelihoodWeightingAdapter(NBNAdapter):
    """NBN with `LikelihoodWeightingEngine` (Monte-Carlo IS, hybrid-friendly)."""
    def __init__(self, device: str = "cpu", n_samples: int = 1024, **kw):
        super().__init__(device=device, engine="lw", n_samples=n_samples, **kw)
        self.name = "nbn_lw"


class NBNHybridRouterAdapter(NBNAdapter):
    """NBN with `HybridRouter` — auto-picks VE for small treewidth, LW otherwise."""
    def __init__(self, device: str = "cpu", **kw):
        super().__init__(device=device, engine="hybrid", **kw)
        self.name = "nbn_hybrid"


class NBNNeuralCategoricalAdapter(NBNAdapter):
    """NBN with `NeuralCategoricalMechanism` (MLP+embedding categorical CPDs)."""
    def __init__(self, device: str = "cpu", **kw):
        kw.setdefault("continuous_mech", "mdn")
        super().__init__(
            device=device, engine="hybrid",
            discrete_mech="neural_categorical", **kw,
        )
        self.name = "nbn_neural_categorical"


class NBNLinearGaussianAdapter(NBNAdapter):
    """NBN with `LinearGaussianMechanism` (closed-form ridge for continuous)."""
    def __init__(self, device: str = "cpu", **kw):
        kw.setdefault("discrete_mech", "categorical_table")
        super().__init__(
            device=device, engine="hybrid",
            continuous_mech="linear_gaussian", **kw,
        )
        self.name = "nbn_linear_gaussian"
