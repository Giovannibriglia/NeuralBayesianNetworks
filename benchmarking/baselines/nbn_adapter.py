"""NBN's own adapter — wraps the in-house engines for a uniform runner."""
from __future__ import annotations

import torch

from nbn import NeuralBayesianNetwork
from benchmarking.baselines.base import BaselineAdapter
from benchmarking.domains.base import BenchmarkProblem, Query
from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
from nbn.inference.tensor_ve import TensorVariableElimination
from nbn.mechanisms import (
    CategoricalTableMechanism,
    MDNMechanism,
)


class NBNAdapter(BaselineAdapter):
    """Uses ``TensorVariableElimination`` for discrete and
    ``LikelihoodWeightingEngine`` for everything else.
    """

    name = "nbn"
    supports = {"discrete", "continuous", "hybrid", "do", "batched"}

    def __init__(self, device: str = "cpu", n_samples: int = 1024) -> None:
        self.device = torch.device(device)
        self.n_samples = n_samples
        self.model: NeuralBayesianNetwork | None = None
        self._engine = None

    def fit(self, problem: BenchmarkProblem) -> None:
        model = NeuralBayesianNetwork(
            problem.dag, variables=problem.variables, device=str(self.device),
        )
        for node, var_spec in problem.variables.items():
            kind, _ = var_spec
            if kind == "discrete":
                mech = CategoricalTableMechanism()
            else:
                mech = MDNMechanism(num_components=3, hidden=(32,))
            model.set_mechanism(node, mech)
        model.fit(problem.train_data, epochs=20, batch_size=512, lr=1e-3)
        self.model = model
        # Pick engine
        all_disc = all(k == "discrete" for k, _ in problem.variables.values())
        self._engine = (
            TensorVariableElimination() if all_disc
            else LikelihoodWeightingEngine(n_samples=self.n_samples)
        )

    def query(self, q: Query) -> torch.Tensor:
        assert self.model is not None and self._engine is not None
        ev = {
            k: (v if isinstance(v, torch.Tensor)
                else torch.tensor(v).to(self.device))
            for k, v in q.evidence.items()
        }
        result = self._engine.query(self.model, list(q.targets), ev)
        if isinstance(result, tuple):
            # (weights, samples) — reduce to mean
            w, s = result
            return (w.unsqueeze(-1) * s).sum(dim=-2).squeeze(0).detach().cpu()
        return result.detach().cpu()

    def teardown(self) -> None:
        self.model = None
        self._engine = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
