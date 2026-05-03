"""pgmpy adapter — wraps VariableElimination for discrete BNs."""
from __future__ import annotations

import torch

from benchmarking.baselines.base import BaselineAdapter
from benchmarking.domains.base import BenchmarkProblem, Query


class PgmpyAdapter(BaselineAdapter):
    """Wraps ``pgmpy.inference.VariableElimination``. Discrete only."""

    name = "pgmpy"
    supports = {"discrete"}

    def __init__(self) -> None:
        self.model = None
        self.infer = None

    def fit(self, problem: BenchmarkProblem) -> None:
        try:
            from pgmpy.inference import VariableElimination
            from pgmpy.models import DiscreteBayesianNetwork
        except ImportError as e:
            raise ImportError("PgmpyAdapter needs pgmpy: pip install pgmpy") from e
        import warnings
        import pandas as pd
        # Discrete only
        for k, (kind, _) in problem.variables.items():
            if kind != "discrete":
                raise ValueError(f"pgmpy can only handle discrete vars; got {k} ({kind}).")
        df = pd.DataFrame({
            k: v.cpu().long().reshape(-1).numpy()
            for k, v in problem.train_data.items()
        })
        bn = DiscreteBayesianNetwork(problem.dag)
        # API drift across pgmpy versions:
        # 1. pgmpy >= 1.1 wants `pgmpy.parameter_estimator.DiscreteMLE()`.
        # 2. older pgmpy accepts the `MaximumLikelihoodEstimator` class.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            try:
                from pgmpy.parameter_estimator import DiscreteMLE
                bn.fit(df, estimator=DiscreteMLE())
            except (ImportError, TypeError):
                from pgmpy.estimators import MaximumLikelihoodEstimator
                bn.fit(df, estimator=MaximumLikelihoodEstimator)
        self.model = bn
        self.infer = VariableElimination(bn)

    def query(self, q: Query) -> torch.Tensor:
        assert self.infer is not None
        ev = {k: int(v.item() if isinstance(v, torch.Tensor) else v)
              for k, v in q.evidence.items()}
        result = self.infer.query(variables=list(q.targets), evidence=ev,
                                   show_progress=False)
        # Return values as a torch tensor
        return torch.from_numpy(result.values).float()

    def teardown(self) -> None:
        self.model = None
        self.infer = None
