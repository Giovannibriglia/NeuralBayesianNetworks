"""Plugin contract for benchmark domains.

A *domain* is one family of test problems.  v0.5 ships a single
synthetic-BN generator (``benchmarking.synthetic.make_synthetic_bn``);
these types are kept so adapters and user code can construct
``BenchmarkProblem`` / ``Query`` instances directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping

import torch


@dataclass(frozen=True)
class GroundTruth:
    """Reference answers for a query battery.

    Either *analytic* (when the true BN is known and an exact engine
    such as pgmpy's ``VariableElimination`` provides the truth) or
    *empirical* (Monte Carlo from the true generative process —
    ``SyntheticBN.ground_truth_samples``).
    """

    marginals: dict[str, torch.Tensor] = field(default_factory=dict)
    conditionals: dict[tuple, torch.Tensor] = field(default_factory=dict)
    samples: torch.Tensor | None = None


@dataclass(frozen=True)
class Query:
    """A single benchmark query.

    Parameters
    ----------
    targets:
        Target node names.
    evidence:
        Observed values (may be batched: ``[B]``).
    kind:
        ``'marginal'`` | ``'conditional'`` | ``'map'`` | ``'sample'`` | ``'do'``.
    """

    targets: tuple[str, ...]
    evidence: Mapping[str, int | float | torch.Tensor]
    kind: str = "marginal"


@dataclass
class BenchmarkProblem:
    """One concrete benchmark instance: DAG + data + queries + (optional) truth."""

    name: str
    dag: list[tuple[str, str]]
    variables: dict[str, tuple[str, int]]   # ('discrete', K) | ('continuous', D)
    train_data: dict[str, torch.Tensor]
    test_data: dict[str, torch.Tensor]
    queries: list[Query]
    ground_truth: GroundTruth | None = None


class BenchmarkDomain(ABC):
    """A domain = one family of test problems.

    Subclasses must implement ``list_problems`` and ``load_problem``.
    The standard query battery (5 kinds) should be emitted by ``load_problem``
    via ``benchmarking.queries.make_query_battery``.
    """

    name: str = "abstract"

    @abstractmethod
    def list_problems(self) -> list[str]: ...

    @abstractmethod
    def load_problem(
        self,
        problem: str,
        *,
        n_train: int,
        n_test: int,
        seed: int,
        device: torch.device,
    ) -> BenchmarkProblem: ...
