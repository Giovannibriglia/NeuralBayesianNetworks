"""Plugin contract for benchmark domains.

A *domain* is one family of test problems.  v0.5 ships a single
synthetic-BN generator (``benchmarking.synthetic.make_synthetic_bn``);
these types are kept so adapters and user code can construct
``BenchmarkProblem`` / ``Query`` instances directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

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
    """One concrete benchmark instance: DAG + data + queries + (optional) truth.

    v0.13 additions (Phase 1b-iv, all additive with defaults — no existing code breaks):

    ``true_model``
        The data-generating model instance (typically a
        ``NeuralBayesianNetwork``).  Required by ``AccuracyAndTiming`` for
        continuous and hybrid oracle sampling via
        ``true_model.sample(n=N, evidence=ev)``.  Set by
        ``SyntheticProblemSource``; ``None`` for problem sources that do not
        expose a ground-truth model (e.g. bnlearn — Phase 4).  When
        ``true_model`` is ``None``, ``AccuracyAndTiming`` skips continuous
        oracle rows and emits ``status="not_supported"``.

    ``family``
        BN family string matching the v0.13 parquet schema:
        ``"discrete"`` | ``"continuous_lg"`` | ``"continuous_nongauss"`` |
        ``"hybrid"``.  Populated by problem sources; empty string for
        problems constructed without a source (e.g. unit-test fixtures).

    ``problem_id``
        Short identifier used as the ``problem_id`` column in the v0.13
        parquet.  For synthetic problems: ``str(n_nodes)`` (e.g. ``"10"``).
        For bnlearn problems: network name (e.g. ``"asia"``).  Populated by
        problem sources; falls back to ``name`` when empty.

    Reference: docs/v0.13-benchmark-redesign.md §4.1, §6.2
    """

    name: str
    dag: list[tuple[str, str]]
    variables: dict[str, tuple[str, int]]   # ('discrete', K) | ('continuous', D)
    train_data: dict[str, torch.Tensor]
    test_data: dict[str, torch.Tensor]
    queries: list[Query]
    ground_truth: GroundTruth | None = None
    # v0.13 additions — all optional with safe defaults
    true_model: Any | None = None
    family: str = ""
    problem_id: str = ""
    seed: int = 0


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
