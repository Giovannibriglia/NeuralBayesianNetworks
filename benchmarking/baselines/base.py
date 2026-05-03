"""Base interface every baseline adapter must implement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch

from benchmarking.domains.base import BenchmarkProblem, Query


class BaselineAdapter(ABC):
    """A uniform interface over external probabilistic libraries.

    The runner calls ``fit(problem)`` once, then iterates ``query(q)`` over
    the problem's query battery.  ``teardown()`` is called for cleanup
    (e.g. releasing GPU memory).

    Subclasses declare which query kinds they ``supports``; unsupported
    queries are skipped and recorded as ``not_supported`` in the result.
    """

    name: str = "abstract"
    supports: set[Literal["discrete", "continuous", "hybrid", "do", "batched"]] = set()

    @abstractmethod
    def fit(self, problem: BenchmarkProblem) -> None: ...

    @abstractmethod
    def query(self, q: Query) -> torch.Tensor: ...

    def query_batch(self, q: Query) -> torch.Tensor:
        """Default: loop over the batch dim and stack."""
        ev = q.evidence
        b = max((v.shape[0] if isinstance(v, torch.Tensor) and v.dim() >= 1 else 1
                 for v in ev.values()), default=1)
        rows = []
        for i in range(b):
            row_ev = {
                k: (v[i] if isinstance(v, torch.Tensor) and v.dim() >= 1 else v)
                for k, v in ev.items()
            }
            rows.append(self.query(Query(targets=q.targets, evidence=row_ev, kind=q.kind)))
        return torch.stack(rows, dim=0)

    def teardown(self) -> None:  # noqa: B027 — intentional empty default
        pass
