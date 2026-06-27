"""Shared types for incremental, no-rehearsal CPD updates.

The :mod:`nbn.update` package lets an already-fitted network fold in new
data with three constraints held throughout:

* **No rehearsal** — ``update`` never sees the original training data; each
  mechanism persists whatever sufficient statistics it needs inside
  ``fit_local`` so the old data can be discarded.
* **Fixed graph** — the DAG and every node's cardinality/shape are frozen;
  only CPD parameters change.
* **Posterior-as-prior (recursive Bayes)** — the posterior left by ``fit``
  becomes the prior for ``update``.  For the exact updaters this is literally
  conjugate accumulation (Dirichlet counts, Gaussian normal-equations), so a
  chunked ``fit``→``update`` reproduces a single ``fit`` on the pooled data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ForgettingConfig:
    """Exponential-forgetting weight applied to the persisted prior state.

    ``factor == 1.0`` is pure recursive Bayes (no forgetting); ``factor < 1.0``
    geometrically fades older sufficient statistics so the model tracks a
    drifting distribution.  Must lie in ``(0, 1]``.
    """

    factor: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < float(self.factor) <= 1.0):
            raise ValueError(f"forgetting factor must be in (0, 1], got {self.factor!r}")
        self.factor = float(self.factor)


@dataclass
class UpdateHistory:
    """Records per-node outcomes of a single ``model.update(...)`` call."""

    node_log_likelihoods: Dict[str, float] = field(default_factory=dict)
    node_methods: Dict[str, str] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    wall_clock_s: float = 0.0
    device: str = "cpu"

    def mean_ll(self, node: str) -> float:
        return self.node_log_likelihoods.get(node, float("nan"))
