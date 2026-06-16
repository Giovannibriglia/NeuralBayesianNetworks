"""Posterior dataclass returned by BaselineAdapter.query().

Reference: docs/v0.13-benchmark-redesign.md §4.1
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Posterior:
    """The result of querying a fitted model for p(target | evidence).

    Two representations supported, depending on target's family:
    - For discrete targets: `probs` is a (n_categories,) tensor;
      `samples` is None.
    - For continuous targets: `samples` is a (n_samples,) tensor of
      posterior samples; `probs` is None.

    The query-evaluation code paths (TV/JSD for discrete, W1 for
    continuous) dispatch on which field is set.
    """

    probs: torch.Tensor | None = None    # discrete: (n_categories,)
    samples: torch.Tensor | None = None  # continuous: (n_samples,)

    # Per-query effective-sample-size fraction ∈ (0, 1] for importance-sampling
    # engines (LW / AIS); None for engines without IS weights (VE, AVI, and the
    # pgmpy / pyro / pomegranate adapters).  Additive — does not participate in
    # the probs/samples XOR validation below.  Lets downstream analysis show ESS
    # distributions and filter out unreliable posteriors (X1).
    ess: float | None = None

    def __post_init__(self) -> None:
        if (self.probs is None) == (self.samples is None):
            raise ValueError(
                "Posterior must have exactly one of `probs` or `samples` set; "
                f"got probs={self.probs is not None}, samples={self.samples is not None}"
            )
