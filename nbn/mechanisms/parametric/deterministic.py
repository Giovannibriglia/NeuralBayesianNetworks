from __future__ import annotations

import torch
from torch.distributions import Distribution

from nbn.mechanisms.base import Mechanism
from nbn.utils.batching import ensure_2d

# Log-probability stand-in for "impossible state" in a tabulated delta CPD.
# A true ``-inf`` propagates NaN through variable elimination's factor
# products (``-inf + inf``, ``0 * -inf``); ``exp(-1e9)`` is exactly 0.0 in
# float32, so this is numerically a hard zero without the NaN hazard, and
# sums of a handful of them stay far inside float32's range.
_LOG_ZERO = -1e9


class _DeltaDistribution(Distribution):
    """A Dirac delta distribution at a fixed value."""

    has_rsample = True

    def __init__(self, value: torch.Tensor) -> None:
        self._value = value
        super().__init__(batch_shape=value.shape[:-1], event_shape=value.shape[-1:], validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        return self._value.expand(*sample_shape, *self._value.shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.sample(sample_shape)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)


class DeterministicMechanism(Mechanism):
    """Deterministic CPD: delta distribution at a fixed value.

    Used by Pearl's do-operator: replacing a node's CPD with
    ``DeterministicMechanism(value)`` implements ``do(X = value)``.

    Parameters
    ----------
    value:
        Fixed output value, shape ``[D_x]`` or ``[B, D_x]``.
    cardinality:
        Number of states when the intervened node is *discrete*.  Supplying it
        makes the mechanism advertise itself as discrete and gives it the
        tabular interface (``n_classes``, ``tabulate``, ``_class_values``) that
        exact inference needs.  Leave ``None`` for continuous nodes, which
        keeps the historical continuous-delta behaviour byte-for-byte.

    Notes
    -----
    Before ``cardinality`` existed this class always inherited
    ``Mechanism.is_discrete = False``, so an ``intervene()``-produced model of
    an all-discrete network was rejected wholesale by
    ``TensorVariableElimination`` ("node 'X' has a continuous mechanism").
    That left discrete networks with no exact interventional path at all.
    """

    def __init__(
        self, value: torch.Tensor, cardinality: int | None = None
    ) -> None:
        super().__init__()
        self.register_buffer("_fixed_value", value)
        self._n_classes = int(cardinality) if cardinality is not None else 0
        # Instance-level override of the class attribute (same pattern as
        # KNNConditionalMechanism): one class serves both variable kinds.
        self.is_discrete = cardinality is not None
        if cardinality is not None:
            self.register_buffer(
                "_class_values",
                torch.arange(int(cardinality), dtype=torch.float, device=value.device),
            )

    def forward(self, parents: torch.Tensor | None) -> _DeltaDistribution:
        if parents is not None and self._fixed_value.dim() == 1:
            b = ensure_2d(parents).shape[0]
            val = self._fixed_value.unsqueeze(0).expand(b, -1)
        else:
            val = self._fixed_value
        return _DeltaDistribution(val)

    def sample(self, parents: torch.Tensor | None, n: int = 1) -> torch.Tensor:
        b = 1 if parents is None else ensure_2d(parents).shape[0]
        val = self._fixed_value
        if val.dim() == 1:
            return val.view(1, 1, -1).expand(b, n, -1)
        return val.unsqueeze(1).expand(-1, n, -1)

    def log_prob(self, x: torch.Tensor, parents: torch.Tensor | None) -> torch.Tensor:
        return torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)

    # Vacuously true: fit_local estimates nothing from the data (the CPD is
    # fixed at construction), so there is no statistic for a weight to bias.
    supports_weights: bool = True

    def fit_local(self, x: torch.Tensor, parents: torch.Tensor | None, **kwargs) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Tabular interface (discrete interventions only)
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Always True: a delta CPD is fully specified at construction.

        ``TensorVariableElimination`` refuses to build a factor for an
        unfitted mechanism, and the base-class default is ``False``, so
        without this an intervened model could never be queried exactly.
        """
        return True

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def tabulate(self, parent_cards: list[int] | None = None) -> torch.Tensor:
        """Return the ``[K]`` log-CPT of the post-intervention (mutilated) node.

        The do-operator severs the node's incoming edges, so the tabulation is
        unconditional: all mass on the intervened state.  ``parent_cards`` is
        accepted for API uniformity and ignored — a mutilated node has no
        parents by construction.
        """
        if not self.is_discrete:
            raise NotImplementedError(
                "DeterministicMechanism.tabulate() is only defined for discrete "
                "interventions; construct it with cardinality=<K>."
            )
        idx = int(self._fixed_value.reshape(-1)[0].item())
        log_cpt = torch.full(
            (self._n_classes,), _LOG_ZERO,
            device=self._fixed_value.device, dtype=torch.float,
        )
        log_cpt[idx] = 0.0
        return log_cpt
