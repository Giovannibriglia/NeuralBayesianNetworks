"""Binning Categorical CPD for hybrid Bayesian networks (PR-B §B.3).

Discrete-target nodes with continuous parents need a way to index their
probability table by integer parent state.  This mechanism wraps
:class:`CategoricalTableMechanism` and discretises every continuous
parent column at fixed quantile thresholds *learned from the training
data* (option (b) of the v0.4b pre-Phase-D agreement) before forwarding
to the underlying categorical machinery.

Crucially, the thresholds are derived **only from train_data quantiles**;
no information from any "true model" leaks into the fitter.  The
contamination-prevention test
``tests/unit/test_binning_categorical_no_contamination.py`` pins this
contract.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism


class BinningCategoricalTable(CategoricalTableMechanism):
    """Categorical CPD whose continuous parents are discretised at empirical
    quantiles of train_data into ``n_bins`` bins.

    Parameters
    ----------
    parent_kinds:
        ``("discrete", k)`` or ``("continuous", 1)`` per parent in DAG
        order.  Discrete parents pass through unchanged; continuous
        parents are bucketised at the learned thresholds.
    n_bins:
        Number of bins for each continuous parent.  Must equal the
        ``cardinality`` parameter the categorical CPT was sized for.
    n_categories:
        Cardinality of the *target* variable (``K`` in
        :class:`CategoricalTableMechanism`).
    """

    is_discrete: bool = True
    # Deferred: the inherited Dirichlet updater indexes the CPT with raw parent
    # values, but this mechanism must bucketise continuous parents first (and
    # reuse the learned bin edges).  Re-enabled in a later PR.
    supports_update: bool = False

    def __init__(
        self,
        *,
        parent_kinds: Sequence[Tuple[str, int]],
        n_bins: int,
        n_categories: int,
    ) -> None:
        super().__init__(alpha=0.0)
        self._parent_kinds: List[Tuple[str, int]] = list(parent_kinds)
        self._n_bins = int(n_bins)
        self._n_categories = int(n_categories)

    def fit_local(  # type: ignore[override]
        self,
        x: torch.Tensor,
        parents: torch.Tensor | None,
        **kwargs,
    ) -> dict:
        """Fit the binner thresholds from ``parents`` (train data), then
        delegate to the parent ``CategoricalTableMechanism.fit_local``.

        The thresholds are stored as a buffer so ``.to(device)`` carries
        them along with the rest of the model.
        """
        if parents is None or parents.shape[-1] == 0:
            # Root node — no binning needed; just call parent.
            empty = torch.zeros((0, self._n_bins - 1))
            self.register_buffer("_thresholds", empty)
            self.register_buffer("_cont_mask", torch.zeros(0, dtype=torch.bool))
            return super().fit_local(x.long().reshape(-1), None, **kwargs)

        n_pa = parents.shape[-1]
        cont_mask = torch.tensor(
            [k == "continuous" for (k, _) in self._parent_kinds], dtype=torch.bool,
        )
        # PR-B §B.3 contamination-prevention contract: thresholds are
        # the empirical quantiles of *train_data*, never of the true
        # model.
        thresholds = torch.full((n_pa, self._n_bins - 1), float("nan"))
        quantile_levels = torch.linspace(
            1.0 / self._n_bins, (self._n_bins - 1) / self._n_bins, self._n_bins - 1,
        )
        for col in range(n_pa):
            if not cont_mask[col]:
                continue
            col_data = parents[..., col].float().reshape(-1)
            thresholds[col] = torch.quantile(col_data, quantile_levels.to(col_data.device))

        self.register_buffer("_thresholds", thresholds)
        self.register_buffer("_cont_mask", cont_mask)

        # Now bin the continuous parent columns and call the parent fitter.
        bucketed = self._bucketize(parents)
        # Force pa cardinalities to n_bins for every column so the
        # underlying CategoricalTable allocates a fixed-shape CPT.
        parent_cards = [self._n_bins] * n_pa
        return super().fit_local(
            x.long().reshape(-1), bucketed, parent_cards=parent_cards, **kwargs,
        )

    def _bucketize(self, parents: torch.Tensor) -> torch.Tensor:
        out = parents.long().clone()
        if not self._cont_mask.any():
            return out
        for col in torch.where(self._cont_mask)[0].tolist():
            thr = self._thresholds[col]
            valid_thr = thr[~torch.isnan(thr)].to(parents.device)
            if valid_thr.numel() == 0:
                continue
            out[..., col] = torch.bucketize(
                parents[..., col].float().contiguous(), valid_thr,
            ).clamp_max(self._n_bins - 1).long()
        return out

    def forward(self, parents):  # type: ignore[override]
        return super().forward(self._bucketize(parents) if parents is not None else None)

    def sample(self, parents, n: int = 1):  # type: ignore[override]
        return super().sample(
            self._bucketize(parents) if parents is not None else None, n=n,
        )

    def log_prob(self, x, parents):  # type: ignore[override]
        return super().log_prob(
            x, self._bucketize(parents) if parents is not None else None,
        )


__all__ = ["BinningCategoricalTable"]
