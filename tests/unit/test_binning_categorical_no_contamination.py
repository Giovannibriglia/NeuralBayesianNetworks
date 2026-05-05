"""PR-B §B.3 contract: BinningCategoricalTable thresholds come from
*train_data quantiles*, never from any "true model" the test code
might happen to have access to.

If a future refactor accidentally wires the truth into the fitter,
this test fails because the thresholds match the truth's quantiles
rather than the deliberately-shifted train data's quantiles.
"""
from __future__ import annotations

import torch

from nbn.mechanisms.binning_categorical import BinningCategoricalTable


def test_thresholds_match_train_quantiles_not_truth() -> None:
    torch.manual_seed(0)
    n_bins = 4
    n_train = 5000

    # "True" continuous parent: standard normal.  Quantile(0.25/0.5/0.75)
    # are roughly -0.674, 0, 0.674.
    true_quantiles = torch.tensor([-0.674, 0.0, 0.674])

    # Train data is shifted +5: the train quantiles are +5 + true.
    parents_train = torch.randn(n_train, 1) + 5.0
    expected_train_q = torch.quantile(
        parents_train.reshape(-1), torch.linspace(1 / n_bins, (n_bins - 1) / n_bins, n_bins - 1),
    )

    # Random integer targets — content doesn't matter for the contract test.
    x_train = torch.randint(0, n_bins, (n_train,)).long()

    mech = BinningCategoricalTable(
        parent_kinds=[("continuous", 1)],
        n_bins=n_bins, n_categories=n_bins,
    )
    mech.fit_local(x_train, parents_train)

    learned = mech._thresholds[0].cpu().float()
    # Thresholds must match train-data quantiles, not the truth.
    assert torch.allclose(learned, expected_train_q, atol=0.05), (
        f"learned thresholds {learned.tolist()} differ from train quantiles "
        f"{expected_train_q.tolist()} by more than 0.05 — fitter may be "
        f"contaminated with truth-side knowledge"
    )
    # And explicitly NOT the truth quantiles (they're shifted by 5).
    assert not torch.allclose(learned, true_quantiles, atol=1.0), (
        f"learned thresholds {learned.tolist()} suspiciously close to "
        f"the un-shifted truth quantiles {true_quantiles.tolist()} — "
        f"the truth has leaked into the fitter"
    )


def test_discrete_parents_pass_through_unchanged() -> None:
    """A discrete parent column must NOT be bucketised — the binner only
    touches columns flagged ``continuous``."""
    torch.manual_seed(1)
    n_bins = 3
    n_train = 1000
    parents = torch.stack(
        [
            torch.randint(0, n_bins, (n_train,)).float(),  # discrete col
            torch.randn(n_train),                          # continuous col
        ], dim=-1,
    )
    x = torch.randint(0, n_bins, (n_train,)).long()
    mech = BinningCategoricalTable(
        parent_kinds=[("discrete", n_bins), ("continuous", 1)],
        n_bins=n_bins, n_categories=n_bins,
    )
    mech.fit_local(x, parents)
    # First column thresholds should be NaN (not used).
    assert torch.isnan(mech._thresholds[0]).all()
    # Second column should have learned thresholds.
    assert not torch.isnan(mech._thresholds[1]).any()
