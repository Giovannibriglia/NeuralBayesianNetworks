"""nbn-cat-bayes baseline registration (Laplace-smoothed CPTs).

At paper scale the alpha=1 estimators (pyro-empirical, pgmpy-bayes) beat
nbn-cat (alpha=0, MLE parity) on BOTH discrete recovery metrics — TV 0.0646
vs 0.0699 and KL 0.0222 vs 0.2467 (param_learning_complete 20260701) — and
on every small-n_train point of the learning curves. The library already
shipped ``SmoothedEmpiricalCategoricalMechanism`` (alpha=1); these tests pin
its exposure as the ``cat-bayes`` benchmark mechanism.
"""
from __future__ import annotations

import pytest

from benchmarking.adapters.nbn_adapter import NBNAdapter
from benchmarking.core.applicability import is_applicable


def test_make_mech_returns_smoothed_categorical():
    from nbn.mechanisms import SmoothedEmpiricalCategoricalMechanism

    adapter = NBNAdapter(mechanism="cat-bayes", engine=None, device="cpu")
    mech = adapter._make_mech("discrete", k=3, parent_kinds=["discrete"])
    assert isinstance(mech, SmoothedEmpiricalCategoricalMechanism)
    assert mech.alpha == 1.0  # Laplace — the point of the label


def test_engine_less_name_matches_applicability_key():
    adapter = NBNAdapter(mechanism="cat-bayes", engine=None, device="cpu")
    assert adapter.name == "nbn-cat-bayes"
    assert is_applicable("nbn-cat-bayes", "discrete")
    assert not is_applicable("nbn-cat-bayes", "continuous_lg")


@pytest.mark.parametrize("engine", ["ve", "lw"])
def test_inference_labels_registered(engine):
    adapter = NBNAdapter(mechanism="cat-bayes", engine=engine, device="cpu")
    assert adapter.name == f"nbn-cat-bayes-{engine}"
    assert is_applicable(adapter.name, "discrete")


def test_smoothing_differs_from_mle_on_sparse_counts():
    """alpha=1 must shift sparse-count CPTs away from the raw MLE tables."""
    import torch

    from nbn.mechanisms import (
        CategoricalTableMechanism,
        SmoothedEmpiricalCategoricalMechanism,
    )

    # One parent config seen 3 times, child always class 0 of 2.
    y = torch.zeros(3, dtype=torch.long)
    pa = torch.zeros(3, 1, dtype=torch.long)
    mle = CategoricalTableMechanism()
    mle.fit_local(y, pa, n_classes=2, parent_cards=[1])
    sm = SmoothedEmpiricalCategoricalMechanism()
    sm.fit_local(y, pa, n_classes=2, parent_cards=[1])

    pa_q = torch.zeros(1, 1, dtype=torch.long)
    p_mle = mle.forward(pa_q).probs.detach().squeeze()
    p_sm = sm.forward(pa_q).probs.detach().squeeze()
    # The exact presented value also folds in the table's never-observed +1
    # presentation adjustment, so pin the estimator ORDERING (the property
    # the label exists for), not a constant: alpha=1 pulls the sparse-count
    # CPT toward uniform relative to the MLE table.
    assert float(p_sm[0]) < float(p_mle[0])
    assert float(p_sm[1]) > float(p_mle[1])
    assert float(p_sm.sum()) == pytest.approx(1.0, abs=1e-5)
