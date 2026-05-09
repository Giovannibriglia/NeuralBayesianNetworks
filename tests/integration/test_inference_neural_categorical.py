"""v0.8-#26 regression: VE inference on networks with NeuralCategoricalMechanism.

Pre-#26: ``TensorVariableElimination._extract_factors`` read ``mech._logits``
directly and gated on ``hasattr(mech, "_logits")``.  ``NeuralCategoricalMechanism``
computes its CPD per-call via an MLP forward pass and has no flat
``_logits`` attribute, so the engine raised
``RuntimeError("Mechanism for node 'X' has not been fitted.")`` even on
a correctly fitted neural mechanism.

Post-#26: ``Mechanism.tabulate(parent_cards)`` introduced in v0.8-#59/#26
materialises the CPD by enumeration for forward-based mechanisms; the
engine reads it via ``tabulate()`` and gates on ``mech.is_fitted``.

This test uses a 3-node chain ``A -> B -> C`` where ``A``, ``B`` are
``CategoricalTableMechanism`` (so the analytical truth ``P(C|A=a)`` is
computable by hand-enumeration over ``B``) and ``C`` is a
``NeuralCategoricalMechanism`` fit on data drawn from a planted
``P(C|B)``.  After fitting:

* The engine must run end-to-end (no ``RuntimeError`` on the neural
  mechanism's "fittedness").
* VE-inferred ``P(C|A=a)`` must match the analytical truth within MC
  noise from the 8000-sample fit (~5% absolute tolerance leaves ample
  headroom on a 2-class CPT).

Pre-fix verification: scratch-revert the engine + ``is_fitted``
property changes and re-run — the test fails with the
``RuntimeError`` from the old ``hasattr(_logits)`` guard.
"""
from __future__ import annotations

import torch

from nbn.core.network import NeuralBayesianNetwork
from nbn.inference.tensor_ve import TensorVariableElimination
from nbn.mechanisms.categorical_table import CategoricalTableMechanism
from nbn.mechanisms.neural_categorical import NeuralCategoricalMechanism


# Hand-picked CPTs for the analytical-truth side of the comparison.
P_A = torch.tensor([0.6, 0.4])
P_B_GIVEN_A = torch.tensor([
    [0.8, 0.2],
    [0.1, 0.9],
])
P_C_GIVEN_B = torch.tensor([
    [0.7, 0.3],
    [0.2, 0.8],
])


def _truth_C_given_A(a_obs: int) -> torch.Tensor:
    """Analytical ``P(C | A=a)`` by enumeration over B."""
    p_b = P_B_GIVEN_A[a_obs]
    p_c = torch.zeros(2)
    for b_val in range(2):
        p_c += p_b[b_val] * P_C_GIVEN_B[b_val]
    return p_c


def test_ve_runs_on_neural_categorical_mechanism() -> None:
    """v0.8-#26 regression: VE inference must run on a network containing
    NeuralCategoricalMechanism and produce posteriors close to truth.
    """
    torch.manual_seed(0)

    nbn = NeuralBayesianNetwork(
        [("A", "B"), ("B", "C")],
        variables={
            "A": ("discrete", 2),
            "B": ("discrete", 2),
            "C": ("discrete", 2),
        },
        device="cpu",
    )
    mA = CategoricalTableMechanism(alpha=0.0)
    mB = CategoricalTableMechanism(alpha=0.0)
    mC = NeuralCategoricalMechanism(n_classes=2, hidden=(16, 16))
    nbn.set_mechanism("A", mA)
    nbn.set_mechanism("B", mB)
    nbn.set_mechanism("C", mC)

    n = 8000
    a = torch.distributions.Categorical(P_A).sample((n,))
    b = torch.distributions.Categorical(P_B_GIVEN_A[a]).sample()
    c = torch.distributions.Categorical(P_C_GIVEN_B[b]).sample()

    mA.fit_local(a, parents=None)
    mB.fit_local(b, parents=a.unsqueeze(-1), parent_cards=[2])
    mC.fit_local(c.long(), parents=b.unsqueeze(-1).float(), epochs=80, lr=5e-3)

    eng = TensorVariableElimination()
    for a_obs in (0, 1):
        truth = _truth_C_given_A(a_obs)
        ve = eng.query(nbn, ["C"], evidence={"A": torch.tensor([a_obs])})
        diff = (ve - truth).abs().max().item()
        assert diff < 0.05, (
            f"P(C | A={a_obs}): VE result {ve.tolist()} drifted from "
            f"analytical truth {truth.tolist()} by {diff:.4f}; "
            f"tolerance 0.05.  Either the tabulate-by-enumeration path "
            f"in _extract_factors is broken, or the neural mechanism's "
            f"learned CPD is far from the planted P(C|B)."
        )


def test_ve_raises_clean_runtime_error_on_unfitted_mechanism() -> None:
    """Companion check: the post-#26 engine guard must still raise
    ``RuntimeError`` (not propagate AssertionError or AttributeError)
    on an unfitted mechanism.

    The old guard read ``hasattr(mech, "_logits") or mech._logits is
    None``; the new guard reads ``mech.is_fitted``.  Both produce the
    same surface error for unfitted mechanisms — this test pins that.
    """
    nbn = NeuralBayesianNetwork(
        [("X", "Y")],
        variables={"X": ("discrete", 2), "Y": ("discrete", 2)},
        device="cpu",
    )
    nbn.set_mechanism("X", CategoricalTableMechanism())  # NOT fit
    nbn.set_mechanism("Y", NeuralCategoricalMechanism(n_classes=2))  # NOT fit
    eng = TensorVariableElimination()
    import pytest
    with pytest.raises(RuntimeError, match="has not been fitted"):
        eng.query(nbn, ["Y"], evidence={"X": torch.tensor([0])})
