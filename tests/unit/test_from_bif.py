"""``NeuralBayesianNetwork.from_bif`` must reproduce the file's CPTs exactly.

This loader was dead code: it called ``model.topological_order``,
``cpd.evidence`` and ``cpd.evidence_card``, all three of which pgmpy has
removed, so it raised ``AttributeError`` on any modern pgmpy.  Underneath that
lay a subtler trap — ``cpd.get_evidence()`` returns the CPT's axis order
*reversed*, so the obvious repair would have silently transposed every
multi-parent CPT and returned confidently wrong probabilities.

The CPTs below are deliberately asymmetric in both cardinality (2 x 3) and
value, so any axis permutation changes the answer.  Every case is checked
against pgmpy's own variable elimination on the same file.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from nbn import NeuralBayesianNetwork as NBN

pytest.importorskip("pgmpy")


@pytest.fixture(scope="module")
def bif_path(tmp_path_factory):
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.readwrite import BIFWriter

    model = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("C", "D")])
    model.add_cpds(
        TabularCPD("A", 2, [[0.3], [0.7]]),
        TabularCPD("B", 3, [[0.2], [0.3], [0.5]]),
        TabularCPD(
            "C", 2,
            [[0.90, 0.80, 0.70, 0.10, 0.20, 0.30],
             [0.10, 0.20, 0.30, 0.90, 0.80, 0.70]],
            evidence=["A", "B"], evidence_card=[2, 3],
        ),
        TabularCPD("D", 2, [[0.95, 0.05], [0.05, 0.95]],
                   evidence=["C"], evidence_card=[2]),
    )
    path = tmp_path_factory.mktemp("bif") / "net.bif"
    BIFWriter(model).write(str(path))
    return str(path), model


def test_from_bif_builds_a_fitted_model(bif_path):
    path, _ = bif_path
    loaded = NBN.from_bif(path, device="cpu")
    assert sorted(loaded.mechanisms.keys()) == ["A", "B", "C", "D"]
    assert all(m.is_fitted for m in loaded.mechanisms.values())
    assert loaded.variables["B"].cardinality == 3


def test_from_bif_preserves_cpt_axis_order(bif_path):
    """parents(C) must line up with the CPT's own axes, not a reversed list."""
    path, _ = bif_path
    loaded = NBN.from_bif(path, device="cpu")
    assert loaded.dag.parents("C") == ["A", "B"]
    # Cards follow the same order: A has 2 states, B has 3.
    assert loaded.mechanisms["C"]._parent_cards == [2, 3]


@pytest.mark.parametrize(
    ("target", "evidence"),
    [
        ("C", {}),
        ("C", {"A": 0}),
        ("C", {"B": 2}),
        ("C", {"A": 1, "B": 0}),
        ("C", {"A": 0, "B": 2}),
        ("C", {"D": 1}),
        ("D", {}),
        ("D", {"A": 0}),
        ("D", {"A": 1, "B": 0}),
        ("A", {"D": 1}),
        ("A", {"B": 2}),
    ],
)
def test_from_bif_marginals_match_pgmpy(bif_path, target, evidence):
    from pgmpy.inference import VariableElimination

    path, model = bif_path
    loaded = NBN.from_bif(path, device="cpu")
    expected = VariableElimination(model).query(
        [target], evidence=evidence, show_progress=False,
    ).values
    got = loaded.query(
        [target], evidence={k: torch.tensor(v) for k, v in evidence.items()},
    )
    np.testing.assert_allclose(got.cpu().numpy(), expected, atol=1e-4)


def test_from_bif_exposes_state_names(bif_path):
    """Callers need the label -> index mapping to build evidence."""
    path, _ = bif_path
    loaded = NBN.from_bif(path, device="cpu")
    assert set(loaded.state_names) == {"A", "B", "C", "D"}
    assert len(loaded.state_names["B"]) == 3


def test_from_bif_round_trips_through_save_load(bif_path, tmp_path):
    path, _ = bif_path
    loaded = NBN.from_bif(path, device="cpu")
    ckpt = str(tmp_path / "m.pt")
    loaded.save(ckpt)
    reloaded = NBN.load(ckpt)
    assert reloaded.dag.parents("C") == ["A", "B"]
    torch.testing.assert_close(
        reloaded.query(["C"], evidence={"A": torch.tensor(1)}),
        loaded.query(["C"], evidence={"A": torch.tensor(1)}),
        atol=1e-6, rtol=1e-6,
    )
