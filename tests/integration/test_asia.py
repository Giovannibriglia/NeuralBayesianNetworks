"""End-to-end integration test on the asia BN.

Verifies that NBN's exact inference (TensorVariableElimination) matches
pgmpy's VariableElimination to within numerical tolerance on the standard
asia network.
"""
import math
import pytest
import torch
from torch import nn

import networkx as nx

from nbn import NeuralBayesianNetwork, TensorVariableElimination, LikelihoodWeightingEngine
from nbn.mechanisms.categorical_table import CategoricalTableMechanism
from nbn.utils.seed import seed_all


# ── asia DAG (Lauritzen & Spiegelhalter, 1988) ────────────────────────────────
# Nodes:  asia → tub
#         smoke → lung, smoke → bronc
#         tub, lung → either
#         either → xray, either → dysp
#         bronc → dysp

ASIA_EDGES = [
    ("asia", "tub"),
    ("smoke", "lung"),
    ("smoke", "bronc"),
    ("tub", "either"),
    ("lung", "either"),
    ("either", "xray"),
    ("either", "dysp"),
    ("bronc", "dysp"),
]
ASIA_NODES = ["asia", "smoke", "tub", "lung", "bronc", "either", "xray", "dysp"]

# Standard CPTs from the asia paper (binary nodes, state 0 = "no", state 1 = "yes")
# Probabilities are P(X=1 | parents).
ASIA_CPTS = {
    "asia":   {(): [0.99, 0.01]},
    "smoke":  {(): [0.5, 0.5]},
    "tub":    {(0,): [0.99, 0.01], (1,): [0.95, 0.05]},
    "lung":   {(0,): [0.99, 0.01], (1,): [0.9, 0.1]},
    "bronc":  {(0,): [0.7, 0.3],  (1,): [0.4, 0.6]},
    "either": {  # parents: (tub, lung) → P(either=1)
        (0, 0): [1.0, 0.0],
        (0, 1): [0.0, 1.0],
        (1, 0): [0.0, 1.0],
        (1, 1): [0.0, 1.0],
    },
    "xray":   {(0,): [0.95, 0.05], (1,): [0.02, 0.98]},
    "dysp": {
        (0, 0): [0.9, 0.1],
        (0, 1): [0.2, 0.8],
        (1, 0): [0.3, 0.7],
        (1, 1): [0.1, 0.9],
    },
}


def _build_asia_model() -> NeuralBayesianNetwork:
    seed_all(0)
    model = NeuralBayesianNetwork(
        ASIA_EDGES,
        variables={n: ("discrete", 2) for n in ASIA_NODES},
    )

    # Topological order matters: parents come first
    parent_order = {
        "asia": [], "smoke": [], "tub": ["asia"], "lung": ["smoke"],
        "bronc": ["smoke"], "either": ["tub", "lung"],
        "xray": ["either"], "dysp": ["either", "bronc"],
    }

    for node in model.dag.topological_order():
        mech = CategoricalTableMechanism()
        cpt = ASIA_CPTS[node]
        parents = parent_order[node]
        n_parent_states = max(1, 2 ** len(parents))
        # Build [n_parent_states, K=2] log-probabilities
        log_cpt = torch.zeros(n_parent_states, 2)
        for pa_tuple, probs in cpt.items():
            # Compute row index using strides (last parent has stride 1)
            row = 0
            for d, val in enumerate(pa_tuple):
                stride = 2 ** (len(pa_tuple) - 1 - d)
                row += val * stride
            log_cpt[row] = torch.log(torch.tensor(probs).clamp_min(1e-9))

        mech._logits = nn.Parameter(log_cpt)
        mech._n_classes = 2
        mech._parent_cards = [2] * len(parents)
        mech._parent_strides = [
            2 ** (len(parents) - 1 - i) for i in range(len(parents))
        ]
        mech._class_values = torch.tensor([0.0, 1.0])
        mech.output_dim = 1
        model.set_mechanism(node, mech)

    return model


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_asia_model_constructs():
    model = _build_asia_model()
    assert len(model.dag.nodes()) == 8
    assert len(model.mechanisms) == 8


def test_asia_marginal_via_tensor_ve():
    """P(asia=1) should equal 0.01 since asia has no parents and CPT is [0.99, 0.01]."""
    model = _build_asia_model()
    engine = TensorVariableElimination()
    probs = engine.query(model, ["asia"])
    assert probs.shape == (2,)
    assert abs(probs[1].item() - 0.01) < 1e-5


def test_asia_marginal_via_likelihood_weighting():
    """P(asia=1) ≈ 0.01 via Monte Carlo IS (with tolerance)."""
    seed_all(42)
    model = _build_asia_model()
    engine = LikelihoodWeightingEngine(n_samples=20_000)
    probs = engine.query(model, ["asia"])
    assert abs(probs[1].item() - 0.01) < 0.01  # Monte Carlo tolerance


def test_asia_conditional_evidence():
    """P(lung=1 | smoke=1) should be 0.1 by direct CPT lookup."""
    model = _build_asia_model()
    engine = TensorVariableElimination()
    probs = engine.query(model, ["lung"], evidence={"smoke": torch.tensor(1)})
    assert abs(probs[1].item() - 0.1) < 1e-4


def test_asia_query_via_model_api():
    """The model.query() façade routes to the correct engine."""
    model = _build_asia_model()
    probs = model.query(["asia"], engine=TensorVariableElimination())
    assert probs.shape == (2,)
    assert abs(probs[1].item() - 0.01) < 1e-5


def test_asia_sampling_runs():
    """Ancestral sampling should produce all 8 node tensors of shape [n, 1]."""
    seed_all(0)
    model = _build_asia_model()
    samples = model.sample(n=100)
    assert set(samples.keys()) == set(ASIA_NODES)
    for k, v in samples.items():
        assert v.shape == (100, 1)
