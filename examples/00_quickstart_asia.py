"""Quickstart: build the asia BN, run exact inference, sample.

Run with:
    python examples/00_quickstart_asia.py
"""
import torch
from torch import nn

from nbn import NeuralBayesianNetwork, TensorVariableElimination, seed_all
from nbn.mechanisms import CategoricalTableMechanism

seed_all(0)

# DAG of the asia network (Lauritzen & Spiegelhalter, 1988)
edges = [
    ("asia", "tub"), ("smoke", "lung"), ("smoke", "bronc"),
    ("tub", "either"), ("lung", "either"),
    ("either", "xray"), ("either", "dysp"), ("bronc", "dysp"),
]
nodes = ["asia", "smoke", "tub", "lung", "bronc", "either", "xray", "dysp"]

model = NeuralBayesianNetwork(edges, variables={n: ("discrete", 2) for n in nodes})

# Build CPTs (binary nodes; values from the asia paper)
cpts = {
    "asia":   {(): [0.99, 0.01]},
    "smoke":  {(): [0.5, 0.5]},
    "tub":    {(0,): [0.99, 0.01], (1,): [0.95, 0.05]},
    "lung":   {(0,): [0.99, 0.01], (1,): [0.9, 0.1]},
    "bronc":  {(0,): [0.7, 0.3],  (1,): [0.4, 0.6]},
    "either": {(0,0): [1.0,0.0], (0,1): [0.0,1.0],
               (1,0): [0.0,1.0], (1,1): [0.0,1.0]},
    "xray":   {(0,): [0.95, 0.05], (1,): [0.02, 0.98]},
    "dysp":   {(0,0): [0.9,0.1], (0,1): [0.2,0.8],
               (1,0): [0.3,0.7], (1,1): [0.1,0.9]},
}
parent_order = {
    "asia": [], "smoke": [], "tub": ["asia"], "lung": ["smoke"],
    "bronc": ["smoke"], "either": ["tub", "lung"],
    "xray": ["either"], "dysp": ["either", "bronc"],
}

for node in model.dag.topological_order():
    parents = parent_order[node]
    n_parent_states = max(1, 2 ** len(parents))
    log_cpt = torch.zeros(n_parent_states, 2)
    for pa, probs in cpts[node].items():
        row = sum(v * (2 ** (len(pa) - 1 - i)) for i, v in enumerate(pa))
        log_cpt[row] = torch.log(torch.tensor(probs).clamp_min(1e-9))
    mech = CategoricalTableMechanism()
    mech._logits = nn.Parameter(log_cpt)
    mech._n_classes = 2
    mech._parent_cards = [2] * len(parents)
    mech._parent_strides = [2 ** (len(parents) - 1 - i) for i in range(len(parents))]
    mech._class_values = torch.tensor([0.0, 1.0])
    mech.output_dim = 1
    model.set_mechanism(node, mech)

print(model)

# Exact inference
ve = TensorVariableElimination()
print("P(asia=1)        =", ve.query(model, ["asia"]).tolist())
print("P(lung=1|smoke=1)=", ve.query(model, ["lung"], evidence={"smoke": torch.tensor(1)}).tolist())

# Sampling
samples = model.sample(n=1000)
mean_lung = samples["lung"].float().mean().item()
print(f"Empirical P(lung=1) ≈ {mean_lung:.3f}")
