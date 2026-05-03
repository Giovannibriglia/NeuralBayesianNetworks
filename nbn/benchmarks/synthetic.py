from __future__ import annotations

import math
from typing import Dict, Tuple

import networkx as nx
import torch

from nbn.core.network import NeuralBayesianNetwork


def _random_dag(n: int, edge_prob: float = 0.15, seed: int = 0) -> nx.DiGraph:
    g = nx.DiGraph()
    rng = torch.Generator().manual_seed(seed)
    nodes = [f"X{i}" for i in range(n)]
    g.add_nodes_from(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            if torch.rand(1, generator=rng).item() < edge_prob:
                g.add_edge(nodes[i], nodes[j])
    return g


def generate_synthetic_hybrid(
    n_nodes: int = 50,
    discrete_frac: float = 0.5,
    edge_prob: float = 0.15,
    seed: int = 0,
) -> Tuple[NeuralBayesianNetwork, Dict[str, torch.Tensor]]:
    """Generate a synthetic hybrid DAG and sample data from it.

    Parameters
    ----------
    n_nodes: int
    discrete_frac: float in [0,1]
        Fraction of discrete nodes.
    edge_prob: float
        Edge probability in the random upper-triangular DAG.
    seed: int
        Reproducibility seed.

    Returns
    -------
    (model, data_dict)
        ``model`` is a fitted NBN; ``data_dict`` is 5_000 IID samples from it.
    """
    from nbn.mechanisms.categorical_table import CategoricalTableMechanism
    from nbn.mechanisms.linear_gaussian import LinearGaussianMechanism
    from nbn.utils.seed import seed_all

    seed_all(seed)
    g = _random_dag(n_nodes, edge_prob=edge_prob, seed=seed)
    nodes = list(nx.topological_sort(g))
    n_disc = int(round(n_nodes * discrete_frac))
    rng = torch.Generator().manual_seed(seed + 1)
    perm = torch.randperm(n_nodes, generator=rng).tolist()
    discrete_set = set(nodes[i] for i in perm[:n_disc])

    variables = {
        n: ("discrete", 3) if n in discrete_set else ("continuous", 1)
        for n in nodes
    }
    model = NeuralBayesianNetwork(g, variables=variables)

    # Generate synthetic data
    data: Dict[str, torch.Tensor] = {}
    n_samples = 5_000
    for node in model.dag.topological_order():
        parents = model.dag.parents(node)
        if node in discrete_set:
            # Random CPT
            n_parent_states = max(1, math.prod([3 if p in discrete_set else 5 for p in parents]) if parents else 1)
            logits = torch.randn(n_parent_states, 3, generator=rng)
            mech = CategoricalTableMechanism()
            mech._logits = torch.nn.Parameter(torch.log_softmax(logits, dim=-1))
            mech._n_classes = 3
            mech._parent_cards = [3 if p in discrete_set else 5 for p in parents]
            strides = []
            stride = 1
            for c in reversed(mech._parent_cards):
                strides.append(stride)
                stride *= c
            mech._parent_strides = list(reversed(strides))
            mech._class_values = torch.tensor([0.0, 1.0, 2.0])
            mech.output_dim = 1
            model.set_mechanism(node, mech)
        else:
            mech = LinearGaussianMechanism()
            d_pa = len(parents)
            x = torch.randn(100, 1)
            if d_pa > 0:
                pa = torch.randn(100, d_pa)
                mech.fit_local(x + pa.sum(-1, keepdim=True), pa)
            else:
                mech.fit_local(x, None)
            model.set_mechanism(node, mech)

    # Forward sample n_samples
    samples = model.sample(n=n_samples)
    return model, samples
