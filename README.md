# NeuralBayesianNetworks (NBN)

**NBN is a PyTorch-native library for learning, sampling from, and querying Bayesian Networks with a known DAG, where each node carries a learnable, batched, GPU-resident *neural mechanism*.**

> NBN is to BNs what GPyTorch is to GPs: a torch-native, batchable, autograd-friendly framework where every conditional distribution is a swappable, learnable module, and every query is a batched tensor operation.

## Installation

```bash
pip install neuralbayesiannetworks
# with all extras:
pip install "neuralbayesiannetworks[bench,neural,gp,mcmc,dev]"
```

## Quickstart

```python
import torch
from nbn import NeuralBayesianNetwork
from nbn.mechanisms import CategoricalTableMechanism, MDNMechanism

dag = [("A", "C"), ("B", "C"), ("C", "D")]
model = NeuralBayesianNetwork(
    dag,
    variables={"A": ("discrete", 2), "B": ("discrete", 3),
               "C": ("discrete", 4), "D": ("continuous", 1)},
)
model.set_mechanism("A", CategoricalTableMechanism())
model.set_mechanism("B", CategoricalTableMechanism())
model.set_mechanism("C", CategoricalTableMechanism())
model.set_mechanism("D", MDNMechanism(num_components=5))

model.fit(data, epochs=100, batch_size=4096, device="cuda")

# Single query
marginal = model.query(["C"], evidence={"A": 0, "B": 2})

# Batched queries — thousands in one GPU launch
batch_marginal = model.query_batch(
    ["C"],
    evidence={"A": torch.tensor([0, 1, 0, 1]), "B": torch.tensor([0, 1, 2, 2])},
)  # shape [4, |C|]

samples = model.sample(n=10_000)
```

## Why NBN?

| Library | Discrete BN | Continuous | GPU-batched queries | Neural CPDs | Autograd |
|---------|------------|------------|---------------------|-------------|---------|
| pgmpy | ✅ exact | ✅ Gaussian only | ❌ | ❌ | ❌ |
| pyAgrum | ✅ exact | ❌ | ❌ | ❌ | ❌ |
| pomegranate | ✅ | ✅ | partial | ❌ | ✅ |
| GPyTorch | ❌ | ✅ GP | ✅ | ✅ | ✅ |
| **NBN** | **✅ exact (VE)** | **✅ MDN/Flow/GP** | **✅** | **✅** | **✅** |

## Features

- **Mechanism zoo**: CategoricalTable, LinearGaussian, MDN, NeuralCategorical, NormalizingFlow, GP (optional), Deterministic
- **Inference engines**: TensorVariableElimination (log-domain einsum + `opt_einsum`), LikelihoodWeighting, AmortizedVariational, HybridRouter
- **Batched-query API**: `query_batch` returns `[Q, K]` in one GPU launch
- **Fully `nn.Module`**: `.to(device)`, `.parameters()`, `torch.compile`, AMP all work
- **Causal extensions**: `do(X=x)` interventions, counterfactuals
- **Benchmarks**: all 23+ bnlearn networks + large hybrid synthetic graphs

## Status

| Phase | Status |
|-------|--------|
| Core (DAG, Variables, Factor) | ✅ |
| Mechanisms (Table, LG, MDN) | ✅ |
| Tensor VE + Likelihood Weighting | ✅ |
| NeuralBayesianNetwork class | ✅ |
| Advanced mechanisms (Flow, NeuralCat) | ✅ |
| Amortized Variational Engine | ✅ |
| Benchmarks suite | ✅ |

## Citation

```bibtex
@software{briglia2025nbn,
  author = {Briglia, Giovanni},
  title  = {NeuralBayesianNetworks: PyTorch-native Bayesian Networks with Neural Mechanisms},
  year   = {2025},
  url    = {https://github.com/Giovannibriglia/NeuralBayesianNetworks},
}
```
