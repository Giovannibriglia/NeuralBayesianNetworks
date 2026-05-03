# NeuralBayesianNetworks (NBN)

**NBN is a PyTorch-native library for learning, sampling from, and querying Bayesian Networks with a known DAG, where each node carries a learnable, batched, GPU-resident *neural mechanism*.**

> NBN is to BNs what GPyTorch is to GPs: a torch-native, batchable, autograd-friendly framework where every conditional distribution is a swappable, learnable module, and every query is a batched tensor operation.

## Installation

```bash
pip install neuralbayesiannetworks
# with all extras:
pip install "neuralbayesiannetworks[bench,neural,gp,mcmc,dev]"
```
or
```bash
git clone https://github.com/Giovannibriglia/NeuralBayesianNetworks.git
pip install -e ".[bench,neural,gp,mcmc,dev]"
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
- **Benchmarks**: top-level [`benchmarking/`](benchmarking/) package — plugin-based runner, 5 baselines (NBN, pgmpy, pomegranate, GPyTorch, Pyro), 15+ metrics, YAML configs.

## Running the crash test and benchmarks

All commands assume you are at the repository root (the directory that
contains `pyproject.toml`).

> **First-time / after-pull**: `pip install -e ".[bench,neural,gp,mcmc]"`
> to register the `benchmarking` package and pull the optional baseline
> deps (pgmpy, pomegranate, gpytorch, pyro). The example scripts also
> include a `sys.path` bootstrap so they still run without an editable
> install if you prefer.

There are two crash tests, focused on different things:

* `examples/crash_test.py` — **parameter learning + serial inference**.
  Each baseline fits a model from training data, then answers a small set
  of queries one at a time. Reports accuracy (TV / MAP-acc) vs ground
  truth and per-query latency. Lineup: 4 NBN variants + pgmpy + pomegranate
  + pyro on alarm; 3 NBN variants + pyro + gpytorch on synthetic_hybrid_50.
* `examples/crash_test_inference.py` — **batched inference throughput**.
  Fits each baseline once, then sweeps batch size `B ∈ {1, 16, 256, 1024,
  4096}`. NBN uses its native `model.query_batch(...)` (single GPU launch);
  pgmpy / pyro have no batched API and loop in Python. Plots throughput
  (q/s) vs B on log-log to make NBN's batched-query advantage visible.

```bash
# Headline crash test: alarm + synthetic-50, all baselines, ~30s on CPU.
# Saves figures under examples/figures/.
python examples/crash_test.py

# Smoke run for CI / quick sanity check (~5s, no figures).
python examples/crash_test.py --smoke

# Inference-throughput crash test: sweeps B in {1, 16, 256, 1024, 4096}.
# Saves the throughput-vs-B figure under examples/figures/.
python examples/crash_test_inference.py
python examples/crash_test_inference.py --smoke   # quick CI variant

# Run a configured benchmark suite (writes parquet under results/).
nbn-bench run benchmarking/configs/discrete_small.yaml
nbn-bench run benchmarking/configs/continuous_small.yaml
nbn-bench run benchmarking/configs/scaling.yaml
```

See [`benchmarking/README.md`](benchmarking/README.md) for the full
reproduction guide, the standard 5-kind query battery, the metrics list,
and instructions for adding your own domain or baseline adapter.

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
