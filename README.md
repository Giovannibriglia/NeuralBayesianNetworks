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

## Crash test (the page-1 figure)

```bash
python examples/crash_test.py        # examples/figures/crash_test_summary.{pdf,svg,png,tex}
python examples/render_throughput_scaling.py  # examples/figures/scaling_nodes_batched_throughput.{pdf,svg,png}
```

Pre-rendered v0.3.0 outputs:
[`crash_test_summary.pdf`](examples/figures/crash_test_summary.pdf),
[`scaling_nodes_batched_throughput.pdf`](examples/figures/scaling_nodes_batched_throughput.pdf).

The v0.3 figure uses **horizontal bars with constrained-layout** (no
tick-label collisions), groups baselines by family with hatched-CPU /
solid-CUDA encoding, and stamps a reproducibility footer (NBN version,
seed, git sha, torch version, GPU name) on every output. Discrete
accuracy is measured against pgmpy's exact VE; continuous accuracy
against MC-rejection ground truth on the synthetic SCM (when
`--with-ground-truth` is in scope; W₁ panel populated when ground truth
is available, otherwise shows the "needs ground-truth-builder" hint).

### Headline numbers (v0.3.0, alarm B=1024, CPU)

`nbn_ve` 134,248 q/s vs `pgmpy` 2,427 q/s — 55× speedup from v0.3.1's
vectorised `query_batch`. NBN-VE is strictly above pgmpy at every
measured n on the bnlearn ladder (cancer, asia, child, alarm). See
[`scaling_nodes_batched_throughput.pdf`](examples/figures/scaling_nodes_batched_throughput.pdf).

For the **batched-inference** crash test on canonical bnlearn networks
(NBN's `query_batch` vs serial-loop competitors), see
`examples/crash_test_inference_bnlearn.py`.  The headline v0.4
batched-inference crash test on synthetic BNs lives at
`examples/crash_test_inference.py`.

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
* `examples/crash_test_inference.py` — **v0.4 synthetic batched
  inference**.  Drives the synthetic-BN runner from
  `benchmarking.crash_test_runner` against the four families
  (`discrete`, `continuous_lg`, `continuous_nongauss`, `hybrid`) at
  fixed `B=1024`; reports throughput and accuracy.
* `examples/crash_test_inference_bnlearn.py` — **v0.2 bnlearn batched
  inference**.  Fits each baseline once on a bnlearn network, then
  sweeps batch size `B ∈ {1, 16, 256, 1024, 4096}` and plots
  throughput-vs-B on log-log.  Kept as the small-network parity demo;
  referenced by CI smoke.

```bash
# Headline crash test: alarm + synthetic-50, all baselines, ~30s on CPU.
# Saves figures under examples/figures/.
python examples/crash_test.py

# Smoke run for CI / quick sanity check (~5s, no figures).
python examples/crash_test.py --smoke

# v0.2 bnlearn inference-throughput crash test (B sweep, kept as parity demo).
python examples/crash_test_inference_bnlearn.py
python examples/crash_test_inference_bnlearn.py --smoke   # quick CI variant

# v0.4 synthetic inference crash test (fixed B=1024, four-family sweep).
python examples/crash_test_inference.py --config benchmarking/configs/crash_test_smoke.yaml

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
| v0.3 page-1 crash test + vectorised batched VE | ✅ — [v0.3.0 release](https://github.com/Giovannibriglia/NeuralBayesianNetworks/releases/tag/v0.3.0) |
| GPU performance work (CUDA-graphs, fused log-CPT) | tracked in [#7](https://github.com/Giovannibriglia/NeuralBayesianNetworks/issues/7) |
| 5-seed multi-replicate error-bar pipeline | tracked in [#7](https://github.com/Giovannibriglia/NeuralBayesianNetworks/issues/7) |
| Lauritzen-Jensen analytic CG ground truth + NUTS gold ground truth | tracked in [#7](https://github.com/Giovannibriglia/NeuralBayesianNetworks/issues/7) |

## Citation

```bibtex
@software{briglia2025nbn,
  author = {Briglia, Giovanni},
  title  = {NeuralBayesianNetworks: PyTorch-native Bayesian Networks with Neural Mechanisms},
  year   = {2025},
  url    = {https://github.com/Giovannibriglia/NeuralBayesianNetworks},
}
```
