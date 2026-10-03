# NeuralBayesianNetworks (NBN)

[![CI](https://github.com/Giovannibriglia/NeuralBayesianNetworks/actions/workflows/ci.yml/badge.svg)](https://github.com/Giovannibriglia/NeuralBayesianNetworks/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nbn.svg)](https://pypi.org/project/nbn/)
[![Python](https://img.shields.io/pypi/pyversions/nbn.svg)](https://pypi.org/project/nbn/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**PyTorch-native Bayesian networks with neural conditional distributions and
GPU-batched inference.**

NBN represents a Bayesian network over a known DAG as an `nn.Module`. Each node
holds a learnable conditional distribution P(X | parents(X)), called a
*mechanism*: a probability table, a linear-Gaussian model, a mixture density
network or a normalising flow. You fit the network from data, then answer
conditional and interventional queries. Thousands of queries run as one
batched tensor operation on CUDA, on Apple-Silicon GPUs (`mps`) or on the CPU.

- **One API for discrete, continuous and hybrid networks.** Mixed
  discrete/continuous networks with non-Gaussian continuous nodes are a
  first-class case.
- **Swappable mechanisms.** Every mechanism is an `nn.Module` behind the same
  interface, from closed-form tables to neural density estimators.
- **Batched inference.** `query_batch` answers B queries with different
  evidence in one call: exact variable elimination for discrete networks,
  likelihood weighting elsewhere, picked automatically.
- **Causal queries.** `do=` interventions work in every engine and can vary
  per batch row.
- **Autograd-friendly.** Likelihoods and samples are differentiable, so NBN
  composes with your own PyTorch models.
- **Reproducible benchmarks.** A suite (`nbn-bench`) compares NBN with pgmpy,
  pomegranate and Pyro on networks with known ground truth.

## Contents

- [Installation](#installation): [Linux](#linux) · [macOS](#macos-apple-silicon-and-intel) · [from source](#from-source)
- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
- [Hardware acceleration](#hardware-acceleration)
- [Benchmark highlights](#benchmark-highlights)
- [Running the benchmarks](#running-the-benchmarks)
- [Development](#development)
- [Citing NBN](#citing-nbn) · [License](#license)

## Installation

NBN needs **Python ≥ 3.10** and **PyTorch ≥ 2.2**. The PyPI package is `nbn`,
and so is the import name.

```bash
pip install nbn                  # core library
pip install "nbn[neural]"        # + normalising flows (zuko)
pip install "nbn[all]"           # + benchmark suite and every baseline library
```

| Extra | Adds |
|---|---|
| `neural` | zuko, for `NormalizingFlowMechanism` / `ConditionalFlowMechanism` |
| `bench` | the `nbn-bench` runner, pgmpy, pomegranate, pandas, plotting |
| `mcmc` | the Pyro baseline for the benchmark suite |
| `all` | everything a benchmark run needs |
| `dev` | tests, linters and docs toolchain (from a checkout) |

Using a virtual environment is recommended.

### Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install "nbn[neural]"
```

For an NVIDIA GPU, install a CUDA build of PyTorch first by following the
selector at [pytorch.org](https://pytorch.org/get-started/locally/). After
that, `pip install nbn` keeps the build you have. Check it with:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # True
```

### macOS (Apple Silicon and Intel)

On Apple-Silicon Macs (M1 and later), NBN uses the GPU through PyTorch's Metal
backend (`mps`). The standard PyPI PyTorch wheel includes it, so there is
nothing extra to install and nothing to configure.

**1. Get Python ≥ 3.10.** macOS does not ship a usable one. Check with
`python3 --version`, and if it is missing or older, install it with either:

```bash
brew install python@3.12        # Homebrew (https://brew.sh)
```

or the official installer from [python.org](https://www.python.org/downloads/macos/).
On macOS the command is `python3` (Homebrew also provides `python3.12`), and
there is no plain `python` until you activate a virtual environment.

**2. Create a virtual environment and install NBN.**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install "nbn[neural]"        # or "nbn[all]" for the benchmark suite
```

**3. Check that the GPU is visible.**

```bash
python -c "import torch; print(torch.backends.mps.is_available())"   # True
```

With the `bench` extra, `nbn-bench check-env` checks the whole environment.
It ends with a line like
`platform darwin; accelerator mps (Apple Metal; float64 unavailable, accumulations run in float32)`.

> **Intel Macs** run NBN on the CPU. PyTorch stopped publishing Intel-macOS
> wheels after 2.2, so pip installs torch 2.2.x there. If you then see a
> NumPy ABI error, run `pip install "numpy<2"`.

### From source

```bash
git clone https://github.com/Giovannibriglia/NeuralBayesianNetworks.git
cd NeuralBayesianNetworks
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,all]"
pytest -m "not slow"             # fast test suite
```

## Quick start

```python
import torch
from nbn import NeuralBayesianNetwork

# Rain → WetGrass ← Sprinkler, and WetGrass → SoilMoisture (continuous)
model = NeuralBayesianNetwork(
    [("Rain", "WetGrass"), ("Sprinkler", "WetGrass"), ("WetGrass", "SoilMoisture")],
    variables={
        "Rain": ("discrete", 2),
        "Sprinkler": ("discrete", 2),
        "WetGrass": ("discrete", 2),
        "SoilMoisture": ("continuous", 1),
    },
    device="auto",  # cuda > mps (Apple Silicon) > cpu
)

# Toy data from a known process
n = 10_000
rain = torch.bernoulli(torch.full((n,), 0.3)).long()
sprinkler = torch.bernoulli(torch.full((n,), 0.4)).long()
wet = ((rain | sprinkler).bool() & (torch.rand(n) < 0.9)).long()
soil = (2.0 * wet + 0.5 * torch.randn(n)).unsqueeze(-1)
data = {"Rain": rain, "Sprinkler": sprinkler, "WetGrass": wet, "SoilMoisture": soil}

model.auto_mechanisms()  # tables for discrete nodes, mixture density networks for continuous
model.fit(data)

# P(Rain | WetGrass = 1)                     ≈ [0.48, 0.52]
print(model.query(["Rain"], evidence={"WetGrass": 1}))

# 1,024 conditional queries in one batched call → shape [1024, 2]
evidence = {"WetGrass": torch.randint(0, 2, (1024,))}
print(model.query_batch(["Rain"], evidence).shape)

# Intervention: P(WetGrass | do(Sprinkler = 1)) ≈ [0.10, 0.90]
print(model.query(["WetGrass"], do={"Sprinkler": 1}))

# Ancestral samples and the (differentiable) log-likelihood of data
samples = model.sample(5)
print(model.log_prob(data).mean())
```

To choose a mechanism per node instead of `auto_mechanisms()`, call
`model.set_mechanism("SoilMoisture", LinearGaussianMechanism())` (all
mechanisms import from `nbn`) before `fit`. More
end-to-end examples (continuous, hybrid, neural, interventional) are in
[`tests/integration/`](tests/integration).

## Core concepts

**Network.** `NeuralBayesianNetwork(edges, variables, device=...)` holds the
DAG, the variable types (`("discrete", cardinality)` or
`("continuous", dim)`) and one mechanism per node. The main methods are
`fit` (all mechanisms, node by node), `update` (fold in new data without
retraining), `query` / `query_batch`, `sample`, `log_prob` and `intervene`.

**Mechanisms** (`nbn.mechanisms`) are the conditional distributions:

| Node type | Mechanisms |
|---|---|
| Discrete | `CategoricalTableMechanism` (closed-form counts), `NeuralCategoricalMechanism` (MLP), `SmoothedEmpiricalCategoricalMechanism`, `BinningCategoricalTable` (continuous parents) |
| Continuous, parametric | `LinearGaussianMechanism` (closed form), `MDNMechanism` (mixture density network), `NormalizingFlowMechanism` / `ConditionalFlowMechanism` (`[neural]` extra) |
| Continuous, non-parametric | `ConditionalKDEMechanism`, `KNNConditionalMechanism`, `FlexCodeMechanism` |
| Fixed | `DeterministicMechanism`, `DiracGaussianMechanism` |

**Inference engines** (`nbn.inference`):

| Engine | Use |
|---|---|
| `TensorVariableElimination` | Exact, log-domain, einsum-based VE for discrete networks; batched over evidence rows |
| `LikelihoodWeightingEngine` | Batched importance sampling; works with any mechanism |
| `HybridRouter` (default) | VE when every node is discrete and the treewidth is ≤ 25, otherwise likelihood weighting |

Pass `engine=` to `query` / `query_batch` to override the default.

**Gradients.** `log_prob` and `sample` (including `sample(n, do=...)`) are
differentiable with respect to the parameters, and parent values or `do=`
values computed by your own `nn.Module` carry gradients back into it.
`query` / `query_batch` are not differentiable (VE detaches at factor build;
likelihood weighting runs under `inference_mode`). `intervene()` returns a
deep copy for querying. To get gradients through an intervention, use
`sample(n, do=...)`.

**Snapshotting.** To revert an optimisation step, copy the state dict:
`snap = copy.deepcopy(mech.state_dict())`. A plain `state_dict()` shares
storage with the live parameters, so an optimiser step silently changes the
snapshot too.

## Hardware acceleration

`device="auto"` (the default for models, `nbn-bench --device` and YAML
configs) picks **CUDA**, then **MPS** (Apple Silicon), then **CPU**. The
results are the same on every backend, up to floating-point precision and
Monte-Carlo noise.

On MPS, NBN works around the Metal backend's gaps for you:

- **No float64 on Metal.** Sufficient statistics (counts, normal equations,
  weighted moments) accumulate in float32 on `mps` and in float64 elsewhere.
- **No `linalg.lstsq` kernel.** The small closed-form Gaussian solves run
  on the CPU. They are `[D_pa+1, D_pa+1]` matrices, so this costs nothing.
- **Other missing kernels.** `import nbn` sets
  `PYTORCH_ENABLE_MPS_FALLBACK=1` on macOS unless you have set it. Any op a
  dependency needs that Metal lacks then runs on the CPU with a one-time
  warning instead of failing. Set it to `0` before importing to opt out.

**When the GPU helps.** GPUs pay off on large batches: many queries per
`query_batch` call, large training sets, wide networks. A toy network like
the quick start is often faster on `device="cpu"`, because kernel-launch
overhead dominates. CI runs the full test suite on Linux and on an
Apple-Silicon macOS runner, where the device-parametrised tests use a real
`mps` device.

## Benchmark highlights

These numbers come from the paper-data run at tag `v0.6c-d` (RTX 4070 Laptop
GPU, 5 seeds, networks of 10–1000 nodes). The full tables and the list of
cells that did not finish are in
[`docs/v0.6c-d/run_summary.md`](docs/v0.6c-d/run_summary.md).

| Setting | NBN | Best external baseline | Result |
|---|---|---|---|
| Inference, continuous linear-Gaussian, 10–1000 nodes | `nbn-lg-lw` | pgmpy `predict` | **8–22× faster** (0.72 s vs 8.5 s at n=1000); W₁ within 0.02 of pgmpy at every size |
| Inference, discrete, n=10 | `nbn-cat-ve` | pgmpy VE | **75× faster** (1.4 ms vs 108 ms) |
| Parameter learning, discrete, n ≥ 50 | `nbn-cat` | pgmpy MLE | **~2.3× lower TV error** (0.146 vs 0.340 at n=1000) |
| Parameter learning, continuous linear-Gaussian | `nbn-lg` | pgmpy | Same accuracy (W₁ ≈ 0.083), **2× faster** at n=1000 |
| Hybrid networks, 10–1000 nodes | `nbn-hybrid` | — | All 25 cells finished (W₁ ≈ 0.08–0.10); no external library in that run had an applicable hybrid baseline |

Pyro's importance sampler was added as a hybrid baseline after that run.
Numbers vary within Monte-Carlo noise across hardware.

## Running the benchmarks

```bash
pip install -U "nbn[all]" && nbn-bench check-env          # must print "environment OK"
nbn-bench inference --config nbn/bench/configs/synthetic/smoke_tests/inference_smoke.yaml
nbn-bench plot results/benchmark_synthetic_smoke_<timestamp> --output-dir results/figures
```

The configs, metrics, the paper-scale runs (`scripts/run_all_benchmarks.sh`),
the figures and how to reproduce the published numbers are documented in
**[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)**. Colab notebooks for each
paper-scale benchmark are in [`notebooks/`](notebooks).

## Development

```bash
pip install -e ".[dev,all]"
pytest -m "not slow"                  # fast suite
pytest                                # everything (slow tests download bnlearn networks)
ruff check nbn/ tests/ scripts/ && mypy nbn/core nbn/mechanisms
```

Repository layout:

    nbn/                 the library: core/, mechanisms/, inference/, learning/, sampling/, update/
    nbn/bench/           benchmark suite: runner, baseline adapters, configs, bnlearn data
    tests/               unit and integration tests
    notebooks/           Colab notebooks for the paper-scale benchmarks
    scripts/             run_all_benchmarks.sh and other helpers
    docs/                benchmark guide, design notes and audits

Releases are cut by pushing a `v*` tag; `.github/workflows/publish.yml` builds
and uploads to PyPI. Known limitations and the roadmap are tracked in the
[issues](https://github.com/Giovannibriglia/NeuralBayesianNetworks/issues).

## Citing NBN

If you use NBN in your research, please cite the software:

```bibtex
@software{briglia_nbn,
  author  = {Briglia, Giovanni},
  title   = {{NeuralBayesianNetworks}: PyTorch-native Bayesian networks with neural mechanisms and GPU-batched inference},
  url     = {https://github.com/Giovannibriglia/NeuralBayesianNetworks},
  license = {Apache-2.0}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
