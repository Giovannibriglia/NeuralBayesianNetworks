# NBN: Neural Bayesian Networks

**PyTorch-native Bayesian networks with neural conditional distributions and
GPU-batched inference.**

NBN represents a Bayesian network over a known DAG as an `nn.Module`. Each node
holds a learnable conditional distribution P(X | parents(X)), called a
*mechanism*: a probability table, a linear-Gaussian model, a mixture density
network or a normalising flow. You fit the network from data, then answer
conditional and interventional queries. Thousands of queries run as one
batched tensor operation on CUDA, on Apple-Silicon GPUs (`mps`) or on the CPU.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting_started/index
:link-type: doc

Install NBN on Linux (CPU or CUDA) or macOS (CPU or Apple MPS), then fit and
query a first network in twenty lines.
:::

:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc

Concepts, the mechanism catalogue, learning, inference engines,
interventions, sampling, devices, persistence, autograd, the SCM layer and
the benchmark suite.
:::

:::{grid-item-card} Examples
:link: examples/index
:link-type: doc

Twenty-one executed notebooks with plots, each checked against a known ground
truth.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Every public class and function, generated from the docstrings.
:::
::::

## Key features

- **One API for discrete, continuous and hybrid networks.** Mixed
  discrete/continuous networks with non-Gaussian continuous nodes are a
  first-class case.
- **Swappable mechanisms.** Every mechanism is an `nn.Module` behind the same
  interface, from closed-form tables to neural density estimators
  ({doc}`catalogue <user_guide/mechanisms>`).
- **Batched inference.** `query_batch` answers B queries with different
  evidence in one call: exact variable elimination for discrete networks,
  likelihood weighting elsewhere, picked automatically
  ({doc}`inference <user_guide/inference>`).
- **Causal queries.** `do=` interventions work in every engine and can vary
  per batch row ({doc}`interventions <user_guide/interventions>`); the
  {doc}`SCM layer <user_guide/scm>` adds explicit noise, abduction and
  counterfactuals.
- **Answers with guarantees.** Certified sampling intervals, exact answers by
  conditioning when memory is short, and PAC answers that also cover the
  learning error ({doc}`inference <user_guide/inference>`,
  {doc}`probability sets <user_guide/probability_sets>`).
- **Incremental updates.** `update()` folds new data into a fitted network
  without the old data, atomically, with optional forgetting
  ({doc}`learning <user_guide/learning>`).
- **Autograd-friendly.** Likelihoods and samples are differentiable, so NBN
  composes with your own PyTorch models ({doc}`autograd <user_guide/autograd>`).
- **Reproducible benchmarks.** A suite (`nbn-bench`) compares NBN with pgmpy,
  pomegranate and Pyro on networks with known ground truth
  ({doc}`benchmarks <user_guide/benchmarks>`).

## Install

```bash
pip install nbn                  # core library
pip install "nbn[neural]"        # + normalising flows (zuko)
pip install "nbn[all]"           # + benchmark suite, every baseline, notebook extras
```

NBN needs Python 3.10 or newer and PyTorch 2.2 or newer. See
{doc}`getting_started/installation` for CUDA, Apple Silicon and the extras.

## Quick start

```python
import torch
from nbn import NeuralBayesianNetwork

# Rain -> WetGrass <- Sprinkler, and WetGrass -> SoilMoisture (continuous)
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

n = 10_000
rain = torch.bernoulli(torch.full((n,), 0.3)).long()
sprinkler = torch.bernoulli(torch.full((n,), 0.4)).long()
wet = ((rain | sprinkler).bool() & (torch.rand(n) < 0.9)).long()
soil = (2.0 * wet + 0.5 * torch.randn(n)).unsqueeze(-1)
data = {"Rain": rain, "Sprinkler": sprinkler, "WetGrass": wet, "SoilMoisture": soil}

model.auto_mechanisms()  # tables for discrete nodes, mixture density networks for continuous
model.fit(data)

print(model.query(["Rain"], evidence={"WetGrass": 1}))          # P(Rain | WetGrass = 1)
print(model.query_batch(["Rain"], {"WetGrass": torch.randint(0, 2, (1024,))}).shape)
print(model.query(["WetGrass"], do={"Sprinkler": 1}))           # P(WetGrass | do(Sprinkler = 1))
```

The {doc}`getting_started/quickstart` walks through it line by line.

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
user_guide/index
examples/index
api/index
development/index
citing
```
