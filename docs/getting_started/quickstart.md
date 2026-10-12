# Quick start

This page builds a four-node hybrid network, fits it from data and asks it
conditional, batched and interventional questions. Every step links to the
page of the user guide that covers it in depth.

## 1. Declare the graph and the variables

```python
import torch
from nbn import NeuralBayesianNetwork, seed_all

seed_all(0)
model = NeuralBayesianNetwork(
    [("Rain", "WetGrass"), ("Sprinkler", "WetGrass"), ("WetGrass", "SoilMoisture")],
    variables={
        "Rain": ("discrete", 2),          # 2 states: 0, 1
        "Sprinkler": ("discrete", 2),
        "WetGrass": ("discrete", 2),
        "SoilMoisture": ("continuous", 1),  # one real value
    },
    device="auto",  # cuda > mps (Apple Silicon) > cpu
)
print(model)
```

The DAG is a list of `(parent, child)` edges (or a `networkx.DiGraph`). A
variable is `("discrete", cardinality)` or `("continuous", dim)`; see
{doc}`../user_guide/concepts`.

## 2. Data

Data are a dict from node name to a tensor with one row per sample: `[N]` or
`[N, 1]` integer states for a discrete node, `[N, D]` floats for a continuous
one. The tensors may live on any device; NBN moves them to the model's.

```python
n = 10_000
rain = torch.bernoulli(torch.full((n,), 0.3)).long()
sprinkler = torch.bernoulli(torch.full((n,), 0.4)).long()
wet = ((rain | sprinkler).bool() & (torch.rand(n) < 0.9)).long()
soil = (2.0 * wet + 0.5 * torch.randn(n)).unsqueeze(-1)
data = {"Rain": rain, "Sprinkler": sprinkler, "WetGrass": wet, "SoilMoisture": soil}
```

## 3. Choose mechanisms and fit

```python
model.auto_mechanisms()   # tables for discrete nodes, mixture density networks for continuous ones
history = model.fit(data)
print({node: type(m).__name__ for node, m in model.mechanisms.items()})
print(f"fitted in {history.wall_clock_s:.1f} s on {history.device}")
```

To choose the mechanism of a node yourself, call `set_mechanism` before
`auto_mechanisms()`, which fills only the nodes left unset:
`model.set_mechanism("SoilMoisture", LinearGaussianMechanism())`.
{doc}`../user_guide/mechanisms` lists every mechanism and when to use it;
{doc}`../user_guide/learning` covers `fit` and `update`.

## 4. Ask questions

```python
# P(Rain | WetGrass = 1): a probability vector over the 2 states of Rain
print(model.query(["Rain"], evidence={"WetGrass": 1}))

# 1024 queries with different evidence in one batched call -> [1024, 2]
evidence = {"WetGrass": torch.randint(0, 2, (1024,))}
print(model.query_batch(["Rain"], evidence).shape)

# Intervention: P(WetGrass | do(Sprinkler = 1))
print(model.query(["WetGrass"], do={"Sprinkler": 1}))

# Ancestral samples and the (differentiable) log-likelihood of data
samples = model.sample(5)
print(model.log_prob(data).mean())
```

The default engine (`HybridRouter`) answered the discrete queries exactly by
variable elimination. With continuous evidence or a continuous target it
switches to likelihood weighting; see {doc}`../user_guide/inference`.

## Next

- The {doc}`../user_guide/index` explains each part of the API.
- The {doc}`../examples/index` are executed notebooks with plots, one per
  feature, each checked against a known ground truth.
