# Mechanisms

A mechanism is the conditional distribution P(X | pa(X)) of one node. All of
them subclass {class}`~nbn.mechanisms.base.Mechanism` and import from `nbn`
or `nbn.mechanisms`. Pick one per node with `set_mechanism`, or let
`auto_mechanisms()` fill every node you did not set.

## Catalogue

The columns *update* and *weights* say whether the mechanism supports
`model.update(...)` ({doc}`learning`) and `fit(..., weights=...)`.

### Discrete nodes

| Mechanism | Model | Update | Weights | When to use |
|---|---|---|---|---|
| `CategoricalTableMechanism(alpha=0.0)` | a table of counts (closed form), optional pseudo-count | yes | yes | discrete parents; the default. Set `alpha > 0` when some parent configurations are rare |
| `SmoothedEmpiricalCategoricalMechanism(alpha=1.0)` | a table with Lidstone smoothing | yes | yes | the same, with smoothing on by default |
| `NeuralCategoricalMechanism(n_classes)` | an MLP from the parents to logits | yes | yes | continuous parents with a smooth effect, or too many parent configurations for a table |
| `BinningCategoricalTable(parent_kinds, n_bins, n_categories)` | a table over quantile bins of continuous parents | no | yes | continuous parents, closed form; what `auto_mechanisms()` assigns there |
| `KNNConditionalMechanism(discrete_child=True)` | neighbour label frequencies | yes | no | non-parametric, mixed parents |

### Continuous nodes

| Mechanism | Model | Update | Weights | When to use |
|---|---|---|---|---|
| `LinearGaussianMechanism()` | `N(W pa + b, diag(sigma^2))`, closed form | yes | yes | linear effects with Gaussian noise; fastest, exact updates |
| `MDNMechanism(num_components=5)` | mixture density network | yes | yes | multimodal or heteroscedastic densities; the `auto_mechanisms()` default |
| `NormalizingFlowMechanism()` / `ConditionalFlowMechanism()` | neural spline flow (`[neural]` extra) | no | yes | flexible, skewed, multivariate densities |
| `ConditionalKDEMechanism()` | Nadaraya-Watson kernel density | yes | yes | low-dimensional parents, enough data, no training loop |
| `KNNConditionalMechanism()` | k-nearest-neighbour density | yes | no | non-parametric baseline, adaptive bandwidth |
| `FlexCodeMechanism()` | orthogonal-series (cosine basis) density | no | yes | smooth densities on a bounded range |

### Fixed

| Mechanism | Model | When to use |
|---|---|---|
| `DeterministicMechanism(value, cardinality=None)` | a point mass | a known constant; `intervene()` uses it for discrete `do=` |
| `DiracGaussianMechanism(value)` | a very narrow Gaussian at `value` | a fixed continuous value; `intervene()` uses it for continuous `do=` |

The structural mechanisms of the SCM layer ({doc}`scm`) are mechanisms too,
so they work everywhere a mechanism does.

## Choosing per node

```python
import torch
from nbn import (CategoricalTableMechanism, LinearGaussianMechanism, MDNMechanism,
                 NeuralBayesianNetwork, seed_all)

seed_all(0)
model = NeuralBayesianNetwork(
    [("Rain", "WetGrass"), ("WetGrass", "Soil"), ("Soil", "Growth")],
    {"Rain": ("discrete", 2), "WetGrass": ("discrete", 2),
     "Soil": ("continuous", 1), "Growth": ("continuous", 1)},
    device="cpu",
)
model.set_mechanism("WetGrass", CategoricalTableMechanism(alpha=1.0))
model.set_mechanism("Soil", MDNMechanism(num_components=3))
model.set_mechanism("Growth", LinearGaussianMechanism())
model.auto_mechanisms()            # fills only the nodes left unset (here: Rain)

n = 2000
rain = torch.bernoulli(torch.full((n,), 0.3)).long()
wet = torch.where(rain == 1, torch.bernoulli(torch.full((n,), 0.9)),
                  torch.bernoulli(torch.full((n,), 0.2))).long()
soil = torch.where(wet == 1, 2.0, -1.0) + 0.5 * torch.randn(n)
growth = 0.8 * soil + 0.5 + 0.2 * torch.randn(n)
data = {"Rain": rain, "WetGrass": wet, "Soil": soil[:, None], "Growth": growth[:, None]}
model.fit(data, consolidate=False)
print({node: type(m).__name__ for node, m in model.mechanisms.items()})
```

`auto_mechanisms(default_discrete="categorical_table", default_continuous="mdn")`
also accepts `"neural_categorical"` and `"lg"` / `"linear_gaussian"`. A
discrete node with a continuous parent gets a `BinningCategoricalTable`
(`n_bins=5` quantile bins) instead of a plain table, which cannot index a
table by a real number.

## Using a mechanism on its own

A fitted mechanism is an ordinary module. `forward(parents)` returns a
`torch.distributions.Distribution`; parents are packed as one `[B, D_pa]`
tensor in `model.dag.parents(node)` order.

```python
growth = model.mechanisms["Growth"]
dist = growth(torch.tensor([[0.0], [2.0]]))       # Soil = 0 and Soil = 2
print(dist.mean.squeeze(-1))                       # about [0.5, 2.1]
print(growth.log_prob(torch.tensor([[0.5]]), torch.tensor([[0.0]])))
```

Discrete mechanisms expose the whole table: `tabulate(parent_cards)` returns
logits with one axis per parent and the node's states last; a softmax over the
last axis gives the probabilities.

```python
table = model.mechanisms["WetGrass"].tabulate([2]).softmax(-1)  # P(WetGrass | Rain)
print(table)
```

## Writing your own

Subclass `Mechanism` and implement `forward(parents)` (return a distribution),
`fit_local(x, parents, **kwargs)` (return a dict of metrics). `log_prob` and
`sample` default to the distribution's own; override them when that is not
enough. A discrete mechanism also implements `tabulate` and `is_fitted`, which
variable elimination reads. Set the class attributes `is_discrete`,
`supports_update` (then implement `update_local`) and `supports_weights` to
describe what it can do; see
{class}`~nbn.mechanisms.base.Mechanism` and the shipped mechanisms in
`nbn/mechanisms/` for the shape contracts.

The notebook {doc}`../examples/01_parameter_learning` fits every family on one
hybrid network and scores it against the true densities;
{doc}`../examples/06_mechanism_gallery` compares the six continuous
mechanisms on a bimodal, a heteroscedastic and a skewed density.
