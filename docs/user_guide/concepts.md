# Core concepts

An NBN model is a Bayesian network over a **known DAG**: you give the graph,
NBN learns the conditional distributions. Four objects carry the whole API.

## The network

`NeuralBayesianNetwork(dag, variables, default_engine="auto", device="auto")`
is an `nn.Module`. It holds the DAG, one {class}`~nbn.core.variables.Variable`
per node and one mechanism per node, so `.to(device)`, `.parameters()` and
`.state_dict()` work as for any PyTorch module.

| Method | Does |
|---|---|
| `set_mechanism(node, mech)` / `auto_mechanisms()` | choose the conditional distribution of each node |
| `fit(data)` | learn every mechanism, node by node ({doc}`learning`) |
| `update(data)` | fold new data in without the old data ({doc}`learning`) |
| `query` / `query_batch` / `map_query` | posterior and interventional inference ({doc}`inference`) |
| `intervene(do)` | the mutilated network as a new model ({doc}`interventions`) |
| `sample(n)` / `log_prob(data)` | ancestral samples and the joint log-likelihood ({doc}`sampling`) |
| `save` / `load` / `from_bif` | persistence ({doc}`persistence`) |

## The DAG

Pass a list of `(parent, child)` edges or a `networkx.DiGraph`. The order of a
node's parents is the order of its edges, and it is the axis order of the
node's table; `model.dag.parents(node)` returns it.

```python
from nbn import NeuralBayesianNetwork

model = NeuralBayesianNetwork(
    [("A", "C"), ("B", "C"), ("C", "D")],
    variables={"A": ("discrete", 2), "B": ("discrete", 3),
               "C": ("discrete", 4), "D": ("continuous", 1)},
    device="cpu",
)
print(model.dag.parents("C"), model.dag.topological_order())
```

Every node of the graph needs a variable spec. A variable that appears in no
edge becomes an isolated node (when the DAG is given as an edge list). Node names are free-form strings (bnlearn names such as `pm2.5`
work).

## Variables and data

| Spec | Meaning | Data tensor |
|---|---|---|
| `("discrete", K)` | states `0 .. K-1` | `[N]` or `[N, 1]` integers |
| `("continuous", D)` | a vector in R^D | `[N, D]` floats (`[N]` when D = 1) |

Data are a `dict` from node name to tensor, one row per sample. NBN moves the
tensors to the model's device; they can be created on the CPU.

## Mechanisms

A mechanism is the conditional distribution P(X | pa(X)) of one node: an
`nn.Module` whose `forward(parents)` returns a `torch.distributions`
distribution, with `log_prob(x, parents)`, `sample(parents, n)` and its own
`fit_local` / `update_local`. The parents arrive packed as one `[B, D_pa]`
tensor in `dag.parents(node)` order. {doc}`mechanisms` lists them all.

## Inference engines

An engine answers `query` / `query_batch`. The default `HybridRouter` uses
exact variable elimination (`TensorVariableElimination`) when every node is
discrete and the elimination fits in memory, and likelihood weighting
(`LikelihoodWeightingEngine`) otherwise. Pass `engine=` per call or
`default_engine=` at construction to choose; see {doc}`inference`.

The notebook {doc}`../examples/01_parameter_learning` builds a hybrid network
with every mechanism family; {doc}`../examples/02_inference` runs every
engine.
