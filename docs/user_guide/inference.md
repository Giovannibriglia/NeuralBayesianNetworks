# Inference

`query` answers one question, `query_batch` answers many at once, and an
*engine* does the work. Two networks serve as examples on this page: a
discrete one, where exact inference applies, and a hybrid one.

```python
import torch
from nbn import MDNMechanism, NeuralBayesianNetwork, seed_all

seed_all(0)
n = 5000
cloudy = torch.bernoulli(torch.full((n,), 0.5)).long()
rain = torch.bernoulli(torch.where(cloudy == 1, 0.8, 0.2)).long()
sprinkler = torch.bernoulli(torch.where(cloudy == 1, 0.1, 0.5)).long()
wet = torch.bernoulli(torch.where((rain | sprinkler) == 1, 0.9, 0.05)).long()
soil = torch.where(wet == 1, 2.0, -1.0) + 0.5 * torch.randn(n)

disc = NeuralBayesianNetwork(
    [("Cloudy", "Rain"), ("Cloudy", "Sprinkler"), ("Rain", "WetGrass"), ("Sprinkler", "WetGrass")],
    dict.fromkeys(["Cloudy", "Rain", "Sprinkler", "WetGrass"], ("discrete", 2)),
    device="cpu",
)
disc.auto_mechanisms()
disc.fit({"Cloudy": cloudy, "Rain": rain, "Sprinkler": sprinkler, "WetGrass": wet})

hybrid = NeuralBayesianNetwork(
    [("Rain", "WetGrass"), ("WetGrass", "Soil")],
    {"Rain": ("discrete", 2), "WetGrass": ("discrete", 2), "Soil": ("continuous", 1)},
    device="cpu",
)
hybrid.set_mechanism("Soil", MDNMechanism(num_components=2))
hybrid.auto_mechanisms()
hybrid.fit({"Rain": rain, "WetGrass": wet, "Soil": soil[:, None]}, consolidate=False)
```

## `query`

```python
p = disc.query(["Rain"], evidence={"WetGrass": 1})
print(p)                     # P(Rain | WetGrass = 1): a [2] probability vector
```

Evidence values are Python scalars or tensors (`[D]` for a `D`-dimensional
continuous node). What comes back depends on the targets:

| Targets | Result |
|---|---|
| one discrete node | `[K]` probabilities |
| several discrete nodes, answered exactly | the joint `[K_1, ..., K_n]`, axes in `targets` order |
| a continuous node, or any query answered by sampling | weighted particles `(weights [1, S], samples [1, S, D])` |

```python
joint = disc.query(["Rain", "Sprinkler"], {"WetGrass": 1})
print(joint.shape, float(joint.sum()))          # [2, 2], sums to 1

weights, samples = hybrid.query(["Soil"], {"Rain": 1}, n_samples=4096)
print(weights.shape, samples.shape)             # [1, 4096] and [1, 4096, 1]
print(float((weights * samples[..., 0]).sum())) # posterior mean of Soil
```

`map_query(targets, evidence)` returns the most probable state of a single
discrete target, as `{target: index}`.

```python
print(disc.map_query(["Rain"], {"WetGrass": 1, "Sprinkler": 0}))
```

## `query_batch`

`query_batch` answers `B` queries with different evidence in one call: each
evidence tensor has a leading batch axis `[B]` (or `[B, D]`), and the result
has one too. It is much faster than a loop of `query` calls, on the CPU and
even more on a GPU.

```python
evidence = {"WetGrass": torch.tensor([0, 1, 1, 0]), "Sprinkler": torch.tensor([0, 0, 1, 1])}
print(disc.query_batch(["Rain"], evidence))      # [4, 2]

grid = torch.linspace(-3.0, 4.0, 8)[:, None]     # continuous evidence: [8, 1]
print(hybrid.query_batch(["Rain"], {"Soil": grid}, n_samples=4096)[:, 1])
```

Every row may set a different `do=` value too ({doc}`interventions`).

## Engines

| Engine | Answers | Notes |
|---|---|---|
| `HybridRouter()` | everything; the default | exact VE when it applies and fits in memory, LW otherwise |
| `TensorVariableElimination()` | discrete networks, exactly | log-domain, einsum-based; batched over evidence rows; caches factors and plans per model |
| `LikelihoodWeightingEngine(n_samples=2048)` | any network | importance sampling with the evidence clamped; error falls like `1 / sqrt(n_samples)` |
| `AmortizedISEngine()` | any network | learns an evidence-aware proposal once, then importance sampling; falls back to LW per query when the proposal is poor |
| `AmortizedVIEngine()` | any network | learns an evidence-conditioned mean-field approximation once, then one forward pass per query; a fit-time quality gate falls back to LW when it is not accurate enough |

Choose per call with `engine=`, or for the model with `default_engine=` at
construction (an instance, or `"tensor_ve"`, `"likelihood_weighting"`,
`"hybrid"`). Sampling engines accept `n_samples=` per call.

```python
from nbn import HybridRouter, LikelihoodWeightingEngine, TensorVariableElimination

exact = disc.query(["Rain"], {"WetGrass": 1}, engine=TensorVariableElimination())
approx = disc.query(["Rain"], {"WetGrass": 1}, engine=LikelihoodWeightingEngine(), n_samples=20_000)
print(exact, approx)

router = HybridRouter()
disc.query(["Rain"], {"WetGrass": 1}, engine=router)
print(router.last_engine)        # "tensor_ve"
hybrid.query(["Rain"], {"Soil": 2.0}, engine=router)
print(router.last_engine)        # "likelihood_weighting"
```

**How `HybridRouter` routes.** Any continuous node, or an elimination whose
treewidth exceeds `treewidth_threshold` (25), goes to likelihood weighting.
Otherwise it prices the actual elimination plan of the query against the
device's memory budget, and uses VE when one evidence row fits (larger
batches are split into chunks that fit; see {doc}`devices`). If VE still runs
out of memory, the query is retried with LW. A discrete query that is
downgraded to LW logs a warning once per query shape, and `last_engine` names
the engine that answered.

**Effective sample size.** The sampling engines return the ESS fraction of
each row with `return_ess=True`; a value near 0 means a few particles carry
all the weight (evidence far in the tails), and more particles or the
amortized IS engine help.

```python
probs, ess = hybrid.query_batch(["Rain"], {"Soil": grid}, n_samples=4096, return_ess=True)
print(ess)
```

**Amortized engines** are trained once per fitted model with
`engine.train_proposal(model)` and then passed as `engine=`:

```python
from nbn.inference.amortized_is import AmortizedISEngine

seed_all(1)
ais = AmortizedISEngine()
info = ais.train_proposal(hybrid, time_budget_s=30)
print(info["proposal_used"], round(info["ess_fraction"], 2))
print(hybrid.query_batch(["Rain"], {"Soil": grid}, engine=ais)[:, 1])
```

`query` and `query_batch` are not differentiable (see {doc}`autograd`). The
notebook {doc}`../examples/02_inference` compares every engine with the exact
answer, including error against particles and a continuous target.
