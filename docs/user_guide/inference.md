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

## Answers with a guarantee, within a budget

Exact inference costs time and memory exponential in the width of the
elimination; sampling scales but gives no error bound. The tools below are all
opt-in (the defaults above do not change) and work on discrete targets.

| Call | Returns | Guarantee |
|---|---|---|
| `certified_query(model, target, evidence)` | an interval `[lower, upper]` for every class | all intervals hold together with probability at least `1 - delta` over the sampling |
| `TensorVariableElimination(on_oom="condition")` | the exact answer when plain VE does not fit in memory | exact (conditioning on a cutset) |
| `conditioned_query(..., enumerate_limit=k)` or `time_budget_s=` | bounds that tighten as more of the cutset is enumerated | holds with probability 1; exact once everything is enumerated |
| `HybridRouter(fallback="certified" or "condition")` | the router's answer, with a certificate when sampling answered | as above |
| `query_many(model, queries, workers=4)` | a list of answers, computed in parallel | the same answers as a loop of `query` |

These bound the *inference* error for the fitted model. They say nothing
about how far the fitted model is from the truth; for that, see the PAC
answers in {doc}`probability_sets`.

**Certified sampling.** `certified_query` runs likelihood weighting a fixed
number of times and turns concentration bounds on the weights into an
interval for every class of the target. The width falls like
`1 / sqrt(n_samples)` and grows like `W / P(e)` for unlikely evidence (`W`
bounds the likelihood weight), never with the tree-width.

```python
from nbn.inference import certified_query

ans = certified_query(disc, "Rain", {"WetGrass": 1}, n_samples=200_000, delta=0.05)
print(ans.lower, ans.upper)       # tensor([0.2963, 0.6925]) tensor([0.3075, 0.7037])
print(ans.estimate)               # inside the intervals; exact VE gives [0.3013, 0.6987]

fast = certified_query(disc, "Rain", {"WetGrass": 1}, time_budget_s=1.0, target_width=0.02)
print(fast.n_samples, fast.max_width)
```

With `time_budget_s=` or `target_width=` the query samples at checkpoints of
doubling size and stops at the first one that meets the width, or when time
runs out. The confidence budget is split over the checkpoints, so the interval
is valid whatever the stopping time. `collapse="auto"` samples only a cutset
and computes the rest exactly; it returns the exact answer when the whole
query fits, and helps most on a GPU.

Evidence nodes must be discrete with discrete parents, so that the weight
bound `W` can be computed from each evidence mechanism's `log_prob` (it is
checked against the mechanism's table). Continuous evidence has unbounded
densities and is refused, and so is evidence with several rows: certify one
row at a time.

**Exact answers when memory is short.** By default, plain VE raises
`OutOfMemoryError` when a query does not fit the memory budget. With
`on_oom="condition"` it picks a small set of variables (a cutset), runs exact
VE for each of their joint values in one batched computation, and adds the
results. The answer is the same as plain VE; the time grows with the number of
cutset values (at most `max_conditioning`), and threads or a GPU divide it.

```python
from nbn import TensorVariableElimination
from nbn.inference import conditioned_query

ve = TensorVariableElimination(memory_budget=2**30, on_oom="condition")
print(disc.query(["Rain"], {"WetGrass": 1}, engine=ve))      # exact

b = conditioned_query(disc, "Rain", {"WetGrass": 1}, enumerate_limit=1)
print(b.exact, b.lower, b.upper)      # sound bounds; exact when everything was enumerated
```

`conditioned_query` with `enumerate_limit=` or `time_budget_s=` stops early
and returns bounds that hold with probability 1 (bounded conditioning). They
are tight only when a few cutset values carry most of the probability; with
unlikely evidence they can stay close to `[0, 1]`.

**Routing and many queries.** `HybridRouter(fallback="certified")` answers a
discrete query that is too large for VE with `certified_query` instead of
plain LW; `router.last_certificate` holds the intervals when that happened
(`None` otherwise). `fallback="condition"` tries conditioning first.
`query_width_check=True` makes the router compute each query's own elimination
width before choosing (some dense networks then get more exact answers).

```python
from nbn.inference import LikelihoodWeightingEngine, query_many

queries = [(["Rain"], {"WetGrass": w}) for w in (0, 1)] + [(["Cloudy"], {"Sprinkler": s}) for s in (0, 1)]
answers = query_many(disc, queries, engine=LikelihoodWeightingEngine(n_samples=20_000), workers=4)
```

`query_many` gives one engine copy to each worker thread. It pays off for
sampling engines, whose per-node loop cannot use more threads on its own. On
Apple MPS it runs the queries one after the other, because Metal does not
accept work from several threads at once.

`query` and `query_batch` are not differentiable (see {doc}`autograd`). The
notebook {doc}`../examples/02_inference` compares every engine with the exact
answer, including error against particles and a continuous target;
{doc}`../examples/19_certified_and_budgeted_inference` shows the certified,
conditioned and parallel answers above on a dense network.
