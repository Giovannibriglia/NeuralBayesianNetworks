# Interventions

`do=` computes P(targets | do(X = x)): the edges into `X` are cut and its value
is set, instead of observed. On a confounded network the two differ.

```python
import torch
from nbn import NeuralBayesianNetwork, seed_all

# Season confounds Sprinkler and Rain: in the dry season the sprinkler runs
# more and it rains less.
seed_all(0)
n = 20_000
season = torch.bernoulli(torch.full((n,), 0.5)).long()                 # 1 = wet season
sprinkler = torch.bernoulli(torch.where(season == 1, 0.1, 0.6)).long()
rain = torch.bernoulli(torch.where(season == 1, 0.7, 0.1)).long()
wet = torch.bernoulli(torch.where((rain | sprinkler) == 1, 0.95, 0.05)).long()

model = NeuralBayesianNetwork(
    [("Season", "Sprinkler"), ("Season", "Rain"), ("Sprinkler", "WetGrass"), ("Rain", "WetGrass")],
    dict.fromkeys(["Season", "Sprinkler", "Rain", "WetGrass"], ("discrete", 2)),
    device="cpu",
)
model.auto_mechanisms()
model.fit({"Season": season, "Sprinkler": sprinkler, "Rain": rain, "WetGrass": wet})

seen = model.query(["Rain"], {"Sprinkler": 1})            # seeing: P(Rain | Sprinkler = 1)
done = model.query(["Rain"], do={"Sprinkler": 1})          # doing:  P(Rain | do(Sprinkler = 1))
print(f"P(Rain=1 | Sprinkler=1) = {float(seen[1]):.2f}, P(Rain=1 | do(Sprinkler=1)) = {float(done[1]):.2f}")
```

Seeing the sprinkler on suggests the dry season, hence less rain (0.19 in the
true process);
switching it on does not change the weather (0.40, the marginal rain
rate).

## Combining with evidence

`evidence=` and `do=` combine: condition on what you observe in the
intervened world.

```python
print(model.query(["Rain"], {"WetGrass": 1}, do={"Sprinkler": 1}))
```

## One intervention per row

In `query_batch`, `do=` values have a batch axis like evidence, so a whole
dose-response sweep is one call:

```python
do_values = torch.tensor([0, 1, 0, 1])
print(model.query_batch(["WetGrass"], {}, do={"Sprinkler": do_values})[:, 1])
```

`do=` works with every engine, discrete or continuous, and continuous nodes
take real values (`do={"Dose": 0.7}`, or a `[B, 1]` tensor in `query_batch`).

## The mutilated network: `intervene`

`intervene(do)` returns a **new model** in which the intervened nodes have no
incoming edges and a point-mass mechanism (`DeterministicMechanism` for a
discrete node, `DiracGaussianMechanism` for a continuous one). The original
model is not changed. Querying the new model without `do=` gives the same
answers as `do=` on the original:

```python
cut = model.intervene({"Sprinkler": 1})
print(cut.dag.parents("Sprinkler"))                      # []
print(torch.allclose(cut.query(["WetGrass"]), model.query(["WetGrass"], do={"Sprinkler": 1})))
```

`intervene` takes one value per node; use `query_batch(do=...)` to sweep values.
It returns a deep copy, so gradients do not flow through it; use
`sample(n, do=...)` for that ({doc}`autograd`).

The notebook {doc}`../examples/03_interventions` compares seeing and doing on
two confounded networks against the truth, sweeps a continuous dose in one
batched call and runs `do=` on every engine. For counterfactuals (what would
*this* unit have done), see {doc}`scm`.
