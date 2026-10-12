# Sampling

`model.sample(n)` draws `n` joint samples by ancestral sampling: each node in
topological order, given its sampled parents. It returns a dict with one
`[n, D]` tensor per node, on the model's device.

```python
import torch
from nbn import LinearGaussianMechanism, NeuralBayesianNetwork, ancestral_sample, seed_all

seed_all(0)
n = 5000
rain = torch.bernoulli(torch.full((n,), 0.3)).long()
wet = torch.bernoulli(torch.where(rain == 1, 0.9, 0.2)).long()
soil = 2.0 * wet + 0.5 * torch.randn(n)
model = NeuralBayesianNetwork(
    [("Rain", "WetGrass"), ("WetGrass", "Soil")],
    {"Rain": ("discrete", 2), "WetGrass": ("discrete", 2), "Soil": ("continuous", 1)},
    device="cpu",
)
model.set_mechanism("Soil", LinearGaussianMechanism())
model.auto_mechanisms()
model.fit({"Rain": rain, "WetGrass": wet, "Soil": soil[:, None]})

draws = model.sample(10_000)
print({k: tuple(v.shape) for k, v in draws.items()})
print("P(Rain = 1) from samples:", float(draws["Rain"].float().mean()))
```

`nbn.ancestral_sample(model, n)` is the function behind the method.

## Log-densities

`return_log_prob=True` also returns the `[n]` joint log-density of each
sample under the model, the same number `log_prob` gives for those rows:

```python
rows, lp = model.sample(5, return_log_prob=True)
print(torch.allclose(lp, model.log_prob(rows)))
```

## Evidence: clamping, not conditioning

`sample(n, evidence=...)` **clamps** the evidence nodes to the given values,
but their ancestors are still drawn from the prior, so the samples are not
from the posterior. `return_log_weight=True` returns the log-likelihood of the
evidence for each sample; weighting by `exp(log_weight)` turns the samples
into a posterior estimate (likelihood weighting):

```python
rows, log_w = model.sample(20_000, evidence={"Soil": 2.0}, return_log_weight=True)
w = torch.softmax(log_w, dim=0)
print("P(Rain = 1 | Soil = 2):", float((w * rows["Rain"].squeeze(-1).float()).sum()))
print("query:                 ", float(model.query(["Rain"], {"Soil": 2.0}, n_samples=20_000)[1]))
```

With both flags the return value is `(samples, log_prob, log_weight)`.

## Interventional samples

`sample(n, do={...})` draws from the intervened network: the node is set to
the value and its descendants follow.

```python
under_do = model.sample(10_000, do={"WetGrass": 1})
print("E[Soil | do(WetGrass = 1)]:", float(under_do["Soil"].mean()))   # about 2.0
```

`do=` takes one value per node here; `query_batch(do=...)` varies it per row.
Unlike `intervene()`, `sample(n, do=...)` uses the live parameters, so the
samples stay differentiable ({doc}`autograd`).

The notebook {doc}`../examples/04_sampling` checks samples against the true
marginals and joint, and the weighted posterior estimate against the exact
one.
