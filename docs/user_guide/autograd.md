# Gradients and autograd

A fitted NBN is an `nn.Module`, and its likelihoods and samples are
differentiable, so a network can sit inside a larger PyTorch model.

| Call | Differentiable with respect to |
|---|---|
| `log_prob(data)`, `log_prob(data, per_node=True)` | the mechanisms' parameters and any tensor in `data` computed by your code |
| `mechanism.log_prob(x, parents)` | the same, for one node |
| `sample(n)`, `sample(n, do=...)` | the parameters, and the `do=` values (for reparameterisable mechanisms such as linear-Gaussian and MDN) |
| `query`, `query_batch` | **not differentiable**: variable elimination detaches when it builds its factors, and likelihood weighting runs without autograd. The results are ordinary tensors |
| `intervene()` | **not differentiable**: it returns a deep copy with fresh parameters |

To differentiate through an intervention, use `sample(n, do=...)`, which
applies it to the live parameters.

```python
import torch
from nbn import LinearGaussianMechanism, NeuralBayesianNetwork, seed_all

seed_all(0)
dose = torch.randn(4000, 1)
lab = {"Dose": dose, "Response": 1.5 * dose + 0.3 * torch.randn(4000, 1)}
model = NeuralBayesianNetwork([("Dose", "Response")],
                              {"Dose": ("continuous", 1), "Response": ("continuous", 1)},
                              device="cpu")
model.set_mechanism("Dose", LinearGaussianMechanism())
model.set_mechanism("Response", LinearGaussianMechanism())
model.fit(lab, consolidate=False)
model.requires_grad_(False)          # a fixed model; only our own parameters train
```

## A likelihood for parameters upstream

A second instrument reports the dose with an unknown gain and offset. Learn
them by maximising the network's log-likelihood of the corrected readings:

```python
raw = (lab["Dose"] - 0.5) / 2.0                        # the miscalibrated readings
calibration = torch.nn.Linear(1, 1)
opt = torch.optim.Adam(calibration.parameters(), lr=0.05)
for _ in range(300):
    opt.zero_grad()
    rows = {"Dose": calibration(raw), "Response": lab["Response"]}
    loss = -model.log_prob(rows).mean()
    loss.backward()
    opt.step()
print(calibration.weight.item(), calibration.bias.item())   # about 2.0 and 0.5
```

## Optimising an intervention

Find the dose that brings the mean response to 1.2, through
`sample(n, do=...)`:

```python
target_dose = torch.zeros(1, requires_grad=True)
opt = torch.optim.Adam([target_dose], lr=0.05)
for _ in range(200):
    opt.zero_grad()
    response = model.sample(512, do={"Dose": target_dose})["Response"]
    loss = (response.mean() - 1.2) ** 2
    loss.backward()
    opt.step()
print(target_dose.item())                             # about 0.8 (= 1.2 / 1.5)
```

## Gradients of the model's own parameters

Without `requires_grad_(False)`, `log_prob` gives gradients with respect to
every mechanism parameter, for sensitivity analysis or custom training loops:

```python
model.requires_grad_(True)
model.log_prob(lab).mean().backward()
print({name: tuple(p.grad.shape) for name, p in model.named_parameters() if p.grad is not None})
```

The notebook {doc}`../examples/08_autograd` does all three on an MDN network
and checks a sensitivity against finite differences and the exact value.
