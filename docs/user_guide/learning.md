# Learning

`fit` learns every mechanism from complete data; `update` folds in new data
later, without the old data. Both run on the model's device.

The examples on this page share one small hybrid network:

```python
import torch
from nbn import LinearGaussianMechanism, NeuralBayesianNetwork, seed_all

EDGES = [("Rain", "WetGrass"), ("WetGrass", "Soil")]
VARIABLES = {"Rain": ("discrete", 2), "WetGrass": ("discrete", 2), "Soil": ("continuous", 1)}


def simulate(n, p_rain=0.3, soil_slope=2.0):
    rain = torch.bernoulli(torch.full((n,), p_rain)).long()
    wet = torch.where(rain == 1, torch.bernoulli(torch.full((n,), 0.9)),
                      torch.bernoulli(torch.full((n,), 0.2))).long()
    soil = soil_slope * wet + 0.5 * torch.randn(n)
    return {"Rain": rain, "WetGrass": wet, "Soil": soil[:, None]}


def new_model():
    model = NeuralBayesianNetwork(EDGES, VARIABLES, device="cpu")
    model.set_mechanism("Soil", LinearGaussianMechanism())
    model.auto_mechanisms()
    return model


seed_all(0)
data = simulate(5000)
```

## `fit`

```python
model = new_model()
history = model.fit(data)
print(f"{history.wall_clock_s:.2f} s on {history.device}")
print({node: round(history.mean_ll(node), 3) for node in VARIABLES})
```

`fit` walks the DAG and calls each mechanism's `fit_local` on its own column
and its parents' columns (`method="local"`, the default; `method="joint"`
trains every parameter with one optimiser). `method="stacked"` trains the
same per-node objectives at the same time, as one batched network per
architecture (MDN, neural categorical, FlexCode, normalising flow). It is the
same estimator: the likelihood splits per node, Adam acts element-wise, and
clipping and early stopping stay per node; only the minibatch order is shared.
It is several times faster on networks with many neural nodes, and it is what
makes a GPU pay off for them. Roots, warm starts and other mechanisms take the
`local` path. It returns a
{class}`~nbn.learning.fit.TrainHistory`: the **in-sample** mean
log-likelihood per node, the wall-clock time and the device. There is no
validation split, so it is a fit trace, not a test score.

| Argument | Meaning |
|---|---|
| `epochs`, `batch_size`, `lr` | training budget of the neural mechanisms. `None` (default) lets each mechanism use its own (flow 300 epochs, MDN 200, neural categorical 100, ...); a value overrides it for every mechanism |
| `consolidate=True` | after the fit, neural mechanisms record what `update()` needs (an extra Fisher pass). Pass `False` when you will never call `update()`, for example when fitting in a loop |
| `weights=None` | per-row multiplicities, see below |
| `warm_start=False` | continue from the current parameters, see below |

## Choosing a mechanism per node

`select_mechanisms` tries several mechanism types for each node and keeps the
one with the best held-out log-likelihood. It trains every candidate briefly,
keeps the best third, trains those longer, and so on (successive halving, the
schedule of Hyperband), so the cost is close to one full fit, not the sum of
all candidates. The held-out likelihood splits per node, so choosing node by
node is exact once the parents are fixed. The winner is installed in the
model.

```python
from nbn import CategoricalTableMechanism, MDNMechanism, NeuralCategoricalMechanism
from nbn.learning import select_mechanisms

model = new_model()
report = select_mechanisms(model, data, {"table": CategoricalTableMechanism,
                                         "neural": NeuralCategoricalMechanism},
                           nodes=["WetGrass"], max_epochs=30)
print(report.chosen)               # {'WetGrass': 'table'}

report = select_mechanisms(model, data, {"linear": LinearGaussianMechanism,
                                         "mdn": lambda: MDNMechanism(num_components=2)},
                           nodes=["Soil"], max_epochs=30)
print(report.rungs["Soil"])        # (candidate, epochs, held-out log-likelihood) per rung
```

- A candidate is a class or a factory with no required argument; a factory
  that takes `n_classes` gets the node's number of classes.
- By default 20% of the rows are held out (`val_fraction`); pass
  `val_data=` to use your own split, and `refit=True` to refit the winner on
  all rows. `report.train_index` and `report.val_index` say which rows were
  used: both count as *fit* rows for any later calibration
  (`fit_ids=`, {doc}`scm`), and a final test score needs rows outside both.

## Weighted fitting

`weights` is a non-negative `[N]` tensor aligned with the rows. A row with
weight 2 counts as two rows, and weight 0 is the same as dropping the row; use
it for importance weights, EM responsibilities, or to undo a biased sample.

```python
# The logger kept every rainy day but only one dry day in three.
seed_all(1)
pool = simulate(9000)
keep = (pool["Rain"] == 1) | (torch.rand(9000) < 1 / 3)
biased = {k: v[keep] for k, v in pool.items()}
w = torch.where(biased["Rain"] == 1, 1.0, 3.0)        # inverse keep rate

plain, weighted = new_model(), new_model()
plain.fit(biased, consolidate=False)
weighted.fit(biased, consolidate=False, weights=w)
print("P(Rain = 1): truth 0.30, unweighted", round(float(plain.query(["Rain"])[1]), 3),
      "weighted", round(float(weighted.query(["Rain"])[1]), 3))
```

Mechanisms that cannot honour weights (`KNNConditionalMechanism`) raise before
any node is fitted.

## Warm starts

`fit(..., warm_start=True)` continues from the parameters a mechanism already
holds instead of a fresh initialisation, so a second `fit` refines the first.
Closed-form mechanisms (tables, linear-Gaussian, KDE) give the same answer
either way; for neural ones (MDN, neural categorical, flow, FlexCode) it is
how you train in rounds:

```python
from nbn import MDNMechanism

seed_all(2)
model = NeuralBayesianNetwork(EDGES, VARIABLES, device="cpu")
model.set_mechanism("Soil", MDNMechanism(num_components=2))
model.auto_mechanisms()
for round_ in range(3):
    h = model.fit(data, epochs=10, consolidate=False, warm_start=round_ > 0)
    print(f"round {round_}: in-sample log-lik of Soil {h.mean_ll('Soil'):.3f}")
```

## `update`: new data without the old

`update(new_data)` folds new rows into a fitted model. The graph and the
cardinalities stay fixed; each mechanism uses the sufficient statistics it
kept during `fit` (the posterior after `fit` is the prior for `update`). For
tables and linear-Gaussian nodes this is exact: `fit` on one batch followed by
`update` on a second gives the model that `fit` on both gives.

```python
seed_all(3)
first, second = simulate(2000), simulate(2000)

streamed = new_model()
streamed.fit(first)
history = streamed.update(second)
print(history.node_methods, "skipped:", history.skipped)

pooled = new_model()
pooled.fit({k: torch.cat([first[k], second[k]]) for k in first})
probe = simulate(1000)                       # same log-likelihood on any rows
print(torch.allclose(streamed.log_prob(probe), pooled.log_prob(probe), atol=1e-4))   # True
```

The returned {class}`~nbn.update.base.UpdateHistory` names the update path of
each node (`node_methods`): Dirichlet counts for tables, recursive least
squares for linear-Gaussian nodes, an online Laplace (EWC) step for neural
mechanisms. Mechanisms without an incremental update (`BinningCategoricalTable`,
flows, FlexCode, fixed mechanisms) keep their fitted distribution and are
listed in `skipped`; refit to change them.

**Atomic by default.** `update` refuses a batch with a missing column or a
NaN before touching anything. Other errors show up only while a node is being
updated; then every mechanism already updated is restored in place, so a
failed update never leaves the network half old, half new. `update(..., atomic=False)`
skips the snapshot, which saves memory for large non-parametric stores (KDE
and k-NN keep their samples).

```python
bad = simulate(100)
bad["WetGrass"][50] = 2          # outside the 2 declared states; Rain is updated first
before = streamed.log_prob(probe)
try:
    streamed.update(bad)
except ValueError as err:
    print("rolled back:", str(err)[:60], "...")
print(torch.equal(before, streamed.log_prob(probe)))   # True: nothing changed
```

**Forgetting.** `update(batch, forgetting=f)` with `0 < f <= 1` multiplies the
stored statistics by `f` before adding the batch, so old data fade
geometrically and the model tracks a drifting process. `f = 1` (the default)
weighs every row the same.

```python
seed_all(4)
tracker = new_model()
tracker.fit(simulate(2000, soil_slope=2.0))
for _ in range(10):                                    # the regime changes
    tracker.update(simulate(500, soil_slope=-1.0), forgetting=0.5)
print(tracker.mechanisms["Soil"](torch.tensor([[1.0]])).mean)   # close to -1.0
```

The notebooks {doc}`../examples/01_parameter_learning` (weights, warm starts,
`TrainHistory`) and {doc}`../examples/05_updating` (streaming, rollback,
forgetting) show these against the true distributions.
