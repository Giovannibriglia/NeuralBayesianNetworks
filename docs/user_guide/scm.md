# Structural causal models

`nbn.scm` reads a Bayesian network as a **structural causal model** (SCM):
every node gets a structural equation `X := f(pa, U)` with an explicit noise
law `P_U`, so the model can abduct the noise of an observed unit and answer
counterfactuals ("what would *this* unit have done under `do(...)`?").

The layer is additive. A {class}`~nbn.scm.model.StructuralCausalModel` is a
`NeuralBayesianNetwork`, and its structural mechanisms are ordinary
mechanisms, so `fit`, `log_prob`, `sample`, `query(..., do=...)`, variable
elimination and likelihood weighting work unchanged, on the CPU, CUDA or MPS.
What NBN does not do is decide which SCM is right: the data identify only the
kernel P(X | pa), and the split into `f` and `U` is a **modelling
convention**. Two SCMs with the same kernel give the same interventional
answers and differ only in counterfactuals.

## Structural mechanisms

| Class | Equation | Noise law |
|---|---|---|
| `DiscreteAdditiveNoiseMechanism(max_offset)` | `x = clip(g(pa) + U, 0, K-1)`, `g` a learned table | categorical over the integer offsets `-max_offset .. max_offset` |
| `AdditiveNoiseMechanism(location, noise)` | `x = g(pa) + U`, `g` linear or an MLP | Gaussian, or a Gaussian mixture |
| `FlowNoiseMechanism()` | `x = T(U; pa)`, a conditional flow (`[neural]` extra) | standard normal; exact abduction through the inverse flow |
| `StructuralCategoricalTable(coupling=...)`, `StructuralNeuralCategorical(coupling=...)` | a categorical kernel read as `X := f(pa, U)` | a declared coupling: `"gumbel_max"` or `"monotone"` |

Each one adds `location(pa)`, `structural(pa, u)`, `abduct(x, pa, n)` (noise
values that explain an observation, `[n, B, D]`: the exact residual for
continuous additive noise, posterior draws for discrete noise) and `noise`
(the law, an `nn.Module`); `residual` (continuous) and `noise_posterior`
(discrete) give the underlying objects.

```python
import torch
from nbn.scm import DiscreteAdditiveNoiseMechanism, StructuralCausalModel

torch.manual_seed(0)
TRUE_NOISE = torch.tensor([0.15, 0.70, 0.15])           # offsets -1, 0, +1


def simulate(n):
    health = torch.randint(0, 4, (n,))
    tired = torch.randint(0, 2, (n,))
    exercise = torch.randint(0, 2, (n,))
    u = torch.multinomial(TRUE_NOISE, n, replacement=True) - 1
    nxt = (health + exercise * (1 - 2 * tired) + u).clamp(0, 3)
    return {"health": health.float(), "tired": tired.float(), "exercise": exercise.float(),
            "health'": nxt.float()}


data = simulate(3000)
model = StructuralCausalModel(
    [("health", "health'"), ("tired", "health'"), ("exercise", "health'")],
    {"health": ("discrete", 4), "tired": ("discrete", 2), "exercise": ("discrete", 2),
     "health'": ("discrete", 4)},
    device="cpu",
)
model.set_mechanism("health'", DiscreteAdditiveNoiseMechanism(max_offset=1))
model.auto_mechanisms()                                   # plain tables elsewhere
model.fit(data)

noise = model.mechanisms["health'"].noise
print(noise.offsets.tolist(), [round(p, 3) for p in noise.probs.tolist()])   # close to the truth
print(model.query(["health'"], {"health": 1}, do={"exercise": 1}))             # still an ordinary NBN
```

**How the discrete fit works.** It maximises the likelihood over the
locations and the shared law jointly, which has local optima (escaping one
needs a location and the law to move together). The search is batched and
staged: one start per offset, then the law shifted by ±1, ±2, then each
feasible location for the busiest rows with the law refitted. Against
brute-force enumeration it found the global optimum in 124 of 125 cases (the
miss: 5e-4 nats per observation); it is a heuristic with no optimality
guarantee. Data that no offset can reach (`max_offset` too small) raise an
error. Locations range over `-m .. K-1+m`, because under clipping a location
past the value range saturates at the edge, which is a law of its own.
Because `(g + s, U - s)` has the same likelihood, the fit is then
canonicalised: the median of the law sits at offset 0, and saturated rows take
the equivalent location closest to the value range, so the result is the same
on every device.

## Counterfactuals

`abduct(data)` returns the noise of every structural node given full
observations; `counterfactual_sample(evidence, do=...)` abducts, applies the
intervention and replays the equations with the same noise.

```python
from nbn.scm import AdditiveNoiseMechanism

torch.manual_seed(1)
dose = torch.rand(4000, 1) * 2 - 1
bimodal = torch.where(torch.rand(4000, 1) < 0.5, -0.4, 0.4) + 0.1 * torch.randn(4000, 1)
rows = {"dose": dose, "level": torch.sin(2 * dose) + bimodal}

scm = StructuralCausalModel([("dose", "level")],
                            {"dose": ("continuous", 1), "level": ("continuous", 1)}, device="cpu")
scm.set_mechanism("dose", AdditiveNoiseMechanism("linear"))
scm.set_mechanism("level", AdditiveNoiseMechanism("mlp", "mixture", n_components=2, epochs=200))
scm.fit(rows)

patients = {"dose": torch.tensor([[-0.5], [0.0]]), "level": torch.tensor([[-1.2], [0.4]])}
world = scm.counterfactual_sample(patients, do={"dose": 0.7})
print(world["level"][0, :, 0])     # each patient's level had the dose been 0.7
```

With continuous additive noise the abduction is exact (one residual per
unit). With discrete noise the observation constrains `U` only to a set, so
`abduct` draws from its posterior. For a categorical kernel the answer
depends on the declared coupling; `counterfactual_bounds` gives the sharp
bounds over **every** SCM compatible with the kernel, and
`counterfactual_bounds_from_sets` widens them over a probability set of
kernels ({doc}`probability_sets`).

## Noise tying and hidden confounders

`model.tie_noise(nodes, name)` makes several structural mechanisms of one
family share **one noise law** (not one noise draw); `fit` then pools their
residuals. A shared law has fewer parameters and gives tighter probability
sets. A shared *draw* is a hidden common cause: declare it with
`StructuralCausalModel(..., bidirected=[("a", "b")])`. Declared confounders
are metadata for diagnostics and for downstream partial identification; NBN
records and tests them but does not fit them.

## Diagnostics

Every test returns a {class}`~nbn.scm.diagnostics.DiagnosticResult` with a
statistic and a p-value; `benjamini_hochberg` controls the false-discovery
rate over a family of tests. Data are moved to the model's device.

| Test | Null hypothesis | Use |
|---|---|---|
| `noise_sharing_test` | the noise law is the same in every parent configuration | allowed to tie or pool |
| `noise_tying_test` | several nodes can share one law | `tie_noise` is supported by the data |
| `freshness_test` | residuals at `t` and `t+1` of one episode are independent | per-step noise is fresh |
| `residual_dependence_test` | the residuals of two nodes are independent | a dependence suggests a hidden common cause |
| `context_invariance_test` | `C ⊥ X | pa(X)`, `C` a dataset or policy id | a rejection flags a confounded logger or drift |
| `noise_independence_test` | the abducted noise is independent of the parents | the additive-noise model fits |
| `calibration_by_bins_test` | the fitted CDF is calibrated in every region of the inputs | the kernel is calibrated |

The discrete sharing and tying tests use a parametric bootstrap that refits
all replicates at once (`R = 100`, `K = 8`, 200 replicates: 0.14 s); under the
null they reject 5.0% and 4.0% of the time at `alpha = 0.05`. The continuous
tests use asymptotic laws (a White-type `n R^2` test, KS with Bonferroni,
Fisher z, HSIC with permutations).

For a discrete node, `context_invariance_test` sums the G statistic over the
parent configurations and, by default, compares it with a **permutation**
reference: within each configuration the context labels are exchangeable
under the null, so each context-by-value table is redrawn `n_perm = 999`
times with its margins fixed, and `p = (1 + #{G_perm >= G_obs}) / (n_perm + 1)`
(so `p >= 0.001`). For independent rows the level holds at any sample size;
for trajectories it is approximate (close to nominal in simulations). To reject
at level `alpha` after correcting for `m` tests, use `n_perm >= m / alpha`;
`confounding_gate` warns when the floor makes a rejection impossible.
`reference="chi2"`
keeps the chi-square approximation, which is fast but over-rejects when many
configurations are sparse: with 64 configurations and two contexts, under the
null at `alpha = 0.05`, the permutation reference rejects 4.9% of 6000
datasets of 2000 rows, the chi-square reference 21% (about 80% with 200 rows). Pass
a CPU `generator` for reproducible p-values. Worked example:
{doc}`../examples/10_confounding_and_noise_diagnostics`, section 1b.

```python
from nbn.scm import noise_sharing_test

result = noise_sharing_test(model, data, "health'", n_boot=100)
print(result.null, "| p =", round(result.p_value, 3))
```

## Calibrated uncertainty for continuous mechanisms

On held-out rows (not the fit data): `pit_band_set` puts a DKW band on the
probability integral transform of any continuous structural mechanism,
`conformal_intervals` gives split-conformal intervals (`"cqr"` or
`"residual"`), and `conformal_sets` gives prediction sets for a discrete
node.

`conformal_sets` ranks classes by decreasing probability and scores the
observed class by the mass ranked above it plus `u` times its own probability.
`score="exclusive"` (`u = 0`, the default) never returns an empty set and stays
informative when a classifier is very confident; `score="inclusive"` (`u = 1`,
Kochenderfer et al., *Algorithms for Validation*, Ex. 12.4) can return empty
sets once probabilities saturate; `score="randomized"` (Romano, Sesia & Candès,
NeurIPS 2020) breaks ties, so coverage is close to `1 - alpha`, and takes a
`generator`. All three have marginal coverage of at least `1 - alpha`.
`HypothesisSet.calibrated_sets(..., discrete_score=...)` passes the choice on.
Example: notebook 13, section 5.

```python
from nbn.scm import conformal_intervals

torch.manual_seed(2)
d2 = torch.rand(2000, 1) * 2 - 1
held_out = {"dose": d2, "level": torch.sin(2 * d2)
            + torch.where(torch.rand(2000, 1) < 0.5, -0.4, 0.4) + 0.1 * torch.randn(2000, 1)}
intervals = conformal_intervals(scm, held_out, "level", alpha=0.1)
print(intervals.level, intervals.guarantee)
```

## Episodes and MDPs

`transitions_from_episodes(states, actions)` flattens logged episodes into
transition rows (`x` and `x'` columns plus `episode` and `time`), the input
for fitting one mechanism per next-state variable, for `freshness_test`, and
for trajectory-valid probability sets ({doc}`probability_sets`).

## Open points

- The polytope of a shared noise law depends on fitted locations: fit them on
  a separate split for an exact `pac_under_model` statement.
- `cdf_band_set` is valid given the location `g`; pass held-out data.
- Continuous probability sets are band or bootstrap only; a finite-sample set
  for neural locations is open.
- Hidden confounders are declared (`bidirected`) and tested for, not fitted;
  partial-identification bounds belong downstream.

The notebooks {doc}`../examples/09_discrete_noise_and_probability_sets`,
{doc}`../examples/10_confounding_and_noise_diagnostics`,
{doc}`../examples/11_continuous_noise_and_counterfactuals`,
{doc}`../examples/12_mdp_trajectories_factored_sets` and
{doc}`../examples/13_discrete_counterfactuals` cover this layer end to end.
