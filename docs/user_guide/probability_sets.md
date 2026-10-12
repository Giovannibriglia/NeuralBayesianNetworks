# Probability sets

`nbn.uncertainty` puts a **guarantee** on learned probabilities: a
probability set contains the true table with probability at least
`1 - delta`, **for all entries at once**. Downstream, such sets become the
uncertainty sets of robust or interval MDPs.

Every set exposes `bounds()` (the `(lower, upper)` of every probability,
always available), `contains(P)`, `level` (`1 - delta`), `guarantee` (`"pac"`,
`"pac_under_model"`, `"bayesian"` or `"asymptotic"`) and `unseen` (rows with
no data, whose bounds are the whole simplex).

## The sets

| Builder | Set | Shape | Guarantee | Built from |
|---|---|---|---|---|
| `interval_set(model, data, node, delta)` | `IntervalSet` | `lo <= P <= hi` per entry | PAC | counts of a discrete node's table |
| `l1_ball_set(model, data, node, delta)` | `L1BallSet` | one ball `||P_row - P_hat_row||_1 <= r` per row | PAC | counts |
| `nbn.scm.noise_parameter_set(model, data, nodes, delta)` | `NoiseParameterSet` | a polytope on a shared noise law `q`; every row of every tied node is a linear image of `q` | PAC under the noise model | pooled counts per distinct expression |
| `dirichlet_set(model, data, node, delta)` | `SampleSet` | Dirichlet posterior draws and their quantile box | Bayesian | counts |
| `bootstrap_set(model, data, statistic, groups)` | `SampleSet` | episode-bootstrap refits of any statistic | asymptotic | refits |
| `nbn.scm.cdf_band_set(model, data, node, delta)` | `CDFBandSet` | a band on the noise CDF of a continuous additive mechanism | PAC given the location `g` | residuals |

`NoiseParameterSet.table_set(node)` gives each table's box (a rectangular
superset), and `CDFBandSet` bounds `P(a < X <= b | pa)` for any `a`, `b`.

```python
import torch
from nbn.scm import DiscreteAdditiveNoiseMechanism, StructuralCausalModel, noise_parameter_set
from nbn.uncertainty import interval_set, l1_ball_set

torch.manual_seed(0)
TRUE_NOISE = torch.tensor([0.15, 0.70, 0.15])


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
model.auto_mechanisms()
model.fit(data)

box = interval_set(model, data, "health'", delta=0.1)                 # 90% for all entries
ball = l1_ball_set(model, data, "health'", delta=0.1)
polytope = noise_parameter_set(model, data, ["health'"], delta=0.1)    # on the shared law

lower, upper = box.bounds()                                            # [16 parent rows, 4 states]
print(box.level, box.guarantee, tuple(lower.shape))
print("mean width: box", round(float((upper - lower).mean()), 3),
      "| polytope (box view)", round(float(polytope.table_set("health'").width().mean()), 3))
print("true noise law inside the polytope:", polytope.contains(TRUE_NOISE.double()))
```

Pooling the shared noise law gives much tighter bounds than one interval per
entry: the 64 entries of the table are linear images of 3 noise probabilities.

## How the guarantee is built

Binomial intervals come from the Chernoff (KL) inequality, inverted by
bisection in torch (batched, no SciPy; the bound KL-UCB inverts, Garivier and
Cappé, COLT 2011). `method="hoeffding"` is also available, and
`method="clopper_pearson"` when SciPy is installed (it is in the `bench`
extra). The confidence budget `delta` is split
over all intervals by a union bound (`union_bound`).

**Trajectory data.** With `sampling="trajectory"` the visit counts are random
and depend on earlier draws, so each interval is made valid for every sample
size at once with `delta_n = delta / (n (n + 1))` (`anytime_delta`; the
coupling argument of Agarwal et al., *RL: Theory and Algorithms*, Lemma 6.2,
with a summable schedule instead of a finite range).

**Factored next states.** Build one set per next-state variable and split
`delta` across them; this is the learning scheme of Schnitzer, Abate and
Parker, *Efficient Solution and Learning of Robust Factored MDPs* (AAAI 2026,
§4.1). `NoiseParameterSet` is the per-expression construction of Schnitzer,
Abate and Parker, *Robust Parameter Learning for Uncertain MDPs*
(arXiv:2605.01339). Solving over products of per-variable sets belongs
downstream, not in NBN.

## Measured coverage

The tests draw hundreds of replicate datasets from a known SCM and check that
the truth is inside the set in at least `1 - delta` of them (minus
Monte-Carlo error), for i.i.d. and for trajectory data; that the pooled set
lies inside the per-row box; and that widths shrink like `n^{-1/2}`. On the
rehab example above (nominal level 0.90, 300 replicates):

| Set | coverage n=400 | coverage n=4000 | mean width n=400 | mean width n=4000 |
|---|---|---|---|---|
| per-entry box (KL) | 0.987 | 0.990 | 0.418 | 0.122 |
| L1 ball (box view) | 1.000 | 1.000 | 0.509 | 0.201 |
| noise polytope (box view) | 0.993 | 0.983 | 0.086 | 0.027 |

Trajectory data (20 episodes x 15 steps, anytime correction): coverage 1.00.
All sets are conservative (union bounds); pooling a shared noise law makes
them about five times tighter.

The notebooks {doc}`../examples/09_discrete_noise_and_probability_sets`
(three sets and a coverage check over 100 datasets) and
{doc}`../examples/12_mdp_trajectories_factored_sets` (trajectory-valid sets
for a factored transition kernel) show these sets at work.
