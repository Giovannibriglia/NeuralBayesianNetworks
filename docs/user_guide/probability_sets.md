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
| `interval_set_from_counts(counts, delta)`, `l1_ball_set_from_counts(counts, delta)` | `IntervalSet`, `L1BallSet` | as above, from a count table `[R, K]` | PAC (unweighted counts only) | any count table, e.g. `count_table_for(variables, data, node, parents)` |
| `dirichlet_set_from_counts(counts, delta)` | `SampleSet` | as above, from a count table | Bayesian | counts or weighted pseudo-counts |
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

### Sets from a count table

The count-based builders only need the table of counts `n(u, x)` of a node per
parent value `u`. Each has a `*_from_counts` twin that takes the table itself,
and `count_table_for` counts a node against **any** tuple of discrete parents,
not only its parents in a model's DAG. This is the building block for comparing
several candidate parent sets of the same node:

```python
from nbn.uncertainty import count_table_for, interval_set_from_counts

candidates = [("health", "exercise"), ("health", "tired", "exercise")]
for parents in candidates:
    counts, cards = count_table_for(model.variables, data, "health'", parents)
    candidate = interval_set_from_counts(counts, delta=0.1 / len(candidates))  # both hold at 90%
    print(parents, tuple(counts.shape), "unseen rows:", int(candidate.unseen.sum()))
```

Rows are in C order (last parent fastest), the layout of
`CategoricalTableMechanism`, so `counts.reshape(*cards, K)` gives one axis per
parent. `interval_set(model, data, node)` is exactly
`interval_set_from_counts(count_table(model, data, node)[0])`, and likewise for
`l1_ball_set` and `dirichlet_set`. When several sets must hold at once (one per
candidate parent set, or one per next-state variable), split `delta` between
them by a union bound. The PAC builders hold only for unweighted counts of
independent draws (a weight is not a number of draws): they reject fractional
entries, but a table built with whole-number weights looks like real counts,
so that check is yours. The Dirichlet builder accepts weighted pseudo-counts.

## Structure uncertainty: several candidate parent sets

When the parents of a node are not known, `nbn.structure.HypothesisSet` takes a
list of **candidate parent sets** per node. A *hypothesis* picks one candidate
per node, so there are `prod_j |P_j|` hypotheses. A *factor* is one pair
`(node, parents)`, and there are only `sum_j |P_j|` factors. Counts,
fitted tables and probability sets depend on one family only, so they are
computed **once per factor** and shared by every hypothesis that uses it
(with complete data the likelihood is a product of one term per family:
Koller & Friedman 2009, §17.2.2, Prop. 17.1).

```python
from nbn.structure import HypothesisSet

hs = HypothesisSet(
    model.variables,
    candidates={"health'": [("health", "exercise"), ("health", "tired", "exercise")]},
)
print(hs)                                   # factors=2, hypotheses=2
sets = hs.factor_sets(data, builder="interval", delta=0.1)
print(sets.level, sets.delta_per_factor)    # 0.9 jointly; 0.05 per factor
for h in hs.hypotheses():
    s = sets.sets_for(h)["health'"]         # a view into the cache, no copy
    print(h["health'"], tuple(s.lower.shape), "unseen:", int(s.unseen.sum()))
chosen = hs.model_for(next(hs.hypotheses()), sets.counts)   # fitted, no pass over the data
```

- **The factor cache.** `hs.counts(data)` counts each node once over the
  union of its candidates' parents, then sums out the parents a candidate does
  not use. This is exact, and it is the usual way to share sufficient statistics
  across families (K&F, Box 18.A). Above `max_union_cells` the candidates
  are counted one by one.
- **Budget.** `delta` is split over the **distinct factors** (union bound).
  A factor's set is the same event in every hypothesis, so on one event of
  probability at least `1 - delta` every factor set holds. For a factor whose
  parents contain the true ones this bounds the true conditional; a factor
  that misses a true parent only describes the data's average over it (see
  below), so the level speaks for the superset factors.
- **Which hypotheses are right.** In a 2-slice model with no intra-slice
  edges, a candidate that contains the true parents, or a superset of them,
  gives the true conditional for any input distribution, including one set by
  behaviour policies (wherever the parent value has data). A candidate that
  misses a true parent is biased toward an average weighted by the data.
  Supersets cost data: more rows, fewer counts per row, wider sets.
- `model_for(h, counts)` returns a fitted `NeuralBayesianNetwork` with
  table mechanisms set from the cached counts, equal to fitting that DAG
  directly.
- Worked example: {doc}`../examples/14_structure_uncertainty_hypothesis_sets`.
- Scope of the count-based sets: discrete nodes and parents. Continuous and
  mixed factors: see the next section.

### Continuous and mixed factors

Count-based sets need discrete nodes and parents. For continuous targets, and
for discrete targets with continuous parents, `HypothesisSet.fit` fits one
mechanism per factor, with **all candidates of a node fitted together**. Each
result is a standard NBN mechanism, so `model_for`, the calibration tools and
counterfactuals work unchanged. Table factors and discrete inputs become
`StructuralCategoricalTable` objects, with the declared `coupling` and the
smoothing `alpha` given to `fit`. A candidate's initial weights depend only on
the seed and the factor itself, so adding or reordering candidates does not
change the other candidates' fits.

```python
fitted = hs.fit(train, location="mlp", epochs=100)        # or location="linear"
cal = hs.calibrated_sets(fitted, calibration, method="cqr", level=0.9)
cal.guarantees                                            # per factor: "marginal"
model = hs.model_for(hypothesis, fitted)                  # a StructuralCausalModel
```

- **Linear location** (`AdditiveNoiseMechanism(location="linear")`). The
  weighted moments of the union of a node's candidate parents are computed
  once, and each candidate's normal equations are sub-blocks of them.
- **MLP location** (`location="mlp"`). The candidates' networks are stacked
  along a candidate axis, and each reads the union of parents through a fixed
  mask, so unused parents never reach it. The loss is the sum of the
  candidates' losses, so each candidate trains as it would alone. After
  training, each candidate becomes an `AdditiveNoiseMechanism` with its own
  standardisation and a noise law fitted on its own residuals. Discrete
  targets with continuous parents use the same trainer with a categorical
  head (`StructuralNeuralCategorical`). Batching pays off with mini-batches
  (many small steps); with full batches on many rows the work is
  compute-bound and both paths cost about the same.
- **Calibrated sets per factor** on held-out rows: split conformal (`"cqr"`
  / `"residual"`, or conformal sets for discrete children) or a PIT band
  (`"pit"`). The level is split over **all** factors of the set by a union
  bound, as in `factor_sets`. So in a mixed set, `calibrated_sets(level)` for
  the continuous and neural factors and `factor_sets(delta = 1 - level)` for
  the count-based ones hold together at `level`. `factor_sets` builds only
  the count-based factors and lists the others in `skipped`.
  - Conformal results are **marginal**: coverage on average over the parents,
    not for each parent value. So they do not support a per-row worst case
    in robust planning.
  - PIT bands bound every conditional only under the declared
    PIT-independence assumption.
  - Keep every candidate. Choosing one by its calibration score and quoting
    its guarantee is not supported; use a fresh split for that.
- Worked example: {doc}`../examples/17_continuous_hypothesis_sets`.

### One set per row: the agreement rule

A worst case over hypotheses falls back to the largest candidate, which has
no data on most of its rows. `agreement_set(variables, data, node, inputs,
max_parents=d)` gives **one** set per row that is valid without knowing the
parents, under a declared bound `|Pa(node)| <= d` (KWIK agreement rule;
k-meteorologists, Diuk, Li & Leffler 2009; offline use with one behaviour
policy, Simão & Spaan 2019).

1. Every parent set of size `d` from `inputs` is a candidate. Each candidate
   and each **one-variable extension** `c ∪ {v}` gets an L1 ball per row, all
   holding together at `1 - delta`.
2. If `c` contains the true parents, so does `c ∪ {v}`, and their conditionals
   are equal, so their balls overlap on every row. **Disjoint balls prove that
   `c` misses a parent**, so `c` is discarded. A superset candidate is never
   discarded.
3. Over the union of the survivors' parents, each row where every survivor
   has data gets a ball containing all of the survivors' balls. Some survivor
   is a superset, so the ball contains the truth. Other rows get the simplex.

```python
from nbn.structure import agreement_set

res = agreement_set(model.variables, data, "health'", inputs=["health", "tired", "exercise"],
                    max_parents=3, delta=0.1)
print(res.survivors, res.parents, res.set.guarantee)          # pac_under_model
print("joint values visited / with a set:", res.flat_visited, "/", res.flat_with_set)
```

The gain is `res.n_rows_gained`: joint input values that no dataset visited
but that still get a set, because each survivor visited its own parent values.
When the inputs encode constrained actions (for example one-hot flags of a
single action), pass the feasible rows as `coverage_rows`. Otherwise the
counts include flag combinations that never occur.
If every candidate is refuted, the bound `d` is contradicted
(`res.assumption_refuted`), and the result falls back to the flat model (all
inputs as parents). Declared assumptions (in `res.assumptions`): `|Pa| <= d`,
`Pa ⊆ inputs`, no intra-slice edges into the node, and the same mechanism for
every logger (no hidden confounding). Several nodes: split `delta` between
them. Worked example: {doc}`../examples/15_agreement_rule`.

## Export for planning and verification tools

Robust value iteration and interval or parametric model checkers (PRISM,
Storm) need the sets **with their sharing**. A factor row `p_j(. | u)` is
shared by every state-action row whose parents take the value `u`. Without
the sharing, a tool can only build per-row boxes, which are sound but
pessimistic; with it, it can build the coupled parametric model.
`export_factored_sets` writes a documented side file:

```python
from nbn.structure import export_factored_sets, joint_row_bounds, load_factored_sets

export_factored_sets("out/kernel", sets, model, state_vars=["health"], action_vars=["exercise"],
                     exact=True)
ex = load_factored_sets("out/kernel")
(h,) = ex.hypotheses()                       # lazily, {node: factor id}
lower, upper, nodes = joint_row_bounds(ex, h, row=5)   # bounds on P(s' | s, a) for one (s, a)
```

- `sets` can be a `FactorSets` (every factor of a `HypothesisSet`; the
  hypotheses are the product of the candidates), or a mapping `node -> set`:
  a `ProbabilitySet` (parents from the model's DAG), an `AgreementSet`, or a
  `NoiseParameterSet` (nodes tied to one noise law share one **parameter
  block**, so the coupling survives the export).
- **`manifest.json`** (`schema: "nbn.factored_sets/1"`, JSON Schema in
  `nbn/structure/schemas/factored_sets_v1.schema.json`) records:
  - the variables and their roles (state, action, next state, reward,
    termination) and the planning space (C order over state then action
    variables);
  - per factor: node, parents, set type, guarantee, level, sampling,
    declared assumptions, and `graph_preserving`. That flag is false when an
    entry can be zero in one member and positive in another, for example on
    unseen rows; some interval-MDP solvers assume it;
  - the parameter blocks, the hypotheses, the composition (union bound over
    all listed sets, with the total level), and the two quantifiers. Robust
    planning is "for all kernels, max policy"; parameter synthesis is
    "exists parameters".
- **`arrays.npz`** holds, per factor, `lower` / `upper` `[R, K]`, `unseen`,
  `center` / `radius` for L1 balls, `samples`, `counts` (visits per factor
  row), and one row map per distinct parent tuple (the factor row of each
  planning row; omitted above `max_rows`, where `factor_row` computes it from
  the layout). With `exact=True` it also holds rational bounds
  `lower_num / D <= lower` and `upper_num / D >= upper`, rounded outward
  exactly, for the factor boxes and the noise-polytope constraints.
- Turning factor sets into joint rows is the tool's job. A joint row is the
  product of its factor rows (multilinear in the factor parameters), and
  `joint_row_bounds` gives the entry-wise box of that product.

Worked example: {doc}`../examples/16_export_factored_sets`.

## Confounded action effects from several regimes

If a logged action may depend on a **hidden** variable that also moves the
next state (`A <- W -> X'`), the fitted table `P(X' | parents)` is not the
interventional row `P(X' | do(A=a), u)` that planning needs. With data from
several regimes (behaviour policies), `confounded_interval_set` gives
per-entry intervals on the interventional rows, valid together with
probability `>= 1 - delta` under declared assumptions:

```python
from nbn.uncertainty import confounded_interval_set, confounding_gate

rows = confounded_interval_set(model, data, node="h'", action="a", regime="logger",
                               method="auto", delta=0.05, sampling="trajectory",
                               declared_unconfounded=[1])       # logger 1 is a randomised trial
rows.set            # IntervalSet in the node's table layout, guarantee "pac_under_model"
rows.method_used    # per configuration of the other parents: "lp" / "intersection" / ...
rows.refuted        # configurations where the data contradict the instrument assumptions
```

1. **Confidence first.** A box on every regime's observables
   `P_k(A=a, X'=y | u)` and `P_k(A=a, X' != y | u)`, simultaneous by
   Bonferroni over a budget fixed in advance, anytime-valid for trajectories.
   The intervals are KL by default (SciPy-free); `interval_method="clopper_pearson"`
   is available.
2. **Rows over the whole box.**
   - `"intersection"`: per-regime Manski rows `[P(a, y), 1 - P(a, X' != y)]`,
     intersected (the natural IV bound).
   - `"lp"`: the IV linear program over response types (Balke–Pearl,
     generalised to finite cardinalities). Regimes in
     `declared_unconfounded` add constraints.
   - `"sensitivity"`: rows under a marginal sensitivity model with parameter
     `lam`.
   - `"auto"`: the LP when it is small and SciPy is installed
     (`pip install "nbn[bounds]"`), else the intersection.

   With a confidence box, the rows are **valid, not sharp**.
3. **Refutation.** An empty intersection or an infeasible LP means the
   regimes are incompatible with one shared environment (the instrumental
   inequality). It is reported (`refuted`, the simplex for that
   configuration), never silently widened.
4. **Assumptions** (in `rows.assumptions`):
   - exclusion: the regime acts on the child only through the action;
   - independence of the regime and the hidden variable given the other
     parents;
   - **fresh** confounding for trajectory data (a new hidden value at every
     step);
   - declared regimes do not read the hidden variable.

**The gate.** `confounding_gate(declared_unconfounded, invariance)` treats a
node as possibly confounded unless it is declared unconfounded. A rejected
`context_invariance_test` (regime as context) forces the confounded rows even
for a declared node, but a test that passes never certifies a node: there is
no statistical test for confounding.

Worked example: {doc}`../examples/18_confounded_rows`.

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
