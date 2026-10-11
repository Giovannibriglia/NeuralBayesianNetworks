# SCM layer and probability sets — design draft

Status: draft for review (branch `feat/scm-noise-confounding`).
Context: NBN becomes the *SCM-learning engine* of the "SCM → MDP family"
pipeline (step S4, parameter learning). The pipeline logic (confounding bounds,
four-source uncertainty sets, compilation to MDPs) stays outside NBN.

## What NBN keeps

Everything added here is **additive**: new subpackages `nbn.scm` and
`nbn.uncertainty`, no change to the core classes. NBN's own message stays:

* discrete **and continuous** variables;
* every operation is a batched PyTorch tensor operation that runs on CPU,
  CUDA or MPS (`device=` as in the rest of NBN);
* new mechanisms are ordinary `Mechanism`s, so `fit`, `log_prob`, `sample`,
  `query(..., do=...)`, variable elimination (`tabulate`) and likelihood
  weighting keep working unchanged.

## 1. Structural mechanisms (noise made explicit)

A `Mechanism` is a conditional distribution `P(X | pa)`. That is the *kernel*,
which many SCMs share. A `StructuralMechanism` adds the structural equation

    X := f(pa, U),   U ~ P_U   (independent of pa)

with extra methods: `location(pa)`, `structural(pa, u)`, `abduct(x, pa, n)`
(noise values that explain an observation, `[n, B, D]`: the exact residual for
continuous additive noise, posterior draws for discrete noise) and `noise` (the
law `P_U`, an `nn.Module`).  `residual` (continuous) and `noise_posterior`
(discrete) give the underlying objects.

| Class | Equation | Noise law |
|---|---|---|
| `DiscreteAdditiveNoiseMechanism` | `x = clip(g(pa) + U, 0, K-1)`, `g` a learned table | categorical over integer offsets |
| `AdditiveNoiseMechanism` | `x = g(pa) + U`, `g` linear or MLP | Gaussian, or Gaussian mixture |

Locations range over `-m .. K-1+m`: under clipping, a location past the value
range saturates at the edge, which is a law of its own (needed, e.g., for
`x = clip((p + q) // 2 + 3 + U)`).

The discrete fit maximises the likelihood over locations and the shared law.
It has local optima, because escaping them needs a location and the law to
move together.  The search is batched and staged: multi-start (one start per
offset), shift polish (the law moved by ±1, ±2), and row-move polish (each
feasible location for the busiest rows, `q` refitted, ascent resumed).
Against brute-force enumeration it found the global optimum in 124 of 125
cases (the miss: 5e-4 nats per observation).  It is a heuristic: no
optimality guarantee is claimed.  Data that no offset can reach (`max_offset`
too small) raise a clear error.

Because `(g + s, q moved by -s)` has exactly the same likelihood, the fit is
then *canonicalised*: the whole group is shifted so the median of `q` sits at
offset 0, and saturated rows take the equivalent location closest to the
value range.  The likelihood is unchanged and the result is the same on CPU
and MPS.  Bootstrap replicates start from the observed fit (they are simulated
from it), which keeps hundreds of replicates cheap: `R = 100`, `K = 8`,
`n_boot = 200` takes 0.14 s.  Under the null the discrete sharing and tying
tests reject 5.0% and 4.0% of the time at `alpha = 0.05`.

The split between `g` and `U` is a **modelling convention**: the data identify
only the kernel. Two SCMs with the same kernel give the same interventional
answers, and differ only in counterfactuals.

## 2. Noise sharing (tying)

`StructuralCausalModel.tie_noise(nodes, name)` makes several mechanisms share
**one noise law** (not one noise draw). Fitting pools their residuals. A shared
law has few parameters, gives tighter uncertainty sets, and becomes a shared
parameter of a parametric MDP downstream. A shared *draw* is a hidden common
cause; it is declared with `bidirected=[(a, b)]`, not with tying.

## 3. Diagnostics (all return a `DiagnosticResult` with a p-value)

| Test | Null hypothesis | Use |
|---|---|---|
| `noise_sharing_test` | the noise law is the same in every parent configuration | allowed to tie / pool |
| `freshness_test` | residuals at `t` and `t+1` of one episode are independent | per-step noise is fresh |
| `residual_dependence_test` | residuals of two nodes are independent | a dependence suggests a hidden common cause (bidirected edge) |
| `context_invariance_test` | `C ⊥ X | pa(X)`, `C` = dataset / policy id | a rejection flags a confounded logger or drift |

Discrete sharing and tying tests use a parametric bootstrap that refits all
replicates at once (batched).  Continuous tests use asymptotic laws (a
White-type `n R^2` test, KS with Bonferroni, Fisher z, HSIC with permutations).
Benjamini–Hochberg is provided for families of tests.  Data are moved to the
model's device.

## 4. Probability sets (uncertainty on the probabilities)

"Safe" means: with probability at least `1 - delta`, the true probabilities
are inside the set, **for all entries at once**. A set exposes:

* `bounds()` → `(lo, hi)`, the min/max of every probability (always available);
* `contains(P)`;
* `level` (`1 - delta`), `guarantee` (`"pac"`, `"pac_under_model"`,
  `"bayesian"`, `"asymptotic"`), and `unseen` per row (bounds = whole simplex).

`NoiseParameterSet` (a polytope on the shared law `q`; `table_set(node)` gives
each table's box, a rectangular superset) and `CDFBandSet` have their own API.

| Set | Shape | Guarantee | Built from |
|---|---|---|---|
| `IntervalSet` | `lo <= P <= hi` per entry | PAC | counts of a categorical CPD |
| `L1BallSet` | `||P_row - P_hat_row||_1 <= r` | PAC | counts |
| `NoiseParameterSet` | polytope on the shared noise law `q`; every row of every tied node is a linear image of `q` | PAC under the noise model | pooled counts per distinct expression |
| `SampleSet` | samples (Dirichlet posterior or episode bootstrap) and their quantile box | Bayesian / asymptotic | counts / refits |
| `CDFBandSet` | band on the noise CDF of a continuous additive mechanism | PAC given the location `g` | residuals |

Binomial intervals come from the Chernoff (KL) inequality, inverted by
bisection in torch — batched, no SciPy (the bound KL-UCB inverts; Garivier &
Cappé, COLT 2011). Clopper–Pearson is available when SciPy
is installed (it is part of the `bench` extra). The confidence budget `delta` is split over all intervals by a
union bound. For **trajectory data** the visit counts are random and depend on
earlier draws, so each interval is made valid for every sample size at once
(`delta_n = delta / (n (n + 1))`; the coupling argument of Agarwal et al.,
*RL: Theory and Algorithms*, Lemma 6.2, with a summable schedule instead of a
finite range).

For a factored next state, build one set per variable and split `delta`
across them; this is the learning scheme of Schnitzer, Abate & Parker,
*Efficient Solution and Learning of Robust Factored MDPs* (AAAI 2026, §4.1).
`NoiseParameterSet` is the per-expression construction of Schnitzer, Abate &
Parker, *Robust Parameter Learning for Uncertain MDPs* (arXiv:2605.01339).
Solving over products of per-variable sets (McCormick relaxations) belongs
downstream, not in NBN.

## 5. Tests

* exact formulas against reference values;
* **coverage**: hundreds of replicate datasets from a known SCM; the truth must
  be inside the set in at least `1 - delta` of them (minus Monte-Carlo error),
  for i.i.d. and for trajectory data;
* soundness and tightness: the pooled set is inside the per-row box; widths
  shrink like `n^{-1/2}`;
* diagnostics: size (type-I error at most `alpha`) and power, over replicates;
* NBN contracts for the new mechanisms: weights (replication equivalence),
  `state_dict` round trip, devices (CPU and MPS/CUDA when present), `do=`,
  variable elimination.

Slow Monte-Carlo tests carry the `slow` marker; a short version runs by default.

### Measured (rehab example, nominal level 0.90, 300 replicates)

| Set | coverage n=400 | coverage n=4000 | mean width n=400 | mean width n=4000 |
|---|---|---|---|---|
| per-entry box (KL) | 0.987 | 0.990 | 0.418 | 0.122 |
| L1 ball (box view) | 1.000 | 1.000 | 0.509 | 0.201 |
| noise polytope (box view) | 0.993 | 0.983 | 0.086 | 0.027 |

Trajectory data (20 episodes x 15 steps, anytime correction): coverage 1.00.
Discrete sharing test under the null: rejection rate 0.050 at alpha 0.05.
All sets are conservative (union bounds); pooling a shared noise law makes them
about five times tighter.

## 6. Open points

* The polytope's expressions depend on fitted locations: fit them on a
  separate split for an exact `pac_under_model` statement.
* `cdf_band_set` is valid given the location `g`; pass held-out data.
* Continuous probability sets are band/bootstrap only; a finite-sample set for
  neural locations is open.
* Hidden confounders are declared (`bidirected`) and detected, not fitted;
  their partial-identification bounds belong to the pipeline (S4).
