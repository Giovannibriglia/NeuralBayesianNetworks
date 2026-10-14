# Examples

Executed Jupyter notebooks, one per NBN feature.  Each one simulates its own
data from a known ground truth, so what NBN learns or infers is compared with
the exact answer.  The outputs and plots are stored in the notebooks, so they
read on GitHub without running anything.

## The core library

| Notebook | What you learn |
|---|---|
| [`01_parameter_learning.ipynb`](01_parameter_learning.ipynb) | Every mechanism family on a hybrid network with known densities: `auto_mechanisms`, `KL(true \|\| learned)` per node and against the training-set size (3 seeds), weighted fitting (`weights=`) to undo a biased sample, fixed mechanisms, and `TrainHistory` with warm starts. |
| [`02_inference.ipynb`](02_inference.ipynb) | Exact variable elimination, likelihood weighting (error against particles, effective sample size), `HybridRouter`, the amortized engines (`AmortizedISEngine`, `AmortizedVIEngine` and its quality gate), evidence on a continuous node, a continuous target (W1 to the truth), `map_query` and joint queries. |
| [`03_interventions.ipynb`](03_interventions.ipynb) | `P(y \| x)` against `P(y \| do(x))` on two confounded networks learned from observational data, a dose-response curve with one intervention per row in `query_batch`, `intervene()` and the mutilated graph, and `do=` on every engine. |
| [`04_sampling.ipynb`](04_sampling.ipynb) | `sample` / `ancestral_sample` against the true marginals and joint, `return_log_prob`, clamped evidence with `return_log_weight` (posterior estimate and effective sample size), and samples under `do=`. |
| [`05_updating.ipynb`](05_updating.ipynb) | `update()` with streaming batches against refitting (same exact KL, constant cost per batch), the update path of each mechanism (Dirichlet counts, recursive least squares, online EWC; `UpdateHistory`), atomic rollback of a failed update, and tracking a change of regime with `forgetting=`. |
| [`06_mechanism_gallery.ipynb`](06_mechanism_gallery.ipynb) | The six continuous mechanisms on a bimodal, a heteroscedastic and a skewed conditional density: fitted against true densities, held-out log-likelihood and fit time. |
| [`07_batching_devices_persistence.ipynb`](07_batching_devices_persistence.ipynb) | `query_batch` against a loop of `query` (timing from 1 to 4096 queries), device selection and `model.to()`, a `save` / `load` round trip with identical answers, and `from_bif` on the Asia network shipped in `tests/data/bnlearn/`. |
| [`08_autograd.ipynb`](08_autograd.ipynb) | Gradients through the network: learn an upstream calibration through `log_prob`, optimise a `do=` value through `sample`, the sensitivity of a downstream answer to a CPD parameter (autograd against finite differences and the exact value), and why `query` is not differentiable. |

## Structural causal models and probability sets

For `nbn.scm` and `nbn.uncertainty` (user guide:
[structural causal models](https://nbn.readthedocs.io/en/latest/user_guide/scm.html),
[probability sets](https://nbn.readthedocs.io/en/latest/user_guide/probability_sets.html)).

| Notebook | What you learn |
|---|---|
| [`09_discrete_noise_and_probability_sets.ipynb`](09_discrete_noise_and_probability_sets.ipynb) | Recover a discrete noise law with `DiscreteAdditiveNoiseMechanism`, query with `do=`, and compare three probability sets with a 90% guarantee (per-entry intervals, L1 balls, shared-noise polytope), including a coverage check over 100 datasets; then sets from a count table (`count_table_for`, `interval_set_from_counts`) for two candidate parent sets, and why a candidate that misses a parent depends on the logger. |
| [`10_confounding_and_noise_diagnostics.ipynb`](10_confounding_and_noise_diagnostics.ipynb) | Detect a confounded logger with `context_invariance_test` (permutation vs chi-square reference on sparse strata), and test noise sharing, tying and freshness with Benjamini–Hochberg, and how to read a rejection. |
| [`11_continuous_noise_and_counterfactuals.ipynb`](11_continuous_noise_and_counterfactuals.ipynb) | Fit an MLP location with mixture noise (`AdditiveNoiseMechanism`), compare it with Gaussian noise, bound `P(a < X <= b \| pa)` with a DKW band, and compute counterfactuals by exact abduction. |
| [`12_mdp_trajectories_factored_sets.ipynb`](12_mdp_trajectories_factored_sets.ipynb) | Turn episodes into transitions, fit one mechanism per next-state variable, and build trajectory-valid probability sets for a factored transition kernel, with the budget split across variables. |
| [`13_discrete_counterfactuals.ipynb`](13_discrete_counterfactuals.ipynb) | Two declared couplings (Gumbel-max, monotone) giving different counterfactuals from one kernel, sharp bounds over every compatible SCM and over a confidence box, and calibration with a randomized-PIT band and conformal sets. |
| [`14_structure_uncertainty_hypothesis_sets.ipynb`](14_structure_uncertainty_hypothesis_sets.ipynb) | Several candidate parent sets per node on a four-machine ring logged by two policies: `HypothesisSet`, the factor cache (12 factors for 81 hypotheses), one trajectory-valid set per factor with the budget split over factors, why supersets are right but wider and a missing parent is biased, and `model_for`. |
| [`15_agreement_rule.ipynb`](15_agreement_rule.ipynb) | One valid set per row when the parents are unknown: `agreement_set` (KWIK agreement rule under a declared bound on the number of parents) on the ring of notebook 14, candidates refuted by their one-variable extensions, 176 never-visited joint values covered per machine, a coverage check over 30 datasets, and what happens when the bound is wrong. |
| [`16_export_factored_sets.ipynb`](16_export_factored_sets.ipynb) | Export the ring's agreement sets for planning and verification tools (`export_factored_sets`, schema `nbn.factored_sets/1`): the manifest, the row map that encodes the sharing (32 planning rows per factor row), outward-rounded rationals, and joint next-state bounds of a never-visited row rebuilt from the file and checked against the truth. |
| [`17_continuous_hypothesis_sets.ipynb`](17_continuous_hypothesis_sets.ipynb) | Continuous and mixed factors: `HypothesisSet.fit` trains all candidates of a node together (stacked MLP, one mask per candidate) into standard mechanisms, `calibrated_sets` puts one conformal set per factor with the level split over factors, marginal vs conditional coverage under parent-dependent noise, conformal sets for a discrete child with a continuous parent, and `model_for`. |
| [`18_confounded_rows.ipynb`](18_confounded_rows.ipynb) | Confounded action effects from two regimes (a doctor who reads a hidden strength and a randomised trial): the biased naive table, `confounding_gate` with `context_invariance_test`, sound rows by intersection, IV linear program and LP with a declared randomised regime, a refuted shared environment, and a coverage check over 20 datasets. |

## Running them

```bash
pip install "nbn[examples]"        # matplotlib, JupyterLab and zuko; or "nbn[all]"
jupyter lab examples/
```

From a checkout: `pip install -e ".[examples]"` (or `uv run --extra examples jupyter lab examples/`).
Section 4 of notebook 07 (`from_bif`) also needs `pgmpy` (`pip install pgmpy`,
or the `bench` extra).  Each notebook runs in under 30 s on a laptop CPU.
The device is chosen automatically (CUDA, then Apple MPS, then the CPU); set
`NBN_EXAMPLES_DEVICE=cpu` to force the CPU.  The stored outputs were made on
the CPU; on an accelerator, results that depend on random draws or neural
training can differ in the last digits, and timings differ.

## In CI

The `notebooks` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
executes every notebook on a Linux CPU runner with
`pytest --nbmake examples/` (`nbmake` is in the `dev` extra).  To run the same
check locally:

```bash
pip install -e ".[all,dev]"
MPLBACKEND=Agg pytest --nbmake examples/
```

Note: any string is a valid node name, including names of `torch.nn.Module`
attributes (e.g. `train`, `eval`, `to`) and names with dots (`pm2.5`): NBN
stores such nodes under an escaped alias and every call keeps the original
name.
