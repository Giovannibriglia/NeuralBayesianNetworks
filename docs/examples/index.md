# Examples

Executed Jupyter notebooks, one per feature. Each simulates its own data from a
known ground truth, so what NBN learns or infers is compared with the exact
answer. The pages below show the stored outputs; the notebooks themselves are
in [`examples/`](https://github.com/Giovannibriglia/NeuralBayesianNetworks/tree/master/examples).

To run them: `pip install "nbn[examples]"` (matplotlib, JupyterLab and zuko;
notebook 07 also needs pgmpy), then `jupyter lab examples/` from a checkout.
Each runs in under 30 s on a laptop CPU; set `NBN_EXAMPLES_DEVICE=cpu` to
force the CPU.

## The core library

| Notebook | What it shows |
|---|---|
| {doc}`01_parameter_learning` | Every mechanism family on a hybrid network with known densities: `auto_mechanisms`, KL to the truth per node and against the training-set size, weighted fitting, fixed mechanisms, `TrainHistory` and warm starts. |
| {doc}`02_inference` | Exact variable elimination, likelihood weighting (error against particles, ESS), `HybridRouter`, the amortized engines, continuous evidence and targets, `map_query` and joint queries. |
| {doc}`03_interventions` | Seeing against doing on two confounded networks, a dose-response curve with one intervention per row, `intervene()` and `do=` on every engine. |
| {doc}`04_sampling` | `sample` against the true marginals and joint, `return_log_prob`, clamped evidence with importance weights, and samples under `do=`. |
| {doc}`05_updating` | `update()` on streaming batches against refitting, the update path of each mechanism, atomic rollback, and tracking a change of regime with `forgetting=`. |
| {doc}`06_mechanism_gallery` | The six continuous mechanisms on a bimodal, a heteroscedastic and a skewed density: fit, held-out log-likelihood and fit time. |
| {doc}`07_batching_devices_persistence` | `query_batch` against a loop of `query` (1 to 4096 queries), devices and `model.to()`, `save` / `load`, and `from_bif` on the Asia network. |
| {doc}`08_autograd` | Learning an upstream calibration through `log_prob`, optimising a `do=` value through `sample`, and a CPD sensitivity checked against finite differences. |

## Structural causal models and probability sets

| Notebook | What it shows |
|---|---|
| {doc}`09_discrete_noise_and_probability_sets` | A discrete noise law with `DiscreteAdditiveNoiseMechanism`, and three 90% probability sets (intervals, L1 balls, shared-noise polytope) with a coverage check; sets from a count table for candidate parent sets. |
| {doc}`10_confounding_and_noise_diagnostics` | Detecting a confounded logger with `context_invariance_test`, and testing noise sharing, tying and freshness with Benjamini-Hochberg. |
| {doc}`11_continuous_noise_and_counterfactuals` | An MLP location with mixture noise, a DKW band on `P(a < X <= b | pa)`, and counterfactuals by exact abduction. |
| {doc}`12_mdp_trajectories_factored_sets` | From logged episodes to a factored transition model with trajectory-valid probability sets, the budget split across variables. |
| {doc}`13_discrete_counterfactuals` | Two declared couplings giving different counterfactuals from one kernel, sharp bounds over every compatible SCM, and calibration with a PIT band and conformal sets. |
| {doc}`14_structure_uncertainty_hypothesis_sets` | Candidate parent sets per node with `HypothesisSet`: the factor cache, one set per factor with the budget split over factors, supersets against a missing parent, and `model_for`. |
| {doc}`15_agreement_rule` | One valid set per row under structure uncertainty with `agreement_set`: refutation by one-variable extensions, coverage of never-visited joint values, a coverage check, a wrong bound, and the union over a given family (`refute=False`). |
| {doc}`16_export_factored_sets` | Export factored sets with their sharing for robust / interval / parametric MDP tools, read them back, and rebuild the joint bounds of a never-visited row. |
| {doc}`17_continuous_hypothesis_sets` | Continuous and mixed factors: batched candidate fits, one calibrated set per factor, marginal vs conditional coverage, and `model_for`. |
| {doc}`18_confounded_rows` | Sound interval rows for a confounded action from several regimes: the gate, intersection vs IV linear program, a declared randomised trial, refutation, and coverage. |

```{toctree}
:hidden:
:maxdepth: 1

01_parameter_learning
02_inference
03_interventions
04_sampling
05_updating
06_mechanism_gallery
07_batching_devices_persistence
08_autograd
09_discrete_noise_and_probability_sets
10_confounding_and_noise_diagnostics
11_continuous_noise_and_counterfactuals
12_mdp_trajectories_factored_sets
13_discrete_counterfactuals
14_structure_uncertainty_hypothesis_sets
15_agreement_rule
16_export_factored_sets
17_continuous_hypothesis_sets
18_confounded_rows
```
