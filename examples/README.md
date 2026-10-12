# Examples: structural causal models and probability sets

Five Jupyter notebooks for `nbn.scm` and `nbn.uncertainty`
(design note: [`docs/scm-design-draft.md`](../docs/scm-design-draft.md)).
Each one simulates its own data from a known ground truth, so you can compare
what is learned with the truth.  The outputs and plots are stored in the
notebooks, so they read on GitHub without running anything.

| Notebook | What you learn |
|---|---|
| [`01_discrete_noise_and_probability_sets.ipynb`](01_discrete_noise_and_probability_sets.ipynb) | Recover a discrete noise law with `DiscreteAdditiveNoiseMechanism`, query with `do=`, and compare three probability sets with a 90% guarantee (per-entry intervals, L1 balls, shared-noise polytope), including a coverage check over 100 datasets. |
| [`02_confounding_and_noise_diagnostics.ipynb`](02_confounding_and_noise_diagnostics.ipynb) | Detect a confounded logger with `context_invariance_test`, and test noise sharing, tying and freshness with Benjamini–Hochberg, and how to read a rejection. |
| [`03_continuous_noise_and_counterfactuals.ipynb`](03_continuous_noise_and_counterfactuals.ipynb) | Fit an MLP location with mixture noise (`AdditiveNoiseMechanism`), compare it with Gaussian noise, bound `P(a < X <= b \| pa)` with a DKW band, and compute counterfactuals by exact abduction. |
| [`04_mdp_trajectories_factored_sets.ipynb`](04_mdp_trajectories_factored_sets.ipynb) | Turn episodes into transitions, fit one mechanism per next-state variable, and build trajectory-valid probability sets for a factored transition kernel, with the budget split across variables. |
| [`05_discrete_counterfactuals.ipynb`](05_discrete_counterfactuals.ipynb) | Two declared couplings (Gumbel-max, monotone) giving different counterfactuals from one kernel, sharp bounds over every compatible SCM and over a confidence box, and calibration with a randomized-PIT band and conformal sets. |

## Running them

```bash
pip install "nbn[examples]"        # matplotlib + JupyterLab; or "nbn[all]"
jupyter lab examples/
```

From a checkout: `pip install -e ".[examples]"` (or `uv run --extra examples jupyter lab examples/`).
Each notebook runs in a few seconds on a laptop CPU.  The device is chosen
automatically (CUDA, then Apple MPS, then the CPU); set
`NBN_EXAMPLES_DEVICE=cpu` to force the CPU.  The stored outputs were made on
the CPU; on an accelerator, results that depend on random draws or neural
training can differ in the last digits.

## In CI

The `notebooks` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
executes every notebook on a Linux CPU runner with
`pytest --nbmake examples/` (`nbmake` is in the `dev` extra).  To run the same
check locally:

```bash
pip install -e ".[all,dev]"
MPLBACKEND=Agg pytest --nbmake examples/
```

Note: node names must not collide with `torch.nn.Module` attributes
(e.g. `train`, `eval`, `to`): NBN stores one mechanism per node in an
`nn.ModuleDict`.  The examples use `exercise` instead of `train`.
