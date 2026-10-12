# Examples: structural causal models and probability sets

Small, self-contained scripts for `nbn.scm` and `nbn.uncertainty`
(design note: `docs/scm-design-draft.md`).  Each one simulates its own data
from a known ground truth, so you can compare what is learned with the truth.

```bash
uv run python examples/01_discrete_noise_and_probability_sets.py
uv run python examples/02_confounding_and_noise_diagnostics.py
uv run python examples/03_continuous_noise_and_counterfactuals.py
uv run python examples/04_mdp_trajectories_factored_sets.py
```

| Script | Shows |
|---|---|
| `01_discrete_noise_and_probability_sets.py` | `DiscreteAdditiveNoiseMechanism` (`x = clip(g(pa) + U)`): recover the noise law; `query` with `do=`; three probability sets with a guarantee (per-entry intervals, L1 balls, shared-noise polytope) and how tight each is |
| `02_confounding_and_noise_diagnostics.py` | episodes from three behaviour policies, one confounded; `context_invariance_test` (policy as context) flags it; `noise_sharing_test`, `noise_tying_test`, `freshness_test` with Benjamini–Hochberg — and how to read a rejection |
| `03_continuous_noise_and_counterfactuals.py` | `AdditiveNoiseMechanism` with an MLP location and mixture noise; held-out likelihood vs Gaussian noise; DKW band on `P(a < X <= b | pa)`; `counterfactual_sample` |
| `04_mdp_trajectories_factored_sets.py` | `transitions_from_episodes`; one mechanism per next-state variable; trajectory-aware sets (`sampling="trajectory"`), the budget split across variables, and bounds on a transition probability of the factored kernel |

All scripts run on the CPU in a few seconds; pass `device="mps"` or
`device="cuda"` to the model to run them on an accelerator.

Note: node names must not collide with `torch.nn.Module` attributes
(e.g. `train`, `eval`, `to`): NBN stores one mechanism per node in an
`nn.ModuleDict`.  The examples use `exercise` instead of `train`.
