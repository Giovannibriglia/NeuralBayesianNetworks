# DNF cells — v0.6c-d paper data

Categorised summary of cells that did not produce a value on RTX 4070
Laptop (8 GB VRAM). Of the 3013 total cells across both runs, 167 (5.5%)
are DNFs. Every DNF falls into one of three categories:

1. **Known v0.7 backlog issue** (#26, #30, #36) — to be addressed in
   subsequent releases.
2. **Out-of-design-domain baseline** — e.g. gpytorch-gp on non-Gaussian
   data, pgmpy on n=10 seeds that trigger factor explosion.
3. **Hardware-bound 8 GB VRAM limit** — fit-time scaling for
   neural-density baselines (nbn-flow, nbn-mdn) on continuous_nongauss
   at n ≥ 50; CPU baseline timeouts at n ≥ 500.

None are method-quality regressions of the v0.7-#37 W₁ fix. Source of
truth for per-row precision is the parquet
(`results/raw/{inference,parameter_learning}_paper_metrics.parquet`).

## Inference run (1571 cells)

STATUS: `ok=501, not_supported=936, error=75, timeout=34, oom=20, no_result=5`.
The 936 `not_supported` cells were skipped pre-dispatch by the registry
applicability gate (e.g. discrete-only baselines on continuous families).
Genuine DNFs total 134.

| family | baseline | status | n_nodes affected | tracking |
| --- | --- | --- | --- | --- |
| continuous_nongauss | gpytorch-gp-predict | error | [100, 500, 1000] | expected (gpytorch on non-Gaussian data; not its design domain) |
| continuous_nongauss | nbn-flow-lw | error | [50, 100, 500, 1000] | expected (flow fit-time scaling on 8 GB; large-n OOM during training) |
| continuous_nongauss | nbn-mdn-lw | error | [50, 100, 500, 1000] | expected (mdn fit-time scaling on 8 GB; large-n OOM during training) |
| discrete | pgmpy-bayes-ve | error | [10] | expected (pgmpy seed-dependent factor explosion on 3/5 seeds at n=10; pgmpy known limit) |
| discrete | pgmpy-mle-ve | error | [10] | expected (same as pgmpy-bayes-ve; both share VE inference path) |
| hybrid | nbn-hybrid-router | error | [10, 50, 100, 500, 1000] | issue #30 (HybridRouter cuda assert) |
| discrete | nbn-cat-ve | oom | [50, 100, 500, 1000] | issue #26 (NeuralCategorical-VE engine refactor; LW handles same family at scale) |
| discrete | pgmpy-bayes-ve | timeout | [500, 1000] | expected (pgmpy seed=2 factor explosion at large n) |
| discrete | pgmpy-mle-ve | timeout | [500, 1000] | expected (same as pgmpy-bayes-ve seed=2) |
| discrete | pomegranate-discrete-ve | timeout | [500, 1000] | expected (CPU baseline timeout at n≥500) |
| discrete | pyro-empirical-importance | timeout | [50, 100, 500, 1000] | issue #36 (pyro paper-config timeout) |
| discrete | pyro-empirical-importance | no_result | [10] | issue #36 (pyro completes but emits no metric value) |

## Parameter-learning run (1442 cells)

STATUS: `ok=484, not_supported=925, error=3, timeout=30`. Genuine DNFs
total 33.

| family | baseline | status | n_nodes affected | tracking |
| --- | --- | --- | --- | --- |
| discrete | nbn-cat | error | [10] | isolated IndexError at n=10 seed=0; new v0.7 issue to file separately if reproduces |
| continuous_nongauss | nbn-flow | error | [50] | expected (training instability at seed=3; NaN in distribution loc) |
| continuous_nongauss | nbn-mdn | error | [50] | expected (same NaN pattern as nbn-flow at seed=3) |
| continuous_nongauss | nbn-flow | timeout | [100, 500, 1000] | expected (flow training time exceeds 600 s per cell on 8 GB) |
| continuous_nongauss | nbn-mdn | timeout | [100, 500, 1000] | expected (same scaling limit as nbn-flow) |

## Per-row precision

For exact per-`(family, baseline, n_nodes, seed)` DNF rows, query the
parquet directly:

```python
import pandas as pd
df = pd.read_parquet('results/raw/inference_paper_metrics.parquet')
dnf = df[df['status'] != 'ok'][['family', 'baseline', 'n_nodes', 'seed', 'status', 'error_msg']]
print(dnf.to_string())
```
