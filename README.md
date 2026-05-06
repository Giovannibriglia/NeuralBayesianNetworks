# NeuralBayesianNetworks (NBN)

A PyTorch-native library for learning, sampling from, and querying Bayesian
Networks with a known DAG, where each node carries a learnable, batched,
GPU-resident neural conditional distribution.

## Why NBN

NBN is to Bayesian Networks what GPyTorch is to Gaussian Processes:
a torch-native, batchable, autograd-friendly framework where every
conditional distribution is a swappable, learnable module, and every
query is a batched tensor operation.

| Library         | Discrete BN  | Continuous       | Batched queries | Neural CPDs | Hybrid native |
|-----------------|:------------:|:----------------:|:---------------:|:-----------:|:-------------:|
| pgmpy           | ✅ exact     | ✅ Gaussian only | ❌              | ❌          | ⚠️ CG only    |
| pomegranate     | ✅           | ✅               | partial         | ❌          | ⚠️ limited    |
| GPyTorch        | ❌           | ✅ GP            | ✅              | ✅          | ❌            |
| Pyro / NumPyro  | ✅ via enum  | ✅               | partial         | ✅          | ✅ universal  |
| **NBN**         | **✅ exact** | **✅ MDN/Flow/GP** | **✅ batched VE** | **✅** | **✅ native** |

## Install

    pip install -e ".[dev,bench,neural]"

## Repository layout

    nbn/                Library code (mechanisms, inference, sampling, core).
    benchmarking/       Crash-test runner, baselines, configs, output figures.
    tests/              Unit + integration tests.
    RESEARCH.md         Paper outline and contribution claims.

## Crash tests

NBN ships two crash tests on synthetic Bayesian networks with **known
ground truth**, sweeping network size on the x-axis:

1. **Parameter-learning crash test** — measures accuracy of fitted CPDs
   against the true generative process. Speed is not measured.
2. **Inference crash test** — measures both accuracy and total time for
   `Q` conditional queries. NBN uses `query_batch(B=Q)` (one batched
   call); other libraries loop over the same `Q` queries in Python.

Each crash test has a smoke config (CI, < 60s) and a paper config
(local reproduction, ~30 min on CPU).

### Reproduce

```bash
# Smoke (runs in CI):
nbn-bench param-learning --config benchmarking/configs/parameter_learning_smoke.yaml
nbn-bench inference      --config benchmarking/configs/inference_smoke.yaml

# Paper:
nbn-bench param-learning --config benchmarking/configs/parameter_learning_paper.yaml
nbn-bench inference      --config benchmarking/configs/inference_paper.yaml
```

Each invocation writes its output under `benchmarking/results/`:

    benchmarking/results/figures/{prefix}_total_time_vs_size.{pdf,svg,png}
    benchmarking/results/figures/{prefix}_accuracy_vs_size.{pdf,svg,png}
    benchmarking/results/raw/{prefix}_metrics.parquet
    benchmarking/results/raw/{prefix}_{timestamp}.log         (gitignored)
    benchmarking/results/raw/{prefix}_{timestamp}.run.json    (gitignored)
    benchmarking/results/tables/                              (placeholder for v0.6c-C)

### Smoke results

(Smoke figures will appear here after PR-B's data-layer fixes land.
This PR ships the structure; figures with all baselines visible
require the data fixes in PR-B.)

### Paper results

Run the paper configs locally and commit the resulting figures to
`benchmarking/results/figures/` for the README to display. CI does
not run paper configs.

## Configuration

Each config is a YAML file with these fields:

    mode:                 'parameter_learning' | 'inference'
    families:             list of families ∈ {discrete, continuous_lg,
                          continuous_nongauss, hybrid}
    n_nodes:              list of network sizes
    n_seeds:              number of seeds per cell (mean ± std reported)
    n_queries_per_cell:   number of queries per cell
    nbn_batch_size:       B for NBN's query_batch (inference mode only)
    baselines:            list of baseline names
    per_cell_timeout_s:   wall-clock cap per (family, n_nodes, seed, baseline)

See `benchmarking/configs/*.yaml` for all four shipped configs.

## Status

| Component                                | Status                |
|------------------------------------------|-----------------------|
| Core (DAG, Variables, Factor)            | ✅                    |
| Mechanisms (Table, LG, MDN, Flow, GP)    | ✅                    |
| Tensor VE + LW + HybridRouter            | ✅                    |
| Vectorised batched query_batch           | ✅ v0.3.1             |
| Synthetic crash-test framework           | ✅ v0.4a              |
| Restructured benchmarking layout         | ✅ v0.5a (this PR)    |
| Inference accuracy plumbing + plotter    | ⏳ v0.5b              |
| GPU-required perf (CUDA graphs etc.)     | ⏳ tracked on #7      |
| 5-seed error bars                        | ⏳ tracked on #7      |
| Strongly-non-Gaussian ablation           | ⏳ v0.5/v0.6          |

## License

Apache 2.0.
