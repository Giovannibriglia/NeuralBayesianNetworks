# v0.6c-d paper-data run

This document pins the canonical paper data for NeuralBayesianNetworks: the
two paper runs (inference + parameter learning) executed on Giovanni's
RTX 4070 Laptop at branch SHA `33ca98c`, against master SHA `8190e35`
(post v0.7-#37 W₁ truncation fix).

> **Audit caveat (added v0.7)** — parameter-learning rows for `nbn-neuralcat`
> and `pgmpy-bayes` are bit-identical to `nbn-cat` and `pgmpy-mle`
> respectively (max abs diff = 0.0 across all 24-25 cells per pair). The
> v0.6c-C-1b runner refactor wires `_param_learning_cell` to dispatch on the
> legacy adapter string (`"nbn"` / `"pgmpy"`), then relabels result rows by
> method-keyed label. The relabeling produces method-keyed columns in the
> parquet, but the underlying fit is shared per family. See
> [`docs/audits/v0.7-43-fit-path-audit.md`](../audits/v0.7-43-fit-path-audit.md)
> for the full trace and v0.8-targeted fix scope.
>
> The paper's headline `nbn-cat` 2.3× quality finding compares `nbn-cat` to
> `pgmpy-mle` — these dispatch to genuinely distinct fit paths
> (`CategoricalTableMechanism` closed-form counting vs pgmpy's
> `MaximumLikelihoodEstimator`). That comparison is honest and stands. The
> `continuous_lg` `nbn-lg` vs `pgmpy-lg` comparison is also unaffected
> (different family, different code path).

## Hardware and software

- **GPU**: NVIDIA GeForce RTX 4070 Laptop (8 GB VRAM)
- **PyTorch**: 2.11.0+cu130
- **Run SHA (this branch)**: `33ca98c99b702ae8ed72b97f778971a5286ee774`
- **Master SHA at run**: `8190e35`
- **Inference wall time**: 11.4 h (41,034 s)
- **Parameter-learning wall time**: 6.5 h (23,376 s)
- **Total wall time**: 17.9 h

## Configuration

The laptop variants differ from the canonical paper YAMLs only in batch
sizing for 8 GB VRAM: the **inference** laptop YAML reduces
`nbn_batch_size: 1024 → 256` (per-cell inference batch); the
**parameter-learning** laptop YAML reduces `batch_size: 4096 → 1024`
(per-fit training minibatch). All other scientific knobs are preserved.

Shared config: `n_seeds=5`, `n_train=10000`, `per_cell_timeout_s=600`,
`n_nodes=[10, 50, 100, 500, 1000]`, `fraction_continuous=0.5`,
`edge_density=0.2`, `cardinality=4`. Inference uses `n_test=1024` and
`nbn_lw_n_samples=4000`. Parameter learning uses `n_test=2000` and
`nbn_lw_n_samples=512`.

## Run details

| Mode | Wall time | Cells | STATUS |
| --- | --- | --- | --- |
| Inference | 11.4 h | 1571 | ok=501, not_supported=936, error=75, timeout=34, oom=20, no_result=5 |
| Parameter learning | 6.5 h | 1442 | ok=484, not_supported=925, error=3, timeout=30 |

`not_supported` cells are skipped pre-dispatch by the registry applicability
gate; they are not failures. Genuine DNFs total 134 (inference) + 33
(parameter-learning) = 167 across 3013 total cells (5.5%). Every DNF maps
to a known v0.7 backlog issue, an out-of-design-domain baseline, or an
8 GB VRAM scaling limit. See `dnf_cells.md` for the categorised table.

## Headline findings

### Inference

The v0.7-#37 W₁ truncation fix is verified at paper scale. All four
`continuous_lg` baselines cluster within 0.02 W₁ at every n:

| n_nodes | pgmpy-lg-predict | nbn-lg-lw | nbn-mdn-lw | nbn-flow-lw | spread |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.0465 | 0.0525 | 0.0560 | 0.0564 | 0.010 |
| 50 | 0.1411 | 0.1490 | 0.1503 | 0.1553 | 0.014 |
| 100 | 0.4694 | 0.4696 | 0.4795 | 0.4666 | 0.013 |
| 500 | 0.5842 | 0.5800 | 0.5858 | 0.5969 | 0.017 |
| 1000 | 0.2059 | 0.2216 | 0.2158 | 0.2227 | 0.017 |

**Speed (continuous_lg, total_time_s)**: NBN-lg-lw is 9–22× faster than
pgmpy-lg-predict across all n_nodes:

| n_nodes | pgmpy-lg-predict | nbn-lg-lw | speedup |
| --- | --- | --- | --- |
| 10 | 0.0426 | 0.0019 | 22× |
| 50 | 0.2197 | 0.0268 | 8× |
| 100 | 0.8065 | 0.0593 | 14× |
| 500 | 3.3412 | 0.3611 | 9× |
| 1000 | 8.5271 | 0.7216 | 12× |

On discrete at n=10, NBN-cat-ve (1.4 ms) is 75× faster than pgmpy-mle-ve
(108 ms).

### Parameter learning

NBN-cat parameter learning is **2.3× more accurate** than pgmpy-mle on
discrete networks at scale (TV-per-node, lower is better):

| n_nodes | nbn-cat | pgmpy-mle | NBN advantage |
| --- | --- | --- | --- |
| 10 | 0.022 | 0.026 | 1.2× |
| 50 | 0.100 | 0.245 | 2.5× |
| 100 | 0.141 | 0.314 | 2.2× |
| 500 | 0.145 | 0.338 | 2.3× |
| 1000 | 0.146 | 0.340 | 2.3× |

pgmpy-mle saturates at TV ≈ 0.34 starting at n=50; NBN-cat plateaus at
TV ≈ 0.14. The gap opens at n=50 and persists through n=1000.

On `continuous_lg`, NBN-lg matches pgmpy-lg quality within MC noise
(W₁-per-node ≈ 0.083 across all n) at 2× the speed:

| n_nodes | nbn-lg | pgmpy-lg | gap |
| --- | --- | --- | --- |
| 10 | 0.0734 | 0.0836 | NBN slightly better |
| 50 | 0.0838 | 0.0829 | tied |
| 100 | 0.0819 | 0.0824 | tied |
| 500 | 0.0831 | 0.0832 | tied |
| 1000 | 0.0827 | 0.0828 | tied |

At n=1000, NBN-lg takes 2.5 s vs pgmpy-lg 5.2 s (2× faster).

On the `hybrid` family, NBN-hybrid completes all 25 (5 seeds × 5 n_nodes)
cells with W₁-per-node ≈ 0.08–0.10. No other library has applicable
baselines for hybrid — this is NBN-only territory.

## Reproducing the run

```bash
git checkout 33ca98c99b702ae8ed72b97f778971a5286ee774
nbn-bench inference \
  --config benchmarking/configs/inference_paper_laptop.yaml
nbn-bench param-learning \
  --config benchmarking/configs/parameter_learning_paper_laptop.yaml
```

Numerical values vary within MC noise; STATUS counts and qualitative
findings (cluster, speedup, quality gap) are stable.
