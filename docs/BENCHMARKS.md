# Benchmark suite

NBN ships a benchmark suite (`nbn.bench`, CLI `nbn-bench`) that compares NBN
against pgmpy, pomegranate and Pyro on Bayesian networks with
**known ground truth**: synthetic networks of controlled size and family,
and the [bnlearn repository](https://www.bnlearn.com/bnrepository/) networks.
This page covers installing, running, plotting and configuring it. To add a
new problem source, baseline or metric, see
[`benchmarks_extending.md`](benchmarks_extending.md).

- [Install and check the environment](#install-and-check-the-environment)
- [What is measured](#what-is-measured)
- [Run](#run)
- [Plot the results](#plot-the-results)
- [Configuration](#configuration)
- [Reproducing the paper numbers](#reproducing-the-paper-numbers)

## Install and check the environment

```bash
pip install -U "nbn[all]"          # or, from a checkout: pip install -U -e ".[all]"
nbn-bench check-env                # must end with "environment OK"
```

`all` = `bench` (runner, pgmpy, pomegranate, plotting) + `neural` (zuko) +
`gp` (gpytorch) + `mcmc` (pyro). Use it for benchmark runs: a baseline whose
library is missing is recorded as `not_supported`, not as an error, so a
partial install silently produces an incomplete run.

The adapters are written against specific library versions (pgmpy ≥ 1.0 for
`DiscreteBayesianNetwork`; scikit-learn ≥ 1.6, which pgmpy 1.x needs but does
not pin; pomegranate ≥ 1.0 for the torch API; …). `nbn-bench check-env`
verifies every declared requirement: installed, importable (including the
submodules the adapters use, e.g. `pgmpy.models`), at the required version,
and not shadowed by a second copy in `~/.local`. It also prints the
accelerator the run will use:

```bash
nbn-bench check-env                        # every requirement
nbn-bench check-env --config <run.yaml>    # only what that config's baselines need
```

`nbn-bench inference` and `nbn-bench param-learning` run the same check for
their config's baselines and **refuse to start** on a problem (exit code 2);
`--skip-env-check` overrides that. Run the check again after every
`git pull` on a benchmark machine.

## What is measured

| Benchmark | Command | x axis | Metrics |
|---|---|---|---|
| **Parameter learning** — accuracy of the fitted CPDs against the true generative process | `nbn-bench param-learning` | `n_nodes` (or `n_train` for learning curves) | `log_likelihood`, `param_recovery_{tv,kl}` (discrete), `calibration_{pit_ks,sd_ratio}` (continuous), `fit_time` |
| **Inference** — accuracy and total time for `Q` conditional queries | `nbn-bench inference` | `n_nodes`, `batch_size`, or bnlearn `network` | `tv_per_node`, `jsd_per_node` (discrete), `w1_per_node` (continuous), `total_query_time`, `fit_time` |

NBN answers the `Q` queries with one batched `query_batch(B=Q)` call; the
other libraries loop over the same queries in Python. Synthetic families are
`discrete`, `continuous_lg` (linear Gaussian), `continuous_nongauss` and
`hybrid`.

Shipped configs (under `nbn/bench/configs/`):

| Config | Command | What it sweeps |
|---|---|---|
| `synthetic/smoke_tests/inference_smoke.yaml` | `inference` | CI smoke, < 90 s on CPU |
| `synthetic/smoke_tests/parameter_learning_smoke.yaml` | `param-learning` | CI smoke |
| `synthetic/complete/inference_complete.yaml` | `inference` | network size |
| `synthetic/complete/inference_scalability_complete.yaml` | `inference` | network size, up to large n (time is the headline) |
| `synthetic/speed/inference_speed.yaml` | `inference` | query batch size |
| `synthetic/complete/parameter_learning_complete.yaml` | `param-learning` | network size |
| `synthetic/learning_curves/learning_curves.yaml` | `param-learning` | training-set size |
| `bnlearn/complete/inference_complete.yaml` | `inference` | bnlearn networks |
| `*/reruns/*.yaml` | either | partial reruns, combined with `nbn-bench merge` |

`learning_curves.yaml` and `parameter_learning_complete.yaml` declare
`metrics: log_likelihood`, so they **must** run under `param-learning`;
`inference` refuses them and prints the command to use.

## Run

```bash
# Smoke tests (what CI runs; about a minute each)
nbn-bench param-learning --config nbn/bench/configs/synthetic/smoke_tests/parameter_learning_smoke.yaml
nbn-bench inference      --config nbn/bench/configs/synthetic/smoke_tests/inference_smoke.yaml

# One full benchmark
nbn-bench inference --config nbn/bench/configs/synthetic/complete/inference_complete.yaml --device auto

# All paper-scale benchmarks, at most three at a time
bash scripts/run_all_benchmarks.sh
MAX_PARALLEL=2 bash scripts/run_all_benchmarks.sh    # fewer in flight (8 GB GPUs)
DEVICE=cpu     bash scripts/run_all_benchmarks.sh    # force CPU
```

`--device` takes `auto` (cuda > mps > cpu), `gpu` (cuda > mps), `cpu`,
`cuda[:i]` or `mps`; per-baseline `device:` keys in the YAML override it.
The shipped configs pin baselines that are too slow on a CPU (KDE/kNN/FlexCode
likelihood weighting, paper-scale neural fits) to `device: gpu`, so the same
config runs on an NVIDIA machine and on an Apple-Silicon Mac. pgmpy is
CPU-only on every platform, and Pyro's importance sampler runs on the CPU
under `auto` on a Mac (it is launch-bound and times out on `mps`). The complete configs are paper-scale: expect 10–20 h each on
a single consumer GPU. `scripts/run_all_benchmarks.sh` works with the
bash 3.2 that macOS ships, runs `check-env` first, and sets
`PYTHONNOUSERSITE=1` so user-site packages cannot shadow the environment.
Launch long runs from a plain terminal (or `tmux` / `nohup`), not an IDE's
embedded terminal. On Linux desktops, systemd-oomd kills the IDE's whole
process group under memory pressure.

The [`notebooks/`](../notebooks) folder has one Google Colab notebook per
paper-scale benchmark, for running on a cloud GPU.

Each invocation writes one run directory; the parquet is the canonical
artefact (figures and tables are never generated automatically):

    results/benchmark_<benchmark>_<config_name>_<YYYYMMDD_HHMMSS>/
        <config_name>_metrics.parquet     one row per (cell, metric)
        metrics.jsonl                     the same rows, streamed while running
        run.log                           per-cell log (fit/query phases, errors)

The runner appends to `metrics.jsonl` after every finished cell, so an
interrupted run keeps its completed cells.

**macOS notes.** On Apple Silicon, `--device auto` resolves to `mps`, and
bnlearn networks download over HTTPS with certifi's CA bundle, so the
python.org Python needs no extra certificate setup.
`gpu_peak_mb` reports the allocation at measurement time (Metal exposes no
high-water mark), so it is a lower bound on the peak. macOS does not enforce
the per-cell `RLIMIT_AS` memory cap, so cells rely on the fit and query time
budgets.

## Plot the results

`nbn-bench plot` turns one or more run directories into figures (PDF) and
LaTeX tables. The x axis is inferred from the parquet (`batch_size` sweep,
`n_train` sweep, bnlearn network, else `n_nodes`):

```bash
nbn-bench plot results/benchmark_synthetic_complete_<timestamp> \
  --output-dir results/figures/complete
# options: --aggregation iqm_iqr|mean_std (default iqm_iqr)
#          --benchmark synthetic|bnlearn  (default: every benchmark in the parquet)
#          --top-nbn N                    (nbn methods shown per x value, default 2)
```

Every figure is a grouped bar plot (one group per x value, one bar per
method, error bar = aggregation band across seeds), in two views:

- `all/`: each method is aggregated over the seeds **it** solved. A bar
  whose method solved fewer than all seeds carries `k/n`; a method that
  solved none shows its failure code (`timeout`, `oom`, `error`).
- `common/`: each method is aggregated over the seeds solved by **every
  shown method** at that x, so the bars in a group are computed on the same
  problems. The x label reports `|C|`, the common-seed count.

Shown methods at each x are every non-NBN baseline applicable to the family
plus the `--top-nbn` best NBN methods. Output tree, per family:

```
<output-dir>/<benchmark>/<family>/
  all/plots/<metric>_vs_<x>.pdf     all/tables/<metric>_vs_<x>.tex
  all/plots/success_rate.pdf        (status breakdown, diagnostic)
  common/plots/<metric>_vs_<x>.pdf  common/tables/<metric>_vs_<x>.tex
  common/common_seeds.txt           the common seeds per metric and x
  selection.txt                     the nbn methods shown per (view, metric, x)
```

A figure is written only when at least one seed was solved for its metric in
that family. If a plot you expect is missing, `nbn-bench plot -v` logs
`skip ...` with the reason, and `run.log` in the run directory has the
per-cell error. To combine a full run with a partial rerun, use
`nbn-bench merge`. The aggregation contract is specified in
[`v0.18-bar-reporting-all-common.md`](v0.18-bar-reporting-all-common.md).

## Configuration

A config is a YAML file. Each shipped smoke config starts with the full
schema as a comment (source of truth: `nbn/bench/core/yaml_config.py`). The
main fields:

```yaml
version: "v0.13"              # schema version (required)
benchmark: synthetic          # synthetic | bnlearn
config_name: smoke            # prefix of the run directory and artefacts
metrics: all                  # all | timing; param-learning configs use log_likelihood
selector: uniform_random      # how query targets/evidence are chosen

source:                       # synthetic shown; bnlearn takes a network list
  families: [discrete, continuous_lg, continuous_nongauss, hybrid]
  n_nodes_list: [5, 10]
  seeds: [0]
  n_train: 2048
  n_test: 512
  n_reference: 5000           # oracle samples for continuous ground truth
  edge_density: 0.20
  max_in_degree: 2
  cardinality: 4
  fraction_continuous: 0.5

baselines:                    # one entry per method
  - {library: pgmpy, mechanism: discrete, param_method: mle, inference_method: ve}
  - {library: nbn,   mechanism: cat,      param_method: mle, inference_method: ve}
  - {library: nbn,   mechanism: mdn,      param_method: mle, inference_method: lw,
     extra_kwargs: {epochs: 20, n_samples: 1024}}

n_queries_per_cell: 2
per_cell_timeout_s: 60.0
```

Each baseline needs `library`, `mechanism` and `param_method`;
`inference_method`, `device` and `extra_kwargs` are optional. See the shipped
configs for every supported combination.

## Reproducing the paper numbers

The headline numbers in the [README](../README.md#benchmark-highlights) come
from the paper-data run anchored at tag **`v0.6c-d`** (commit `2e0dd32`):

| | |
|---|---|
| Hardware | NVIDIA GeForce RTX 4070 Laptop (8 GB VRAM) |
| PyTorch | 2.11.0+cu130 |
| Wall time | 11.4 h inference + 6.5 h parameter learning |
| Shared config | 5 seeds, `n_train=10000`, `n_nodes ∈ {10, 50, 100, 500, 1000}`, 600 s per-cell timeout |

To regenerate comparable numbers on the current code, run the shipped
configs that succeed the paper ones:

```bash
nbn-bench inference      --config nbn/bench/configs/synthetic/complete/inference_complete.yaml
nbn-bench param-learning --config nbn/bench/configs/synthetic/complete/parameter_learning_complete.yaml
```

On a GPU with little memory, lower the `BATCH_SIZE_FIT` anchor at the top of
a copy of the config (the NBN training minibatch, default 1024).

[`v0.6c-d/run_summary.md`](v0.6c-d/run_summary.md) has the full headline
tables and status counts, including an audit caveat on two pairs of
parameter-learning rows. [`v0.6c-d/dnf_cells.md`](v0.6c-d/dnf_cells.md)
categorises every cell that did not finish. Numerical values vary within
Monte-Carlo noise across hardware; status counts and the qualitative findings
(speed-up, quality gap) are stable.
