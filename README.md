# NeuralBayesianNetworks (NBN)

A PyTorch-native Bayesian network library where each mechanism is learnt
by a neural network. Every node carries a learnable, batched, GPU-resident
conditional distribution; every query is a batched tensor operation;
inference and parameter learning both run end-to-end on cuda.

NBN is **9-22× faster** than pgmpy on continuous Linear Gaussian inference
and **2.3× more accurate** on discrete parameter learning at scale, and is
the only library in our benchmark suite that handles hybrid (mixed
continuous-discrete) networks at scale.

## Quick start (from a checkout)

```bash
pip install -U -e ".[all]" && nbn-bench check-env
```

This installs the library plus everything the benchmark suite needs at the
pinned versions, then verifies the environment (`nbn-bench check-env` must
print `environment OK`; the run commands refuse to start otherwise). Run it
again after every `git pull` on a benchmark machine.

## Install

```bash
pip install nbn                 # the library
pip install "nbn[neural]"       # + zuko-backed flows / MDNs (NBN's headline mechanisms)
pip install "nbn[bench]"        # + the benchmark suite (`nbn-bench`, `nbn.bench`)
pip install "nbn[all]"          # everything a benchmark run needs (bench + neural + gp + mcmc)
```

The import name is `nbn`. The `[neural]` extra adds zuko-backed flows/MDNs;
`bench` pulls in the benchmark runner's own dependencies (pandas, pyarrow,
scipy, yaml, tqdm), the external-baseline libraries (pgmpy, pomegranate) and
plotting (matplotlib, seaborn); `gp`/`mcmc` add the gpytorch and pyro
baselines; `all` is the union. Use `all` for benchmark runs: a baseline whose
library is absent is recorded as `not_supported`, not as an error.

**Check the environment before a run.** The adapters are written against
specific library versions (pgmpy ≥ 1.0 for `DiscreteBayesianNetwork`,
scikit-learn ≥ 1.6 because pgmpy 1.x needs it but does not pin it,
pomegranate ≥ 1.0 for the torch API, …). `nbn-bench check-env` verifies
every declared requirement — installed, importable (including the submodules
the adapters use, e.g. `pgmpy.models`), at the required version, and not a
second copy shadowing the environment's (a stale pgmpy in `~/.local` lost
one run all its pgmpy cells; an old scikit-learn there lost another):

```bash
nbn-bench check-env                        # every requirement
nbn-bench check-env --config <run.yaml>    # only what that config's baselines need
```

`nbn-bench inference` and `nbn-bench param-learning` run the same check for
their config's baselines and **refuse to start** on a problem (exit code 2);
`--skip-env-check` overrides that. `scripts/run_all_benchmarks.sh` runs it
too and sets `PYTHONNOUSERSITE=1` so user-site packages cannot shadow the
environment. If the check fails, `pip install -U "nbn[all]"` (or
`pip install -U -e ".[all]"` from a checkout) brings everything up to the
pinned versions.

Working from a checkout (development, or reproducing the paper runs, whose
YAML configs are addressed by repo-relative path):

```bash
git clone https://github.com/Giovannibriglia/NeuralBayesianNetworks.git
cd NeuralBayesianNetworks
pip install -e ".[dev,all]"
nbn-bench check-env
```

`dev` is the test and docs toolchain. Releases are cut by pushing a `v*` tag;
`.github/workflows/publish.yml` builds from the tag and uploads to PyPI.

### Run the benchmark suite

From the repo root, launch the six paper-scale benchmarks. At most three run
in parallel; the next starts as soon as one finishes:

```bash
bash scripts/run_all_benchmarks.sh
```

Override the pool size or device via `MAX_PARALLEL=2 bash …` or
`DEVICE=cpu bash …` (three GPU benchmarks in flight can exceed an 8 GB card).
Each benchmark leaves a run directory under `results/`; turn it into figures
and tables with `nbn-bench plot` (see [Plot the results](#plot-the-results)).

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

## Headline results

These numbers come from the canonical paper-data run at tag `v0.6c-d`
(see [Reproducibility](#reproducibility) below).

### Inference: 9-22× faster on continuous Linear Gaussian networks

![Inference total time vs network size](results/benchmark_synthetic_learning_curves_20260908_092714/figures/inference_paper_total_time_vs_size.png)

NBN-lg-lw vs pgmpy-lg-predict on continuous Linear Gaussian networks:
22× faster at n=10 (1.9 ms vs 42 ms), 12× at n=1000 (0.72 s vs 8.5 s).
Accuracy matches pgmpy within 0.02 W₁ at every n_nodes — speed gain
comes without quality regression.

On discrete networks at n=10, NBN-cat-ve runs at 1.4 ms vs pgmpy-mle-ve
at 108 ms (75× faster).

### Parameter learning: 2.3× more accurate on discrete networks at scale

![Parameter learning accuracy vs network size](results/benchmark_synthetic_learning_curves_20260908_092714/figures/parameter_learning_paper_accuracy_vs_size.png)

On discrete Bayesian networks, NBN-cat reaches TV ≈ 0.14 across all
n_nodes ≥ 50; pgmpy-mle saturates at TV ≈ 0.34. The quality gap opens
at n=50 (0.10 vs 0.25) and persists through n=1000 (0.146 vs 0.340).
NBN's gradient-based fitting scales past pgmpy's sample-complexity wall.

On continuous Linear Gaussian networks, NBN matches pgmpy quality
(W₁ ≈ 0.083 across all n) at 2× the speed.

### Hybrid networks

NBN-hybrid handles mixed continuous-discrete networks across all n_nodes
in our benchmark (n ∈ {10, 50, 100, 500, 1000}). Among the external
libraries, only pyro covers hybrid inference (Importance sampler); pgmpy,
gpytorch, and pomegranate have no applicable hybrid baselines.

## Quick start

```python
import torch
from nbn import NeuralBayesianNetwork, TensorVariableElimination
from nbn.mechanisms import CategoricalTableMechanism

# A → B → C, all categorical with cardinality 4
edges = [("A", "B"), ("B", "C")]
model = NeuralBayesianNetwork(
    edges,
    variables={"A": ("discrete", 4), "B": ("discrete", 4), "C": ("discrete", 4)},
)

# Fit each node's mechanism from data
data = {"A": torch.randint(0, 4, (10_000,)),
        "B": torch.randint(0, 4, (10_000,)),
        "C": torch.randint(0, 4, (10_000,))}
for node in model.dag.topological_order():
    parents = model.dag.parents(node)
    pa = torch.stack([data[p] for p in parents], dim=-1).float() if parents else None
    mech = CategoricalTableMechanism()
    mech.fit_local(data[node], pa, parent_cards=[4] * len(parents))
    model.set_mechanism(node, mech)

# Batched query: P(C | A=a) for 4 evidence rows at once
engine = TensorVariableElimination()
posterior = engine.query_batch(model, ["C"], {"A": torch.tensor([0, 1, 2, 3])})
# posterior shape: (B=4, 1, 4) — a distribution over C for each evidence row
```

For continuous, hybrid, and neural-mechanism examples, see the test suite
under `tests/integration/`.

### Gradients

Parent values and `do=` values are gradient-transparent — a parent computed by
your own `nn.Module` carries autograd back into it:

| path | differentiable |
|---|---|
| `mechanism.log_prob(x, parents)` | yes |
| `model.log_prob(data)`, incl. `per_node=True` | yes |
| `model.sample(n)` / `model.sample(n, do=v)` | yes, w.r.t. parameters **and** `v` |
| `model.query` / `model.query_batch` | **no** — VE detaches at factor build, LW runs under `torch.inference_mode()` |
| `model.intervene(do=...)` | **no** — returns a `deepcopy`, so its parameters are fresh leaves |

The last two are deliberate. **When you need gradients through an
intervention, use `model.sample(n, do=...)`**, which applies it against the
live parameters; `intervene()` is for building a mutilated model to *query*.
The contract is pinned by `tests/unit/test_parent_gradient_contract.py`.

### Snapshotting parameters

To take an optimisation step and be able to reject it (backtracking an M-step
that decreased the objective, say), use torch's `state_dict` / `load_state_dict`
— but **copy the snapshot**:

```python
snap = copy.deepcopy(mech.state_dict())   # NOT mech.state_dict()
...                                       # optimiser step
mech.load_state_dict(snap)                # reverts exactly
```

`state_dict()` returns tensors sharing storage with the live parameters, and
optimisers update in place, so an uncopied snapshot is mutated by the step it
is meant to undo — silently, with nothing raising. Pinned by
`tests/unit/test_parameter_snapshot_contract.py`.

## Repository layout

    nbn/                Library code (mechanisms, inference, sampling, core).
    nbn/bench/          Benchmark suite: runner, baseline adapters, configs, data.
    notebooks/          Colab-ready notebooks for the six paper-scale benchmarks.
    scripts/            run_all_benchmarks.sh and other operational helpers.
    results/            Where nbn-bench writes runs (gitignored).
    tests/              Unit + integration tests.
    RESEARCH.md         Paper outline and contribution claims.

## Reproducibility

The headline numbers above are anchored at tag `v0.6c-d`
(commit `2e0dd32`):

```bash
git checkout v0.6c-d          # the suite lived at benchmarking/ at this tag
nbn-bench inference \
  --config benchmarking/configs/inference_paper_laptop.yaml
nbn-bench param-learning \
  --config benchmarking/configs/parameter_learning_paper_laptop.yaml
```

- **Hardware**: NVIDIA GeForce RTX 4070 Laptop (8 GB VRAM)
- **PyTorch**: 2.11.0+cu130
- **Wall time**: 11.4 h inference + 6.5 h parameter-learning
- **Paper data anchor**: see [`docs/v0.6c-d/run_summary.md`](docs/v0.6c-d/run_summary.md) for full headline tables and [`docs/v0.6c-d/dnf_cells.md`](docs/v0.6c-d/dnf_cells.md) for the DNF table

Numerical values vary within MC noise across hardware; STATUS counts
and qualitative findings (cluster, speedup, quality gap) are stable.
The committed parquets, tables, and figures under
`results/{raw,tables,figures}/` are the canonical paper
artefacts.

## Crash tests

NBN ships two crash tests on synthetic Bayesian networks with **known
ground truth**, sweeping network size on the x-axis:

1. **Parameter-learning crash test** — measures accuracy of fitted CPDs
   against the true generative process. Speed is not measured.
2. **Inference crash test** — measures both accuracy and total time for
   `Q` conditional queries. NBN uses `query_batch(B=Q)` (one batched
   call); other libraries loop over the same `Q` queries in Python.

Each crash test has a smoke config (CI, < 60s) and a paper config
(local reproduction, ~17.9 h on RTX 4070 Laptop 8 GB; CPU not supported for paper-config).

### Reproduce

```bash
# Smoke (runs in CI):
nbn-bench param-learning --config nbn/bench/configs/synthetic/smoke_tests/parameter_learning_smoke.yaml
nbn-bench inference      --config nbn/bench/configs/synthetic/smoke_tests/inference_smoke.yaml

# Paper (8 GB VRAM, the laptop variant used for v0.6c-d paper data):
nbn-bench param-learning --config nbn/bench/configs/synthetic/complete/parameter_learning_complete_laptop.yaml
nbn-bench inference      --config nbn/bench/configs/synthetic/complete/inference_complete_laptop.yaml

# Paper (≥16 GB VRAM, canonical config without batch reductions):
nbn-bench param-learning --config nbn/bench/configs/synthetic/complete/parameter_learning_complete.yaml
nbn-bench inference      --config nbn/bench/configs/synthetic/complete/inference_complete.yaml
```

Each invocation writes one run directory under `results/`; the parquet is
the single canonical artefact of a run (figures and tables are never
generated automatically):

    results/benchmark_<benchmark>_<config_name>_<YYYYMMDD_HHMMSS>/
        <config_name>_metrics.parquet     one row per (cell, metric)
        metrics.jsonl                     the same rows, streamed while running
        run.log                           per-cell log (fit/query phases, errors)

### Plot the results

`nbn-bench plot` turns one (or more) run directories into paper figures and
LaTeX tables. The same command serves every benchmark; the x grid is decided
by what the parquet contains (`batch_size` sweep, `n_train` sweep, bnlearn
network, else `n_nodes`), so you never pick a plotter:

```bash
nbn-bench plot results/benchmark_synthetic_learning_curves_20260908_092714 \
  --output-dir results/figures/learning_curves
# options: --aggregation iqm_iqr|mean_std (default iqm_iqr)
#          --benchmark synthetic|bnlearn  (default: every benchmark in the parquet)
#          --top-nbn N                    (nbn methods shown per x value, default 2)
```

Every figure is a **grouped bar plot** (one group per x value, one bar per
method, error bar = aggregation band across seeds) and every figure/table
comes in two views:

- `all/` — each method aggregated over the seeds **it** solved; a bar or
  table cell whose method solved fewer than all seeds carries `k/n`; a method
  that solved none shows its failure code (`timeout`, `oom`, `error`).
- `common/` — each method aggregated over the seeds solved by **every shown
  method** at that x, so the bars in a group are computed on the same
  problems; the table footer and the x label report `|C|`, the common-seed
  count.

Shown methods at each x are every non-nbn baseline applicable to the family
plus the `--top-nbn` best nbn methods (ranked on the `all` view and kept
identical in `common`; nbn rows carry a dagger in the tables). Output tree, per family:

```
<output-dir>/<benchmark>/<family>/
  all/plots/<metric>_vs_<x>.pdf     all/tables/<metric>_vs_<x>.tex
  all/plots/success_rate.pdf        (status breakdown, diagnostic)
  common/plots/<metric>_vs_<x>.pdf  common/tables/<metric>_vs_<x>.tex
  common/common_seeds.txt           the common seeds per metric and x
  selection.txt                     the nbn methods shown per (view, metric, x)
```

The run-directory name uses the config's `config_name`
(`complete`, `scalability_complete`, `batch_speed`, `param_learning_complete`,
`learning_curves`, `bnlearn_complete`), not the YAML file name. Per benchmark
(`<metric>` below is each accuracy metric present plus the timing ones):

| Benchmark (config) | Run with | x grid | Metrics rendered |
|---|---|---|---|
| Synthetic inference (`synthetic/complete/inference_complete.yaml`) | `nbn-bench inference --config …` | `n_nodes` | `tv_per_node`, `jsd_per_node` (discrete), `w1_per_node` (continuous), `total_query_time`, `fit_time` |
| Inference scalability (`synthetic/complete/inference_scalability_complete.yaml`) | `nbn-bench inference --config …` | `n_nodes` | same; the time figures are the headline |
| Inference speed / batching (`synthetic/speed/inference_speed.yaml`, a `batch_sizes` sweep) | `nbn-bench inference --config …` | `batch_size` | `query_time` (per-query time); non-batchable baselines only have a `B=1` bar and read `--` beyond |
| Parameter learning (`synthetic/complete/parameter_learning_complete.yaml`) | `nbn-bench param-learning --config …` | `n_nodes` | `log_likelihood`, `param_recovery_{tv,kl}` (discrete), `calibration_{pit_ks,sd_ratio}` (continuous), `fit_time` |
| Learning curves / sample efficiency (`synthetic/learning_curves/learning_curves.yaml`, an `n_train_sweep`) | `nbn-bench param-learning --config …` | `n_train` | same as parameter learning |
| bnlearn inference (`bnlearn/complete/inference_complete.yaml`) | `nbn-bench inference --config …` | `network` (sorted by size; split into `_partK` files beyond 8 networks) | as synthetic inference |
| Calibration vs accuracy divergence (no config: combine two runs) | one `param-learning` run + one `inference` run on the same families | — | `all/plots/divergence_calibration_pit_ks_vs_w1_per_node.pdf` per continuous family (rows are concatenated; engine suffixes such as `-lw` are stripped to align `nbn-mdn-lw` with `nbn-mdn`) |

Two things that bite:

- `learning_curves.yaml` and `parameter_learning_complete.yaml` declare
  `metrics: log_likelihood` and their baselines carry no `inference_method`,
  so they **must** run under `param-learning`. Under `inference` the loader
  refuses them and prints the command to use.
- A figure is only written when at least one seed was solved for its metric
  in that family (a seed with any timed-out query counts as unsolved for that
  cell; an `ok` row with a NaN value counts as unsolved too). If a plot you
  expect is missing, `nbn-bench plot -v` logs `skip ...` with the reason, and
  `run.log` in the run directory has the per-cell error.

The aggregation contract lives in
[`docs/v0.18-bar-reporting-all-common.md`](docs/v0.18-bar-reporting-all-common.md).

## Configuration

Each config is a YAML file with these fields:

    mode:                 'parameter_learning' | 'inference'
    families:             list of families ∈ {discrete, continuous_lg,
                          continuous_nongauss, hybrid}
    n_nodes:              list of network sizes
    n_seeds:              number of seeds per cell (mean ± std reported)
    n_queries_per_cell:   number of queries per cell
    nbn_batch_size:       B for NBN's query_batch (inference mode only)
    baselines:            list of baseline spec dicts, each with required
                          fields {library, mechanism, param_method} plus
                          optional inference_method and device (cpu|cuda|auto)
    per_cell_timeout_s:   wall-clock cap per (family, n_nodes, seed, baseline)

See `nbn/bench/configs/**/*.yaml` for all shipped configs.

## Status

Current release: **v0.6c-d** (paper-data anchor). The library is in
publishable empirical state.

| Component | Status |
| --- | --- |
| Core (DAG, Variables, Factor) | ✅ |
| Mechanisms (Categorical, NeuralCategorical, LG, MDN, Flow, GP, Hybrid) | ✅ |
| Tensor VE + LW + HybridRouter | ✅ |
| Vectorised batched `query_batch` | ✅ |
| Synthetic crash-test framework | ✅ |
| Method-keyed baseline registry | ✅ v0.6c-C |
| Aggregator + tables (CSV/MD/parquet/TEX) | ✅ v0.6c-C-3 |
| Paper-grade figures + paper-data anchor | ✅ v0.6c-d |
| Multi-library baselines (pgmpy, gpytorch, pomegranate, pyro) | ✅ |
| Per-baseline YAML device override | ✅ v0.12 |
| README enrichment | ✅ v0.6d |

Active backlog (v0.7, none paper-blocking):

- Plotter polish: W₁ band lower-clip (#42), parameter-learning accuracy panels for non-discrete families (#44)
- Adapter audits: pgmpy-mle vs pgmpy-bayes / nbn-cat vs nbn-neuralcat fit-path distinctness (#43)
- HybridRouter cuda assert at hybrid n ≥ 10 (#30)
- NeuralCategorical-VE engine refactor (#26)
- pyro inference speedup (v0.8 candidate) — current Importance sampler is Python-bound and CPU-only; GPU is 11× slower at benchmark scale, so speedup requires `pyro.plate` vectorisation or alternative inference modes (SVI, NUTS). See `docs/audits/v0.12-pyro-gpu-investigation.md`.

See the [open issues](https://github.com/Giovannibriglia/NeuralBayesianNetworks/issues) for the full v0.7 backlog.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
