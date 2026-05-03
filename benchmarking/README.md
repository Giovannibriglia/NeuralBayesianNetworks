# NBN Benchmarking Suite

This directory holds the **plugin-based benchmark runner** for NBN: domains,
baseline adapters, query battery, metrics, plotter, and YAML configs.

It lives outside the main `nbn/` package so you can find, run, and extend
experiments without digging into the library code.

---

## TL;DR — reproduce everything

From the **repository root**:

```bash
# 1. One-shot install (CPU-only is fine; CUDA picked up automatically)
pip install --prefer-binary -e ".[dev,bench,neural,gp,mcmc]"

# 2. Headline crash test (alarm + synthetic_50, all baselines, ~30s on CPU)
python examples/crash_test.py

# 3. Focused benchmark suites (writes parquet under results/)
nbn-bench run benchmarking/configs/discrete_small.yaml
nbn-bench run benchmarking/configs/continuous_small.yaml
nbn-bench run benchmarking/configs/scaling.yaml
```

Figures land in `examples/figures/` (crash test) or `results/` (suites).

---

## What gets compared

### Domains

| Domain | File | What it ships |
|---|---|---|
| **bnlearn** | [`domains/bnlearn.py`](domains/bnlearn.py) | 14 canonical discrete BNs (asia, cancer, alarm, child, insurance, hailfinder, …). Ground-truth marginals from pgmpy's exact VE. |
| **synthetic_hybrid** | [`domains/synthetic_hybrid.py`](domains/synthetic_hybrid.py) | Random DAGs of size n ∈ {10, 50, 200, 1000} with mixed discrete + non-Gaussian continuous CPDs (skew-normal mixtures). Empirical ground truth via samples from the true SCM. |

### Baselines

#### NBN variants (every public engine × mechanism family)

| Variant | Engine | Mechanisms | Notes |
|---|---|---|---|
| **`nbn`** / **`nbn_hybrid`** | `HybridRouter` | `CategoricalTable` + `MDN` | Default — auto-picks VE on small treewidth, LW elsewhere |
| **`nbn_ve`** | `TensorVariableElimination` | `CategoricalTable` + `MDN` | Exact log-domain einsum VE; discrete-network workhorse |
| **`nbn_lw`** | `LikelihoodWeightingEngine` | `CategoricalTable` + `MDN` | Batched ancestral importance sampling; hybrid-friendly |
| **`nbn_neural_categorical`** | `HybridRouter` | `NeuralCategorical` + `MDN` | MLP + embedding categorical CPDs (large parent spaces) |
| **`nbn_linear_gaussian`** | `HybridRouter` | `CategoricalTable` + `LinearGaussian` | Closed-form ridge regression for continuous CPDs |

All NBN variants share a single configurable adapter
([`baselines/nbn_adapter.py`](baselines/nbn_adapter.py)) — the table above
just enumerates the registered presets. To create your own combination:

```python
from benchmarking.baselines.nbn_adapter import NBNAdapter
NBNAdapter(
    device="cuda",
    engine="lw",                      # or "ve" / "hybrid"
    discrete_mech="neural_categorical",
    continuous_mech="linear_gaussian",
    n_samples=4096,                   # for engine='lw'
)
```

#### External baselines

| Baseline | File | Supports | Inference method |
|---|---|---|---|
| **`pgmpy`** | [`baselines/pgmpy_adapter.py`](baselines/pgmpy_adapter.py) | discrete | Exact `VariableElimination` |
| **`pomegranate`** | [`baselines/pomegranate_adapter.py`](baselines/pomegranate_adapter.py) | discrete | torch-backed v1.x BN inference (`predict_proba` with NaN sentinel for unobserved nodes) |
| **`pyro`** | [`baselines/pyro_adapter.py`](baselines/pyro_adapter.py) | discrete, continuous, hybrid | `pyro.infer.Importance` over a generative Pyro model with `pyro.poutine.condition` |
| **`gpytorch`** | [`baselines/gpytorch_adapter.py`](baselines/gpytorch_adapter.py) | continuous | Independent SVGP per continuous node (parents as features) |

#### Why is GPyTorch not used on discrete networks?

GPyTorch implements **Gaussian Processes**. Its built-in likelihoods
(`GaussianLikelihood`, `BernoulliLikelihood`, `MultitaskGaussianLikelihood`)
all assume a continuous regression target or a single-bit binary outcome.
There is no first-class way to model a multi-class categorical CPT inside a
GP without a continuous relaxation (e.g. the Polya-Gamma trick), and that
relaxation is itself a research project rather than a fair benchmark
adapter. To prevent meaningless numbers, the `GPyTorchAdapter` declares
`supports = {"continuous"}` and the runner records discrete-evidence
queries as `not_supported` with the reason in the parquet output.

---

## Standard query battery (the 5-kind taxonomy)

`make_query_battery` (in [`queries.py`](queries.py)) emits a fixed, seeded
battery of queries. Every baseline answers identical queries so results are
directly comparable.

| Kind | Description | Default count per problem |
|---|---|---|
| `marginal` | `P(X)` for every node | one per node |
| `conditional` (single) | `P(X | E_j = e)` | 20 (bnlearn) / 10 (hybrid) |
| `conditional` (multi) | `P(X | E = e)`, `|E| ∈ {2, 4, 8}` | 10 each (bnlearn) / 5 each (hybrid) |
| `map` | `argmax_x P(x | E)` | 10 / 5 |
| `do` | `P(X | do(Y = y))` | 10 / 5 |

Each `Query.kind` is recorded in the parquet output, so the plotter can
split metrics per kind.

---

## Metrics ([`metrics.py`](metrics.py))

| Metric | Applies to | Formula |
|---|---|---|
| `kl_divergence` | discrete | `Σ p log(p/q)` with `eps=1e-12` |
| `js_divergence` | discrete | symmetric Jensen-Shannon |
| `tv_distance` | discrete | total-variation `0.5·Σ|p-q|` |
| `mae_marginals` | discrete | mean absolute error |
| `wasserstein_1d` | continuous (1-D) | sorted-CDF empirical Wasserstein-1 |
| `energy_distance` | continuous (n-D) | Monte Carlo `2·E‖X-Y‖ - E‖X-X'‖ - E‖Y-Y'‖` |
| `mmd_rbf` | continuous (n-D) | unbiased MMD² with median-heuristic bandwidth |
| `held_out_nll` | any | `-mean(log_prob(x_test))` |
| `map_accuracy` | MAP queries | exact-match argmax |
| `brier_score` | discrete predictions | `Σ (p_pred - 1[y=k])²` |
| `crps` | continuous predictions | empirical-quantile Continuous Ranked Probability Score |
| `query_latency_ms` | speed | median wall-clock |
| `batched_throughput` | speed | queries / s |
| `gpu_peak_mb` | memory | `torch.cuda.max_memory_allocated()` |

All metrics are torch-native and device-aware.

---

## Adding a new domain or baseline

See [`docs/benchmarks_extending.md`](../docs/benchmarks_extending.md). In
short:

```python
# benchmarking/domains/my_domain.py
from benchmarking.domains import BenchmarkDomain, BenchmarkProblem
from benchmarking.queries import make_query_battery

class MyDomain(BenchmarkDomain):
    name = "my_domain"
    def list_problems(self) -> list[str]: ...
    def load_problem(self, problem, *, n_train, n_test, seed, device):
        return BenchmarkProblem(
            name=problem, dag=..., variables=...,
            train_data=..., test_data=...,
            queries=make_query_battery(...),
            ground_truth=...,
        )
```

Register in `benchmarking/domains/__init__.py::_DOMAIN_REGISTRY`. The runner
will pick it up automatically; every existing baseline + metric works on
your problem family.

---

## YAML config schema

```yaml
domain: bnlearn               # or synthetic_hybrid (or your own plugin)
problems: [asia, cancer]      # subset of domain.list_problems()
n_train: 5000
n_test: 1000
seed: 0
devices: [cpu, cuda]          # cuda auto-skipped if unavailable
baselines: [nbn, pgmpy, pomegranate, pyro]   # see registry above
query_kinds: [marginal, conditional, map, do]
output: results/discrete_small.parquet
```

---

## Reproducing the headline crash-test result

`examples/crash_test.py` produces the four publication-ready figures used
in the paper draft. Output (CPU, no `--smoke`):

```
==== NBN crash test ====
-- Discrete (alarm, 37 nodes), device=cpu, baselines=['nbn', 'pgmpy', 'pomegranate', 'pyro']
-- Hybrid (hybrid_50), device=cpu, baselines=['nbn', 'pyro', 'gpytorch']

==== Results ====
  nbn         cpu   alarm     :   37q in  0.114s ( 3.09 ms/q) | TV=0.0074 | MAP-acc=1.000
  pgmpy       cpu   alarm     :   37q in  0.021s ( 0.57 ms/q) | TV=0.0064 | MAP-acc=1.000
  pomegranate cpu   alarm     :   37q in  ...      ...        | TV=...    | MAP-acc=...
  pyro        cpu   alarm     :   37q in  ...      ...        | TV=...    | MAP-acc=...
  nbn         cpu   hybrid_50 :   50q in  1.224s (24.48 ms/q)
  ...
```

Acceptance gate: NBN on alarm must reach `TV < 0.05` and `MAP-acc > 0.85`,
otherwise the script exits non-zero. CI runs the same script with
`--smoke` (10 queries, no figures) on every push.
