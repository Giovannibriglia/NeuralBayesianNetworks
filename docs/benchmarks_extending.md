# Extending NBN's benchmarking suite (v0.13)

The v0.13 benchmark runner uses a composition-based architecture.
To add a new test domain, baseline, or measurement, subclass the
appropriate protocol interface and register the new component.

Reference: docs/v0.13-benchmark-redesign.md

## Adding a new problem source

Implement `benchmarking.core.ProblemSource` and add it to the dispatch
table in `benchmarking/core/yaml_config.py`:

```python
from benchmarking.core.interfaces import ProblemSource
from benchmarking.domains.base import BenchmarkProblem

class MyProblemSource:
    def iter_problems(self, config) -> Iterable[BenchmarkProblem]:
        for problem_id in config.problem_ids:
            yield self._load(problem_id, config)
```

Then add a `benchmark="my_domain"` branch to `load_runner_config` in
`benchmarking/core/yaml_config.py`.

## Adding a new baseline adapter

Implement `benchmarking.core.BaselineAdapter` and register the library
in `benchmarking/adapters/__init__.py` and `build_adapter` in
`benchmarking/core/config.py`:

```python
from benchmarking.core.interfaces import BaselineAdapter
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.domains.posterior import Posterior

class MyAdapter:
    name = "my-lib-mle-lw"   # canonical label key

    def fit(self, problem: BenchmarkProblem, **kwargs) -> None:
        ...   # train on problem.train_data

    def query(self, q: Query) -> Posterior:
        ...   # return Posterior(samples=...) or Posterior(probs=...)

    def is_applicable(self, problem: BenchmarkProblem) -> bool:
        return problem.family in {"discrete", "continuous_lg"}
```

Add the label to `benchmarking/core/applicability.py`:

```python
BASELINE_FAMILY_APPLICABILITY["my-lib-mle-lw"] = BaselineApplicability(
    families=frozenset({"discrete", "continuous_lg"}),
)
```

Then wire it in `build_adapter`:

```python
if lib == "my_lib":
    from benchmarking.adapters.my_adapter import MyAdapter
    return MyAdapter(**kw)
```

## Standard v0.13 YAML schema

```yaml
version: v0.13
benchmark: synthetic
config_name: my_run
metrics: all           # all (AccuracyAndTiming) | timing (TimingOnly)
selector: uniform_random

source:
  families: [discrete, continuous_lg]
  n_nodes_list: [10, 50, 100]
  seeds: [0, 1, 2]
  n_train: 5000
  n_test: 1000
  n_reference: 10000
  edge_density: 0.20
  max_in_degree: 4
  cardinality: 4
  fraction_continuous: 0.5

baselines:
  - {library: nbn, mechanism: cat, param_method: mle, inference_method: ve}
  - {library: pgmpy, mechanism: discrete, param_method: mle, inference_method: ve}
  - {library: nbn, mechanism: mdn, param_method: mle, inference_method: lw,
     extra_kwargs: {n_samples: 1024}}
  - {library: pyro, mechanism: empirical, param_method: mle,
     inference_method: importance, device: cpu}

n_queries_per_cell: 16
per_cell_timeout_s: 120.0
fit_timeout_s_multiplier: 10.0
```

Run with:
```bash
nbn-bench inference --config my_config.yaml [--device cpu|cuda]
```

## Built-in metrics

`AccuracyAndTiming` (metrics: "all") emits per-query:
- **Discrete targets**: TV, JSD accuracy + `total_time_s`
- **Continuous targets**: W1 accuracy + `total_time_s`

`TimingOnly` (metrics: "timing") emits:
- `total_time_s` only — no oracle calls, much faster

See `benchmarking/measurements/` for the implementation.
