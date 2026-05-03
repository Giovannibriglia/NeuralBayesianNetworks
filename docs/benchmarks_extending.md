# Extending NBN's benchmarking suite

The benchmark runner is plugin-based.  To add a new test domain, subclass
`benchmarking.domains.base.BenchmarkDomain`, register it in
`benchmarking/domains/__init__.py`, and you're done — every existing
baseline + metric works on your new problem family.

## Domain plugin contract

```python
from benchmarking.domains import BenchmarkDomain, BenchmarkProblem
from benchmarking.queries import make_query_battery

class MyDomain(BenchmarkDomain):
    name = "my_domain"

    def list_problems(self) -> list[str]:
        return ["small", "medium", "large"]

    def load_problem(self, problem, *, n_train, n_test, seed, device):
        # 1. Build / fetch the DAG, variable specs, and data tensors.
        # 2. Generate a query battery (use `make_query_battery` for the standard taxonomy).
        # 3. Optionally provide ground truth for accuracy metrics.
        return BenchmarkProblem(
            name=problem, dag=..., variables=...,
            train_data=..., test_data=...,
            queries=make_query_battery(...),
            ground_truth=...,
        )
```

Then add it to `_DOMAIN_REGISTRY` in `benchmarking/domains/__init__.py`.

## Standard query battery

`make_query_battery(...)` produces five kinds of queries with a fixed seed:

1. **Univariate marginals** — `P(X)` for every node.
2. **Conditional marginals (single)** — `P(X | E_j = e)`.
3. **Conditional marginals (multi)** — `P(X | E = e)` with `|E| ∈ {2, 4, 8}`.
4. **MAP** — `argmax_x P(x | E)`.
5. **`do`** — interventional `P(X | do(Y = y))`.

All baselines answer identical queries.  Each `Query.kind` is recorded, so the
plotter can split metrics per kind.

## Adding a baseline

Subclass `benchmarking.baselines.base.BaselineAdapter` and register it in
`benchmarking/baselines/__init__.py`.  Adapters declare which query kinds
they support; unsupported queries are skipped (e.g. GPyTorch can't do discrete
evidence — it raises `NotImplementedError` and the runner records the skip).

## Built-in metrics

See `benchmarking/metrics.py` — KL / JS / TV on discrete marginals,
Wasserstein-1 / energy distance / MMD-RBF on continuous, MAP accuracy /
Brier / CRPS on predictions, plus latency, throughput, GPU peak memory.

## YAML config

```yaml
domain: my_domain
problems: [small, medium]
n_train: 5000
n_test: 1000
seed: 0
devices: [cpu, cuda]
baselines: [nbn, pgmpy]
query_kinds: [marginal, conditional, map, do]
output: results/my_run.parquet
```

Run with `nbn-bench run my_config.yaml`.
