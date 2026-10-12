# Benchmark suite

NBN ships a benchmark suite (`nbn.bench`, command `nbn-bench`, the `bench`
extra) that compares NBN with pgmpy, pomegranate and Pyro on Bayesian
networks with **known ground truth**: synthetic networks of controlled size
and family, and the bnlearn repository networks. It measures inference
accuracy and speed and parameter-learning quality, runs every cell in its own
subprocess, and draws the paper figures.

```bash
pip install -U "nbn[all]" && nbn-bench check-env          # must print "environment OK"
nbn-bench inference --config nbn/bench/configs/synthetic/smoke_tests/inference_smoke.yaml
nbn-bench plot results/benchmark_synthetic_smoke_<timestamp> --output-dir results/figures
```

| Page | Covers |
|---|---|
| {doc}`../BENCHMARKS` | installing, what is measured, running, plotting, the YAML configuration, reproducing the paper numbers |
| {doc}`../SERVER_RUN` | full paper-scale runs on a remote server: monitoring, interruption and recovery, finalisation |
| {doc}`../benchmarks_extending` | adding a problem source, a baseline adapter or a metric |
| {doc}`../v0.13-oracle-methodology` | how the ground-truth posterior every accuracy metric is scored against is computed |

```{toctree}
:hidden:

../BENCHMARKS
../SERVER_RUN
../benchmarks_extending
../v0.13-oracle-methodology
```
