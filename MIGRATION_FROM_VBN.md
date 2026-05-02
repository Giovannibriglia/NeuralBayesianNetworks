# Migration from VectorizedBayesianNetwork (VBN) to NeuralBayesianNetworks (NBN)

## 1. Inventory

| File | LOC | Purpose |
|------|-----|---------|
| `vbn/vbn.py` | 824 | Main `VBN` class |
| `vbn/core/base.py` | 108 | `BaseCPD`, `Query`, `CPDOutput` ABCs |
| `vbn/core/dags.py` | 55 | `StaticDAG` wrapping `networkx.DiGraph` |
| `vbn/core/utils.py` | 128 | Shape helpers, device resolution |
| `vbn/core/registry.py` | ~60 | Registry dicts + decorators |
| `vbn/core/cpd_handle.py` | 428 | Property-accessor façade over fitted CPDs |
| `vbn/cpds/categorical_table.py` | 417 | Tabular categorical CPD (count-based MLE) |
| `vbn/cpds/linear_gaussian.py` | 217 | Linear Gaussian (closed-form ridge) |
| `vbn/cpds/mdn.py` | 272 | Mixture Density Network |
| `vbn/cpds/gaussian_nn.py` | 288 | MLP Gaussian CPD |
| `vbn/cpds/softmax_nn.py` | 788 | MLP softmax categorical CPD |
| `vbn/cpds/categorical_embedded_softmax.py` | 511 | Embedding + softmax categorical |
| `vbn/cpds/rff_gaussian.py` | 291 | Random Fourier Feature Gaussian |
| `vbn/cpds/kde.py` | 182 | Kernel Density Estimation CPD |
| `vbn/inference/likelihood_weighting.py` | 82 | Batched ancestral IS |
| `vbn/inference/categorical_exact.py` | 128 | Exact lookup when parents observed |
| `vbn/inference/gaussian_exact.py` | 183 | Closed-form Gaussian exact |
| `vbn/inference/rao_blackwellized_marginalization.py` | 324 | Rao-Blackwellized marginalization |
| `vbn/inference/monte_carlo_marginalization.py` | 92 | Monte Carlo marginalization |
| `vbn/inference/importance_sampling.py` | 93 | Plain IS |
| `vbn/inference/resampled_importance_sampling.py` | 105 | SIR resampled IS |
| `vbn/inference/lbp.py` | ~80 | Loopy Belief Propagation (draft) |
| `vbn/learning/node_wise.py` | 191 | Per-node sequential MLE |
| `vbn/learning/amortized.py` | ~60 | Amortized learning stub |
| `vbn/sampling/ancestral.py` | ~80 | Batched ancestral sampling |
| `vbn/sampling/gibbs.py` | 92 | Gibbs sampler |
| `vbn/sampling/hmc.py` | 141 | HMC sampler |
| `vbn/update/` | ~300 | EMA, online SGD, replay buffer, streaming stats |
| `vbn/benchmarking/` | ~600 | bnlearn download, query gen, model wrappers |
| **Total** | **~7 431** | |

Public symbols: `VBN`, `BaseCPD`, `CPDOutput`, `Query`, `StaticDAG`, all CPD/inference/learning/sampling classes, registries, `defaults`.

---

## 2. Core abstractions found

| Concept | VBN |
|---------|-----|
| DAG | `StaticDAG(networkx.DiGraph)` — validates acyclicity, caches topo order + parents dict |
| CPD | `BaseCPD(nn.Module)` with `input_dim`, `output_dim`, `device` — abstract `sample(parents, n_samples)`, `log_prob(x, parents)`, `fit(parents, x)`, `update(parents, x)` |
| Network | `VBN` — **not** `nn.Module`; holds `dag`, `nodes: Dict[str, BaseCPD]`; methods set separately via `set_learning_method`, `set_inference_method`, `set_sampling_method` |
| Query | `Query(target: str, evidence: Dict[str, Tensor], do: Dict[str, Tensor])` |
| fit | `vbn.fit(data)` → delegates to `_learning.fit(vbn, tensor_data)` → returns `Dict[str, BaseCPD]` stored in `vbn.nodes` |
| infer | `vbn.infer_posterior(query)` → `(pdf [B,S], samples [B,S,D])` — always detach |
| sample | `vbn.sample(query, n_samples)` → `Dict[str, Tensor]` or `Tensor` |

Tensor convention throughout: `[B, S, D]` where B=batch, S=particles, D=output dim.

---

## 3. Parameter-learning approach

- **Categorical table**: vectorized count accumulation + Dirichlet(α) Laplace smoothing; pure torch; closed-form.
- **Linear Gaussian**: closed-form ridge regression via `torch.linalg.lstsq`; per-node.
- **MDN / GaussianNN / SoftmaxNN**: mini-batch Adam on negative log-likelihood; training loop inside each CPD.
- **NodeWiseLearner**: iterates topo order, instantiates each CPD, calls `cpd.fit(parent_tensor, x)`. No shared optimizer.
- **AmortizedLearner**: stub — not implemented.
- **Missing-data handling**: not implemented.

---

## 4. Inference approach

| Engine | Mechanism |
|--------|-----------|
| `likelihood_weighting` | Ancestral sampling; evidence nodes scored by log_prob; softmax normalization of weights; returns `[B, S]` weights and `[B, S, D]` target samples |
| `categorical_exact` | Direct CPT lookup when all parents observed; falls back to LW |
| `gaussian_exact` | Closed-form Gaussian when parents observed; falls back to LW |
| `rao_blackwellized_marginalization` | Samples latent ancestors, analytically marginalizes target distribution (Gaussian or Categorical); falls back to LW |
| `monte_carlo_marginalization` | Samples all latents, collects empirical target distribution |
| `importance_sampling` | Proposal = prior; IS weights by evidence likelihood |
| `resampled_importance_sampling` | SIR: IS + systematic resampling |
| `lbp` | Loopy BP — draft, not fully functional |

**No tensor variable elimination (einsum-based VE).** All methods are sampling-based or direct-lookup. The absence of log-domain einsum-VE is the primary inference gap.

---

## 5. Benchmark setup

Located in `benchmarking/`. Pipeline:
1. `01_download_data.py` — downloads `.bif`/`.net` files from bnlearn repo.
2. `02_generate_benchmark_queries.py` — generates fixed query sets per network.
3. `03_generate_data.py` — samples training data from ground-truth nets.
4. `04_run_benchmark.py` — runs VBN, pgmpy, numpyro, pyro, gpytorch wrappers.
5. `05_report_results.py` — aggregates results, produces plots.

Baseline wrappers: `pgmpy`, `numpyro`, `pyro`, `gpytorch` (all in `benchmarking/models/`).
Metrics: KL divergence on marginals, wall-clock, GPU mem (in `benchmarking/metrics/`).

---

## 6. Strengths to keep

- **`[B, S, D]` tensor convention** — clean, GPU-friendly, consistent. NBN adopts it as `[B, N_samples, D]`.
- **`StaticDAG` pattern** — validates acyclicity on construction, caches topo order and parents. NBN's `DAG` wraps this.
- **`InferenceState` caching** (`vbn/inference/_core.py`) — hashes query signature, pre-computes slices, parent indices, evidence masks. NBN replicates and extends this.
- **`CategoricalTableCPD`** — vectorized stride-based multi-parent CPT indexing; Dirichlet smoothing; correct `[B,S,D]` output. NBN's `CategoricalTableMechanism` is a direct clean-room rewrite.
- **`LinearGaussianCPD`** — closed-form ridge regression via `torch.linalg.lstsq`; exact MLE. Rewritten cleanly in NBN.
- **`MDNCPD`** — proper log-sum-exp stable mixture log-prob; reparameterized sampling. Rewritten in NBN.
- **`LikelihoodWeighting` + `RaoBlackwellizedMarginalization`** — solid IS implementations. NBN's `LikelihoodWeightingEngine` generalizes these.
- **NodeWiseLearner** — parallel-ready per-node fitting. NBN's `NodeWiseFitter` improves with true parallelism.
- **Benchmarking pipeline** — bnlearn download + query generation scripts are salvageable. NBN's `benchmarks/` reuses the bnlearn network URLs and query format.
- **Query dataclass** — `Query(target, evidence, do)` maps cleanly to NBN's `NBNQuery`.

---

## 7. Weaknesses / bugs to discard

- **`VBN` is not `nn.Module`** (`vbn/vbn.py:184`). Cannot use `.to(device)`, `.parameters()`, `torch.compile`, AMP, or gradient checkpointing at the network level. **NBN's `NeuralBayesianNetwork` is `nn.Module`.**
- **No tensor variable elimination** — all inference is sampling-based. Sampling-based inference is fundamentally inexact for discrete networks and cannot match pgmpy's `VariableElimination` accuracy at the same computational cost. **NBN adds `TensorVariableElimination` with `opt_einsum` and log-einsum-exp.**
- **No batched-query API** — `infer_posterior` takes a single `Query` with a scalar target. Answering 10 000 queries requires a Python loop. **NBN adds `query_batch` returning `[Q, K]` in one GPU launch.**
- **No `torch.distributions.Distribution` return** — CPDs return raw tensors; inference engines return `(pdf, samples)` pairs always detached. No autograd through inference. **NBN's `Mechanism.forward()` returns a `Distribution`; inference engines are differentiable where the mechanism permits.**
- **No normalizing flows** — only Gaussian-family continuous CPDs. **NBN adds `NormalizingFlowMechanism` and `ConditionalFlowMechanism`.**
- **`set_learning_method` / `set_inference_method` boilerplate** (`vbn/vbn.py:210-335`) — 125 lines of repeated dispatch code for 4 methods; brittle registry string lookup. **NBN uses direct constructor injection.**
- **Registry + YAML config indirection** (`vbn/configs/`, `vbn/defaults.py`, `vbn/config_cast.py`) — adds 300 LOC of glue without adding expressiveness. **NBN removes YAML-driven CPD configs; users pass Python objects.**
- **LBP inference is a draft** (`vbn/inference/lbp.py`) — passes on complex graphs; incomplete message-passing schedule. Discarded; NBN schedules loopy BP correctly or uses the hybrid router.
- **Amortized learning is a stub** (`vbn/learning/amortized.py`) — `NotImplementedError`. Discarded and reimplemented.
- **`cpd_handle.py`** (428 LOC) — a property-accessor façade that duplicates CPD state; brittle type detection via `hasattr`. Discarded; NBN exposes CPDs directly via `model.mechanisms`.

---

## 8. Salvageable utilities

| VBN path | What | NBN destination |
|----------|------|-----------------|
| `vbn/core/dags.py` | Acyclicity check, topo cache | `nbn/core/dag.py` (clean-room rewrite) |
| `vbn/inference/_core.py` | `InferenceState`, `get_inference_state`, `prepare_fixed_values` | `nbn/inference/state.py` |
| `vbn/core/utils.py` | `ensure_2d`, `flatten_samples`, `broadcast_samples` | `nbn/utils/batching.py` |
| `benchmarking/I_data_download/download_bnlearn.py` | bnlearn URL map + download logic | `nbn/benchmarks/bnlearn_loader.py` |
| `benchmarking/metadata/bnlearn.json` | Network metadata (node counts, edge counts) | `nbn/benchmarks/bnlearn_meta.json` |
| `benchmarking/metrics/divergences.py` | KL/JS on discrete marginals | `nbn/benchmarks/metrics.py` |
| `benchmarking/models/pgmpy.py` | pgmpy wrapper for baselines | `nbn/benchmarks/baselines.py` |

---

## 9. Identifier rename table

| VBN symbol | NBN symbol |
|-----------|-----------|
| `VBN` | `NeuralBayesianNetwork` |
| `BaseCPD` | `Mechanism` |
| `CPDOutput` | removed (use `torch.distributions.Distribution`) |
| `BaseLearning` | `LearningStrategy` |
| `BaseInference` | `InferenceEngine` |
| `BaseSampling` | `SamplingStrategy` |
| `StaticDAG` | `DAG` |
| `Query` | `NBNQuery` |
| `register_cpd` | `register_mechanism` |
| `register_inference` | `register_engine` |
| `CPD_REGISTRY` | `MECHANISM_REGISTRY` |
| `INFERENCE_REGISTRY` | `ENGINE_REGISTRY` |
| `LEARNING_REGISTRY` | `LEARNER_REGISTRY` |
| `node_wise` (learner) | `node_wise` |
| `categorical_table` | `categorical_table` |
| `linear_gaussian` | `linear_gaussian` |
| `mdn` | `mdn` |
| `likelihood_weighting` | `likelihood_weighting` |
| `categorical_exact` | removed (absorbed into `TensorVariableElimination`) |

sed command for files: `find . -name "*.py" | xargs sed -i 's/\bvbn\b/nbn/g; s/\bVBN\b/NBN/g'`

---

## 10. API translation

| VBN call | NBN equivalent |
|----------|----------------|
| `vbn = VBN(dag)` | `model = NeuralBayesianNetwork(dag, variables={...})` |
| `vbn.set_learning_method("node_wise", nodes_cpds={...})` | `model.set_mechanism("X", CategoricalTableMechanism())` |
| `vbn.fit(data)` | `model.fit(data)` |
| `vbn.infer_posterior({"target": "X", "evidence": {...}})` | `model.query(["X"], evidence={...})` |
| `vbn.sample({"target": "X"}, n_samples=1000)` | `model.sample(n=1000)` |
| `vbn.to_device("cuda")` | `model.to("cuda")` (standard `nn.Module`) |
| `vbn.save("ckpt")` | `model.save("ckpt.pt")` |
| `VBN.load("ckpt")` | `NeuralBayesianNetwork.load("ckpt.pt")` |
| batched queries (Python loop) | `model.query_batch(["X"], evidence={"A": tensor([0,1,...])})` |
