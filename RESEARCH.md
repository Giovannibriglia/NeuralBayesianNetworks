# NBN Research Document

## Problem Statement

Existing exact PGM methods (pgmpy, pyAgrum, JT/VE) are SOTA on small discrete networks but fail to scale beyond a few hundred nodes and cannot represent non-Gaussian continuous variables. GPyTorch is SOTA for continuous Gaussian regression but is not a BN/PGM library and cannot represent discrete variables or DAG-factorized joint distributions. Pyro provides tensor variable elimination but is a universal PPL with a steep learning curve.

**NBN closes the gap** via per-node neural mechanisms over a known DAG, executed by a vectorized PyTorch inference engine that batches thousands of conditional queries on the GPU, scaling to ≥1000-node hybrid (discrete + non-Gaussian continuous) Bayesian networks while remaining exact on tractable sub-graphs and amortized-approximate elsewhere.

## Claimed Contributions

1. A single torch-native API for hybrid (discrete + non-Gaussian continuous) BNs with a known DAG — the first library to unify both regimes under `nn.Module`.
2. A unified `Mechanism` abstraction covering categorical, MDN, normalising-flow, and GP CPDs, all batched and autograd-compatible via `torch.distributions.Distribution`.
3. A log-domain einsum-based tensor VE engine with treewidth-aware fallback to amortized variational inference.
4. Native batched-query interface (`query_batch`) returning thousands of conditional answers in a single GPU launch.
5. The first benchmark covering all 23+ bnlearn networks AND large hybrid non-Gaussian synthetic graphs (≥1000 nodes), with reproducible scripts comparing NBN to pgmpy, pyAgrum, pomegranate, and GPyTorch.

## Ablations for the Paper

- **Mechanism choice on continuous CPDs** (Linear-Gaussian vs MDN vs NSF vs GP): log-likelihood & sample quality on sachs-continuous, UCI-boston, UCI-wine.
- **Inference engine on discrete bnlearn networks** (TensorVE vs JT vs LikelihoodWeighting): accuracy/time Pareto on asia, alarm, child, insurance, hailfinder.
- **Treewidth-threshold τ for the HybridRouter**: accuracy/time Pareto.
- **Batch size in `query_batch`**: throughput scaling curve (queries/sec vs batch size).
- **GPU vs CPU scaling**: wall-clock vs n_nodes for discrete, continuous, hybrid.

## Roadmap to Publication

1. **Library paper** → JMLR MLOSS (target Q3/Q4 2025).
2. **Benchmark paper** → NeurIPS Datasets-&-Benchmarks Track 2025.
3. **Methodology paper** (hybrid mechanisms + treewidth-aware router + amortized fallback with formal guarantees) → UAI / AISTATS 2026.

## Open Questions / Future Work

- **Structure learning**: NBN currently assumes the DAG is given. GFlowNet-based structure learning (Deleu et al. 2022/2023) would allow uncertain-DAG extension.
- **Dynamic/temporal NBN**: DBN-NBN for time-series and multi-step RL settings.
- **Counterfactual sampling**: twin-network construction and abduction-action-prediction under the structural causal model framework.
- **Causality-driven RL plug-in**: NBN as a differentiable world model in model-based RL, connecting to the PhD research program on Causality-Driven RL.
- **Identifiability / do-calculus interface**: formal do-calculus query engine mirroring DoWhy/Pyro.
- **Amortized BI on graphs**: BayesFlow-style amortized inference for large hybrid networks.
