# v0.6c-A round-1 findings

Two diagnostics, both reproduced on this cpu sandbox. Each routes to a hypothesis with concrete evidence; neither needs additional instrumentation before round 2.

---

## Finding 1 — `nbn_lw` continuous_lg W₁ ≈ 0.83 at paper config

The reviewer brief routed three hypotheses (A: importance-resampling collapse over flattened B×n_samples weights; B: forward-clamp oracle B-scaling; C: LW engine itself degrading). The empirical sweep refutes A/B/C and routes to **Other — config drift**.

### Configs disagree on the per-query LW sample budget

```
benchmarking/configs/inference_smoke.yaml :  nbn_lw_n_samples: 4000
benchmarking/configs/inference_paper.yaml :  (field omitted)
benchmarking/_crash_test_utils.py        :  default = 512
```

Smoke explicitly sets `nbn_lw_n_samples: 4000`; paper YAML omits the field, so `CrashTestConfig` uses its default of **512**. That's an 8× sample-budget reduction; for an importance-sampling estimator with effective sample size that drops sharply on multi-evidence continuous queries, an 8× budget cut produces a much-greater-than-8× variance increase.

`_compute_inference_accuracy` caps the inner loop at `min(B, 16)` queries — so even at `B=1024` the runner only iterates 16 (target, evidence) rows. `_baseline_posterior_for_query` is called per row with single-row evidence (`{k: v[i] for k, v in q.evidence.items()}`); the `n_lw_samples` budget applies to each row independently. **B does not enter the per-query sample budget at all.**

### Sweep 1 — vary B at fixed `n_lw_samples=4000` (the smoke setting)

| B    | n_unique | pred_std | oracle_std | W₁ |
|-----:|---------:|---------:|-----------:|------:|
| 1    | 1555     | 0.8396   | 0.8127     | 0.040 |
| 4    | 1572     | 0.8246   | 0.8441     | 0.050 |
| 16   | 1552     | 0.8245   | 0.8215     | 0.051 |
| 64   | 1580     | 0.8371   | 0.8096     | 0.028 |
| 256  | 1578     | 0.8312   | 0.8281     | 0.032 |
| 1024 | 1583     | 0.8244   | 0.8111     | 0.029 |

W₁ stays in `[0.028, 0.051]` — a **1.8× range** which is just MC noise. **Hypothesis A refuted**: B has no detectable effect on the per-query posterior fidelity.

### Sweep 2 — vary `n_lw_samples` at fixed `B=16` (the smoke setting)

| n_lw_samples | n_unique | pred_std | W₁ |
|------------:|---------:|---------:|------:|
| 128         |  74      | 0.8184   | 1.698 |
| 256         |  148     | 0.8543   | 1.267 |
| 512         |  314     | 0.8150   | 0.962 |
| 1024        |  623     | 0.8277   | 0.725 |
| 2000        | 1254     | 0.8629   | 0.044 |
| 4000        | 1581     | 0.8228   | 0.038 |
| 8000        | 1797     | 0.8220   | 0.038 |

W₁ swings from 1.70 (at 128 samples) down to 0.038 (at 4000+ samples) — a **44.7× range**. The phase transition is between 1024 and 2000 samples; at 2000 W₁ already settles to its asymptotic value.

The `n_unique` column corroborates: at small budgets a few effective samples dominate the posterior (n_unique stays tiny relative to budget), exactly the signature of importance-weight collapse on multi-evidence queries.

### Sweep 3 — corner test

| corner | n_unique | W₁ | matches paper? |
|---|---:|------:|:---:|
| smoke-like (B=16, n_lw_samples=4000)         | 1555 | 0.047 | (smoke baseline) |
| **paper-like (B=1024, n_lw_samples=512)**    | 323  | 0.993 | **✓ reproduces 0.83 ± 0.16** |
| paper-B + smoke-samples (B=1024, n=4000)     | 1590 | 0.021 | clean — same as smoke |
| smoke-B + paper-samples (B=16, n=512)        | 313  | 1.029 | reproduces paper W₁ at smoke B |

The paper-like and smoke-B+paper-samples corners both produce W₁ ≈ 1.0; the paper-B+smoke-samples corner produces W₁ ≈ 0.02 (cleaner than smoke). **B is not the axis; `n_lw_samples` is.**

### Hypothesis routing

- [ ] Hypothesis A — importance resampling on flattened `B × n_samples` weights → per-query collapse
- [ ] Hypothesis B — forward-clamp oracle has B-scaling issue
- [ ] Hypothesis C — LW engine `query_batch` genuinely degrades at large B
- [x] **Other — paper YAML omits `nbn_lw_n_samples`, picks up `CrashTestConfig.nbn_lw_n_samples = 512` default; smoke explicitly sets 4000.**

### Justification

Vary-B sweep produces W₁ range ratio 1.8× (= MC noise floor). Vary-n_lw_samples sweep produces W₁ range ratio 44.7× (= the entire effect). Smoke-B + paper-samples corner reproduces the paper W₁ ≈ 1.0 at B=16 — confirming B is unrelated. Paper-B + smoke-samples corner produces W₁ = 0.021 — confirming B at scale is harmless when budget is adequate. `_compute_inference_accuracy` per-row + `min(B, 16)` cap makes the brief's flattened-weights hypothesis structurally impossible to manifest.

### Estimated round-2 patch size

- **LOC**: 1 line (add `nbn_lw_n_samples: 4000` to `inference_paper.yaml`). Optionally bump higher (8000 or 20000) for stronger paper-grade tightness; the sweep shows diminishing returns past 2000.
- **Files touched**: `benchmarking/configs/inference_paper.yaml` only.
- **Risk**: low. No engine change, no test impact. Smoke parquet is unaffected (smoke YAML already sets 4000 explicitly). Paper run wall-clock cost: linear in `n_lw_samples`, so 8× the LW time per query — but LW is a small fraction of paper-cell wall time so the total impact is modest.
- **Optional companion**: rename `n_samples` → `n_lw_samples` in `_baseline_posterior_for_query`'s call site so the kwarg name in the call matches the config field; or remove the silent default by making `nbn_lw_n_samples` a required field. Either prevents this drift class from re-occurring. Suggest the agent decides in round 2.

---

## Finding 2 — MDN `Categorical(probs=pi)` simplex violation on `continuous_nongauss`

The reviewer brief surfaced three hypotheses (1: softmax overflow → NaN/Inf; 2: division-by-zero in renormalisation; 3: another unsafe op). The audit confirms **Hypothesis 1**, but with a precise mechanism: NaN propagates from upstream parent values through a `linear(parents)` projection whose logit-row weights are zero, because `0 * NaN = NaN` in PyTorch arithmetic.

### Reproduction

The agent retried six (n_nodes, seed) combinations on cpu; the first reliable reproducer was **`continuous_nongauss n=5000 seed=0`** during data generation (`make_synthetic_bn`'s ancestral sample chain). The `n=100 seed=1` cell flagged in the paper run did not reproduce on cpu — likely because cpu vs cuda numerics differ on edge cases, but the *same bug class* fires at `n=5000` cpu, so the diagnostic and fix are the same.

The patch must be installed *before* `make_synthetic_bn` because for `continuous_nongauss` the failure fires inside the data-generation chain, not the posterior query — `make_synthetic_bn` ancestrally samples `train_data` and `ground_truth_samples` from the very same MDN whose `sample()` raises later in the runner.

### Audit of the `probs` tensor at the failing call

| field | value |
|---|---|
| shape                  | `(2000, 3)` |
| has_nan                | **true** |
| n_nan                  | **3** |
| has_inf                | false |
| has_neg                | false |
| min_value (finite)     | 0.139 |
| max_value (finite)     | 0.546 |
| row_sums (finite range)| `[1.0, 1.0]` |
| **first_nan_row_idx**  | **560** |
| **nan_rows_count**     | **1** out of 2000 |

So out of 2000 rows of mixture-component probabilities, 1999 are perfectly valid simplex points (all identically `[0.546, 0.139, 0.314]`) and exactly **one row at index 560 is `[NaN, NaN, NaN]`**. PyTorch's truncated error message displayed only the first non-conforming row — but Simplex's check fires on *any* non-conforming row, and the truncation hid which row that was.

### Sample non-conforming row + sample conforming rows

```
non-conforming:
  idx 560:  row=[NaN, NaN, NaN]   row_sum=NaN

conforming (representative — all 1999 conforming rows are identical):
  idx 0:  row=[0.5464, 0.1393, 0.3143]   row_sum=1.0
  idx 1:  row=[0.5464, 0.1393, 0.3143]   row_sum=1.0
  idx 2:  row=[0.5464, 0.1393, 0.3143]   row_sum=1.0
```

The fact that all 1999 conforming rows are *identical* is itself diagnostic: it means mixture logits are constant across parent inputs.

### Root cause

`benchmarking/synthetic.py::_build_mdn_mechanism` lines 503-527 set the linear projection's weight rows for the logit slots to **zero**:

```python
out_dim = k + k * d_x + k * d_x  # logits + locs + log_scales
weight = torch.zeros(out_dim, p)         # ← rows 0..k all zero (logits)
bias   = torch.zeros(out_dim)
bias[:k] = _dirichlet_logits(k, gen)     # only bias contributes to logits
mean_w = torch.randn(k, d_x, p, ...)
weight[k:k + k * d_x, :] = mean_w.reshape(...)  # locs only
```

So algebraically the mixture pi *should* be constant in parents (only the bias contributes). But `linear(parents)` computes `weight @ parents + bias` for *every* output dim, and for the logit slots `weight[0:k, :] @ parents = 0 * parents`. In PyTorch arithmetic, `0 * NaN = NaN` — so when one or more parents in row 560 are NaN, the logit slot becomes NaN despite the math saying it should be the bias regardless.

Then `softmax(NaN) = NaN`, `clamp_min(1e-7, NaN) = NaN` (clamp doesn't touch NaN), and `NaN / sum(NaN) = NaN`. The non-NaN rows produce the constant `softmax(bias) ≈ [0.546, 0.139, 0.314]`.

### Why a parent becomes NaN at n=5000

The ancestral chain depth at `n=5000` exposes long-tail variance amplification in the MDN's mean parameters. `_build_mdn_mechanism` already tightened constants twice (1/√P → 0.5/√P → 0.3/√P; skew_bias 1.0 → 0.5 → 0.3; scale_high 1.0 → 0.8 → 0.7) for n=100. At n=5000, even those tightened constants compound enough over a 5000-deep chain that some sample drifts to ±Inf or ±NaN at some upstream node. That single bad sample then propagates downstream through every mechanism that consumes it as a parent.

### Hypothesis routing

- [x] **Hypothesis 1 — softmax overflow → NaN/Inf in probs**, with the additional mechanism that NaN propagates through zero-weight `linear(parents)` rows because `0 * NaN = NaN`.
- [ ] Hypothesis 2 — division-by-zero in renormalisation (`pi / pi.sum`). Refuted: row sums are 1.0 (or NaN if upstream NaN), never 0; clamp_min(1e-7) ensures finite divisor when input is finite.
- [ ] Hypothesis 3 — another unsafe op. No evidence.

### Justification

The audit shows exactly 1 NaN row out of 2000, with `[NaN, NaN, NaN]` in all three components. Non-NaN rows are identical (constant `softmax(bias)`), confirming logit weights are zero (per `_build_mdn_mechanism`). The only way for one specific row to become all-NaN under a constant logit is upstream NaN propagation through the zero-weighted projection — `0 * NaN = NaN` in PyTorch.

### Estimated round-2 patch size

- **LOC**: ~10-15 (a defensive guard in `MDNMechanism.sample`/`forward`/`log_prob`).
- **Files touched**: `nbn/mechanisms/mdn.py` (3 methods that call `_params_from_parents`).
- **Risk**: low-medium.

Recommended fix shape (round 2 to decide):

(a) **Sanitise parents at the entry of MDN methods** — replace NaN/Inf in parents with `0.0`, log a warning at the engine level. Robust to any upstream numerical drift, requires no understanding of "why" a parent becomes NaN. ~5 LOC. *Concern*: silently masks deeper numerical issues; warning-on-trigger mitigates this.

(b) **Compute `pi` directly from `bias` when logit weights are zero** — short-circuit the linear projection for the logit rows. Cleaner mathematically but synthetic-side only (production MDNs may have non-zero logit weights). ~5 LOC in `_build_mdn_mechanism` + helper. *Concern*: doesn't fix the general "parent becomes NaN" class; only this specific synthetic-side artifact.

(c) **Tighten MDN constants further** for very deep chains (n≥1000). Pre-compute the maximum n_nodes the constants can sustain without NaN drift and clamp downward. *Concern*: doesn't address the general robustness story; just buys more depth.

I'd suggest **(a)** as the round-2 fix — it's the most defensive and generalises. (b) and (c) are optional follow-ups; (b) is a clean-up of synthetic.py's redundant linear projection, (c) is a v0.7 numerical-stability sweep.

The bug only fires on **3 cells out of 360** in the paper run (`continuous_nongauss n=100 seed=1, seed=3, n=5000 seed=0`) — about 0.8% — so the runner's `run_with_guard` already catches it as `status='error'` and the figure DNFs those cells. (a) flips them to `status='ok'` cleanly; the run-level damage was bounded by the existing classification.

---

## Bonus — structured run logging

`benchmarking/_run_logging.py` adds `setup_run_logging(cfg)` and `finalise_run_logging(meta, status_summary)`. Every `nbn-bench` run now writes:

- `{output_dir}/{output_prefix}_{timestamp}.log` — INFO+ logs, file-level
- `{output_dir}/{output_prefix}_{timestamp}.run.json` — git SHA, torch version, cuda device, full config, wall time, status counts

Both gitignored (`*.log` was already in `.gitignore`; this PR adds `*.run.json`). Best-effort on every step; never blocks the run on a logging failure.

Verified on this sandbox via `nbn-bench inference --config benchmarking/configs/inference_smoke.yaml`:

```
benchmarking/figures/inference_smoke_20260505_174743.log       (gitignored ✓)
benchmarking/figures/inference_smoke_20260505_174743.run.json  (gitignored ✓)
```

The `.run.json` includes `git_sha`, `torch_version`, `cuda_available`, `cuda_device`, the full `config` snapshot (so `nbn_lw_n_samples` would have been visible in any prior run.json had this existed), and post-run `wall_time_s` + `status_summary`.
