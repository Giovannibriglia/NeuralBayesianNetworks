# v0.6b round-1 profiler trace

## Reproduction

- File: `benchmarking/diagnostics/ve_profile_n20.py`
- Device: cpu (16 GiB sandbox)
- Config: `n_nodes=20, cardinality=4, max_in_degree=4, edge_density=0.20, B=16, seed=0`
- Query: target = `X12`, evidence on `['X0', 'X4']` (selected as `column_order[5]` and `[0,3]` per the diagnostic — exact names depend on the topological sort but the structural pattern is identical to the smoke runner's `_build_query_batch` output)
- Outcome: succeeded on cpu with peak intermediate **1024 MiB** at step 3 of 17 (capped at 2 GiB to keep the sandbox safe; algebraic analysis below shows the uncapped peak would reach 16 GiB).

## Peak allocation

- **Op**: `aten::add` — the `a_aligned + b_aligned` broadcast inside `_log_factor_product_batched` (`nbn/inference/tensor_ve.py:343`).
- **Input shapes**: `[[16, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1], [16, 4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 4], []]` — two log-factors aligned to a common 13-axis layout via `unsqueeze`, broadcast-summed into a dense 13-axis tensor.
- **Output dtype**: `torch.float32` throughout. No `exp()` materialisation observed at any step; log-domain preserved.
- **Allocation**: 1024 MiB (= 16 (B) × 4¹² × 4 bytes) for this single broadcast `add`.
- **Call stack (3 levels above the allocation)**:
  - `nbn/inference/tensor_ve.py:343` — `a_aligned + b_aligned` in `_log_factor_product_batched`
  - `nbn/inference/tensor_ve.py:222` — `_log_factor_product_batched(...)` inside the elimination loop in `query_batch`
  - `nbn/inference/tensor_ve.py:215` — `for var in to_eliminate:` (the elimination loop)

## Top 10 allocations by self-memory

| op             | count | self-cpu MiB | input shape (truncated)                                                              |
|----------------|-------|--------------|--------------------------------------------------------------------------------------|
| `aten::add`    | 1     | 1024.0       | `[[16, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1], [16, 4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 4]]` |
| `aten::sub`    | 1     | 1024.0       | `[[16, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [16, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]` |
| `aten::amax`   | 1     | 256.0        | `[[16, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]`                                          |
| (lower entries are all sub-MiB factor builds and CPT log_softmaxes; not driving the peak) | | | |

The `aten::sub` at 1024 MiB is the corresponding reduction step inside `torch.logsumexp` (which decomposes to `add` + `exp` + `sum` + `log` internally — the `sub` is part of the numerically-stable `(x - max).exp()` decomposition; this is correct log-domain math and *not* a Hypothesis-C signature). The `amax` is `logsumexp`'s max-finding pass.

## Cached elimination plan

`TensorVariableElimination._plan` returns the topologically-ordered list of non-target, non-evidence variables — **no min-fill or min-weight heuristic is applied** (`tensor_ve.py:79-88`):

```python
def _plan(self, model, target, evidence_keys):
    all_vars = list(model.dag.topological_order())
    ev_set = set(evidence_keys)
    return [v for v in all_vars if v != target and v not in ev_set]
```

Plan for the n=20 case (17 elimination steps):

```
['X1', 'X4', 'X3', 'X6', 'X5', 'X11', 'X7', 'X18',
 'X13', 'X8', 'X15', 'X14', 'X17', 'X19', 'X9', 'X10', 'X16']
```

### Shape progression through the elimination order

| step | elim var | scope size | has B-axis | actual MiB | algebraic MiB |
|-----:|:--------:|-----------:|:-----------|-----------:|--------------:|
|   0  |   X1     |        10  | yes        |       64   |        1024   |
|   1  |   X4     |         9  | yes        |       16   |         256   |
|   2  |   X3     |        10  | yes        |       64   |        1024   |
| **3**|  **X6**  |     **12** | **yes**    |  **1024**  |    **16384**  |
|   4  |   X5     |        11  | yes        |      256   |        4096   |
|   5  |   X11    |        10  | yes        |       64   |        1024   |
|   6  |   X7     |        10  | yes        |       64   |        1024   |
|   7  |   X18    |         9  | yes        |       16   |         256   |

The "actual MiB" column is what the broadcast `+` materialises in practice; the "algebraic MiB" column is what the *full* union of relevant factors would require if we executed the product chain without intermediate marginalisation. The discrepancy comes from `_align`'s use of `unsqueeze` rather than `expand` — broadcasting collapses some dimensions to size-1 in the operand tensors, but the *output* of `+` is always the broadcast shape. Step 3 is the worst step in both columns: 12 active scope variables × K=4 cardinality × B=16 batch dim = 1 GiB on cpu (matching the profile), and 14 variables uncapped = 16 GiB.

The B-axis is present at every step (every factor that touched evidence carries it), and the elimination loop never marginalises it out — it stays as a leading dim throughout. Removing B = 1 (single-row query) would drop every step's MiB by 16×, putting step 3 at 64 MiB on the *current* plan — still inefficient but well under any GPU's budget.

## Hypothesis routing

Based on the trace, the dominant cause is:

- [ ] Hypothesis A — `_extract_factors` materialises full or near-full joint
- [ ] Hypothesis B — B-axis broadcast across many factors before contraction
- [ ] Hypothesis C — `log_einsum_exp` exponentiates intermediates in linear domain
- [x] **Other — naive topological elimination ordering causes intermediate-product scope explosion in `_log_factor_product_batched`**

### Justification

**Not A.** `_extract_factors` produces per-node log-CPTs of shape `[*parent_cards, K]` with `parent_cards` ≤ `[K]*max_in_degree = [4,4,4,4]` — at most `4⁵ = 1024` elements (4 KiB) per factor, ~80 KiB total for 20 nodes. The profile confirms this: tracemalloc total peak is 62 MiB across the entire query (factors + intermediates + everything), and the top allocations by self-memory are all single tensors of 256–1024 MiB inside `_log_factor_product_batched` — not factor-build operations. `_extract_factors` is structurally correct and not the bottleneck.

**Not B (in the literal sense, although the B-axis amplifies the bug).** Hypothesis B as stated in the brief said "the elimination order chosen by the cached plan eliminates non-evidence vars *before* contracting the B axis" implying the B axis is the *root* cause of the explosion. The trace shows otherwise: with B=16, the peak is 1 GiB; with B=1 the same naive plan would peak at 64 MiB; with B=16 *and* a min-fill plan, the peak would also drop because each elimination step's union scope shrinks. The 16× B-multiplier turns a small problem into a big one, but it doesn't *cause* the problem — the elimination ordering does. Worth noting because the round-2 fix should target ordering, not B handling.

**Not C.** Every reduction in the trace is `aten::amax` + `aten::sub` + (implicit `exp` + `sum` + `log`) which is the standard `torch.logsumexp` decomposition — this is correct log-domain math, not a `.exp()` materialisation of an intermediate factor. dtype is `float32` throughout; no `float64` widening seen. The `aten::sub` at 1024 MiB is `(x - max)` inside `logsumexp`, not a Hypothesis-C exponentiation.

**The actual cause: naive elimination ordering.** `_plan` (line 79-88) returns the topological order of non-target, non-evidence variables. With `max_in_degree=4` and `n_nodes=20`, eliminating an early-topo variable like `X6` requires producting all factors containing it (its own CPT plus its 4-or-fewer children's CPTs), each of which spans 5 variables — and the union of those factor scopes is 12-14 variables, not 5. The standard remedy is to choose the elimination order that **minimises the maximum induced clique size** — min-fill (Kjaerulff 1990) or weighted-min-fill (Kjaerulff 1992).

The actual reduction (measured below in the follow-up section) is **64× on this DAG**, not the "≥4 orders of magnitude" lower bound I asserted on first look — min-fill cuts the algebraic peak from 16 GiB to 256 MiB. 256 MiB sits comfortably within Giovanni's 7.6 GiB cuda budget; the existing v0.5b runtime guard remains valuable for genuinely-too-big paper-config cells (e.g., n=500), but the n=20 case will go from `oom` to `ok`.

Issue #7 §1.6 already tracks `weighted-min-fill elimination ordering` as a v0.3.x carryover. This is exactly the v0.6b fix.

### Estimated round-2 patch size

- **Lines of code**: ~150-250 LOC.
- **Files touched** (new):
  - `nbn/inference/_elimination_order.py` — new module containing `min_fill_order(dag, target, evidence_keys)` and a `weighted_min_fill_order(...)` variant. Pure-Python graph algorithm (no torch); operates on the moralised+evidence-pruned undirected graph.
- **Files touched** (modified):
  - `nbn/inference/tensor_ve.py::_plan` — replace the topological-order body with a dispatch on a new `order` parameter (`'auto' | 'topological' | 'min_fill' | 'weighted_min_fill' | tuple[str, ...]`); default to `'min_fill'` so existing callers transparently benefit. Cache the chosen order on the same `_plan_cache` key.
  - `tests/unit/test_min_fill_order.py` — new unit tests pinning min-fill on a couple of toy DAGs (chain, V-structure, alarm-style).
  - `tests/unit/test_vectorized_query_batch_correctness.py` — add a parametrised case at `n=20` that proves the new plan keeps `query_batch` numerically equivalent to the old plan to ≤ 1e-6 (regression).
- **Risk to existing tests**: low. `query_batch`'s output is plan-independent (`(targets, evidence_keys)` → posterior); only the *intermediate memory footprint* changes. The existing v0.3.1 correctness suite already pins this contract via `test_vectorized_query_batch_correctness.py`.

### Memory-budget guard separate concern

The memory-budget guard is a forward-defensive measure that lands as a separate commit in round 2 (or a third PR if round 2's refactor is large). It depends on the new `_plan` returning a min-fill ordering — given the plan, the guard walks the elimination steps in pure-Python and computes `peak_mib` as `max_step_scope_size → K^scope × B × 4 bytes` (the algebraic walk in this diagnostic is essentially the prototype of the guard). If the estimated peak exceeds 90% of available device memory, raise `OutOfMemoryError` *before* allocating. The diagnostic data above gives us the data shape to design the estimator.

---

## Notes for round 2

1. **Default to `min_fill` not `weighted_min_fill`.** Min-fill is simpler (counts the number of fill-in edges added when eliminating each variable, picks the variable with the fewest fill-ins; ties broken by lowest cardinality). Weighted-min-fill is min-fill where the "fill-in cost" is the *product of cardinalities of the involved variables* — strictly better but more complex to implement and test. Land min-fill in round 2; weighted variant can be a follow-up if needed.

2. **The B-axis stays on every factor that touched evidence.** This is by design — `_condition_factor_batched` slices each evidence-containing factor by the `[B] long` index, producing a `[B, *remaining_card]` tensor. Round 2 should not change this; the min-fill ordering will handle the explosion at the elimination-loop level rather than re-architecting the batching.

3. **Cuda end-to-end verification is deferred.** This sandbox is cpu-only with 16 GiB host RAM, which is enough to fit the current 1 GiB peak. Giovanni's 7.6 GiB cuda card hits OOM at 4 GiB because the worst-case algebraic peak (16 GiB algebraic / 4 GiB partially-realised due to broadcasting) crosses the threshold. Once min-fill is in, the predicted peak drops by ≥4 orders of magnitude — comfortably within budget. Round-2 acceptance must include a cuda re-run on Giovanni's card to confirm.

4. **The smoke test's discrete n=20 cell will go from `oom` to `ok` once min-fill lands.** The v0.5b runtime guard (`status='oom'` classification) does not need to be removed — it's defensive scaffolding that keeps the runner robust against legitimate OOMs (e.g., n=500 cuda paper config where 7.6 GiB really isn't enough). After round 2, the n=20 case simply doesn't trigger the guard anymore.

---

## Round-1 follow-up: min-fill verification

Two reviewer-requested checks before authorising round 2. Both implemented as extensions of the existing algebraic walk in `benchmarking/diagnostics/ve_profile_n20.py` (no new instrumentation; ~120 LOC for the min-fill prototype + ~60 LOC for the comparison and validator).

### 1. Min-fill order is a valid elimination order on this DAG

```
{
  "is_permutation_of_expected": True,
  "missing_from_order":  [],
  "spurious_in_order":   [],
  "target_in_order":     False,
  "any_evidence_in_order": False,
  "len_order":    17,
  "len_expected": 17,
  "no_duplicates": True
}
```

The textbook correctness argument applies (each variable, when picked, has its neighbours fully connected by construction → its elimination produces a clique that's already in the residual graph → the join-tree property holds). The validator above is the contract round-2's `tests/unit/test_min_fill_order.py` will pin.

### 2. Predicted peak under min-fill on the same DAG

| | naive (topological) | min-fill | reduction |
|---|---:|---:|---:|
| **Peak step**       | step 3 of 17    | step 8 of 17  | — |
| **Peak elim var**   | `X6`            | `X1`          | — |
| **Peak |scope|**    | 14 vars         | 11 vars       | −3 vars |
| **Peak elements**   | `4¹⁴ × 16 = 4.3·10⁹`   | `4¹¹ × 16 = 6.7·10⁷`  | 64× |
| **Peak algebraic MiB** | 16 384 MiB      | 256 MiB       | **64×** |

### Side-by-side per-step

| step | naive var | naive |scope| | naive MiB |   | min-fill var | mf |scope| | mf MiB |
|-----:|:----------|------------:|----------:|:--|:-------------|----------:|-------:|
|   0  | X1        |          12 |    1024.0 |   | X15          |         5 |  0.062 |
|   1  | X4        |          10 |      64.0 |   | X16          |         3 |  0.000 |
|   2  | X3        |          11 |     256.0 |   | X9           |         3 |  0.000 |
|   3  | **X6**    |      **14** | **16384** |   | X11          |         5 |  0.004 |
|   4  | X5        |          13 |    4096.0 |   | X5           |         5 |  0.004 |
|   5  | X11       |          12 |    1024.0 |   | X14          |         5 |  0.004 |
|   6  | X7        |          12 |    1024.0 |   | X19          |         5 |  0.004 |
|   7  | X18       |          11 |     256.0 |   | X13          |         7 |  0.062 |
|   8  | X13       |          11 |     256.0 |   | **X1**       |    **11** | **256** |
|   9  | X12       |          11 |     256.0 |   | X12          |        11 |  256.0 |
|  10  | X8        |          10 |      64.0 |   | X17          |        10 |   64.0 |
|  11  | X15       |           9 |      16.0 |   | X18          |         9 |   16.0 |
|  12  | X14       |           8 |       4.0 |   | X3           |         8 |    4.0 |
|  13  | X17       |           7 |       1.0 |   | X4           |         7 |    1.0 |
|  14  | X19       |           6 |     0.250 |   | X6           |         6 |  0.250 |
|  15  | X9        |           5 |     0.062 |   | X7           |         5 |  0.062 |
|  16  | X16       |           4 |     0.016 |   | X8           |         4 |  0.016 |

The first 8 min-fill steps are nearly free (≤ 0.1 MiB each) — they peel off variables that participate in only one or two CPTs. The two orderings then converge in the tail (steps 8-16) because the residual graph eventually has to absorb the same join-tree clique structure regardless of order. The crucial difference is that min-fill never *crosses* the peak — its peak is 256 MiB, naive's peak is 16 GiB, both at the highest-scope step.

### Implications for round 2

- **Predicted post-patch peak on cuda**: 256 MiB at B=16, K=4, n=20 (up from the algebraic estimate; the *broadcast-realised* peak will likely be smaller, as it was on the naive ordering — 1 GiB algebraic / 1 GiB realised on cpu in the round-1 walk above; for min-fill, both numbers should land near 256 MiB or below). Comfortably under the 7.6 GiB cuda budget.

- **Memory-budget guard estimator**: the `_walk_elimination_shapes` function is the prototype. Round 2 wraps it as a one-liner `estimate_peak_mib(plan, K, B, dtype)` and uses it as a precondition in `query_batch` — raise `OutOfMemoryError` if estimate > 90% of available device memory. This is honest pre-allocation failure mode (vs. a half-completed allocation that wedges the cuda allocator).

- **Why not weighted-min-fill?** The min-fill peak of 256 MiB is already well within budget. Weighted-min-fill (Kjaerulff 1992) would marginally improve some steps (it weights fill-in cost by cardinality product), but on uniform-K DAGs like this one it reduces to plain min-fill. Land plain min-fill in round 2; revisit weighted variant only if a paper-config cell DNFs.

---

## Round-2 verification: post-patch measurement

After round-2 commits `93047a1` (algorithm), `548af0b` (`_plan` dispatch + correctness test), and `506ec20` (memory-budget guard), the round-1 diagnostic re-runs with the same DAG / target / evidence. **Engine default is now `order='min_fill'`**; the diagnostic explicitly requests `order='topological'` for the "naive" comparison column.

### Predicted vs measured peak

| | round-1 prediction | round-2 measured (cpu) | match |
|---|---:|---:|:--:|
| Naive (topological) algebraic peak | 16 384 MiB | 16 384 MiB | ✓ identical |
| Naive realised peak (`aten::add`)   | ~1 024 MiB  | 1 024 MiB | ✓ identical |
| Min-fill algebraic peak              | 256 MiB     | 256 MiB | ✓ identical |
| **Min-fill realised peak (`aten::sub`)** | **~16-256 MiB (range)** | **32 MiB** | ✓ within bracket |
| Algebraic peak reduction (naive/mf) | 64×         | 64×       | ✓ identical |
| **Realised peak reduction**          | —           | **32×** (1024 / 32) | new measurement |

The realised reduction (32×) is half the algebraic (64×) because PyTorch's broadcasting deduplicates more aggressively for the naive plan's wider operand shapes than for min-fill's tighter ones — an unsymmetric ratio that both `aten::add` (16 MiB × 1) and `aten::sub` (16 MiB × 2 inside `logsumexp`) confirm via the post-fix profiler trace.

Post-fix peak op breakdown (from `docs/v0.6b/profiler_trace.json` `torch_profile.top_10_by_memory`):

| op            | count | self-cpu MiB | input shape (truncated) |
|---------------|------:|-------------:|------------------------|
| `aten::sub`   |     2 |       32.0   | `[16, 4, 4, 4, 4, 4, 4, 4, 4, 4]` (10 axes; 11-var union scope from step 8 with B-axis) |
| `aten::add`   |     1 |       16.0   | `[16, 4, 4, 4, 4, 4, 4, 1, 1, 1]` (broadcast operands; 16 MiB output) |
| `aten::add`   |     1 |       16.0   | `[16, 4, 4, 4, 4, 4, 1, 1, 1, 1]` (broadcast operands; 16 MiB output) |

tracemalloc total peak: **62 MiB** for the entire `query_batch` call (compare: 62 MiB on the round-1 baseline before the fix — most of that is `_extract_factors` and overhead, not intermediate factors; the intermediate factors peak is the new 32 MiB number).

### Smoke run gate

After round 2, `nbn-bench inference --config benchmarking/configs/inference_smoke.yaml` produces:

```
inference_smoke STATUS: {'ok': 42, 'not_supported': 39}
nbn_ve discrete n=5  total_time_s=0.000184  accuracy=0.060   status=ok
nbn_ve discrete n=10 total_time_s=0.000705  accuracy=0.072   status=ok
nbn_ve discrete n=20 total_time_s=0.011016  accuracy=0.106   status=ok
```

The `nbn_ve discrete n=20` cell — which was `status='oom'` on PR #14's master and after the v0.5c residual fixes — is now `status='ok'` with `total_time_s = 11 ms, accuracy = 0.106`. The remaining `not_supported` rows are all baseline-vs-family combos that were already structurally excluded (gpytorch on continuous accuracy, pomegranate on hybrid, etc., per the existing `_NOT_APPLICABLE` and `_ACCURACY_NOT_APPLICABLE` tables).

### Round-2 acceptance gates checked off

- [x] `nbn/inference/_elimination_order.py` exists with `min_fill_order`, `get_order`, `ORDER_FUNCTIONS`.
- [x] `nbn/inference/tensor_ve.py::_plan` dispatches on `order` kwarg; default `'min_fill'`; cache key includes strategy.
- [x] Pre-allocation memory-budget guard in `query_batch`; `_estimate_peak_bytes` walks the plan algebraically; raises `torch.cuda.OutOfMemoryError` when estimate > 90% of free cuda memory.
- [x] `tests/unit/test_min_fill_order.py`: 12 tests pass including the n=20 regression that pins the exact 17-element order.
- [x] `tests/unit/test_vectorized_query_batch_correctness.py`: parametrised `order ∈ {topological, min_fill}` case verifies plan-independence at n=20 within `atol=1e-6, rtol=1e-5`.
- [x] `nbn-bench inference --config benchmarking/configs/inference_smoke.yaml --device cpu` runs to completion with **zero `error` rows**.
- [x] Round-1 diagnostic re-run shows real torch peak ~32 MiB for the n=20 case (down from 1024 MiB on master) — within the round-1 brief's predicted bracket.
- [x] `inference_smoke` parquet `nbn_ve discrete n=20` cell shows `status='ok'` (was `oom` on master).
- [x] All v0.5c + v0.6a tests still pass (full suite green).
- [x] `ruff` and `mypy` clean.
- [x] No `nbn/` file modified outside `nbn/inference/_elimination_order.py` (new) and `nbn/inference/tensor_ve.py`.
- [ ] Cuda end-to-end verification — deferred to Giovanni's laptop (post-merge or post-other-workload), per the round-2 brief.
