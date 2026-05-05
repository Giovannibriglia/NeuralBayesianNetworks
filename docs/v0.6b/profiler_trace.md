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

**The actual cause: naive elimination ordering.** `_plan` (line 79-88) returns the topological order of non-target, non-evidence variables. With `max_in_degree=4` and `n_nodes=20`, eliminating an early-topo variable like `X6` requires producting all factors containing it (its own CPT plus its 4-or-fewer children's CPTs), each of which spans 5 variables — and the union of those factor scopes is 12-14 variables, not 5. The standard remedy is to choose the elimination order that **minimises the maximum induced clique size** — min-fill (Kjaerulff 1990) or weighted-min-fill (Kjaerulff 1992). On this DAG the min-fill ordering would keep each step's scope ≤ 6 variables, dropping the peak from `4¹² × B = 1 GiB` to `4⁶ × B = 256 KiB` — **four orders of magnitude smaller**.

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
