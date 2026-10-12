# VE peak-memory model (#178)

`TensorVariableElimination` refuses, chunks or (through `HybridRouter`) reroutes
a query whose estimated peak exceeds the device budget. The estimate must
therefore be an **upper bound** of what the call makes the device hold. Until
#178 it was "largest elimination factor x 3.0" (`_LIVESET_MULTIPLIER`, #177),
which has been removed. The model now lives in `nbn/inference/_ve_memory.py`.

## The model

### 1. The tape

`_plan_cost` walks the plan exactly as `query_batch` runs it and records every
allocation and free, in program order. Sizes are `(batch_free, per_row)`
element counts, so one tape prices every `B`.

| phase | allocations |
|---|---|
| conditioning | a factor touched by evidence becomes a new `[B, ...]` tensor; an untouched factor is a view of its cached CPT |
| product `P_j = P_{j-1} + r_j` | `P_j` is allocated. The operands are broadcast *views*. Each input is freed once it has been multiplied in (`_multiply_batched` consumes its list). |
| `logsumexp` | max, `abs` and mask (freed), the full-size `input - max` temporary, and the result; then the temporary, the max and the input are freed |
| answer | the product of what is left, then `softmax`; a joint answer adds a reshape copy and the one-hot embedding of clamped targets |

### 2. Tensors that depend on the call

`_PeakModel` adds these to the tape:

- **evidence and do values** (`evidence_prelude`):
  - Python lists or scalars are materialised by `torch.as_tensor`;
  - values that are not already int64 on the device are copied to int64;
  - `check_discrete_values` makes a float32 copy (unless the input is already
    float32), a rounding and a mask, freed before elimination;
  - int64 values already on the device cost nothing;
  - a node known only by name is priced as a list of floats (4 + 8 B per row);
- on a **cold factor cache**, every CPT's tabulation (`tables_tape`): the
  mechanism's working memory (`Mechanism.tabulate_working_bytes`, overridden
  for table, deterministic, neural-categorical, kNN and SCM mechanisms), the
  logits, and the cached CPT.

### 3. The device's allocator, replayed over the tape (`_device_peak`)

| device | rule |
|---|---|
| CPU | the running sum of live bytes, 64 B aligned |
| MPS | each live buffer is charged its share of the heap it lives in, plus one partly filled heap per size class in use. Heap classes: buffers up to 1 MiB use 8 MiB heaps; up to 10 MiB, 32 MiB heaps; up to 512 MiB, 1 GiB heaps (a 342 MiB buffer fits only twice); larger buffers get an exact heap rounded to 16 MiB. These classes were measured on Apple unified memory with torch 2.14. |
| CUDA | a simulation of `CUDACachingAllocator` in its default configuration. A request is rounded to 512 B. It takes the best-fitting cached free block of its pool (requests up to 1 MiB use the small pool), and that block is split when the rest is at least 512 B (small pool) or more than 1 MiB (large pool). Otherwise a new segment is created: 2 MiB, 20 MiB for requests under 10 MiB, or the request rounded up to 2 MiB. Freed blocks merge with free neighbours and stay cached. The estimate is the peak *reserved* bytes, which includes freed segments that later requests cannot reuse. |

The result is multiplied by `_LIVESET_SAFETY = 1.10`.

### Chunking

A batch whose peak exceeds the budget is split into chunks.

- **What stays alive next to each chunk.** The batch's evidence tensors (each
  chunk indexes views of them), the finished answers, and their final
  concatenation. This reserve is subtracted from the budget and passed down,
  so each chunk re-checks itself against what is left.
- **Refusal.** If even one-row chunks cannot fit the *stable* budget next to
  the reserve, the batch is refused with `OutOfMemoryError`. Before, it ran as
  1-row chunks that exceeded the budget.
- **Chunk size.** It is the largest one whose own estimate fits
  (`_max_fitting_rows`, a binary search over the model).

## Validation

`scripts/ve_memory_validation.py` covers:

- random DAGs with treewidth about 9 to 10, 4x12 and 6x10 grids, naive Bayes
  models, and an HMM;
- evidence at the tail, scattered, or absent;
- evidence passed as int64 tensors, float tensors, or Python lists;
- `do()`, joint queries with clamped targets, single-row `query()`, and cold
  caches, including a neural-categorical node with 7 parents;
- budget-forced chunking;
- B from 1 to 2.8 M.

On MPS and CUDA every case runs in a fresh process. Before each measurement,
only the CPT cache is warmed (`_extract_factors`); allocator state left by
earlier work would absorb allocations and hide under-estimates.

Results as estimate / measured, min / median / max (budget-forced rows
excluded; torch 2.14.1, macOS 26, Apple M-series, 32 GB, 2026-10-12):

| device | reading | new model | old 3x rule |
|---|---|---|---|
| CPU | exact: profiler allocation events (37 cases) | **1.10 / 1.10 / 2.47** | 0.00 / 1.12 / 1.25 |
| MPS | heap level: `driver_allocated_memory` growth (29 cases) | **1.15 / 1.62 / 34.8** | 0.00 / 0.28 / 2.12 |
| MPS, element cap lifted | heap level, 2 to 8 GiB plans (4 cases) | **1.27 / 1.30 / 1.54** | 0.00 / 0.91 / 1.07 |
| CUDA | `max_memory_allocated` and reserved growth | to be re-run (#301) | |

Notes on the table:

- **CPU tightness.** Every CPU case is at 1.10, which is exactly the margin.
  The one exception is the cold neural-categorical case at 2.47, where the
  tabulation hook is conservative.
- **MPS looseness.** Large MPS ratios come from the 1 GiB heap reserve on
  small plans. That reserve is real: plans measured 1.03 GiB of driver growth
  for a 20 MiB live set.
- **Budget-forced cases.** Each one either ran within its budget or was
  refused:
  - CPU: 760 MiB <= 845 MiB, and 45 MiB <= 54 MiB;
  - MPS: 72 MiB <= 81 MiB;
  - refused: naive Bayes with float evidence at 0.15x and 0.25x, and `wide30`
    at 0.3x on MPS, where one row needs a 1 GiB heap.

Reproduce:

```bash
uv run --extra all python scripts/ve_memory_validation.py --device cpu           # add --big-only for the 2-6 GiB cases
uv run --extra all python scripts/ve_memory_validation.py --device mps --op-probe
uv run --extra all python scripts/ve_memory_validation.py --device mps --big-only --mps-uncapped
uv run --extra all python scripts/ve_memory_validation.py --device cuda --json ve_memory_cuda.json
```

Out of scope:

- mechanisms that do not override `tabulate_working_bytes` get a default of
  four table-sized float32 temporaries;
- MPS heap packing was measured, not read from source;
- the CUDA simulation assumes the default allocator configuration (no
  `max_split_size`, no `garbage_collection_threshold`).
