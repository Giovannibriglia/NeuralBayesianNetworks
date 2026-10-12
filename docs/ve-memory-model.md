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
  - a node known only by name is priced as a list of floats (4 + 8 B per row).
    This is the costliest input, so on evidence-heavy queries with few
    factors it can be many times the real peak: 16x for a naive Bayes with 29
    observed children at B=20000. `estimate_peak_bytes` therefore accepts the
    call's `{node: value}` mappings in place of names; pass them whenever the
    values are known;
- on a **cold factor cache**, every CPT's tabulation (`tables_tape`): the
  mechanism's working memory (`Mechanism.tabulate_working_bytes`, overridden
  for table, deterministic, neural-categorical, kNN and SCM mechanisms), the
  logits, and the cached CPT. The hook returns one size per buffer, in the
  parameters' dtype, so a device prices each buffer in its own size class.

### 3. The device's allocator, replayed over the tape (`_device_peak`)

| device | rule |
|---|---|
| CPU | the running sum of live bytes, 64 B aligned |
| MPS | a simulation of Metal heaps under PyTorch's MPS allocator, with the same segment machinery as CUDA. Heap classes: buffers up to 1 MiB use 8 MiB heaps; up to 10 MiB, 32 MiB heaps; up to 512 MiB, 1 GiB heaps; larger buffers get an exact heap rounded to 16 MiB. Buffers are rounded to 256 B, or to 1/64 of the next power of two above 1 MiB. With a sync after every op, freed buffers return their space to the heap, holes coalesce, and a request that no single hole fits opens a new heap. In a real call no free is credited at all: MPS runs asynchronously, and a buffer freed while queued GPU work still uses it cannot be reused yet. Measured on barley q2 at B=64: 73 MiB of driver growth unsynchronised, 33 MiB synchronised. Everything here was measured on Apple unified memory with torch 2.14. |
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
- **Cold cache.** The CPTs are tabulated once, before chunking, if they fit
  next to the reserve. They then stay resident: they join the reserve, and
  every chunk is priced warm. Before, each chunk re-entered the guard cold,
  re-chunked about 150 levels deep, and the batch was refused.
- **Chunk size.** It is the largest one whose own estimate fits
  (`_max_fitting_rows`, a binary search over the model). The chunks run in a
  loop, never by re-entering the guard, and each one is re-sized against the
  live headroom before it runs.

### Cost per call

The plan's replay is memoised on the cached plan cost per `(B, dtype,
device)`, so a repeated query shape replays only its small evidence tapes.
The guard's factor-cache check is reused by the same call's
`_extract_factors`. Single-row `query()` (median of 5 queries x 50 calls,
CPU): alarm 0.066 ms against 0.063 on master, barley 0.44 against 0.45, link
0.36 against 0.41.

## Validation

`scripts/ve_memory_validation.py` covers:

- random DAGs with treewidth about 9 to 10, 4x12 and 6x10 grids, naive Bayes
  models, and an HMM;
- evidence at the tail, scattered, or absent;
- evidence passed as int64 tensors, float tensors, or Python lists;
- `do()`, joint queries with clamped targets, single-row `query()`, and cold
  caches, including a neural-categorical node with 7 parents;
- budget-forced chunking;
- B from 1 to 2.8 M;
- bnlearn barley, two queries with 12 evidence variables
  (`UniformRandomSelector(n_evidence=12)`, seed 0, queries 2 and 7) at B=64
  to 715. Query 7 is the one the round-2 review measured.

On MPS and CUDA every case runs in a fresh process. Before each measurement,
only the CPT cache is warmed (`_extract_factors`); allocator state left by
earlier work would absorb allocations and hide under-estimates.

Results as estimate / measured, min / median / max (budget-forced rows
excluded; torch 2.14.1, macOS 26, Apple M-series, 32 GB, 2026-10-12):

| device | reading | new model | old 3x rule |
|---|---|---|---|
| CPU | exact: profiler allocation events (49 cases) | **1.10 / 1.10 / 2.47** | 0.00 / 1.20 / 1.25 |
| MPS | heap level: `driver_allocated_memory` growth (41 cases) | **1.15 / 1.65 / 11.5** | 0.00 / 0.26 / 2.12 |
| MPS, element cap lifted | heap level, 2 to 8 GiB plans | **1.26 / 1.38 / 3.31** | |
| CUDA | `max_memory_allocated` and reserved growth | to be re-run (#301 phase 2) | |

Barley q7 on MPS, with driver growth measured in a fresh process (the
round-2 review's counterexample):

| B | before round 2 | now |
|---:|---|---|
| 256 | 1716 / 2097 MiB = **0.82** | 2508 / 2097 = 1.20 |
| 384 | 2054 / 2097 = **0.98** | 3670 / 2097 = 1.75 |
| 512 | 2480 / 3113 = **0.80** | 4814 / 3113 = 1.55 |
| 700 | 2562 / 4137 = **0.62** | 5905 / 4137 = 1.43 |

On CPU the same query is exact (1.10) at every B.

Notes on the table:

- **CPU tightness.** CPU is exact (1.10, the margin). The cold
  neural-categorical case is the exception, because the tabulation hook is
  conservative.
- **MPS looseness.** MPS is loose by design: frees are not credited, so long
  plans sum their allocations. The HMM with 60 steps at B=65536 is 11.5x.
  Crediting frees once the GPU has caught up would need the command-buffer
  timeline.
- **Budget-forced cases.** They either ran within budget or were refused:
  - CPU: 760 MiB <= 845 MiB, and 45 MiB <= 54 MiB;
  - MPS heap level: 1024 MiB <= 2722 MiB, and 96 MiB <= 158 MiB;
  - refused: naive Bayes with float evidence at 0.15x and 0.25x.

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
