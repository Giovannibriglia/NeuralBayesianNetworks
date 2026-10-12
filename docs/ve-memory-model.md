# VE peak-memory model (#178)

`TensorVariableElimination` refuses, chunks or (through `HybridRouter`) reroutes
a query whose estimated peak exceeds the device budget. The estimate must
therefore be an **upper bound** of what the call allocates. Until #178 it was
"largest elimination factor x 3.0" (`_LIVESET_MULTIPLIER`, #177). It is now a
walk of the live set.

## The model (`nbn/inference/tensor_ve.py`, `_plan_cost`)

The walk follows `query_batch` step by step. Element counts are kept as
`(batch_free, per_row)` pairs, so one walk prices every `B`. At each
allocation the live set is:

| phase | alive |
|---|---|
| conditioning | each factor touched by evidence becomes a new `[B, ...]` tensor; an untouched factor is a view of its cached CPT |
| product `P_j = P_{j-1} + r_j` | the rest of the pool, the step's inputs not yet multiplied in, `P_{j-1}`, `r_j`, `P_j`. The operands are aligned as broadcast *views*; only the sum is materialised. |
| `logsumexp` over the eliminated variable | the input, one full-size temporary (`input - max`, exponentiated in place), and the max and the result (both reduced) |
| answer | the product of what is left, then `softmax` (`[B, K]`); a joint answer adds a reshape copy, and clamped joint targets add the one-hot embedding |

Terms added on top of the walk (`_PeakModel.bytes`):

- the dtype of the parameters (`float64` models cost twice as much);
- `8 B` per row for each evidence or do variable, the int64 index copy;
- on a cold factor cache, every CPT of the model, which the call tabulates;
- one allocator rounding per tensor that can be alive at once: 64 B on CPU,
  512 B on CUDA, 16 KiB on MPS;
- the caching allocator's reserve: one partly filled heap or segment per size
  class that the largest tensor reaches. On MPS, buffers of 10 MiB to 1 GiB
  are carved from 1 GiB heaps (measured). On CUDA the segments are 2 MiB and
  20 MiB;
- `_LIVESET_SAFETY = 1.10`.

The issue described three union-scope copies per product (`a_aligned`,
`b_aligned`, the sum). The code does not allocate them: the aligned operands
are `permute` / `unsqueeze` views. Measured on both CPU and MPS, `+` and
advanced indexing allocate only their output. `logsumexp` allocates exactly
one temporary the size of its input.

The engine change that goes with the model: products now *consume* their
input list (`_multiply_batched`, `_multiply`). Each input is freed as soon as
it is multiplied in. Before, lists and loop variables kept every input alive
until the elimination step ended. The CPU peak drops by up to about 9 % (for
example 704 to 640 MiB on `wide30`, B=16).

**Chunking.** The estimate is a maximum of terms affine in `B`, so it is
convex in `B`. `_max_chunk_rows` sizes chunks by the chord through `B=1` and
the full `B`. The chord lies above a convex function, so the chunk sizes it
gives are safe. The heap reserve is a step function of `B`, so for chunking it
is priced at the full batch. Chunks also leave room for the rows that earlier
chunks already finished. The old rule, `budget // (peak_B / B)`, over-sized
chunks whenever part of the peak does not grow with `B`.

`_LIVESET_MULTIPLIER` is **removed**, with no fallback. The walk reproduces the
CPU peak exactly, and the multiplier was below the measured peak in exactly
the cases the walk handles (see below), so a fallback could not be both sound
and useful.

Out of scope: the working memory inside a neural mechanism's `tabulate()`
(MLP activations over all parent configurations), which belongs to the
mechanism. The produced tables themselves are priced.

## Validation

`scripts/ve_memory_validation.py` runs 35 problems:

- random DAGs with in-degree 3 to 4 and treewidth about 9 to 10;
- 4x12 and 6x10 grids;
- naive Bayes with 40 observed children;
- an HMM with 60 observed steps;
- evidence at the tail, scattered, or absent;
- `do()`, a joint query with a clamped target, a single-row `query()`, a cold
  cache, and budget-forced chunking;
- B from 1 to 131072.

Peaks are measured during warm calls, relative to the allocation level just
before the call. `tests/perf/test_ve_memory_validation.py` runs the cases
under 1 GiB on every available device.

Ratios are estimate / measured, given as min / median / max.

| device | what is measured | new model | old 3x rule |
|---|---|---|---|
| CPU (exact: `torch.profiler` allocation events) | 35 cases | **1.10 / 1.10 / 1.66** | 0.00 / 1.09 / 1.24 (on master's own peaks) |
| MPS, heap level (`driver_allocated_memory` growth) | 28 cases | **1.05 / 1.53 / 2.87** | 0.00 / 0.33 / 2.91 |
| MPS, heap level, element cap lifted (single buffers >= 1 GiB, exact heaps) | 3 cases, 2 to 3 GiB | **1.17 / 1.26 / 1.46** | 0.82 / 1.00 / 1.07 |
| CUDA | not measured here; see the PR's "CUDA validation requested" | | |

Notes:

- **CPU is exact.** The walk lands within the 1.10 margin on every factor-heavy
  case. The cases at 1.6 are dominated by index tensors: the estimate assumes
  each evidence tensor is copied to int64, but these callers already passed
  int64. With float evidence the copy happens, and the ratio returns to 1.10.
- **The old rule under-estimated.** It priced naive Bayes at 0 bytes (nothing
  is eliminated) and the HMM at 0.17x its peak. It also under-estimated the
  4x12 grid at B=1 (0.92), and the MPS footprint whenever a plan used a 1 GiB
  heap. At B=131072, the naive-Bayes chunked run used 86 MiB against a budget
  of 0.
- **MPS** has no peak counter. The polled tensor-level reading
  (`current_allocated_memory` after every op) is a lower bound only: it cannot
  see temporaries freed inside an op, and the estimate is 1.44 or more times
  above it. A per-op probe at 1 GiB, where heaps are exact, shows the same
  temporaries as the CPU kernels: `logsumexp` 1x its input, and none for
  `+`, `softmax` or advanced indexing. What MPS adds is the heap reserve. With
  the reserve modelled, the estimate bounds the heap-level growth on every
  case.

Reproduce:

```bash
uv run --extra all python scripts/ve_memory_validation.py --device cpu
uv run --extra all python scripts/ve_memory_validation.py --device mps --op-probe
uv run --extra all python scripts/ve_memory_validation.py --device mps --big-only --mps-uncapped
uv run --extra all python scripts/ve_memory_validation.py --device cuda
```

The validation ran on 2026-10-12 with torch 2.14.1 on macOS 26, Apple
M-series, 32 GB.
