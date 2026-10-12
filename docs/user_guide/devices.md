# Devices, batching and memory

## Choosing a device

Every model lives on one device, set at construction and changed with
`.to()`:

```python
import torch
from nbn import NeuralBayesianNetwork, seed_all
from nbn.utils.device import accelerator_summary, resolve_device

print(resolve_device("auto"), "|", accelerator_summary())

model = NeuralBayesianNetwork([("A", "B")], {"A": ("discrete", 2), "B": ("discrete", 3)},
                              device="cpu")              # or "auto", "cuda", "cuda:1", "mps"
print(model.device)
```

`device="auto"` (the default for models, `nbn-bench --device` and the YAML
configs) picks **CUDA**, then **MPS** (Apple Silicon), then the **CPU**. Data
passed to `fit`, `update`, `query`, `sample` and `log_prob` are moved to the
model's device for you; results come back on it. Move a fitted model with
`model.to("cuda")` (or `.cpu()`, `.cuda(1)`); cached inference state is
rebuilt on the new device. On a machine with several GPUs, give each model
its own (`device="cuda:1"`); `realise_device` pins the index so a later
`torch.cuda.set_device` does not move the model's memory accounting.

Results are the same on every backend up to floating-point precision and
Monte-Carlo noise.

## Apple Silicon (MPS)

NBN works around the gaps of PyTorch's Metal backend:

- **No float64 on Metal.** Sufficient statistics (counts, normal equations,
  weighted moments) accumulate in float32 on `mps` and in float64 elsewhere
  (`nbn.utils.device.accum_dtype`).
- **No `linalg.lstsq` kernel.** The small closed-form Gaussian solves
  (`[D_pa + 1, D_pa + 1]` matrices) run on the CPU.
- **Other missing kernels.** `import nbn` sets `PYTORCH_ENABLE_MPS_FALLBACK=1`
  on macOS unless you set it yourself, so an op a dependency needs and Metal
  lacks runs on the CPU with a one-time warning instead of failing. Set it to
  `0` before importing NBN to opt out.

## Batching

`query_batch` answers `B` queries in one call ({doc}`inference`). Variable
elimination contracts all rows at once; likelihood weighting draws
`B x n_samples` particles at once. The speed-up over a loop of `query` calls
grows with `B`; GPUs pay off on large batches, large training sets and wide
networks, while a toy network is often faster on the CPU because kernel-launch
overhead dominates.

```python
import time

seed_all(0)
x = torch.randint(0, 2, (5000,))
model.auto_mechanisms()
model.fit({"A": x, "B": (x + torch.randint(0, 2, (5000,))) % 3})

evidence = {"B": torch.randint(0, 3, (4096,))}
start = time.perf_counter()
batched = model.query_batch(["A"], evidence)
t_batch = time.perf_counter() - start
start = time.perf_counter()
looped = torch.stack([model.query(["A"], {"B": int(b)}) for b in evidence["B"][:256]])
t_loop = (time.perf_counter() - start) * 4096 / 256
print(f"query_batch: {t_batch * 1e3:.1f} ms; loop of query (extrapolated): {t_loop * 1e3:.0f} ms")
print(torch.allclose(batched[:256], looped, atol=1e-6))
```

## The memory guard

Before it allocates anything, variable elimination estimates the peak memory
of the query's elimination plan and compares it with a budget: on CUDA a
fraction of the live free memory, on the CPU and MPS the stable memory
capacity (RAM capped by the container's cgroup limit; Metal's recommended
working set).

- If one evidence row does not fit, `TensorVariableElimination` raises
  `torch.cuda.OutOfMemoryError` (the device-generic OOM type) before
  allocating; `HybridRouter` answers such a query with likelihood weighting
  instead.
- If one row fits but the whole batch does not, the batch is split into the
  largest row-chunks that fit. Chunking is exact: it changes only the speed.
- `TensorVariableElimination(memory_budget=bytes)` and
  `HybridRouter(memory_budget=bytes)` cap the budget further, for example on
  a shared machine.

The notebook {doc}`../examples/07_batching_devices_persistence` times
`query_batch` against a loop from 1 to 4096 queries and moves a model between
devices.
