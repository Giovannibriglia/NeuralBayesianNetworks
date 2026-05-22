# Server benchmark run procedure

Covers launching and managing full paper-grade benchmark runs on a
remote server (or any machine that is not the development laptop).

## 1. Prerequisites

- **Python ≥ 3.10**
- **CUDA**: recommended (NBN mechanisms use GPU tensors). CPU-only runs
  work but are significantly slower for NBN baselines. pyro is
  CPU-forced regardless (see §8).
- **VRAM**: ≥ 8 GB. The configs cap `n_nodes` at 1000; n=5000 was
  dropped due to CUDA OOM at 7.6 GiB (noted in config comments).
- **Disk**: `benchmarking/results/` accumulates parquets, JSONL
  sidecars, figures, and tables. Allow ~500 MB per complete run.

## 2. Installation

Clone the repo and install with all required extras:

```bash
git clone https://github.com/Giovannibriglia/NeuralBayesianNetworks.git
cd NeuralBayesianNetworks
pip install -e ".[dev,bench,neural,gp,mcmc]"
```

The `gp` and `mcmc` extras install gpytorch and pyro respectively.
**All four extras (`bench`, `neural`, `gp`, `mcmc`) are required** for
paper-grade runs. Using only `.[dev,bench,neural]` silently omits
gpytorch and pyro — those cells emit `not_supported` rather than
erroring, so the omission is easy to miss.

## 3. Quick sanity check

```bash
python -c "import nbn, gpytorch, pyro, pgmpy, pomegranate; print('all imports ok')"
```

If any import fails, install the corresponding extra and re-check
before launching a multi-hour run.

## 4. Running the full paper benchmark

**All commands must be run from the repository root.** The configs use
`output_dir: benchmarking/results`, which is a relative path resolved
from wherever `nbn-bench` is invoked. Running from a subdirectory
will create output in the wrong location or fail.

```bash
cd /path/to/NeuralBayesianNetworks   # always from repo root

# Inference benchmark: ~15-20 h on a CUDA server (NBN baselines on GPU,
# pyro on CPU; pyro timeouts at n≥100 for discrete/continuous_lg/hybrid
# add ~3-5 h regardless of CUDA). ~30-50 h on a CPU-only server.
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
nohup nbn-bench inference \
  --config benchmarking/configs/inference_paper.yaml \
  > benchmarking/results/raw/inference_paper_${TIMESTAMP}.log 2>&1 &
echo "PID=$!"

# Parameter-learning benchmark (~6 h on CUDA)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
nohup nbn-bench param-learning \
  --config benchmarking/configs/parameter_learning_paper.yaml \
  > benchmarking/results/raw/parameter_learning_paper_${TIMESTAMP}.log 2>&1 &
echo "PID=$!"
```

Do not launch both simultaneously — they share VRAM and the combined
memory pressure may cause OOM at large n.

### Config selection

| Hardware | Inference config | Param-learning config |
|---|---|---|
| Server (≥16 GB VRAM or CPU-only) | `inference_paper.yaml` | `parameter_learning_paper.yaml` |
| Laptop (8 GB VRAM) | `inference_paper_laptop.yaml` | `parameter_learning_paper_laptop.yaml` |

The laptop variants lower `nbn_batch_size` (256 vs 1024) and
`batch_size` (1024 vs 4096) to fit 8 GB. Use the canonical
(`_paper.yaml`) variants on the server unless VRAM is limited.

**Output prefix collision:** both `inference_paper.yaml` and
`inference_paper_laptop.yaml` share `output_prefix: inference_paper`
and write to the same parquet. Only run one config per machine per run.

## 5. Monitoring progress

```bash
# Is it still running?
ps aux | grep "nbn-bench\|crash_test" | grep -v grep

# How many cells have completed?
wc -l benchmarking/results/raw/inference_paper_metrics.jsonl

# CUDA memory in use?
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

# Live log tail
tail -f benchmarking/results/raw/inference_paper_<TIMESTAMP>.log

# Current status breakdown from JSONL (works mid-run)
python3 -c "
import json
from collections import Counter
c = Counter()
with open('benchmarking/results/raw/inference_paper_metrics.jsonl') as f:
    for line in f:
        try:
            r = json.loads(line)
            c[r.get('status')] += 1
        except: pass
print(c)
"
```

## 6. Interruption and recovery

a. **The JSONL sidecar is the source of truth.** During a run, every
   completed cell is appended to `inference_paper_metrics.jsonl`
   immediately. The parquet is written only at natural completion.

b. **If the process is killed or crashes**, the JSONL is intact up to
   the last flushed cell. Reconstruct the parquet from it:

   ```python
   from pathlib import Path
   from benchmarking._crash_test_utils import jsonl_to_parquet
   jsonl_to_parquet(
       Path("benchmarking/results/raw/inference_paper_metrics.jsonl"),
       Path("benchmarking/results/raw/inference_paper_metrics.parquet"),
   )
   ```

c. **Killing the process is safe for the parquet.** `write_parquet` is
   called after the `try/finally` block; a SIGTERM does not reach it.
   The existing parquet on disk is left untouched.

d. **Restarting `nbn-bench` will re-run all cells from scratch**, NOT
   resume from where it left off. There is no skip/resume logic. The
   new run overwrites the parquet with only the current run's rows.

e. **To avoid losing in-progress data when relaunching**: archive the
   JSONL before restarting, then manually merge the JSONLs afterward:

   ```bash
   # Before killing / before relaunch
   cp benchmarking/results/raw/inference_paper_metrics.jsonl \
      benchmarking/results/raw/_archive_$(date +%Y%m%d_%H%M%S).jsonl

   # After both runs finish, merge and rebuild parquet
   cat benchmarking/results/raw/_archive_*.jsonl \
       benchmarking/results/raw/inference_paper_metrics.jsonl \
       > benchmarking/results/raw/_merged.jsonl

   python3 -c "
   from pathlib import Path
   from benchmarking._crash_test_utils import jsonl_to_parquet
   jsonl_to_parquet(
       Path('benchmarking/results/raw/_merged.jsonl'),
       Path('benchmarking/results/raw/inference_paper_metrics.parquet'),
   )
   "
   ```

   Note: `jsonl_to_parquet` does not deduplicate — if both runs cover
   the same cell, both rows will be present. Inspect for duplicates
   before using the merged parquet for figures.

## 7. Finalization

When the run completes naturally, the parquet and figures are written
automatically. Spot-check before committing results:

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('benchmarking/results/raw/inference_paper_metrics.parquet')
print('Total rows:', len(df))
print()
print(df.groupby(['family', 'baseline', 'status']).size().to_string())
"
```

Expected: `ok` rows for all applicable (family, baseline) pairs,
`not_supported` for non-applicable pairs, `timeout` acceptable for
pyro at large n (see §8). No `error` rows.

If the run was for parameter learning, substitute
`parameter_learning_paper_metrics.parquet`.

### Investigating error rows

If the status breakdown shows any `error` rows (not `oom`, `timeout`,
or `not_supported`), these are unexpected — `run_with_guard` caught
something the registry did not gate out. Inspect:

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('benchmarking/results/raw/inference_paper_metrics.parquet')
errs = df[df.status == 'error']
print(errs[['family', 'n_nodes', 'seed', 'baseline', 'metric', 'error_msg']].to_string())
"
```

Common causes: a baseline that should have been registry-gated but
wasn't, a config-loaded baseline that isn't installed, or a
device-specific issue (e.g., CUDA-only lstsq driver on a CPU-only
server). File an issue or check the run log before proceeding to
figures.

## 8. Known caveats

- **pyro is CPU-forced** via `device: cpu` in the YAML baseline entry.
  This is intentional: the Importance sampler's Python-bound loop makes
  GPU ~10× slower at benchmark scale (measured; see
  `docs/audits/v0.12-pyro-gpu-investigation.md`).

- **pyro timeouts at large n**: on CPU, pyro-empirical-importance takes
  ~95 s at n=10 and scales linearly with n. Cells at n ≥ 100 on
  discrete and continuous_lg will hit the 600 s timeout and emit
  `status=timeout` rows. This is expected and acceptable — it documents
  pyro's scalability limit.

- **pyro continuous_lg and hybrid at n=10/50**: ~95–360 s/cell on CPU,
  5 seeds each ≈ 8–30 min per (family, n_nodes) slice.

- **`hybrid` family**: only pyro (inference) and NBN-hybrid have
  applicable baselines. pgmpy, gpytorch, and pomegranate have no hybrid
  support. This is correct, not a gap.

- **`continuous_nongauss` family**: NBN-only (mdn-lw, flow-lw). No
  third-party baselines are applicable to non-Gaussian continuous
  families.

- **gpytorch accuracy is not supported**: gpytorch-gp-predict emits
  speed rows only (`accuracy_supported=False` in the registry). This is
  architectural — the SVGP baseline evaluates the local conditional
  at zero-parent values, not a posterior given evidence. Speed timing
  is still useful for comparison.

- **`nbn-cat-ve` OOM at n ≥ 50 on discrete**: VE factor tables grow
  exponentially with n. Expected; cells emit `status=oom`.
