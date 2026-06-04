# Server benchmark run procedure (v0.13)

Covers launching and managing full paper-grade benchmark runs on a
remote server (or any machine that is not the development laptop).

## 1. Prerequisites

- **Python ≥ 3.10**
- **CUDA**: recommended (NBN mechanisms use GPU tensors). CPU-only runs
  work but are significantly slower for NBN baselines. pyro is
  CPU-forced regardless (see §8).
- **VRAM**: ≥ 8 GB. The configs cap `n_nodes` at 1000; n=5000 was
  dropped due to CUDA OOM at 7.6 GiB (noted in config comments).
- **Disk**: `results/` accumulates parquets, JSONL sidecars, figures,
  and tables. Allow ~500 MB per complete run.

## 2. Installation

Clone the repo and install with all required extras:

```bash
git clone https://github.com/Giovannibriglia/NeuralBayesianNetworks.git
cd NeuralBayesianNetworks
python -m venv .venv
source .venv/bin/activate
```

Install torch for your platform first, then the package:

```bash
pip install -e ".[dev,bench,neural,gp,mcmc]"
```

The `gp` and `mcmc` extras install gpytorch and pyro respectively.
**All four extras (`bench`, `neural`, `gp`, `mcmc`) are required** for
paper-grade runs. Using only `.[dev,bench,neural]` silently omits
gpytorch and pyro — those cells emit `not_supported` rather than
erroring, so the omission is easy to miss.

## 3. Quick sanity check

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

```bash
python -c "import nbn, gpytorch, pyro, pgmpy, pomegranate; print('all imports ok')"
```

If any import fails, install the corresponding extra and re-check
before launching a multi-hour run.

## 4. Running the full paper benchmark

**All commands must be run from the repository root.**

```bash
cd /path/to/NeuralBayesianNetworks   # always from repo root

# Inference benchmark: ~15-20 h on a CUDA server (NBN baselines on GPU,
# pyro on CPU; pyro timeouts at n≥100 for discrete/continuous_lg/hybrid
# add ~3-5 h regardless of CUDA). ~30-50 h on a CPU-only server.
nohup nbn-bench inference \
  --config benchmarking/configs/synthetic/complete/inference_complete.yaml \
  > /tmp/inference_complete.log 2>&1 &
echo "PID=$!"
```

Do not launch both inference and parameter-learning simultaneously —
they share VRAM and the combined memory pressure may cause OOM at large n.

**Note:** `param-learning` is not yet implemented in v0.13. Use
`nbn-bench inference` for now (see issue #109).

### Config selection

| Hardware | Inference config |
|---|---|
| Any (server or laptop) | `synthetic/complete/inference_complete.yaml` |

### Output location

v0.13 output goes to `results/benchmark_synthetic_<config_name>_<timestamp>/`.
The directory is created automatically; `--config` does not need an
`output_dir` field.

## 5. Monitoring progress

```bash
# Is it still running?
ps aux | grep "nbn-bench" | grep -v grep

# How many cells have completed?
wc -l results/benchmark_synthetic_paper_*/metrics.jsonl

# CUDA memory in use?
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

# Current status breakdown from JSONL (works mid-run)
python3 -c "
import json
from collections import Counter
import glob
jsonl = sorted(glob.glob('results/benchmark_synthetic_paper_*/metrics.jsonl'))[-1]
c = Counter()
with open(jsonl) as f:
    for line in f:
        try:
            r = json.loads(line)
            c[r.get('status')] += 1
        except: pass
print(c)
"
```

## 6. Interruption and recovery

a. **The JSONL file is the source of truth.** During a run, every
   completed cell is appended to `metrics.jsonl` immediately
   (line-buffered). The parquet is written only at natural completion.

b. **If the process is killed or crashes**, the JSONL is intact up to
   the last flushed cell. Reconstruct the parquet from it:

   ```python
   from pathlib import Path
   from benchmarking.core.output import jsonl_to_parquet
   jsonl_to_parquet(
       Path("results/benchmark_synthetic_paper_<timestamp>/metrics.jsonl"),
       Path("results/benchmark_synthetic_paper_<timestamp>/metrics.parquet"),
   )
   ```

c. **Restarting `nbn-bench` re-runs all cells from scratch**. There is
   no skip/resume logic. A fresh run creates a new timestamped directory.

d. **To avoid losing in-progress data when relaunching**: copy the
   JSONL before restarting, then manually merge:

   ```bash
   cp results/benchmark_synthetic_paper_<ts1>/metrics.jsonl /tmp/run1.jsonl

   # After both runs finish, merge and rebuild parquet
   cat /tmp/run1.jsonl results/benchmark_synthetic_paper_<ts2>/metrics.jsonl \
       > /tmp/merged.jsonl

   python3 -c "
   from pathlib import Path
   from benchmarking.core.output import jsonl_to_parquet
   jsonl_to_parquet(Path('/tmp/merged.jsonl'), Path('/tmp/merged.parquet'))
   "
   ```

   Note: `jsonl_to_parquet` does not deduplicate — if both runs cover
   the same cell, both rows will be present. Inspect for duplicates
   before using the merged parquet for figures.

## 7. Finalization

When the run completes naturally, the parquet and aggregated figures are
written automatically. Spot-check before committing results:

```bash
python3 -c "
import pandas as pd, glob
parquet = sorted(glob.glob('results/benchmark_synthetic_paper_*/metrics.parquet'))[-1]
df = pd.read_parquet(parquet)
print('Total rows:', len(df))
print()
print(df.groupby(['family', 'baseline', 'status']).size().to_string())
"
```

Expected: `ok` rows for all applicable (family, baseline) pairs,
`not_supported` for non-applicable pairs, `timeout` acceptable for
pyro at large n (see §8). No `error` rows.

### Investigating error rows

If the status breakdown shows any `error` rows (not `oom`, `timeout`,
or `not_supported`), inspect:

```bash
python3 -c "
import pandas as pd, glob
parquet = sorted(glob.glob('results/benchmark_synthetic_paper_*/metrics.parquet'))[-1]
df = pd.read_parquet(parquet)
errs = df[df.status == 'error']
print(errs[['family', 'n_nodes', 'seed', 'baseline', 'metric', 'error_msg']].to_string())
"
```

Common causes: a baseline that should have been registry-gated but
wasn't, a config-loaded baseline that isn't installed, or a
device-specific issue.

## 8. Known caveats

- **pyro is CPU-forced** via `device: cpu` in the YAML baseline entry.
  This is intentional: the Importance sampler's Python-bound loop makes
  GPU ~10× slower at benchmark scale (measured; see
  `docs/audits/v0.12-pyro-gpu-investigation.md`).

- **pyro timeouts at large n**: on CPU, pyro-empirical-importance takes
  ~95 s at n=10 and scales linearly with n. Cells at n ≥ 100 on
  discrete and continuous_lg will hit the 600 s timeout and emit
  `status=timeout` rows. This is expected and acceptable.

- **`hybrid` family**: only pyro (inference) and NBN-hybrid have
  applicable baselines. pgmpy, gpytorch, and pomegranate have no hybrid
  support. This is correct, not a gap.

- **`continuous_nongauss` family**: NBN-only (mdn-lw, flow-lw). No
  third-party baselines are applicable to non-Gaussian continuous
  families.

- **gpytorch accuracy is not supported**: gpytorch-gp-predict emits
  speed rows only (`accuracy_supported=False` in the registry). This is
  architectural — the SVGP baseline evaluates the local conditional
  at zero-parent values, not a posterior given evidence.

- **`nbn-cat-ve` OOM at n ≥ 50 on discrete**: VE factor tables grow
  exponentially with n. Expected; cells emit `status=oom`.

## 9. Optional: inference scalability run

**Purpose:** different question from the paper benchmark. Instead of
"how accurate at fixed scales?", this benchmark asks "at what n_nodes
does each baseline's full cell pipeline overflow the 60s budget?"

**Launch:**

```bash
nohup nbn-bench inference \
  --config benchmarking/configs/synthetic/complete/inference_scalability_complete.yaml \
  > /tmp/inference_scalability_complete.log 2>&1 &
echo "PID $!"
```

**Expected output:** `results/benchmark_synthetic_scalability_<ts>/metrics.jsonl`
with rows spanning `status ∈ {ok, timeout, oom, not_supported}`.
Most baselines produce `ok` at n=5–100 and `timeout` from n=200–500.
