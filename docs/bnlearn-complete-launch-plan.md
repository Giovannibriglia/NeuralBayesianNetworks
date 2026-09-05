# bnlearn_complete launch plan

**Target run:** paper §5 critical-path data — full bnlearn networks × 5 seeds × 12 baselines
**Config:** `nbn/bench/configs/bnlearn/complete/inference_complete.yaml`
**Expected duration:** 2–3 days wall-clock
**Author:** Giovanni
**Date:** 2026-06-05 (post-v0.13 cutover, post-#147 GPU autodetect)

---

## Why this doc exists

The bnlearn_complete launch has been deferred for four days. The first attempt (2026-06-03) died at hour 7 — silent stop on `munin1`, caused by two simultaneous bugs:

1. **URL bug** (`bnlearn.py:_bif_url`) — `munin1/2/3` networks live at `/munin4/` directory on bnlearn.com, but the URL builder didn't know
2. **No exception handler** — `iter_problems` had no try/except, traceback went to stderr (not `run.log`), runner stopped silently

Both fixed in PR #145. PR #147 also fixed a related bug (GPU baselines silently running on CPU).

**This doc exists so the next launch doesn't fail in a new silent way.** It captures pre-launch checks, monitoring plan, abort criteria, and recovery procedures for this config and this network history. Works for laptop or server.

---

## 1. Pre-launch checklist

Run through this top-to-bottom before launching. If anything fails, stop and investigate — do not launch with items unchecked.

**Hardware state:**
- [ ] GPU memory clear: `nvidia-smi --query-gpu=memory.used,memory.free --format=csv` shows < 500 MiB used (no leftover processes from prior runs)
- [ ] Disk space: `df -h .` shows >50 GB free (run produces ~3 GB; safety margin for run.log + JSONL + parquet)
- [ ] RAM headroom: `free -h` shows >40 GB available

**Software state:**
- [ ] Master branch up-to-date: `git status` clean, `git log --oneline -1` shows expected SHA
- [ ] All required v0.13 PRs merged: #145 (runner robustness), #146 (config cleanup), #147 (GPU autodetect)
- [ ] CUDA visible: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` → `True N` (N ≥ 1)
- [ ] No stale `nbn-bench` processes: `ps aux | grep nbn-bench | grep -v grep` is empty
- [ ] Scalability benchmark from 2026-06-04 has **finished** (we chose to wait, not run concurrently)

**Session persistence:**
- [ ] Running inside a persistent session: tmux (laptop) or screen (server). `nohup` with logfile is acceptable but a multiplexer is preferred (lets you re-attach, see live tqdm).
- [ ] If using tmux: `tmux new -s bnlearn`, verify `echo $TMUX` returns a non-empty value inside the session
- [ ] If using screen: `screen -S bnlearn`, verify `echo $STY` returns a non-empty value inside the session
- [ ] **Host stays awake for the run's expected duration** — no suspend/sleep/power-down (your responsibility per machine)

**Config state:**
- [ ] Config file inspected: `cat nbn/bench/configs/bnlearn/complete/inference_complete.yaml`
- [ ] `pyro` is pinned to `cpu` (per PR #102 finding — Importance sampler is 11× slower on GPU)
- [ ] `fit_timeout_s` and `per_cell_timeout_s` are reasonable for paper-scale (defaults per #143)
- [ ] Output directory is `results/` (the default; the runner creates a timestamped subdirectory)

**Logging:**
- [ ] Will launch with `-v` (verbose) so `run.log` captures enough for post-mortem


## 2. Expected timeline & shape of the run

**Total cells:** 31 networks × 5 seeds × 12 baselines = **1,860 cells**

**Per-cell time budget** (from config):
- `fit_timeout_s = 1200` — max 20 min to fit a model on a network
- `per_cell_timeout_s = 256` and `n_queries_per_cell = 256` — up to ~256s of total query time per cell (~1s/query budget)
- Theoretical worst case per cell: ~24 minutes if every timeout maxes out

**Theoretical envelope:** `1,860 × 24 min = ~31 days` if every cell timed out. That's the outer bound, not the expectation. Real run expected to finish in **2-3 days** because:
- Small/medium networks (asia, cancer, earthquake, alarm, child, …) finish in seconds per cell
- Large networks (munin1-4, link, pathfinder, andes) take longer; some baselines hit timeouts
- Pyro is the slowest baseline by far (CPU-pinned per #102); it dominates wall-clock on networks where it doesn't time out

**Expected shape:**
- First few hours: small networks complete (the partial 2026-06-03 run made it through asia/cancer/earthquake before hitting the URL bug at hour 7)
- Middle of the run: medium-to-large networks (alarm through hailfinder) — bulk of the work
- End of the run: massive networks (link, pathfinder, munin1-4, diabetes, pigs) — most timeouts/OOMs concentrated here

**These estimates are NOT from a prior complete bnlearn run.** No complete bnlearn run exists yet — the 2026-06-03 attempt died at hour 7 from the URL bug. **Refine this section after the first successful complete run.**

## 3. Launch procedure

Follow these steps in order. Each step is a single command unless noted.

```bash
# Step 1: Confirm you're in the right working directory
cd <path-to-repo>/NeuralBayesianNetworks
git status                                     # should be clean
git log --oneline -1                           # confirm expected SHA

# Step 2: Start the persistent session
tmux new -s bnlearn                            # laptop
# OR
screen -S bnlearn                              # server

# Step 3: Inside the session, activate venv and confirm CUDA
source .venv/bin/activate
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"

# Step 4: Final pre-launch check (paste output into a note somewhere if useful)
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
df -h .
date

# Step 5: Launch
nbn-bench inference \
    --config nbn/bench/configs/bnlearn/complete/inference_complete.yaml \
    --device auto \
    -v

# Step 6: Detach
# tmux:   Ctrl-B then D
# screen: Ctrl-A then D

# Step 7: Verify the process is still running AFTER detach
ps aux | grep nbn-bench | grep -v grep
# Should show the PID. If output is empty, the run died on detach — investigate before relaunching.
```

**Re-attach later:**
- tmux: `tmux attach -t bnlearn`
- screen: `screen -r bnlearn`

**Note on `--device auto`:** per PR #147, `auto` resolves to CUDA-if-available-else-CPU per adapter, with pyro pinned to CPU regardless. This is the intended behavior for the complete run.

## 4. Abort criteria

Kill the run if any of these is true.

**Hard kill (immediate):**
- The process has been "stuck" on a single cell for > `fit_timeout_s + per_cell_timeout_s + 10 min` (~35 minutes). Means the timeout machinery failed. Investigate before relaunching.
- Same network name in tqdm for hours with no progression to other baselines on that network. Cell is hung past the safety net.
- Disk fills up (`df -h .` < 5 GB free). Killing now is better than letting it crash mid-write.
- GPU enters a bad state (`nvidia-smi` shows all memory locked, kernel timeouts, or unidentified processes). Driver-level issue, restart needed.

**Soft kill (consider over the next hour):**
- Same error repeats > 50 times in `run.log` across different cells. Pattern suggests a systematic issue, not isolated cell failures.
- Process exists in `ps aux` but tqdm has been silent for > 30 min and `metrics.jsonl` row count isn't growing. Process alive but not making progress.

**Do NOT kill if:**
- A single cell hits `timeout` or `oom` — that's expected; PR #145 records it as a status row and continues.
- The run is "slow" — pyro on large networks is genuinely slow; slow ≠ broken.
- You see `not_supported` rows — that's expected for baselines outside their applicability.

**How to kill:**

```bash
# Find the PID
ps aux | grep nbn-bench | grep -v grep

# Graceful kill first (let the run flush JSONL and write partial parquet if possible)
kill <PID>
sleep 30

# Verify it died
ps aux | grep nbn-bench | grep -v grep

# If still alive after 30s, force-kill
kill -9 <PID>
```

**After a kill, before relaunching:**
- Read the last 200 lines of `run.log` to understand why
- Confirm the partial JSONL exists and is readable (`wc -l metrics.jsonl`)
- Rename or move the partial run dir if you intend to relaunch with the same config (so the new run gets a fresh timestamped dir)
- Do NOT relaunch without understanding what happened — that's how you get the same failure twice

## 5. Known expected outcomes

**These are normal and expected. Do not panic.**

**OOM rows on large networks:**
- Expected on networks with > 200 nodes: `link`, `pathfinder`, `munin1-4`, `diabetes`, `pigs`, `andes`.
- Most likely on nbn baselines that use GPU (8 GB GPU memory caps batch size on large networks).
- Recorded as `status=oom` per PR #127 / #145 taxonomy. Run continues.

**Timeout rows on large networks:**
- Expected on pyro-empirical-importance for any non-trivial network (pyro is CPU-pinned per #102; Importance sampler is 11× slower than CPU pgmpy on small networks, even worse at scale).
- Expected on pomegranate for the largest networks (the scalability data showed pomegranate hitting > 1000s at n=10000).
- Expected on pgmpy variants for the largest networks (variance bands in the scalability run show wild timing variability past n=1000).
- Recorded as `status=timeout`. Run continues.

**`not_supported` rows for baseline/family mismatches:**
- `nbn-flow-lw`, `nbn-mdn-lw`, `nbn-lg-lw` → all `not_supported` on discrete networks (continuous-only mechanisms).
- `nbn-hybrid-router` → `not_supported` on any pure-family network (hybrid-only by design).
- `pgmpy-mle-predict` → `not_supported` on discrete networks (continuous-only path).
- Recorded as `status=not_supported`. Expected.

**`nbn-neuralcat-lw` accuracy will be `not_supported`:**
- Per issue #153, `nbn-neuralcat-lw` silently fails to produce accuracy metrics (missing `_class_values` on `NeuralCategorical`).
- Timing rows for this baseline will be `ok`; accuracy rows will be `not_supported`.
- Known v0.13 bug; do not be surprised by it in the output.
- Affected cells can be re-run after #153 is fixed.

**Sentinel rows have `query_role='random'`:**
- Per issue #154, sentinel rows from `_not_supported_sentinel` / `_fit_failure_rows` / `_rows_to_cellresults` are labeled `query_role='random'` (incorrect placeholder).
- All `metric='status'` rows have `query_role='random'`; there's a 1:1 correspondence.
- Downstream analyzers must filter `metric=='status'` before doing role decompositions.
- Known v0.13 cleanup; data is correct, just labeled with a confusing placeholder.

**munin1/2/3 download:**
- These networks live at `/munin4/` on bnlearn.com (quirk of the upstream archive).
- PR #145's URL override (`_URL_DIRECTORY_OVERRIDES`) handles this; downloads should succeed without manual intervention.
- If you see a download failure for munin1-3 specifically, suspect the URL override has regressed.

**Pyro and pomegranate device choices:**
- Pyro is config-pinned to cpu (correct).
- Pomegranate auto-detects to cuda per #147 (correct for the discrete inference path).

## 6. Post-launch verification

**Immediately after the run completes** (tqdm reaches 100% or you see "done" in the log):

```bash
RUNDIR=$(ls -dt results/benchmark_bnlearn_* | head -1)
echo "Verifying: $RUNDIR"

# 1. Process exited cleanly
ps aux | grep nbn-bench | grep -v grep
# Expected: empty (process is done)

# 2. Both output files exist
ls -lh $RUNDIR/
# Expected: metrics.jsonl, metrics.parquet, run.log all present
# (parquet is written once at the end — if absent, the run was killed before completing)

# 3. Row count sanity check
python3 -c "
import pandas as pd
df = pd.read_parquet('$RUNDIR/metrics.parquet')
print(f'rows: {len(df):,}')
print(f'cells (unique problem×seed×baseline): {df.groupby([\"problem_id\",\"seed\",\"baseline\"]).ngroups}')
print(f'expected cells: 1,860')
"
# Expected: cells = 1,860

# 4. Status distribution
python3 -c "
import pandas as pd
df = pd.read_parquet('$RUNDIR/metrics.parquet')
print(df.groupby('status').size())
"
# Expected: ok dominates; not_supported / timeout / oom present in modest quantities

# 5. Per-baseline ok-rate sanity
python3 -c "
import pandas as pd
df = pd.read_parquet('$RUNDIR/metrics.parquet')
ok_rate = (df.groupby('baseline')
             .apply(lambda g: (g['status']=='ok').mean())
             .sort_values(ascending=False))
print(ok_rate)
"
# Expected: nbn-cat-lw, nbn-cat-ve near 100%
# pyro, pomegranate likely 70-95%
# nbn-flow-lw / nbn-mdn-lw / nbn-lg-lw will be ~0% on discrete-only families (expected, not a bug)

# 6. Check for unexpected errors
grep -i "traceback\|critical\|unexpected" $RUNDIR/run.log | head -20
# Expected: empty, or only known-issue tracebacks

# 7. Confirm no networks were silently skipped
python3 -c "
import pandas as pd
df = pd.read_parquet('$RUNDIR/metrics.parquet')
print(sorted(df['problem_id'].unique()))
print(f'network count: {df[\"problem_id\"].nunique()}')
"
# Expected: all 31 bnlearn networks present
```

**If all of the above pass**, the run is a paper-quality success.

**If any fail**, before declaring the run usable:
- Note which check failed.
- Identify whether it's a recoverable issue (some baselines failed but data is otherwise complete) or a fundamental issue (rows missing, fewer networks than expected).
- If recoverable: proceed with paper-figure generation, note the caveat.
- If fundamental: investigate the run.log, decide whether to use partial data or relaunch.

**Then plot:**

```bash
nbn-bench plot $RUNDIR/metrics.parquet --output-dir $RUNDIR/figures
```

Compare the figures against the synthetic scalability run from 2026-06-04 as a sanity reference (different benchmark, but similar shape — NBN should still be fast, accuracy still flat, status taxonomy still clean).

---

## Summary

This doc captures the launch plan for bnlearn_complete. Pre-launch checklist, expected timeline, launch procedure, abort criteria, known expected outcomes, and post-launch verification. Refine after the first successful complete run — especially Section 2's timeline numbers.

