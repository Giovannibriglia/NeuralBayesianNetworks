# SLURM launcher — paper-scale suite under a 24 h wall-time cap

European HPC centres cap single jobs at 24 h. This directory replaces the
single-allocation `benchmarking/run_all_benchmarks.sh` with **one independent
SLURM array chain per benchmark**, each resuming across jobs at **cell
granularity** (a cell = one problem × baseline, already subprocess-isolated).

## Quick start

```bash
# from the repo root, on the login node
bash benchmarking/slurm/submit_all_benchmarks.sh            # all benchmarks
bash benchmarking/slurm/submit_all_benchmarks.sh inference_speed   # a subset
N_RESTARTS=8 bash benchmarking/slurm/submit_all_benchmarks.sh      # longer chains
```

Before the first submit, edit the **EDIT ME** blocks in `nbn_bench.slurm`
(partition / qos / account / GPU constraint / module loads) for your cluster.

## How it works

Three layers, exactly as the cluster maintainers suggest:

1. **SLURM** — each benchmark is submitted as `--array=1-N%1`: N queued 24 h
   attempts, one running at a time. `--signal=B:USR1@900` delivers SIGUSR1 to
   the batch shell ~15 min before the limit.
2. **Batch shell** (`nbn_bench.slurm`) — traps SIGUSR1/SIGTERM and forwards
   them to the Python process; interprets its exit code (see contract below);
   drops a `DONE` marker when the benchmark finishes and cancels the pending
   attempts.
3. **Python** (`benchmarking/core/checkpoint.py`) — the CLI runs with
   `--results-dir benchmarking/results/slurm/<name> [--resume]`.
   The runner already streams every result row to `metrics.jsonl`; on top of
   that, each **fully completed cell** is recorded in a
   `completed_cells.jsonl` sidecar. On `--resume`, completed cells are
   skipped, and partial rows from the cell that was in flight when the
   previous job died are compacted out of `metrics.jsonl` first — so no cell
   is ever duplicated in the final parquet. On SIGUSR1/SIGTERM the runner
   finishes the in-flight cell, checkpoints, and exits.

No model/optimizer/RNG state needs serialising: a cell is recomputed from its
`(family, problem_id, seed, n_train)` identity deterministically, and the cell
is the atomic unit of work (bounded by `fit_timeout_s` + per-cell query
budgets). The cell key includes `n_train` because the learning-curves sweep
yields problems differing only in training-set size.

### Exit-code contract

| Python exit | Meaning                          | Batch script action                    |
|-------------|----------------------------------|----------------------------------------|
| `0`         | benchmark complete               | `touch DONE`, cancel pending attempts  |
| `124`       | preempted (SIGUSR1/SIGTERM)      | exit 0 → next attempt resumes          |
| other       | genuine failure                  | exit rc → next attempt retries anyway (crash-resume) |

Array attempts are **not** assumed to run in numeric order — the checkpoint,
not `SLURM_ARRAY_TASK_ID`, is the source of truth for what remains.

## Differences vs `run_all_benchmarks.sh`

* Each benchmark has the whole GPU allocation to itself (no `MAX_PARALLEL=3`
  VRAM sharing) and its own queue position, logs, and results dir.
* Results land in a **stable** dir per benchmark:
  `benchmarking/results/slurm/<name>/` (not a fresh timestamped dir), because
  restarts must find the previous state. Move/rename the dir to archive a run
  and start fresh.
* The local script remains untouched and is still the right tool on a
  workstation without wall-time limits.

## Monitoring & operations

```bash
squeue --me                                                  # queue state
wc -l benchmarking/results/slurm/*/completed_cells.jsonl     # cells done per benchmark
tail -f logs/nbn_inference_speed_*.out                       # live batch-shell log
tail -f benchmarking/results/slurm/inference_speed/run.log   # full INFO run log (appends across attempts)
```

* **Ran out of attempts?** Re-run `submit_all_benchmarks.sh` (optionally only
  the unfinished names) — finished benchmarks exit immediately on their
  `DONE` marker, unfinished ones resume.
* **Re-run a benchmark from scratch?** Remove (or archive) its
  `benchmarking/results/slurm/<name>/` dir and resubmit.
* **Resume manually (no SLURM)?**
  `python -m benchmarking.cli inference --config <yaml> --results-dir <dir> --resume`

## Testing the resume machinery locally

```bash
# start a smoke run into a pinned dir, Ctrl-C (or kill -USR1) it mid-run …
python -m benchmarking.cli inference \
  --config benchmarking/configs/synthetic/smoke_tests/inference_smoke.yaml \
  --results-dir /tmp/nbn_resume_test
# … then resume; already-completed cells are skipped:
python -m benchmarking.cli inference \
  --config benchmarking/configs/synthetic/smoke_tests/inference_smoke.yaml \
  --results-dir /tmp/nbn_resume_test --resume
```
