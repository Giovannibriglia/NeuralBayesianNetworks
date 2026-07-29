#!/bin/bash
# =============================================================================
# nbn_bench.slurm — run ONE benchmark with checkpoint/resume across 24h jobs.
# =============================================================================
#
# One SLURM array chain per benchmark (submitted by submit_all_benchmarks.sh):
#
#   sbatch --job-name="nbn_<name>" --array=1-4%1 \
#          --export=ALL,BENCH_NAME=<name>,BENCH_SUB=<subcommand>,BENCH_CFG=<yaml> \
#          benchmarking/slurm/nbn_bench.slurm
#
# `--array=1-4%1` queues 4 sequential attempts of which ONE runs at a time.
# Each attempt resumes from the shared results dir
#   benchmarking/results/slurm/<BENCH_NAME>/
# at CELL granularity (completed_cells.jsonl, see benchmarking/core/checkpoint.py).
# When an attempt finishes the whole benchmark (exit 0) it drops a DONE marker
# and cancels the remaining pending attempts, so over-provisioning the array
# is free. If 4x24h turns out not to be enough, just submit the same command
# again — it picks up where the chain left off.
#
# Exit-code contract with the Python CLI:
#   0   -> benchmark complete            -> touch DONE, scancel pending siblings
#   124 -> preempted (SIGUSR1/SIGTERM)   -> exit 0; next attempt resumes
#   *   -> genuine failure               -> exit rc; next attempt still retries
#                                           from the checkpoint (crash-resume)
#
# `--signal=B:USR1@900` asks SLURM to send SIGUSR1 to the BATCH SHELL ~15 min
# before the wall-time limit (can arrive up to 60 s early). The trap below
# forwards it to the Python process, which finishes the in-flight cell,
# checkpoints, and exits 124. Cells are subprocess-isolated and bounded by
# fit_timeout_s + per-cell query budgets, so 15 min is normally enough to
# drain the in-flight cell; if it is not, the hard SIGKILL only loses that
# one cell — its partial rows are compacted away on the next resume.
#
# =============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# ── site-specific: EDIT ME for your cluster ─────────────────────────────────
#SBATCH --partition=all_usr_prod
#SBATCH --qos=qos_biz_lprod
#SBATCH --account=pnrr_fair_zambonelli
#SBATCH --constraint="gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
# ────────────────────────────────────────────────────────────────────────────

set -uo pipefail

: "${BENCH_NAME:?export BENCH_NAME (e.g. inference_speed) via sbatch --export}"
: "${BENCH_SUB:?export BENCH_SUB (inference | param-learning) via sbatch --export}"
: "${BENCH_CFG:?export BENCH_CFG (path to the YAML config) via sbatch --export}"
DEVICE="${DEVICE:-auto}"

echo "Job ID       : ${SLURM_JOB_ID:-?} (array ${SLURM_ARRAY_JOB_ID:-none}, attempt ${SLURM_ARRAY_TASK_ID:-none})"
echo "Node         : ${SLURMD_NODENAME:-?}"
echo "Benchmark    : ${BENCH_NAME} (${BENCH_SUB}, ${BENCH_CFG})"

# ── run from the repo root (configs + results paths are repo-relative) ──────
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

RESULTS_DIR="benchmarking/results/slurm/${BENCH_NAME}"
DONE_MARKER="${RESULTS_DIR}/DONE"

# A previous attempt already finished the whole benchmark: nothing to do.
# (Array tasks are not guaranteed to run in numeric order, so the checkpoint
# — not the task id — is the source of truth for what remains.)
if [[ -f "${DONE_MARKER}" ]]; then
  echo "Benchmark already complete (${DONE_MARKER}); exiting."
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}" 2>/dev/null || true
  fi
  exit 0
fi

# ── site modules: EDIT ME for your cluster ──────────────────────────────────
if command -v module >/dev/null 2>&1; then
  module load cuda/12.6.3-none-none
  module load cudnn/8.9.7.29-12-none-none-cuda-12.6.3
  module load python/3.11.11-gcc-11.4.0
  module list
fi

echo "Checking GPU..."
nvidia-smi || true

# ── pick a Python interpreter (same fallback chain as run_all_benchmarks.sh)
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi

mkdir -p "${RESULTS_DIR}"

# Resume when a previous attempt already wrote rows into this dir.
RESUME_FLAG=()
if [[ -f "${RESULTS_DIR}/metrics.jsonl" ]]; then
  RESUME_FLAG=(--resume)
  echo "Resuming from existing checkpoint in ${RESULTS_DIR}"
fi

# ── launch, forwarding SLURM's early-warning signal to Python ───────────────
# The CLI installs SIGUSR1/SIGTERM handlers that stop the cell loop on the
# next cell boundary and exit 124. --signal=B:USR1@900 only signals this
# batch shell, so the trap forwards it to the child.
"${PY}" -m benchmarking.cli "${BENCH_SUB}" \
  --config "${BENCH_CFG}" \
  --device "${DEVICE}" \
  --results-dir "${RESULTS_DIR}" \
  "${RESUME_FLAG[@]}" &
pid=$!
trap 'echo "[$(date +%H:%M:%S)] wall-time warning — forwarding to python"; kill -USR1 "${pid}" 2>/dev/null' USR1 TERM

# `wait` returns >128 when interrupted by the trap; re-wait until the child
# has actually exited so we read its real exit code.
wait "${pid}"; rc=$?
while (( rc > 128 )) && kill -0 "${pid}" 2>/dev/null; do
  wait "${pid}"; rc=$?
done

echo "[$(date +%H:%M:%S)] python exited with rc=${rc}"

if (( rc == 0 )); then
  touch "${DONE_MARKER}"
  echo "Benchmark COMPLETE — cancelling remaining attempts."
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}" 2>/dev/null || true
  fi
  exit 0
elif (( rc == 124 )); then
  echo "Preempted at wall-time limit — next attempt will resume."
  exit 0
else
  echo "FAILED (rc=${rc}) — next attempt will retry from the checkpoint."
  exit "${rc}"
fi
