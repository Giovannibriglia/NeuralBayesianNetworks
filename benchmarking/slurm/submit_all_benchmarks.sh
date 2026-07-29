#!/usr/bin/env bash
# =============================================================================
# submit_all_benchmarks.sh — SLURM launcher for the paper-scale suite.
# =============================================================================
#
# The 24h-wall-time replacement for `bash benchmarking/run_all_benchmarks.sh`:
# instead of one shared allocation running six benchmarks through a worker
# pool, each benchmark gets its OWN independent SLURM array chain
# (`--array=1-N%1`: N sequential 24h attempts, one running at a time), and
# each attempt resumes the benchmark at cell granularity from
# benchmarking/results/slurm/<name>/ (see benchmarking/core/checkpoint.py).
#
# This also means each benchmark has the whole GPU to itself (the old
# MAX_PARALLEL=3 sharing is gone) and can be monitored / cancelled / re-run
# independently.
#
# Usage (from the repo root, on the cluster login node):
#   bash benchmarking/slurm/submit_all_benchmarks.sh                # all
#   bash benchmarking/slurm/submit_all_benchmarks.sh inference_speed learning_curves
#   N_RESTARTS=8 bash benchmarking/slurm/submit_all_benchmarks.sh   # longer chains
#   DEVICE=cpu    bash benchmarking/slurm/submit_all_benchmarks.sh
#
# Re-submitting is always safe: a finished benchmark exits immediately on its
# DONE marker, an unfinished one resumes from its checkpoint.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Sequential 24h attempts per benchmark. Over-provisioning is free (the DONE
# marker cancels pending attempts); under-provisioning just means submitting
# again later.
N_RESTARTS="${N_RESTARTS:-4}"

mkdir -p logs

# ── job table: "name|subcommand|config" — mirrors run_all_benchmarks.sh ─────
#   learning_curves and parameter_learning_complete declare metrics:
#   log_likelihood, so they MUST use the `param-learning` subcommand.
JOBS=(
  "inference_speed|inference|benchmarking/configs/synthetic/speed/inference_speed.yaml"
  "learning_curves|param-learning|benchmarking/configs/synthetic/learning_curves/learning_curves.yaml"
  "parameter_learning_complete|param-learning|benchmarking/configs/synthetic/complete/parameter_learning_complete.yaml"
  # "inference_complete|inference|benchmarking/configs/synthetic/complete/inference_complete.yaml"
  "inference_scalability_complete|inference|benchmarking/configs/synthetic/complete/inference_scalability_complete.yaml"
  "bnlearn_inference_complete|inference|benchmarking/configs/bnlearn/complete/inference_complete.yaml"
)

# Optional positional filter: submit only the named benchmarks.
FILTER=("$@")

submitted=0
for entry in "${JOBS[@]}"; do
  IFS='|' read -r name sub cfg <<<"${entry}"
  if (( ${#FILTER[@]} > 0 )); then
    keep=0
    for f in "${FILTER[@]}"; do
      [[ "${f}" == "${name}" ]] && keep=1
    done
    (( keep )) || continue
  fi
  if [[ ! -f "${cfg}" ]]; then
    echo "ERROR: config not found: ${cfg}" >&2
    exit 1
  fi
  jid=$(sbatch --parsable \
    --job-name="nbn_${name}" \
    --array="1-${N_RESTARTS}%1" \
    --export=ALL,BENCH_NAME="${name}",BENCH_SUB="${sub}",BENCH_CFG="${cfg}",DEVICE="${DEVICE:-auto}" \
    benchmarking/slurm/nbn_bench.slurm)
  echo "submitted ${name}: array job ${jid} (1-${N_RESTARTS}%1)"
  submitted=$(( submitted + 1 ))
done

if (( submitted == 0 )); then
  echo "Nothing submitted — no benchmark matched: $*" >&2
  exit 1
fi

echo "------------------------------------------------------------"
echo "Monitor  : squeue --me"
echo "Progress : wc -l benchmarking/results/slurm/*/completed_cells.jsonl"
echo "Logs     : logs/nbn_<name>_<arrayid>_<attempt>.out"
echo "Results  : benchmarking/results/slurm/<name>/<config>_metrics.parquet"
