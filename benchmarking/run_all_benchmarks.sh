#!/usr/bin/env bash
# =============================================================================
# run_all_benchmarks.sh — launch the paper-scale benchmark suite in parallel.
# =============================================================================
#
# Runs the six paper-scale benchmarks through the benchmarking CLI with a
# bounded worker pool: at most MAX_PARALLEL (default 3) run concurrently, and
# the next job launches as soon as any running one finishes (sliding window,
# not fixed waves of three).
#
# Each job's stdout/stderr is captured to a per-job launch log under
#   benchmarking/results/_parallel_runs/<timestamp>/<job>.log
# and the CLI additionally writes its own run.log + *_metrics.parquet into a
# fresh timestamped results dir (see benchmarking/core/output.make_results_dir).
#
# Usage (from anywhere — the script cd's to the repo root itself):
#   bash benchmarking/run_all_benchmarks.sh
#   MAX_PARALLEL=2 bash benchmarking/run_all_benchmarks.sh     # fewer in flight
#   DEVICE=cpu     bash benchmarking/run_all_benchmarks.sh     # force CPU
#
# NOTE on VRAM: three GPU benchmarks in flight can exceed an 8 GB card. If you
# hit CUDA OOM, drop to MAX_PARALLEL=2 (or 1), or set DEVICE=cpu for the run.
#
# Requires bash >= 4.3 (for `wait -n`).
# =============================================================================
set -uo pipefail

# ── always run from the repo root (configs + results paths are repo-relative)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MAX_PARALLEL="${MAX_PARALLEL:-3}"
DEVICE="${DEVICE:-auto}"

# ── pick a Python interpreter: prefer an active `python` (e.g. an activated
#    venv), then the repo's own .venv, then system python3. This lets the
#    script work via `bash benchmarking/run_all_benchmarks.sh` even when no
#    venv is activated, as long as the package is installed in .venv.
if command -v python >/dev/null 2>&1; then
  PY=python
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PY="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "ERROR: no python interpreter found (need python or python3 on PATH," >&2
  echo "       or a repo .venv). Install the package first:" >&2
  echo "       pip install -e \".[bench,neural,gp,mcmc]\"" >&2
  exit 1
fi
# `<py> -m benchmarking.cli` is equivalent to the `nbn-bench` console script
# but does not depend on the entry point being on PATH.
BENCH_CMD=("${PY}" -m benchmarking.cli)

# ── preflight: is the CLI importable / runnable?
if ! "${BENCH_CMD[@]}" --help >/dev/null 2>&1; then
  echo "ERROR: cannot run '${BENCH_CMD[*]}'. Install the package first:" >&2
  echo "       pip install -e \".[bench,neural,gp,mcmc]\"" >&2
  exit 1
fi

# ── per-launch log dir
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="benchmarking/results/_parallel_runs/${STAMP}"
mkdir -p "${LOG_DIR}"

# ── job table: "name|subcommand|config"
#   learning_curves and parameter_learning_complete declare metrics:
#   log_likelihood, so they MUST use the `param-learning` subcommand (the
#   `inference` command only accepts metrics 'all' | 'timing').
JOBS=(
  "inference_speed|inference|benchmarking/configs/synthetic/speed/inference_speed.yaml"
  "learning_curves|param-learning|benchmarking/configs/synthetic/learning_curves/learning_curves.yaml"
  "parameter_learning_complete|param-learning|benchmarking/configs/synthetic/complete/parameter_learning_complete.yaml"
  "inference_complete|inference|benchmarking/configs/synthetic/complete/inference_complete.yaml"
  "inference_scalability_complete|inference|benchmarking/configs/synthetic/complete/inference_scalability_complete.yaml"
  "bnlearn_inference_complete|inference|benchmarking/configs/bnlearn/complete/inference_complete.yaml"
)

echo "Repo root    : ${REPO_ROOT}"
echo "Max parallel : ${MAX_PARALLEL}"
echo "Device       : ${DEVICE}"
echo "Launch logs  : ${LOG_DIR}"
echo "Jobs queued  : ${#JOBS[@]}"
echo "============================================================"

launch() {
  local name="$1" sub="$2" cfg="$3"
  echo "[$(date +%H:%M:%S)] START  ${name}"
  (
    "${BENCH_CMD[@]}" "${sub}" --config "${cfg}" --device "${DEVICE}" \
      >"${LOG_DIR}/${name}.log" 2>&1
    rc=$?
    echo "${rc}" >"${LOG_DIR}/${name}.status"
    if [[ ${rc} -eq 0 ]]; then
      echo "[$(date +%H:%M:%S)] DONE   ${name} (exit 0)"
    else
      echo "[$(date +%H:%M:%S)] FAIL   ${name} (exit ${rc}) — see ${LOG_DIR}/${name}.log"
    fi
  ) &
}

# ── bounded worker pool: keep at most MAX_PARALLEL jobs in flight; launch the
#    next as soon as `wait -n` reports one has finished.
running=0
for entry in "${JOBS[@]}"; do
  IFS='|' read -r name sub cfg <<<"${entry}"
  while (( running >= MAX_PARALLEL )); do
    wait -n
    running=$(( running - 1 ))
  done
  launch "${name}" "${sub}" "${cfg}"
  running=$(( running + 1 ))
done

# drain the remaining jobs
wait

# ── summary (non-zero exit if any job failed)
echo "============================================================"
echo "Summary:"
fail=0
for entry in "${JOBS[@]}"; do
  IFS='|' read -r name _ _ <<<"${entry}"
  status_file="${LOG_DIR}/${name}.status"
  rc="$( [[ -f "${status_file}" ]] && cat "${status_file}" || echo '?' )"
  if [[ "${rc}" == "0" ]]; then
    printf '  OK    %s\n' "${name}"
  else
    printf '  FAIL  %s (exit %s)\n' "${name}" "${rc}"
    fail=1
  fi
done
echo "Launch logs : ${LOG_DIR}"
echo "Parquets    : benchmarking/results/benchmark_*_<timestamp>/*_metrics.parquet"
exit "${fail}"
