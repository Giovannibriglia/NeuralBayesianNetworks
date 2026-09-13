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
#   results/_parallel_runs/<timestamp>/<job>.log
# and the CLI additionally writes its own run.log + *_metrics.parquet into a
# fresh timestamped results dir (see nbn/bench/core/output.make_results_dir).
#
# Usage (from anywhere — the script cd's to the repo root itself):
#   bash scripts/run_all_benchmarks.sh
#   MAX_PARALLEL=2 bash scripts/run_all_benchmarks.sh     # fewer in flight
#   DEVICE=cpu     bash scripts/run_all_benchmarks.sh     # force CPU
#
# NOTE on VRAM: three GPU benchmarks in flight can exceed an 8 GB card. If you
# hit CUDA OOM, drop to MAX_PARALLEL=2 (or 1), or set DEVICE=cpu for the run.
#
# NOTE on RAM / where to launch from: run long jobs from a plain terminal
# (or under nohup / tmux), NOT from an IDE's embedded terminal. On a systemd
# desktop the IDE and every process it spawned share one cgroup; when the
# post-run JSONL -> parquet step (or the IDE itself) pushes that cgroup into
# memory pressure, systemd-oomd kills the WHOLE scope -- the IDE and the
# benchmark together -- with no traceback in run.log (2026-09-12 batch_speed
# run: 20 h of cells completed, then killed during conversion).
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
#    script work via `bash scripts/run_all_benchmarks.sh` even when no
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
  echo "       pip install -e \".[all]\"" >&2
  exit 1
fi
# `<py> -m nbn.bench.cli` is equivalent to the `nbn-bench` console script
# but does not depend on the entry point being on PATH.
BENCH_CMD=("${PY}" -m nbn.bench.cli)

# ── never let a stale copy in ~/.local shadow the environment's packages
#    (a pre-1.0 pgmpy there turned every discrete pgmpy cell of the
#    2026-09-10 run into not_supported).
export PYTHONNOUSERSITE=1

# ── preflight: is the CLI importable, and are the libraries the adapters
#    need installed at the right versions? (`nbn-bench check-env`)
if ! "${BENCH_CMD[@]}" --help >/dev/null 2>&1; then
  echo "ERROR: cannot run '${BENCH_CMD[*]}'. Install the package first:" >&2
  echo "       pip install -e \".[all]\"" >&2
  exit 1
fi
if ! "${BENCH_CMD[@]}" check-env; then
  echo "ERROR: environment check failed (see table above). Install the" >&2
  echo "       missing/outdated libraries: pip install -U -e \".[all]\"" >&2
  exit 1
fi

# ── per-launch log dir
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/_parallel_runs/${STAMP}"
mkdir -p "${LOG_DIR}"

# ── job table: "name|subcommand|config"
#   learning_curves and parameter_learning_complete declare metrics:
#   log_likelihood, so they MUST use the `param-learning` subcommand (the
#   `inference` command only accepts metrics 'all' | 'timing').
JOBS=(
  "inference_speed|inference|nbn/bench/configs/synthetic/speed/inference_speed.yaml"
  "learning_curves|param-learning|nbn/bench/configs/synthetic/learning_curves/learning_curves.yaml"
  "parameter_learning_complete|param-learning|nbn/bench/configs/synthetic/complete/parameter_learning_complete.yaml"
  # "inference_complete|inference|nbn/bench/configs/synthetic/complete/inference_complete.yaml"
  "inference_scalability_complete|inference|nbn/bench/configs/synthetic/complete/inference_scalability_complete.yaml"
  "bnlearn_inference_complete|inference|nbn/bench/configs/bnlearn/complete/inference_complete.yaml"
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
echo "Parquets    : results/benchmark_*_<timestamp>/*_metrics.parquet"
exit "${fail}"
