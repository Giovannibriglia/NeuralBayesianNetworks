#!/usr/bin/env bash
# Round 3: does NBN use a larger budget?  (a) CPU threads 1-8, branch vs master;
# (b) GPU (MPS) vs CPU; (c) frontier under a large budget (8 threads, 16 GB, 120 s)
# vs the reference budget (1 thread, 4 GB, 30 s; rounds 1-2).
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
R=benchmarks/scaling/results
M=${NBN_MASTER:-../NeuralBayesianNetworks}
export PYTHONPATH=$PWD
run() { $PY benchmarks/scaling/run.py "$@" --slack 60; }

# cells: sweep axis topology size method      (sizes where 1 thread takes ~1-20 s)
BRANCH_CELLS=(
  "discrete_infer deg er 3 nbn_ve"
  "discrete_infer deg er 2 pgmpy_ve"
  "discrete_infer n local 1000 nbn_lw"
  "discrete_infer_cert n local 500 nbn_cert"
  "discrete_fit n er 5000 nbn"
  "discrete_fit n er 5000 pgmpy"
  "discrete_sets deg er 8 nbn_intervals_batched"
  "cont_fit n er 500 nbn_stacked"
  "cont_fit n er 100 pyro_svi"
  "cont_infer n er 5000 nbn_lw"
  "cont_infer n er 1000 pyro_vec"
  "r2_fit_ncat n er 1000 ncat_stacked"
  "r2_fit_ncat n er 500 pyro_ncat"
  "r2_fit_flex n er 1000 flex_stacked"
  "r2_fit_flow n er 200 flow_stacked"
  "r2_knn_query n er 100 knn_lw"
  "r2_flex_query n er 1000 flex_lw"
  "r2_ais n local 1000 ais"
  "r2_avi n er 500 avi"
)
MASTER_CELLS=(
  "discrete_infer deg er 3 nbn_ve"
  "discrete_infer n local 1000 nbn_lw"
  "discrete_fit n er 5000 nbn"
  "cont_fit n er 200 nbn_local"
  "r2_fit_ncat n er 500 ncat_local"
  "r2_fit_flex n er 200 flex_local"
  "r2_fit_flow n er 100 flow_local"
  "r2_knn_query n er 10 knn_lw"
  "r2_flex_query n er 50 flex_lw"
  "r2_ais n local 1000 ais"
  "r2_avi n er 500 avi"
)
cell() {  # $1 = cell spec, rest = extra args
  read -r sw ax topo size meth <<<"$1"; shift
  run "$sw" --axes "$ax" --topologies "$topo" --sizes "$size" --methods "$meth" "$@"
}

P=${PART:-abc}
if [[ $P == *a* ]]; then
  for t in 1 2 4 8; do
    for c in "${BRANCH_CELLS[@]}"; do
      cell "$c" --threads $t --budget-s 120 --mem-gb 16 --opt mem_gb=16 --out $R/r3_threads.jsonl
    done
    for c in "${MASTER_CELLS[@]}"; do
      cell "$c" --threads $t --budget-s 120 --mem-gb 16 --opt mem_gb=16 --code $M \
        --out $R/r3_threads.jsonl
    done
  done
fi

if [[ $P == *b* ]]; then
  for c in "${BRANCH_CELLS[@]}"; do
    case "$c" in *pgmpy*) continue;; esac
    cell "$c" --threads 1 --device mps --budget-s 120 --mem-gb 16 --opt mem_gb=16 \
      --out $R/r3_mps.jsonl
  done
fi

if [[ $P == *c* ]]; then
  BIG="--threads 8 --budget-s 120 --mem-gb 16 --opt mem_gb=16 --out $R/r3_big.jsonl"
  BIGN="10 20 50 100 200 500 1000 2000 5000 10000"
  run discrete_infer --axes deg --sizes 2 3 4 4.5 5 5.5 6 8 --methods nbn_ve pgmpy_ve $BIG
  run discrete_infer --axes deg --sizes 4.5 5 5.5 --methods nbn_ve pgmpy_ve \
    --out $R/r3_small_deg.jsonl
  run discrete_infer --axes n --topologies er --sizes $BIGN --methods nbn_ve pgmpy_ve nbn_lw $BIG
  run discrete_infer --axes n --topologies local --sizes $BIGN --methods nbn_lw $BIG
  run discrete_infer_cert --axes n --topologies local --sizes $BIGN $BIG
  run cont_fit --axes n --sizes $BIGN --methods nbn_stacked pyro_svi $BIG
  for sw in r2_fit_ncat r2_fit_flex r2_fit_flow r2_knn_query r2_flex_query r2_ais r2_avi; do
    run $sw --sizes 10 20 50 100 200 500 1000 2000 5000 \
      --methods ncat_stacked pyro_ncat flex_stacked flow_stacked knn_lw flex_lw ais avi $BIG
  done
fi
