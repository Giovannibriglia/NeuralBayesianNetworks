#!/usr/bin/env bash
# Round 3, clean rerun of the large-budget node sweeps and the particle study.
# Each command waits until the machine is idle (1-min load < 2), and the load at
# start is logged, so a contended cell can be spotted.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
R=benchmarks/scaling/results
export PYTHONPATH=$PWD
idle() { while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}');
  awk -v l="$l" 'BEGIN{exit !(l<2.0)}' && break; sleep 20; done
  echo "# $(date +%T) load=$(sysctl -n vm.loadavg)"; }
run() { idle; $PY benchmarks/scaling/run.py "$@" --slack 60; }
BIG="--threads 8 --budget-s 120 --mem-gb 16 --opt mem_gb=16 --out $R/r3_big.jsonl"
BIGN="10 20 50 100 200 500 1000 2000 5000 10000"
run discrete_infer_cert --axes n --topologies local --sizes 500 --methods nbn_cert --threads 1 \
  --device mps --budget-s 120 --mem-gb 16 --opt mem_gb=16 --out $R/r3_mps.jsonl
run discrete_infer --axes n --topologies er --sizes $BIGN --methods nbn_ve pgmpy_ve nbn_lw $BIG
run discrete_infer --axes n --topologies local --sizes $BIGN --methods nbn_lw $BIG
run discrete_infer_cert --axes n --topologies local --sizes $BIGN $BIG
run cont_fit --axes n --sizes $BIGN --methods nbn_stacked pyro_svi $BIG
for sw in r2_fit_ncat r2_fit_flex r2_fit_flow r2_knn_query r2_flex_query r2_ais r2_avi; do
  run $sw --sizes 10 20 50 100 200 500 1000 2000 5000 \
    --methods ncat_stacked pyro_ncat flex_stacked flow_stacked knn_lw flex_lw ais avi $BIG
done
# particles: does a larger batch use 8 threads?
O=$R/r3_particles.jsonl
run discrete_infer --axes n --topologies local --sizes 1000 --methods nbn_ve --threads 8 \
  --budget-s 120 --mem-gb 16 --opt mem_gb=16 --out $O
for S in 100000 300000 1000000; do for t in 1 8; do
  run discrete_infer --axes n --topologies local --sizes 1000 --methods nbn_lw --threads $t \
    --budget-s 300 --mem-gb 16 --opt n_samples=$S --tag S$S --out $O
done; done
for S in 10000 100000 300000; do for t in 1 8; do
  run cont_infer --axes n --topologies er --sizes 5000 --methods nbn_lw --threads $t \
    --budget-s 300 --mem-gb 16 --opt n_samples=$S --tag S$S --out $O
done; done
