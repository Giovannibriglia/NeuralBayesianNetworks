#!/usr/bin/env bash
# Round 2: other mechanisms and engines, branch vs first scaling branch (v1) vs master.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
R=benchmarks/scaling/results
M=${NBN_MASTER:-../NeuralBayesianNetworks}
V1=${NBN_V1:-../NBN_scaling_v1}
export PYTHONPATH=$PWD
S="--slack 300"
for sw in r2_fit_ncat r2_fit_flex r2_fit_flow; do
  $PY benchmarks/scaling/run.py $sw $S --out $R/$sw.jsonl
  $PY benchmarks/scaling/run.py $sw $S --code $M --methods ncat_local flex_local flow_local --out $R/$sw.jsonl
done
for sw in r2_knn_query r2_flex_query r2_ais r2_avi; do
  $PY benchmarks/scaling/run.py $sw $S --out $R/$sw.jsonl
  $PY benchmarks/scaling/run.py $sw $S --code $V1 --tag @v1 --out $R/$sw.jsonl
  $PY benchmarks/scaling/run.py $sw $S --code $M --out $R/$sw.jsonl
done
echo R2_DONE
