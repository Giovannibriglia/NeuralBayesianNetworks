#!/usr/bin/env bash
# Round 4: budget -> better answer, with the new options (conditioning, anytime
# certified sampling, bounded conditioning).  One config per process.
# Usage: CONFIGS="cpu:1 cpu:20 cuda:4" bash benchmarks/scaling/r4.sh
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
R=${R:-benchmarks/scaling/results}
mkdir -p $R
export PYTHONPATH=$PWD
CONFIGS=${CONFIGS:-"cpu:1 cpu:20 cuda:4"}
for cfg in $CONFIGS; do
  dev=${cfg%%:*}; thr=${cfg##*:}
  for mem in ${MEMS:-1 4 16}; do
    [[ $dev == cuda && $mem == 16 ]] && mem=10      # 12 GB card
    $PY benchmarks/scaling/r4_budget.py e1 --device $dev --threads $thr --mem-gb $mem \
      --max-inst ${MAXI:-1000000} --degs ${DEGS:-3 4.5 6} \
      --query-limit-s ${QLIM:-20} --out $R/r4_e1.jsonl
  done
  $PY benchmarks/scaling/r4_budget.py e2 --device $dev --threads $thr --mem-gb 4 \
    --times 0.25 1 4 16 --out $R/r4_e2.jsonl
  $PY benchmarks/scaling/r4_budget.py e3 --device $dev --threads $thr --mem-gb 1 \
    --times 1 4 16 64 --n-q 5 --out $R/r4_e3.jsonl
done
echo R4_DONE
