#!/usr/bin/env bash
# Follow-up runs after the main sequential sweep (see README).
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
R=benchmarks/scaling/results
M=${NBN_MASTER:-../NeuralBayesianNetworks}
export PYTHONPATH=$PWD

# 1. LW rows on the final engine (particle passes landed mid-run): replace.
$PY benchmarks/scaling/run.py discrete_infer --seeds 0 1 2 --methods nbn_lw --out $R/_lw.jsonl
$PY - <<'EOF'
import json, pathlib
R = pathlib.Path("benchmarks/scaling/results")
old = [json.loads(l) for l in (R / "discrete_infer.jsonl").read_text().splitlines() if l.strip()]
new = [json.loads(l) for l in (R / "_lw.jsonl").read_text().splitlines() if l.strip()]
keep = [r for r in old if r["method"] != "nbn_lw"]
(R / "discrete_infer.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep + new))
(R / "_lw.jsonl").unlink()
EOF

# 2. master exact VE on the other two seeds (fair exact-answer comparison).
$PY benchmarks/scaling/run.py discrete_infer --code $M --methods nbn_ve --seeds 1 2 \
    --topologies er --axes n --out $R/discrete_infer_master.jsonl

# 3. batched interval sets with the final pass size: replace.
$PY benchmarks/scaling/run.py discrete_sets --methods nbn_intervals_batched --out $R/_sets.jsonl
$PY - <<'EOF'
import json, pathlib
R = pathlib.Path("benchmarks/scaling/results")
old = [json.loads(l) for l in (R / "discrete_sets.jsonl").read_text().splitlines() if l.strip()]
new = [json.loads(l) for l in (R / "_sets.jsonl").read_text().splitlines() if l.strip()]
keep = [r for r in old if r["method"] != "nbn_intervals_batched"]
(R / "discrete_sets.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep + new))
(R / "_sets.jsonl").unlink()
EOF

# 4. GPU arm on the strong non-Gaussian family, with 5. references in parallel.
$PY benchmarks/scaling/run.py cont_fit --device mps --methods nbn_stacked nbn_local pyro_svi \
    --out $R/cont_fit_mps.jsonl &
OMP_NUM_THREADS=2 $PY benchmarks/scaling/ref.py --threads 2 --particles 4000000 > $R/log_ref.txt 2>&1
wait

# 6. full machine (10 threads, 24 GB): how much of the gap is the 1-thread cap?
$PY benchmarks/scaling/run.py discrete_infer --threads 10 --mem-gb 24 --topologies er --axes n \
    --methods nbn_ve pgmpy_ve nbn_lw --opt mem_gb=24 --out $R/discrete_infer_full.jsonl
$PY benchmarks/scaling/run.py cont_fit --threads 10 --mem-gb 24 --axes n \
    --methods nbn_local nbn_stacked pyro_svi --out $R/cont_fit_full.jsonl
echo POST_DONE
