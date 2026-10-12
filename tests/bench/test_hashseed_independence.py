"""Query selection and cell scores do not depend on PYTHONHASHSEED (R3, D6).

The topological selector built evidence pools from frozensets of node names
and shuffled / tie-broke them in iteration order — string-hash order, which
Python randomises per process. Every cell worker is a fresh process, so the
baselines of one cell were scored on different queries (and the same-query
averaging paired mismatched queries). Each check runs in subprocesses with
different PYTHONHASHSEED values and compares the output.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_SELECT = r"""
import json, torch
from nbn.bench.problems import SyntheticConfig, SyntheticProblemSource
from nbn.bench.selectors.topological import TopologicalAllocator
from nbn.bench.selectors.heaviest import HeaviestQueryByRole
cfg = SyntheticConfig(families=["discrete"], n_nodes_list=[24], seeds=[3], n_train=200,
                      n_test=200, n_reference=100, edge_density=0.3, max_in_degree=4)
p = next(iter(SyntheticProblemSource().iter_problems(cfg)))
out = []
for sel in (TopologicalAllocator(), HeaviestQueryByRole()):
    for q in sel.select(p, 40, 3):
        out.append([q.targets[0], q.query_kind, q.evidence_strategy,
                    [[k, None if v is None else float(v)] for k, v in q.evidence.items()]])
print(json.dumps(out))
"""

_CELL = r"""
import json
from nbn.bench.core import cell_worker
from nbn.bench.core.config import BaselineSpec
from nbn.bench.measurements.accuracy_timing import AccuracyAndTiming
from nbn.bench.problems import SyntheticConfig, SyntheticProblemSource
from nbn.bench.selectors.topological import TopologicalAllocator
cfg = SyntheticConfig(families=["discrete"], n_nodes_list=[12], seeds=[1], n_train=300,
                      n_test=100, n_reference=500, edge_density=0.35, max_in_degree=3)
p = next(iter(SyntheticProblemSource().iter_problems(cfg)))
ctx = {"problem": p, "baseline_spec": BaselineSpec("nbn", "cat", "mle", "lw",
       extra_kwargs={"n_samples": 256}), "seed": p.seed,
       "selector": TopologicalAllocator(), "measurement": AccuracyAndTiming(),
       "benchmark": "synthetic", "n_queries_per_cell": 12, "per_cell_timeout_s": 600.0,
       "fit_budget_s": 1000.0, "default_role": "random", "batch_sizes": [1],
       "fit_role": "standalone", "cache_path": None}
rows = cell_worker._run_cell(ctx)
print(json.dumps([[r["query_idx"], r["metric"], r["value"]] for r in rows
                  if r["metric"] in ("tv_per_node", "jsd_per_node")]))
"""


def _run(code: str, hashseed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    return res.stdout.strip().splitlines()[-1]


@pytest.mark.subprocess
def test_selected_queries_identical_across_hash_seeds():
    outs = {_run(_SELECT, s) for s in ("1", "2", "3")}
    assert len(outs) == 1, "query selection depends on PYTHONHASHSEED"
    assert json.loads(outs.pop())


@pytest.mark.subprocess
@pytest.mark.slow
def test_cell_scores_identical_across_hash_seeds():
    outs = [_run(_CELL, s) for s in ("1", "2")]
    assert outs[0] == outs[1], "cell scores depend on PYTHONHASHSEED"
    assert json.loads(outs[0])
