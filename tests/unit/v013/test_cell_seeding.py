"""Cell-level determinism (D6): a cell rerun gives bit-identical numbers.

``_run_cell`` used to run with whatever global RNG state the process had, so
LW / AIS posteriors (and every neural fit) changed from run to run, and the
oracle's draws were unseeded too. Seeds are now derived from (problem, seed,
baseline) with a collision-free mixer.
"""
from __future__ import annotations

import torch

from nbn.bench.core.config import BaselineSpec
from nbn.bench.core.seeding import derive_seed
from nbn.bench.measurements.accuracy_timing import AccuracyAndTiming
from nbn.bench.problems import SyntheticConfig, SyntheticProblemSource
from nbn.bench.selectors.uniform import UniformRandomSelector


def _ctx(problem, spec):
    return {
        "problem": problem, "baseline_spec": spec, "seed": problem.seed,
        "selector": UniformRandomSelector(),
        "measurement": AccuracyAndTiming(n_oracle_samples=200),
        "benchmark": "synthetic", "n_queries_per_cell": 4,
        "per_cell_timeout_s": 600.0, "fit_budget_s": 1000.0,
        "default_role": "random", "batch_sizes": [1],
        "fit_role": "standalone", "cache_path": None,
    }


def _values(rows):
    return [(r["metric"], r["value"]) for r in rows
            if r["metric"] in ("tv_per_node", "w1_per_node", "ks_per_node")
            and r["status"] == "ok"]


def test_cell_rerun_is_bit_identical_for_a_sampling_engine():
    from nbn.bench.core import cell_worker

    cfg = SyntheticConfig(families=["hybrid"], n_nodes_list=[5], seeds=[0],
                          n_train=200, n_test=20, n_reference=300)
    problem = next(iter(SyntheticProblemSource().iter_problems(cfg)))
    spec = BaselineSpec(library="nbn", mechanism="lg", param_method="mle",
                        inference_method="lw", extra_kwargs={"n_samples": 64})
    torch.manual_seed(1)
    a = _values(cell_worker._run_cell(_ctx(problem, spec)))
    torch.manual_seed(2)            # different ambient state must not matter
    b = _values(cell_worker._run_cell(_ctx(problem, spec)))
    assert a and a == b


def test_derive_seed_has_no_linear_collisions():
    # seed*997 + idx collides at (0, 997) vs (1, 0); the mixer does not.
    assert derive_seed(0, 997) != derive_seed(1, 0)
    assert derive_seed("cell", "10", 0, "nbn-cat-lw") != derive_seed(
        "cell", "10", 0, "nbn-cat-ve")
    assert derive_seed("x", 3) == derive_seed("x", 3)
