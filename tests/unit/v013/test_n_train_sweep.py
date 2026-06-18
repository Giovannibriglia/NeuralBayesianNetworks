"""n_train sensitivity sweep — learning curves (#109 PR 6).

Covers:
  * iter_problems produces exactly |families|×|n_nodes|×|seeds|×|n_train_sweep|
    problems;
  * the invariant — same (family, n_nodes, seed) across n_train values yields a
    bit-identical true_model (only train_data size varies);
  * config validation — n_train / n_train_sweep mutual exclusion (actionable
    messages) and bnlearn's synthetic-only rejection;
  * end-to-end — a sweep run stamps the n_train column with distinct values;
  * the recovery weight cache fires once per (seed, n_nodes) across n_train
    (n_train is not part of the cache key).
"""
from __future__ import annotations

import textwrap

import pytest
import torch

from benchmarking.domains.base import BenchmarkProblem
from benchmarking.measurements import ParamLearningMeasurement
from benchmarking.problems import SyntheticConfig, SyntheticProblemSource


def _cfg(**over):
    base = dict(
        families=["discrete"], n_nodes_list=[4], seeds=[0],
        edge_density=0.5, max_in_degree=2, cardinality=3,
        fraction_continuous=0.0, n_test=50, n_reference=100, device="cpu",
    )
    base.update(over)
    return SyntheticConfig(**base)


# ---- iteration cardinality --------------------------------------------------

def test_iter_problems_sweep_cardinality():
    cfg = _cfg(
        families=["discrete"], n_nodes_list=[4, 6], seeds=[0, 1],
        n_train_sweep=[30, 100, 300],
    )
    problems = list(SyntheticProblemSource().iter_problems(cfg))
    assert len(problems) == 1 * 2 * 2 * 3            # families×nodes×seeds×sweep
    # every swept n_train appears for each (n_nodes, seed)
    sizes = sorted({next(iter(p.train_data.values())).shape[0] for p in problems})
    assert sizes == [30, 100, 300]


def test_iter_problems_no_sweep_uses_scalar():
    cfg = _cfg(n_train=200, n_nodes_list=[4], seeds=[0, 1])
    problems = list(SyntheticProblemSource().iter_problems(cfg))
    assert len(problems) == 2                        # 1×1×2 seeds, single n_train
    assert all(next(iter(p.train_data.values())).shape[0] == 200 for p in problems)


# ---- the invariant: same true_model across n_train --------------------------

def test_same_true_model_across_n_train():
    cfg = _cfg(n_nodes_list=[4], seeds=[7], n_train_sweep=[50, 5000])
    problems = list(SyntheticProblemSource().iter_problems(cfg))
    assert len(problems) == 2
    p_small, p_big = problems

    def _cpts(p):
        return {n: m.cpt.clone() for n, m in p.true_model.mechanisms.items()
                if hasattr(m, "cpt")}

    cs, cb = _cpts(p_small), _cpts(p_big)
    assert set(cs) == set(cb) and len(cs) > 0
    for node in cs:
        assert torch.equal(cs[node], cb[node]), node     # bit-identical true CPTs
    assert next(iter(p_small.train_data.values())).shape[0] == 50
    assert next(iter(p_big.train_data.values())).shape[0] == 5000


# ---- config validation ------------------------------------------------------

def test_validation_mutual_exclusion_and_synthetic_only():
    from benchmarking.core.yaml_config import (
        _parse_bnlearn_config,
        _parse_synthetic_config,
    )

    base = dict(
        families=["discrete"], n_nodes_list=[4], seeds=[0], n_reference=100,
        edge_density=0.5, max_in_degree=2, cardinality=3, fraction_continuous=0.0,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _parse_synthetic_config({**base, "n_train": 100, "n_train_sweep": [30, 100]}, "x")
    with pytest.raises(ValueError, match="exactly one of n_train"):
        _parse_synthetic_config({**base}, "x")
    with pytest.raises(ValueError, match=r"non-empty list"):
        _parse_synthetic_config({**base, "n_train_sweep": []}, "x")
    with pytest.raises(ValueError, match=r">= 1"):
        _parse_synthetic_config({**base, "n_train_sweep": [0, 10]}, "x")
    # valid sweep
    cfg = _parse_synthetic_config({**base, "n_train_sweep": [30, 100]}, "x")
    assert cfg.n_train_sweep == [30, 100]

    with pytest.raises(ValueError, match="synthetic-only"):
        _parse_bnlearn_config(
            {"networks": ["asia"], "seeds": [1], "n_train_sweep": [30]}, "x")


# ---- end-to-end: n_train column stamped with distinct values ----------------

@pytest.mark.slow
def test_sweep_stamps_n_train_column(tmp_path):
    from benchmarking.core.runner import Runner
    from benchmarking.core.yaml_config import load_runner_config

    yaml = tmp_path / "lc.yaml"
    yaml.write_text(textwrap.dedent("""
        version: "v0.13"
        benchmark: synthetic
        config_name: lc_test
        metrics: log_likelihood
        selector: uniform_random
        source:
          families: [discrete]
          n_nodes_list: [4]
          seeds: [0]
          n_train_sweep: [20, 60]
          n_test: 40
          n_reference: 100
          edge_density: 0.5
          max_in_degree: 2
          cardinality: 3
          fraction_continuous: 0.0
        baselines:
          - {library: nbn, mechanism: cat, param_method: mle}
        n_queries_per_cell: 2
        per_cell_timeout_s: 60.0
    """))
    cfg = load_runner_config(
        yaml, jsonl_path=tmp_path / "out.jsonl",
        measurement_override=ParamLearningMeasurement(),
    )
    rows = list(Runner().run(cfg))
    metric_rows = [r for r in rows if r.metric.startswith(("log_likelihood", "param_recovery"))]
    assert metric_rows
    # n_train populated on every row, with both swept values present.
    n_trains = {r.n_train for r in metric_rows}
    assert n_trains == {20, 60}
    # the two n_train cells are distinct rows for the same (baseline, metric).
    tv20 = [r for r in metric_rows if r.metric == "param_recovery_tv" and r.n_train == 20]
    tv60 = [r for r in metric_rows if r.metric == "param_recovery_tv" and r.n_train == 60]
    assert len(tv20) == 1 and len(tv60) == 1


# ---- recovery weight cache fires once per (seed, n_nodes) across n_train -----

def test_weight_cache_hits_across_n_train():
    cfg = _cfg(n_nodes_list=[4], seeds=[0], n_train_sweep=[30, 100, 300, 1000])
    problems = list(SyntheticProblemSource().iter_problems(cfg))
    assert len(problems) == 4

    m = ParamLearningMeasurement()
    calls = {"n": 0}
    orig = m._compute_weights

    def _counting(problem, true_cpts):
        calls["n"] += 1
        return orig(problem, true_cpts)

    m._compute_weights = _counting

    class _Recovery:
        name = "nbn-cat"
        supports_scoring = True
        supports_param_recovery = True

        def __init__(self, prob):
            from benchmarking.adapters import NBNAdapter
            self._a = NBNAdapter(mechanism="cat", engine=None, device="cpu")
            self._a.fit(prob)

        def score_data(self, td):
            return self._a.score_data(td)

        def extract_learned_cpts(self):
            return self._a.extract_learned_cpts()

    for p in problems:
        m.measure(p, _Recovery(p), [], seed=p.seed)

    # 4 n_train problems share (name, problem_id, seed) -> one weight draw.
    assert calls["n"] == 1
