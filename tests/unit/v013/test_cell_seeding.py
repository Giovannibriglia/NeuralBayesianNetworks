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


def _mock_cuda(monkeypatch, calls):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "init", lambda: calls.append("init_all"))
    monkeypatch.setattr(torch.cuda, "get_rng_state_all",
                        lambda: calls.append("save_all") or [])
    monkeypatch.setattr(torch.cuda, "get_rng_state",
                        lambda i=None: calls.append(("save", i)) or "state")
    monkeypatch.setattr(torch.cuda, "set_rng_state",
                        lambda st, i=None: calls.append(("restore", st, i)))
    monkeypatch.setattr(torch.cuda, "manual_seed",
                        lambda s: calls.append(("seed", s)))
    monkeypatch.setattr(torch.cuda, "manual_seed_all",
                        lambda s: calls.append("seed_all"))

    class _Dev:
        def __init__(self, i):
            self.i = i

        def __enter__(self):
            calls.append(("on", self.i))

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(torch.cuda, "device", _Dev)


def test_rng_scope_touches_only_the_cell_cuda_device(monkeypatch):
    """rng_scope initialised and saved EVERY visible GPU on a CUDA host (a
    context per GPU even in CPU-only cells; fails under EXCLUSIVE_PROCESS).
    Now only the given device's generator is saved, seeded and restored -- and
    saving it first keeps the leak fix (the scope's seed cannot stay in a
    generator the body created)."""
    from nbn.bench.core import seeding

    calls = []
    _mock_cuda(monkeypatch, calls)
    with seeding.rng_scope(7, "cuda:1"):
        calls.append("body")
    assert calls == [("save", 1), ("on", 1), ("seed", 7), "body",
                     ("restore", "state", 1)]


def test_rng_scope_on_cpu_never_touches_cuda(monkeypatch):
    from nbn.bench.core import seeding

    calls = []
    _mock_cuda(monkeypatch, calls)
    before = torch.random.get_rng_state()
    with seeding.rng_scope(7, "cpu"):
        a = torch.rand(3)
    with seeding.rng_scope(7):
        b = torch.rand(3)
    assert calls == []
    assert torch.equal(a, b)
    assert torch.equal(torch.random.get_rng_state(), before)
