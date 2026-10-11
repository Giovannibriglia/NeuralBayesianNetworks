"""Round-2 regression tests for the inference review fixes (R5-R7, R14b, CUDA-by-reading).

Ground truth is brute-force joint enumeration (``brute_posterior`` from the
round-1 module).
"""
from __future__ import annotations

import pytest
import torch

from tests.conftest import available_devices
from tests.unit.test_review_fixes_inference import _random_discrete, brute_posterior

# --------------------------------------------------------------------------- #
# R7 — discrete evidence / do must be valid integral states on every engine
# --------------------------------------------------------------------------- #


def _engines():
    from nbn import TensorVariableElimination
    from nbn.inference.amortized_is import AmortizedISEngine
    from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine

    return {
        "ve": TensorVariableElimination(),
        "lw": LikelihoodWeightingEngine(64),
        "ais": AmortizedISEngine(n_samples=64),
    }


@pytest.mark.parametrize("engine", ["ve", "lw", "ais"])
@pytest.mark.parametrize("bad", [-1, 3, 1.5])
@pytest.mark.parametrize("kind", ["evidence", "do"])
def test_r7_invalid_discrete_values_are_rejected(engine, bad, kind):
    eng = _engines()[engine]
    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    single = {nodes[2]: torch.tensor(bad)}
    batch = {nodes[2]: torch.tensor([1, bad])}
    for method, spec in (("query", single), ("query_batch", batch)):
        ev, kw = ({}, {"do": spec}) if kind == "do" else (spec, {})
        with pytest.raises(ValueError, match="integer state|outside"):
            getattr(eng, method)(model, [nodes[-1]], ev, **kw)


def test_r7_integral_float_evidence_still_accepted():
    """1.0 as a float is a valid state (bench evidence arrives as float)."""
    eng = _engines()["ve"]
    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    out = eng.query_batch(model, [nodes[-1]], {nodes[0]: torch.tensor([[1.0], [2.0]])})
    ref = torch.stack([brute_posterior(model, nodes[-1], {nodes[0]: v}) for v in (1, 2)])
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# R14b — a model (and its engine) stays picklable after queries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("engine", ["auto", "tensor_ve", "likelihood_weighting"])
def test_r14b_torch_save_after_query(engine, tmp_path):
    import copy
    import pickle

    model = _random_discrete()
    model._engine_spec = engine
    nodes = list(model.dag.topological_order())
    ev = {nodes[0]: torch.tensor(1)}
    model.query([nodes[-1]], ev)
    model.query_batch([nodes[-1]], {nodes[0]: torch.tensor([0, 1])})
    path = tmp_path / "m.pt"
    torch.save(model, path)
    loaded = torch.load(path, weights_only=False)
    torch.testing.assert_close(
        loaded.query([nodes[-1]], ev, engine=_engines()["ve"]),
        brute_posterior(model, nodes[-1], {nodes[0]: 1}), atol=1e-5, rtol=1e-5)
    # The engine itself pickles and deep-copies, with its memos reset.
    eng = model._engine
    assert pickle.loads(pickle.dumps(eng)) is not None
    copy.deepcopy(eng)


# --------------------------------------------------------------------------- #
# R5 — stable CPU/MPS budget, visible reroutes; CUDA-by-reading #1 / #2
# --------------------------------------------------------------------------- #


def test_r5_host_budget_ignores_momentary_free_memory(monkeypatch):
    """Same query, same route, whether the machine is idle or under pressure."""
    from nbn import TensorVariableElimination
    from nbn.inference import _memory
    from nbn.inference.hybrid import HybridRouter

    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    eng = TensorVariableElimination()
    idle = eng.memory_budget_bytes("cpu")
    monkeypatch.setattr(_memory, "host_available_bytes", lambda: 1)  # pressure
    assert eng.memory_budget_bytes("cpu") == idle
    router = HybridRouter()
    router.query(model, [nodes[-1]], {nodes[0]: torch.tensor(1)})
    assert router.last_engine == "tensor_ve"


def test_r5_direct_ve_rejects_only_beyond_capacity(monkeypatch):
    from nbn import TensorVariableElimination
    from nbn.inference import tensor_ve
    from nbn.inference.hybrid import HybridRouter

    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    t, ev = nodes[-1], {nodes[0]: torch.tensor(1)}
    eng = TensorVariableElimination()
    peak = eng.estimate_peak_bytes(model, t, tuple(ev))
    assert peak > 0
    # Capacity between the router's 0.75 share and the full peak: direct VE
    # answers exactly (the plan fits the machine), the router prefers LW.
    monkeypatch.setattr(tensor_ve, "device_capacity_bytes", lambda d: int(peak / 0.9))
    torch.testing.assert_close(eng.query(model, [t], ev),
                               brute_posterior(model, t, {nodes[0]: 1}), atol=1e-5, rtol=1e-5)
    router = HybridRouter(n_samples=64)
    router.query(model, [t], ev)
    assert router.last_engine == "likelihood_weighting"
    # Beyond the whole capacity, direct VE refuses up front.
    monkeypatch.setattr(tensor_ve, "device_capacity_bytes", lambda d: int(peak / 2))
    with pytest.raises(torch.cuda.OutOfMemoryError, match="pre-allocation guard"):
        eng.query(model, [t], ev)


def test_r5_reroute_warns_once_per_query_shape(monkeypatch, caplog):
    import logging

    from nbn.inference import tensor_ve
    from nbn.inference.hybrid import HybridRouter

    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    monkeypatch.setattr(tensor_ve, "device_capacity_bytes", lambda d: 1)
    router = HybridRouter(n_samples=64)
    with caplog.at_level(logging.DEBUG, logger="nbn.inference.hybrid"):
        for v in (0, 1, 2):
            router.query(model, [nodes[-1]], {nodes[0]: torch.tensor(v)})
        router.query(model, [nodes[-1]], {nodes[1]: torch.tensor(0)})  # new shape
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2 and "likelihood weighting" in warnings[0].getMessage()
    assert router.last_engine == "likelihood_weighting"


def _cgroup_tree(root, files):
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


@pytest.mark.parametrize("files,expected", [
    ({"memory.max": "1073741824\n", "memory.current": "629145600\n",
      "memory.stat": "anon 1\ninactive_file 104857600\n"},
     (1 << 30, 629145600 - 104857600)),
    ({"memory.max": "max\n", "memory.current": "1000\n"}, (None, 1000)),
    ({"memory/memory.limit_in_bytes": "9223372036854771712\n",
      "memory/memory.usage_in_bytes": "5000\n",
      "memory/memory.stat": "total_inactive_file 1000\n"}, (None, 4000)),
    ({"memory/memory.limit_in_bytes": "2147483648\n",
      "memory/memory.usage_in_bytes": "1073741824\n"}, (1 << 31, 1 << 30)),
    ({}, (None, None)),
])
def test_cuda_reading_1_cgroup_limits_bound_host_memory(tmp_path, monkeypatch, files, expected):
    from nbn.inference import _memory

    _cgroup_tree(tmp_path, files)
    monkeypatch.setattr(_memory, "_CGROUP_ROOT", str(tmp_path))
    assert _memory.cgroup_memory() == expected
    limit, usage = expected
    total = _memory.host_total_bytes()
    avail = _memory.host_available_bytes()
    if limit is not None:
        assert total is not None and total <= limit
        if usage is not None:
            assert avail is not None and avail <= limit - usage


def test_cuda_reading_2_lw_fallback_runs_after_ve_tensors_are_released(monkeypatch):
    import weakref

    from nbn.inference import hybrid
    from nbn.inference.hybrid import HybridRouter

    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    router = HybridRouter(n_samples=64)
    seen = {}

    def _ve_oom(*a, **k):
        big = torch.empty(1024)  # stands in for VE's intermediate factors
        seen["ref"] = weakref.ref(big)
        raise RuntimeError("CUDA out of memory. Tried to allocate 3.3 TiB")

    real_lw = router._lw.query

    def _lw(*a, **k):
        seen["alive_during_lw"] = seen["ref"]() is not None
        return real_lw(*a, **k)

    released = []
    monkeypatch.setattr(router._ve, "query", _ve_oom)
    monkeypatch.setattr(router._lw, "query", _lw)
    monkeypatch.setattr(hybrid, "release_cached_memory", released.append)
    out = router.query(model, [nodes[-1]], {nodes[0]: torch.tensor(1)})
    assert out.shape == (3,)
    assert seen["alive_during_lw"] is False
    assert released == [model.device]
    assert router.last_engine == "likelihood_weighting"


def test_cuda_reading_2_release_cached_memory_targets_the_device():
    from unittest import mock

    from nbn.inference._memory import release_cached_memory

    with mock.patch("torch.cuda.device") as ctx, \
         mock.patch("torch.cuda.empty_cache") as empty:
        release_cached_memory(torch.device("cuda:1"))
    ctx.assert_called_once_with(torch.device("cuda:1"))
    empty.assert_called_once()
    for dev in available_devices():
        release_cached_memory(torch.device(dev))  # real backends: no error
