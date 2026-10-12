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
    monkeypatch.setattr(_memory, "_PROC_SELF_CGROUP", str(tmp_path / "no-proc"), raising=False)
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


# --------------------------------------------------------------------------- #
# R6 — exact joint posterior for several discrete targets
# --------------------------------------------------------------------------- #


def brute_joint(model, targets, evidence=None, do=None) -> torch.Tensor:
    """Exact ``P(targets | evidence, do(...))`` ``[K_1, …, K_n]`` by enumeration."""
    from tests.unit.test_review_fixes_inference import _joint_table

    evidence, do = dict(evidence or {}), dict(do or {})
    configs, per_node, nodes = _joint_table(model)
    idx = {n: i for i, n in enumerate(nodes)}
    log_joint = sum(v for n, v in per_node.items() if n not in do)
    keep = torch.ones(configs.shape[0], dtype=torch.bool)
    for n, v in {**evidence, **do}.items():
        keep &= configs[:, idx[n]] == int(v)
    cards = [model.mechanisms[t].n_classes for t in targets]
    w = torch.where(keep, log_joint, torch.tensor(-float("inf"))).exp()
    out = torch.zeros(cards)
    flat = torch.zeros(1, dtype=torch.long).expand(configs.shape[0]).clone()
    for t, k in zip(targets, cards):
        flat = flat * k + configs[:, idx[t]]
    out.view(-1).index_add_(0, flat, w)
    return out / out.sum()


@pytest.mark.parametrize("seed", range(4))
def test_r6_ve_joint_matches_enumeration(seed):
    from nbn import TensorVariableElimination

    model = _random_discrete(seed=seed, n_nodes=6)
    nodes = list(model.dag.topological_order())
    g = torch.Generator().manual_seed(seed)
    eng = TensorVariableElimination()
    for n_t in (2, 3):
        perm = torch.randperm(len(nodes), generator=g).tolist()
        targets = [nodes[i] for i in perm[:n_t]]
        e, x = nodes[perm[n_t]], nodes[perm[n_t + 1]]
        ev, do = {e: 2}, ({x: 1} if seed % 2 else {})
        out = eng.query(model, targets, {e: torch.tensor(2)},
                        do={k: torch.tensor(v) for k, v in do.items()})
        assert out.shape == (3,) * n_t
        torch.testing.assert_close(out, brute_joint(model, targets, ev, do),
                                   atol=1e-5, rtol=1e-5)
        # Batched, per-row evidence.
        vals = torch.tensor([0, 2, 1])
        batched = eng.query_batch(model, targets, {e: vals},
                                  do={k: torch.tensor([v] * 3) for k, v in do.items()})
        assert batched.shape == (3,) + (3,) * n_t
        for i, v in enumerate(vals.tolist()):
            torch.testing.assert_close(batched[i], brute_joint(model, targets, {e: v}, do),
                                       atol=1e-5, rtol=1e-5)


def test_r6_joint_with_observed_and_intervened_targets():
    from nbn import TensorVariableElimination

    model = _random_discrete(seed=2)
    nodes = list(model.dag.topological_order())
    eng = TensorVariableElimination()
    a, b, c = nodes[1], nodes[3], nodes[5]
    out = eng.query_batch(model, [a, b, c], {b: torch.tensor([0, 2])},
                          do={c: torch.tensor([1, 1])})
    assert out.shape == (2, 3, 3, 3)
    for i, bv in enumerate((0, 2)):
        torch.testing.assert_close(out[i], brute_joint(model, [a, b, c], {b: bv}, {c: 1}),
                                   atol=1e-5, rtol=1e-5)
    # Every target clamped: a product of point masses.
    out = eng.query(model, [a, b], {a: torch.tensor(1), b: torch.tensor(2)})
    expected = torch.zeros(3, 3)
    expected[1, 2] = 1.0
    torch.testing.assert_close(out, expected)


def test_r6_router_answers_small_joint_exactly():
    from nbn.inference.hybrid import HybridRouter

    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    router = HybridRouter()
    out = router.query(model, [nodes[0], nodes[4]], {nodes[-1]: torch.tensor(1)})
    assert router.last_engine == "tensor_ve"
    torch.testing.assert_close(out, brute_joint(model, [nodes[0], nodes[4]], {nodes[-1]: 1}),
                               atol=1e-5, rtol=1e-5)
    model._engine_spec = "auto"
    model._engine = None
    via_model = model.query([nodes[0], nodes[4]], {nodes[-1]: torch.tensor(1)})
    torch.testing.assert_close(via_model, out)
    # Same marginal as the single-target answer.
    single = router.query(model, [nodes[0]], {nodes[-1]: torch.tensor(1)})
    torch.testing.assert_close(out.sum(1), single, atol=1e-5, rtol=1e-5)


def test_r6_joint_is_priced_and_chunked_like_single_targets():
    from unittest import mock

    from nbn import TensorVariableElimination

    model = _random_discrete(seed=3)
    nodes = list(model.dag.topological_order())
    targets, e = [nodes[0], nodes[2]], nodes[-1]
    probe = TensorVariableElimination()
    B = 6
    p1 = probe.estimate_peak_bytes(model, targets, (e,), B=1)
    pb = probe.estimate_peak_bytes(model, targets, (e,), B=B)
    assert pb > p1 >= 3 * 4 * 9  # the 3x3 joint table itself is priced
    eng = TensorVariableElimination(memory_budget=(p1 + pb) / 2)
    vals = torch.tensor([0, 1, 2, 2, 1, 0])
    with mock.patch.object(eng, "_query_batch_chunked",
                           wraps=eng._query_batch_chunked) as spy:
        out = eng.query_batch(model, targets, {e: vals})
    assert spy.called and out.shape == (B, 3, 3)
    for i, v in enumerate(vals.tolist()):
        torch.testing.assert_close(out[i], brute_joint(model, targets, {e: v}),
                                   atol=1e-5, rtol=1e-5)
