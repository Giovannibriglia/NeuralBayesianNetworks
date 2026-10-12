"""Round-4 regression tests: MPS large-tensor guard; cached cgroup reads."""
from __future__ import annotations

import logging
from unittest import mock

import pytest
import torch

from tests.conftest import available_devices
from tests.unit.test_review_fixes_inference import _random_discrete, brute_posterior
from tests.unit.test_review_fixes_inference_r2 import brute_joint

GiB = 1 << 30

# --------------------------------------------------------------------------- #
# MPS — intermediate factors stay below the size MPS reduces correctly
# --------------------------------------------------------------------------- #


def _per_row(model, target, ev_keys):
    from nbn import TensorVariableElimination

    return TensorVariableElimination()._peak_profile(model, target, ev_keys, (), "min_fill")


@pytest.fixture
def mps_cap(monkeypatch):
    """Force the MPS branch on any device with a chosen element cap."""
    from nbn.inference import tensor_ve

    def _set(cap):
        monkeypatch.setattr(tensor_ve, "_max_tensor_elements", lambda d: cap)
    return _set


def test_mps_cap_chunks_rows_and_stays_exact(mps_cap):
    from nbn import TensorVariableElimination

    model = _random_discrete(seed=3)
    nodes = list(model.dag.topological_order())
    t, e = nodes[-1], nodes[0]
    batch_free, per_row = _per_row(model, t, (e,))
    assert per_row > 0
    mps_cap(max(batch_free, 2 * per_row) + 1)  # room for exactly 2 rows
    eng = TensorVariableElimination()
    vals = torch.tensor([0, 1, 2, 2, 1, 0, 1])
    with mock.patch.object(eng, "_query_batch_chunked",
                           wraps=eng._query_batch_chunked) as spy:
        out = eng.query_batch(model, [t], {e: vals})
    assert spy.call_args.kwargs["chunk"] == 2
    ref = torch.stack([brute_posterior(model, t, {e: v}) for v in range(3)])
    torch.testing.assert_close(out, ref[vals], atol=1e-5, rtol=1e-5)


def test_mps_cap_applies_to_joint_queries(mps_cap):
    from nbn import TensorVariableElimination

    model = _random_discrete(seed=3)
    nodes = list(model.dag.topological_order())
    targets, e = [nodes[0], nodes[2]], nodes[-1]
    eng = TensorVariableElimination()
    prof = eng._peak_profile(model, tuple(targets), (e,), (), "min_fill")
    mps_cap(max(prof[0], prof[1]) + 1)  # one row per chunk
    vals = torch.tensor([0, 2, 1])
    with mock.patch.object(eng, "_query_batch_chunked",
                           wraps=eng._query_batch_chunked) as spy:
        out = eng.query_batch(model, targets, {e: vals})
    assert spy.call_args.kwargs["chunk"] == 1
    for i, v in enumerate(vals.tolist()):
        torch.testing.assert_close(out[i], brute_joint(model, targets, {e: v}),
                                   atol=1e-5, rtol=1e-5)


def test_mps_single_row_beyond_cap_refused_by_ve_rerouted_by_router(mps_cap, caplog):
    from nbn import TensorVariableElimination
    from nbn.inference.hybrid import HybridRouter
    from nbn.inference.tensor_ve import MPSTensorLimitError

    model = _random_discrete(seed=3)
    nodes = list(model.dag.topological_order())
    t, e = nodes[-1], nodes[0]
    _, per_row = _per_row(model, t, (e,))
    mps_cap(per_row)  # not even one row fits
    eng = TensorVariableElimination()
    with pytest.raises(MPSTensorLimitError, match="MPS"):
        eng.query(model, [t], {e: torch.tensor(1)})
    with pytest.raises(MPSTensorLimitError, match="MPS"):
        eng.query_batch(model, [t], {e: torch.tensor([1, 2])})
    router = HybridRouter(n_samples=4096)
    with caplog.at_level(logging.WARNING, logger="nbn.inference.hybrid"):
        out = router.query(model, [t], {e: torch.tensor(1)})
    assert router.last_engine == "likelihood_weighting"
    assert any("MPS" in r.getMessage() for r in caplog.records)
    torch.testing.assert_close(out, brute_posterior(model, t, {e: 1}), atol=0.05, rtol=0)


def test_mps_router_falls_back_if_ve_refuses_anyway(mps_cap, monkeypatch):
    """Belt and braces: an MPSTensorLimitError from VE is retried with LW."""
    from nbn.inference.hybrid import HybridRouter
    from nbn.inference.tensor_ve import MPSTensorLimitError

    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    router = HybridRouter(n_samples=256)

    def _refuse(*a, **k):
        raise MPSTensorLimitError("too big for MPS")

    monkeypatch.setattr(router._ve, "query_batch", _refuse)
    out = router.query_batch(model, [nodes[-1]], {nodes[0]: torch.tensor([0, 1])})
    assert out.shape == (2, 3) and router.last_engine == "likelihood_weighting"


def test_mps_cap_only_on_mps():
    from nbn.inference.tensor_ve import _MPS_MAX_ELEMENTS, _max_tensor_elements

    assert _max_tensor_elements(torch.device("mps")) == _MPS_MAX_ELEMENTS
    for dev in ("cpu", "cuda", "cuda:1"):
        assert _max_tensor_elements(torch.device(dev)) is None


@pytest.mark.skipif("mps" not in available_devices(), reason="needs MPS")
def test_mps_real_device_chunks_under_the_cap(monkeypatch):
    from nbn import TensorVariableElimination
    from nbn.inference import tensor_ve

    model = _random_discrete(seed=3)
    nodes = list(model.dag.topological_order())
    t, e = nodes[-1], nodes[0]
    batch_free, per_row = _per_row(model, t, (e,))
    monkeypatch.setattr(tensor_ve, "_MPS_MAX_ELEMENTS", max(batch_free, 3 * per_row) + 1)
    model.to("mps")
    eng = TensorVariableElimination()
    vals = torch.tensor([0, 1, 2, 2, 1, 0, 1, 2], device="mps")
    with mock.patch.object(eng, "_query_batch_chunked",
                           wraps=eng._query_batch_chunked) as spy:
        out = eng.query_batch(model, [t], {e: vals})
    assert spy.call_args.kwargs["chunk"] == 3
    ref = torch.stack([brute_posterior(model, t, {e: v}) for v in range(3)])
    torch.testing.assert_close(out.cpu(), ref[vals.cpu()], atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# Linux perf — the cgroup layout is read once per process, usage only on demand
# --------------------------------------------------------------------------- #

_JOB = "system.slice/slurmstepd.scope/job_42"


@pytest.fixture
def slurm_tree(tmp_path, monkeypatch):
    """7-level fake SLURM v2 tree; returns ``(module, reads)`` recording file reads."""
    from nbn.inference import _memory

    files = {
        "memory.current": f"{20 * GiB}\n",
        "system.slice/memory.max": "max\n",
        "system.slice/slurmstepd.scope/memory.max": "max\n",
        f"{_JOB}/memory.max": f"{64 * GiB}\n",
        f"{_JOB}/memory.current": f"{GiB}\n",
        f"{_JOB}/memory.stat": "inactive_file 0\n",
        f"{_JOB}/step_0/memory.max": "max\n",
        f"{_JOB}/step_0/user/memory.max": "max\n",
        f"{_JOB}/step_0/user/task_0/memory.max": "max\n",
    }
    for rel, content in files.items():
        p = tmp_path / "sys" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text(f"0::/{_JOB}/step_0/user/task_0\n")
    monkeypatch.setattr(_memory, "_CGROUP_ROOT", str(tmp_path / "sys"))
    monkeypatch.setattr(_memory, "_PROC_SELF_CGROUP", str(proc))
    _memory.refresh_cgroup_cache()
    reads: list[str] = []
    real_read, real_stat, real_paths = _memory._read_int, _memory._stat_value, _memory._own_cgroup_paths
    monkeypatch.setattr(_memory, "_read_int", lambda p: (reads.append(p), real_read(p))[1])
    monkeypatch.setattr(_memory, "_stat_value",
                        lambda p, k: (reads.append(p), real_stat(p, k))[1])
    monkeypatch.setattr(_memory, "_own_cgroup_paths",
                        lambda: (reads.append("/proc/self/cgroup"), real_paths())[1])
    yield _memory, reads
    _memory.refresh_cgroup_cache()


def _usage_reads(reads):
    return [p for p in reads if p.endswith(("memory.current", "memory.usage_in_bytes"))]


def test_cgroup_capacity_walk_once_per_process(slurm_tree, monkeypatch):
    mem, reads = slurm_tree
    assert mem.host_total_bytes() <= 64 * GiB
    assert reads  # first call discovers the layout
    reads.clear()
    for _ in range(5):
        mem.host_total_bytes()
        mem.cgroup_limit()
    assert reads == []  # capacity never re-reads a file
    # Headroom reads usage, and only on the limited level.
    mem.host_available_bytes()
    assert _usage_reads(reads) == [str(mem._levels(mem._CGROUP_ROOT, _JOB)[0]) + "/memory.current"]
    assert "/proc/self/cgroup" not in reads
    # A forked child (new pid) re-discovers its own layout.
    reads.clear()
    monkeypatch.setattr(mem.os, "getpid", lambda: -1)
    mem.host_total_bytes()
    assert "/proc/self/cgroup" in reads


def test_ve_query_reads_usage_at_most_once_and_never_for_tiny_plans(slurm_tree, monkeypatch):
    from nbn import TensorVariableElimination
    from nbn.inference import tensor_ve
    from nbn.inference.hybrid import HybridRouter

    mem, reads = slurm_tree
    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    t, ev = nodes[-1], {nodes[0]: torch.tensor([0, 1, 2])}
    eng = TensorVariableElimination()
    eng.query_batch(model, [t], ev)  # warm: layout discovery, plan caches
    reads.clear()
    eng.query_batch(model, [t], ev)
    eng.query(model, [t], {nodes[0]: torch.tensor(1)})
    router = HybridRouter()
    router.query_batch(model, [t], ev)
    reads_tiny = list(reads)
    assert _usage_reads(reads_tiny) == []  # tiny plan: no headroom needed
    assert "/proc/self/cgroup" not in reads_tiny
    # A plan that is not tiny next to the budget: one usage read per query.
    monkeypatch.setattr(tensor_ve, "_HEADROOM_CHECK_FRACTION", 0.0)
    reads.clear()
    eng.query_batch(model, [t], ev)
    assert len(_usage_reads(reads)) == 1
    reads.clear()
    router.query_batch(model, [t], ev)
    assert len(_usage_reads(reads)) <= 1
