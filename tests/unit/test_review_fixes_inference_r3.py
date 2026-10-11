"""Round-3 regression tests for the inference review fixes (N1, N3, N6, N8, SLURM cgroups)."""
from __future__ import annotations

import pytest
import torch

from tests.unit.test_review_fixes_inference import _random_discrete, brute_posterior

GiB = 1 << 30

# --------------------------------------------------------------------------- #
# SLURM — the job's cgroup is a nested path, not the mount root
# --------------------------------------------------------------------------- #


def _tree(root, files):
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _point_at(monkeypatch, tmp_path, proc_cgroup: str | None):
    from nbn.inference import _memory

    monkeypatch.setattr(_memory, "_CGROUP_ROOT", str(tmp_path / "sys"))
    proc = tmp_path / "proc_self_cgroup"
    if proc_cgroup is not None:
        proc.write_text(proc_cgroup)
    monkeypatch.setattr(_memory, "_PROC_SELF_CGROUP", str(proc), raising=False)
    return _memory


def test_slurm_cgroup_v2_nested_job_limit(tmp_path, monkeypatch):
    job = "system.slice/slurmstepd.scope/job_42"
    _tree(tmp_path / "sys", {
        # v2 root cgroup has no memory.max; intermediate levels are "max".
        "memory.current": f"{20 * GiB}\n",
        "system.slice/memory.max": "max\n",
        "system.slice/slurmstepd.scope/memory.max": "max\n",
        f"{job}/memory.max": f"{4 * GiB}\n",
        f"{job}/memory.current": f"{1 * GiB}\n",
        f"{job}/memory.stat": f"anon 5\ninactive_file {GiB // 4}\n",
        f"{job}/step_0/memory.max": "max\n",
        f"{job}/step_0/memory.current": f"{GiB // 2}\n",
        f"{job}/step_0/user/task_0/memory.max": "max\n",
    })
    mem = _point_at(monkeypatch, tmp_path, f"0::/{job}/step_0/user/task_0\n")
    limit, usage = mem.cgroup_memory()
    assert limit == 4 * GiB
    assert limit - usage == 4 * GiB - (GiB - GiB // 4)  # tightest headroom
    assert mem.host_total_bytes() <= 4 * GiB
    assert mem.host_available_bytes() <= limit - usage


def test_slurm_cgroup_v1_nested_job_limit(tmp_path, monkeypatch):
    big = "9223372036854771712\n"
    job = "memory/slurm/uid_1000/job_42"
    _tree(tmp_path / "sys", {
        "memory/memory.limit_in_bytes": big,
        "memory/memory.usage_in_bytes": f"{30 * GiB}\n",
        "memory/slurm/memory.limit_in_bytes": big,
        f"{job}/memory.limit_in_bytes": f"{2 * GiB}\n",
        f"{job}/memory.usage_in_bytes": f"{GiB // 2}\n",
        f"{job}/step_0/memory.limit_in_bytes": big,
        f"{job}/step_0/memory.usage_in_bytes": f"{GiB // 4}\n",
    })
    mem = _point_at(
        monkeypatch, tmp_path,
        "12:pids:/slurm/uid_1000/job_42\n4:memory,cpuacct:/slurm/uid_1000/job_42/step_0\n",
    )
    assert mem.cgroup_memory() == (2 * GiB, GiB // 2)


@pytest.mark.parametrize("proc", [None, "0::/\n", "garbage line\n0::/missing/path\n"])
def test_cgroup_robust_to_missing_files(tmp_path, monkeypatch, proc):
    _tree(tmp_path / "sys", {"memory.max": f"{3 * GiB}\n", "memory.current": f"{GiB}\n"})
    mem = _point_at(monkeypatch, tmp_path, proc)
    assert mem.cgroup_memory() == (3 * GiB, GiB)


def test_cgroup_absent_everywhere(tmp_path, monkeypatch):
    mem = _point_at(monkeypatch, tmp_path, "0::/user.slice/x\n")
    assert mem.cgroup_memory() == (None, None)


# --------------------------------------------------------------------------- #
# N1 — chunk sizes respect live headroom on CPU / MPS
# --------------------------------------------------------------------------- #


class _FakeVM:
    def __init__(self, total, available):
        self.total, self.available = total, available


def _fake_psutil(monkeypatch, total, available):
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVM(total, available))


def test_n1_chunks_fit_live_headroom_not_total_ram(monkeypatch):
    """32 GiB machine with little free: B=400 must be chunked to what is free."""
    from unittest import mock

    from nbn import TensorVariableElimination

    model = _random_discrete(seed=3)
    nodes = list(model.dag.topological_order())
    t, e = nodes[-1], nodes[0]
    B = 400
    eng = TensorVariableElimination()
    peak_b = eng.estimate_peak_bytes(model, t, (e,), B=B)
    per_row = peak_b / B
    available = int(peak_b / 40 / 0.9)  # room for ~10 rows
    _fake_psutil(monkeypatch, total=32 * GiB, available=available)
    vals = torch.arange(B) % 3
    with mock.patch.object(eng, "_query_batch_chunked",
                           wraps=eng._query_batch_chunked) as spy:
        out = eng.query_batch(model, [t], {e: vals})
    assert spy.called, "sized chunks to total RAM, not free memory"
    chunk = spy.call_args.kwargs["chunk"]
    assert chunk >= 1 and chunk * per_row <= 0.9 * available
    ref = torch.stack([brute_posterior(model, t, {e: v}) for v in range(3)])
    torch.testing.assert_close(out, ref[vals], atol=1e-5, rtol=1e-5)


def test_n1_at_least_one_row_when_free_memory_is_tiny(monkeypatch):
    from unittest import mock

    from nbn import TensorVariableElimination

    model = _random_discrete(seed=3)
    nodes = list(model.dag.topological_order())
    _fake_psutil(monkeypatch, total=32 * GiB, available=1)
    eng = TensorVariableElimination()
    with mock.patch.object(eng, "_query_batch_chunked",
                           wraps=eng._query_batch_chunked) as spy:
        out = eng.query_batch(model, [nodes[-1]], {nodes[0]: torch.tensor([0, 1, 2])})
    assert spy.call_args.kwargs["chunk"] == 1  # exact, just slower — never refused
    assert out.shape == (3, 3)


def test_n1_router_chunks_at_its_host_fraction(monkeypatch):
    from unittest import mock

    from nbn import TensorVariableElimination
    from nbn.inference.hybrid import HybridRouter

    model = _random_discrete(seed=3)
    nodes = list(model.dag.topological_order())
    t, e = nodes[-1], nodes[0]
    B = 64
    probe = TensorVariableElimination()
    peak_b = probe.estimate_peak_bytes(model, t, (e,), B=B)
    cap = int(peak_b / 0.85)  # 0.75 * cap < peak_b <= 1.0 * cap
    _fake_psutil(monkeypatch, total=cap, available=cap)
    ev = {e: torch.arange(B) % 3}
    direct = TensorVariableElimination(memory_safety=1.0)
    with mock.patch.object(direct, "_query_batch_chunked") as spy:
        direct.query_batch(model, [t], ev)
    assert not spy.called  # VE alone: fits the whole capacity
    router = HybridRouter()
    router._ve.memory_safety = 1.0
    with mock.patch.object(router._ve, "_query_batch_chunked",
                           wraps=router._ve._query_batch_chunked) as spy:
        out = router.query_batch(model, [t], ev)
    assert router.last_engine == "tensor_ve" and spy.called
    assert out.shape == (B, 3)


# --------------------------------------------------------------------------- #
# N3 / N6 / N8 — clear errors and visible downgrades
# --------------------------------------------------------------------------- #


def test_n3_ve_on_hybrid_net_raises_the_discrete_error():
    from nbn import TensorVariableElimination
    from tests.unit.test_review_fixes_inference import _hybrid_mdn_parent_model

    model = _hybrid_mdn_parent_model()
    eng = TensorVariableElimination()
    with pytest.raises(ValueError, match="only supports discrete"):
        eng.query(model, ["D"], {"A": torch.tensor(1)})
    with pytest.raises(ValueError, match="only supports discrete"):
        eng.query_batch(model, ["D"], {"A": torch.tensor([1, 2])})


def test_n6_treewidth_downgrade_warns_once(caplog):
    import logging

    from nbn.inference.hybrid import HybridRouter

    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    router = HybridRouter(treewidth_threshold=0, n_samples=64)
    with caplog.at_level(logging.DEBUG, logger="nbn.inference.hybrid"):
        for v in (0, 1):
            router.query(model, [nodes[-1]], {nodes[0]: torch.tensor(v)})
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "treewidth" in warnings[0].getMessage()
    assert router.last_engine == "likelihood_weighting"


def test_n8_router_empty_targets_is_a_clear_error():
    from nbn.inference.hybrid import HybridRouter

    model = _random_discrete()
    with pytest.raises(ValueError, match="targets"):
        HybridRouter().query(model, [], {})
    with pytest.raises(ValueError, match="targets"):
        HybridRouter().query_batch(model, [], {})


def test_n8_ve_query_with_a_batch_points_to_query_batch():
    from nbn import TensorVariableElimination

    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    eng = TensorVariableElimination()
    for kw in ({"evidence": {nodes[0]: torch.tensor([0, 1])}},
               {"evidence": {}, "do": {nodes[0]: torch.tensor([0, 1])}}):
        with pytest.raises(ValueError, match="query_batch"):
            eng.query(model, [nodes[-1]], **kw)
        with pytest.raises(ValueError, match="query_batch"):
            eng.query(model, [nodes[-1], nodes[-2]], **kw)


def test_n8_reroute_message_keeps_interventions_apart(monkeypatch, caplog):
    import logging

    from nbn.inference import tensor_ve
    from nbn.inference.hybrid import HybridRouter

    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    monkeypatch.setattr(tensor_ve, "device_capacity_bytes", lambda d: 1)
    router = HybridRouter(n_samples=64)
    with caplog.at_level(logging.WARNING, logger="nbn.inference.hybrid"):
        router.query(model, [nodes[-1]], {nodes[1]: torch.tensor(0)},
                     do={nodes[0]: torch.tensor(1)})
    msg = caplog.records[-1].getMessage()
    assert f"P({nodes[-1]} | {nodes[1]}; do({nodes[0]}))" in msg
