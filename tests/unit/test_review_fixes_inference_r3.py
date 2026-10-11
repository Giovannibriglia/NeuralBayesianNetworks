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
