"""Device memory budgets and out-of-memory recognition, for every backend.

Exact variable elimination allocates intermediate factors whose size is the
product of their variables' cardinalities, so a plan can need terabytes even
on a modest network.  The engines therefore compare an estimated peak with a
budget for the device the query runs on, and the router treats an allocation
failure on any backend as "use likelihood weighting instead".

Two quantities, used for different backends
-------------------------------------------
* **Headroom** — what a new allocation can use *right now*.  Used on CUDA,
  where device memory belongs to the process(es) on the card and the pre-
  guard behaviour already compared with live free memory:
  ``torch.cuda.mem_get_info`` free memory *plus* the blocks PyTorch's
  caching allocator holds reserved but unused (``memory_reserved -
  memory_allocated``).  The driver reports those cached blocks as used, yet
  this process can reuse them; ignoring them made the guard over-chunk after
  any large op (#264: 151 MiB "free" on an 8 GB card with 7.5 GB cached).
* **Capacity** — a *stable* size of the memory pool.  Used on CPU and MPS:
  momentary free host memory moves with every other process on the machine,
  so a budget built from it made the same query exact on an idle machine and
  approximate (LW) under memory pressure (R5).  CPU: physical RAM, bounded by
  the cgroup limit (Docker / Slurm / Kubernetes); MPS:
  ``torch.mps.recommended_max_memory()`` (Metal's working-set
  recommendation), bounded by the host capacity.

Both return ``None`` when unknown; callers then skip the memory check (the
pre-guard behaviour) unless an explicit budget was configured.
"""
from __future__ import annotations

import gc
import os

import torch

# Raised by the pre-allocation guards on *every* device.  It is torch's
# device-generic OOM type (``torch.OutOfMemoryError`` in recent releases is
# the same class), so the benchmark runner's existing classification and
# ``except torch.cuda.OutOfMemoryError`` callers keep working.
OutOfMemoryError = torch.cuda.OutOfMemoryError

# Lower-cased fragments of allocation-failure messages that do not come with
# a dedicated exception type: MPS ("MPS backend out of memory", "Invalid
# buffer size" for a single tensor over the Metal buffer limit), the CPU
# allocator ("DefaultCPUAllocator: can't allocate memory"), and older CUDA
# builds that raised a plain RuntimeError.
_OOM_MESSAGES = (
    "out of memory",
    "can't allocate memory",
    "cannot allocate memory",
    "not enough memory",
    "invalid buffer size",
    "failed to allocate",
)

# cgroup filesystem root (a module attribute so tests can point it at a fake
# tree).  v2 exposes memory.max / memory.current at the root of the process's
# cgroup namespace; v1 under ``memory/``.
_CGROUP_ROOT = "/sys/fs/cgroup"
# cgroup v1 reports "no limit" as a huge page-aligned number (2^63 - 4096).
_CGROUP_UNLIMITED = 1 << 60


def is_oom_error(exc: BaseException) -> bool:
    """True when ``exc`` is an allocation failure on any backend."""
    if isinstance(exc, (OutOfMemoryError, MemoryError)):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(m in msg for m in _OOM_MESSAGES)
    return False


def _read_int(path: str) -> int | None:
    try:
        with open(path) as fh:
            raw = fh.read().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _stat_value(path: str, key: str) -> int:
    """One counter of a cgroup ``memory.stat`` file (0 if unknown)."""
    try:
        with open(path) as fh:
            for line in fh:
                name, _, value = line.partition(" ")
                if name == key:
                    return int(value)
    except (OSError, ValueError):
        pass
    return 0


def cgroup_memory() -> tuple[int | None, int | None]:
    """``(limit, usage)`` of this process's memory cgroup, ``None`` if absent.

    Containers and batch schedulers cap a job far below the host's RAM, and
    exceeding the cap gets the job OOM-killed, so host-wide figures overstate
    what this process may use.  Usage excludes inactive page cache, which the
    kernel reclaims before killing anything (the ``docker stats`` rule).
    """
    v2_limit = _read_int(os.path.join(_CGROUP_ROOT, "memory.max"))
    v2_usage = _read_int(os.path.join(_CGROUP_ROOT, "memory.current"))
    if v2_limit is not None or v2_usage is not None:
        if v2_usage is not None:
            v2_usage -= _stat_value(
                os.path.join(_CGROUP_ROOT, "memory.stat"), "inactive_file")
        return v2_limit, v2_usage
    v1 = os.path.join(_CGROUP_ROOT, "memory")
    limit = _read_int(os.path.join(v1, "memory.limit_in_bytes"))
    usage = _read_int(os.path.join(v1, "memory.usage_in_bytes"))
    if limit is not None and limit >= _CGROUP_UNLIMITED:
        limit = None
    if usage is not None:
        usage -= _stat_value(os.path.join(v1, "memory.stat"), "total_inactive_file")
    return limit, usage


def host_total_bytes() -> int | None:
    """Stable host memory capacity: physical RAM, capped by the cgroup limit."""
    total = None
    try:
        import psutil  # optional (bench extra)

        total = int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 - missing or broken psutil → sysconf
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            total = None
    limit, _ = cgroup_memory()
    if limit is not None:
        total = limit if total is None else min(total, limit)
    return total


def host_available_bytes() -> int | None:
    """Memory the OS can still hand this process (``None`` if unknown)."""
    avail = None
    try:
        import psutil  # optional (bench extra)

        avail = int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - missing or broken psutil → /proc
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) * 1024
                        break
        except (OSError, ValueError, IndexError):
            pass
    limit, usage = cgroup_memory()
    if limit is not None and usage is not None:
        room = max(0, limit - usage)
        avail = room if avail is None else min(avail, room)
    return avail


def device_capacity_bytes(device) -> int | None:
    """Stable memory capacity of a host-memory device (CPU / MPS), else ``None``."""
    device = torch.device(device)
    if device.type == "cpu":
        return host_total_bytes()
    if device.type == "mps":
        host = host_total_bytes()
        rec = getattr(getattr(torch, "mps", None), "recommended_max_memory", None)
        if rec is None:
            return host
        try:
            cap = int(rec())
        except RuntimeError:
            return host
        return cap if host is None else min(cap, host)
    return None


def device_headroom_bytes(device) -> int | None:
    """Bytes a new allocation on ``device`` can use now (see module docstring)."""
    device = torch.device(device)
    if device.type == "cuda":
        try:
            free, _total = torch.cuda.mem_get_info(device)
        except (RuntimeError, AssertionError):
            return None
        try:
            cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        except (RuntimeError, AssertionError):
            cached = 0
        return int(free + max(0, cached))
    if device.type == "mps":
        mps = getattr(torch, "mps", None)
        rec = getattr(mps, "recommended_max_memory", None)
        cur = getattr(mps, "current_allocated_memory", None)
        drv = getattr(mps, "driver_allocated_memory", None)
        host = host_available_bytes()
        if rec is None or cur is None:
            return host
        try:
            allocated = int(cur())
            head = int(rec()) - allocated
            cached = max(0, int(drv()) - allocated) if drv is not None else 0
        except RuntimeError:
            return host
        if host is not None:
            head = min(head, host + cached)
        return max(0, head)
    if device.type == "cpu":
        return host_available_bytes()
    return None


def release_cached_memory(device) -> None:
    """Return freed-but-cached blocks after an OOM, before retrying elsewhere.

    Called once the failed attempt's exception (and with it the traceback
    frames holding its intermediate tensors) has been dropped: the caching
    allocators keep those blocks reserved, and the fallback engine on the
    same device needs them back.
    """
    gc.collect()  # frames of the failed attempt may sit in reference cycles
    device = torch.device(device)
    if device.type == "cuda":
        try:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        except (RuntimeError, AssertionError):
            pass
    elif device.type == "mps":
        empty = getattr(getattr(torch, "mps", None), "empty_cache", None)
        if empty is not None:
            try:
                empty()
            except RuntimeError:
                pass
