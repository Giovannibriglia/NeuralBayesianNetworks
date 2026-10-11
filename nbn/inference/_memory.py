"""Device memory headroom and out-of-memory recognition, for every backend.

Exact variable elimination allocates intermediate factors whose size is the
product of their variables' cardinalities, so a plan can need terabytes even
on a modest network.  The engines therefore compare an estimated peak with
the *headroom* of the device the query runs on, and the router treats an
allocation failure on any backend as "use likelihood weighting instead".

Headroom per backend
--------------------
* **cuda** (any index, e.g. ``cuda:1``): driver-free memory from
  ``torch.cuda.mem_get_info`` *plus* the blocks PyTorch's caching allocator
  holds reserved but unused (``memory_reserved - memory_allocated``).  The
  driver reports those cached blocks as used, yet this process can reuse
  them without a new driver allocation; ignoring them made the guard
  over-chunk after any large op (#264: 151 MiB "free" on an 8 GB card with
  7.5 GB cached → 30-500 passes instead of one).
* **mps**: ``torch.mps.recommended_max_memory() - current_allocated_memory()``
  where available, further bounded by host available memory plus MPS's own
  cached blocks (unified memory is shared with the CPU).
* **cpu**: host available memory (``psutil`` when installed, otherwise
  ``MemAvailable`` from ``/proc/meminfo``).

``None`` means "unknown"; callers then skip the memory check (the pre-guard
behaviour) unless an explicit budget was configured.
"""
from __future__ import annotations

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


def is_oom_error(exc: BaseException) -> bool:
    """True when ``exc`` is an allocation failure on any backend."""
    if isinstance(exc, (OutOfMemoryError, MemoryError)):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(m in msg for m in _OOM_MESSAGES)
    return False


def host_available_bytes() -> int | None:
    """Memory the OS can still hand this process (``None`` if unknown)."""
    try:
        import psutil  # optional (bench extra)

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - missing or broken psutil → next source
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def device_headroom_bytes(device) -> int | None:
    """Bytes a new allocation on ``device`` can use (see module docstring)."""
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
