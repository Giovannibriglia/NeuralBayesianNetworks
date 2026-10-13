"""Peak-memory model of exact variable elimination (#178).

``TensorVariableElimination`` refuses, chunks or (through ``HybridRouter``)
reroutes a query whose estimated peak exceeds the device budget, so the
estimate must be an *upper bound* of the memory the call makes the device
hold.  It is computed in three layers:

1. **The tape** (``_plan_cost``): a walk of the elimination plan that records,
   in order, every allocation and free ``query_batch`` performs — conditioned
   factors, each product of a step (operands are broadcast *views*; only the
   sum is materialised), ``torch.logsumexp``'s max / mask / temporary /
   result, and the answer (softmax; a joint answer's reshape copy and one-hot
   embedding).  Sizes are ``(batch_free, per_row)`` element counts, so one
   walk prices every ``B``.  Call-dependent tensors are added by
   ``_PeakModel``: evidence materialised from Python lists, int64 index
   copies, the validation temporaries, and — on a cold factor cache — every
   CPT's tabulation (``Mechanism.tabulate_working_bytes``).
2. **The device's allocator**, replayed over the tape (``_device_peak``):
   * CPU — the live bytes (64 B alignment): the running sum of the tape;
   * MPS — each live buffer is charged its share of the heap it lives in:
     buffers <= 1 MiB share 8 MiB heaps, <= 10 MiB 32 MiB heaps, <= 512 MiB
     1 GiB heaps (a 342 MiB buffer fits only twice in one), larger ones get
     an exact heap rounded to 16 MiB (measured, Apple unified memory), plus
     one partly filled heap per size class in use;
   * CUDA — a simulation of ``CUDACachingAllocator`` (best-fit reuse of
     cached blocks, splitting, 2 MiB / 20 MiB segments for blocks <= 1 MiB /
     < 10 MiB, 2 MiB-rounded segments beyond), whose peak *reserved* bytes
     bound what the call takes from the driver, freed-but-unusable cached
     segments included.  The settings of ``PYTORCH_CUDA_ALLOC_CONF`` that
     change the reserve are followed (``_cuda_policy``: ``max_split_size_mb``,
     ``max_non_split_rounding_mb``, ``roundup_power2_divisions``,
     ``expandable_segments``, ``large_segment_size_mb``); another backend
     or an unknown key falls back to a bound that credits no free.
3. **A margin**: ``_LIVESET_SAFETY`` (10 %).

``scripts/ve_memory_validation.py`` measures the bound on CPU (exact), MPS
(heap level) and CUDA (reserved); ``docs/ve-memory-model.md`` has the numbers.
"""
from __future__ import annotations

import bisect
import functools
import math
import os
from typing import Dict, Mapping, NamedTuple, Sequence

import numpy as np
import torch

_LIVESET_SAFETY = 1.10
_KiB, _MiB, _GiB = 1 << 10, 1 << 20, 1 << 30

# Bytes per element of the tapes' tensors: factors take the CPTs' dtype
# (``_FACTOR``, resolved per call); the rest have a fixed element size.
_FACTOR = -1
_INT64, _BOOL = 8, 1


# --------------------------------------------------------------------------- #
# The tape
# --------------------------------------------------------------------------- #

class _Tape:
    """Allocations and frees in program order; sizes ``a + B*r`` elements."""

    __slots__ = ("a", "r", "es", "sign", "ref")

    def __init__(self) -> None:
        self.a: list[int] = []
        self.r: list[int] = []
        self.es: list[int] = []
        self.sign: list[int] = []
        self.ref: list[int] = []   # for a free: the index of its allocation

    def alloc(self, a: int, r: int, es: int = _FACTOR) -> int:
        self.a.append(a)
        self.r.append(r)
        self.es.append(es)
        self.sign.append(1)
        self.ref.append(len(self.ref))
        return len(self.ref) - 1

    def free(self, token: int) -> None:
        self.a.append(self.a[token])
        self.r.append(self.r[token])
        self.es.append(self.es[token])
        self.sign.append(-1)
        self.ref.append(token)

    def frozen(self) -> tuple:
        return (tuple(self.a), tuple(self.r), tuple(self.es),
                tuple(self.sign), tuple(self.ref))


def _fold(tape: tuple, rows: int) -> tuple:
    """``tape`` with its row count fixed at ``rows`` (sizes become constants)."""
    a, r, es, sign, ref = tape
    return tuple(x + rows * y for x, y in zip(a, r)), (0,) * len(a), es, sign, ref


def _concat(tapes: Sequence[tuple]) -> tuple:
    """Concatenate frozen tapes, re-basing the free -> alloc references."""
    a, r, es, sign, ref = [], [], [], [], []
    for t in tapes:
        if not t:
            continue
        off = len(ref)
        a += t[0]
        r += t[1]
        es += t[2]
        sign += t[3]
        ref += [i + off for i in t[4]]
    return a, r, es, sign, ref


# --------------------------------------------------------------------------- #
# Device allocators
# --------------------------------------------------------------------------- #

def _round_up(x: np.ndarray, q: int) -> np.ndarray:
    return np.ceil(x / q) * q


def _accept_any(block: int, req: int, pool) -> bool:
    return True


class _Policy(NamedTuple):
    """How a caching allocator rounds, pools and grows (``_segment_peak``)."""

    round: object      # request bytes -> block bytes
    pool: object       # block bytes -> pool key
    segment: object    # (block bytes, pool key) -> new segment bytes
    split: object      # (rest bytes, pool key, block bytes) -> keep the rest as a free block
    accept: object = _accept_any  # (cached block bytes, block bytes, pool key) -> may it serve?


def _mps_round(s: float) -> int:
    """MPS buffer sizes (measured): 256 B granules up to 1 MiB, then a 64th
    of the next power of two (16 KiB at least)."""
    if s <= _MiB:
        return max(256, math.ceil(s / 256) * 256)
    q = max(16 * _KiB, (1 << math.ceil(math.log2(s))) // 64)
    return math.ceil(s / q) * q


def _mps_pool(req: int):
    if req <= _MiB:
        return "small"
    if req <= 10 * _MiB:
        return "medium"
    return "large" if req <= 512 * _MiB else ("own", req)


_MPS = _Policy(
    round=_mps_round,
    pool=_mps_pool,
    segment=lambda req, pool: {"small": 8 * _MiB, "medium": 32 * _MiB,
                               "large": _GiB}.get(pool, req),  # type: ignore[arg-type]
    split=lambda rest, pool, req: rest >= 256,
)


# --------------------------------------------------------------------------- #
# CUDA allocator configuration (``PYTORCH_CUDA_ALLOC_CONF``)
# --------------------------------------------------------------------------- #

# Keys of ``CUDACachingAllocator`` that change what the allocator reserves
# (``_cuda_policy`` models them) ...
_CUDA_MODELLED = frozenset({"max_split_size_mb", "max_non_split_rounding_mb",
                            "roundup_power2_divisions", "expandable_segments",
                            "large_segment_size_mb"})
# ... keys that do not: garbage collection only releases cached blocks (the
# reserve can only be lower), the rest configure host (pinned) memory, locks,
# graph capture or the per-process cap.
_CUDA_NO_EFFECT = frozenset({
    "garbage_collection_threshold", "release_lock_on_cudamalloc",
    "release_lock_on_hipmalloc", "pinned_use_cuda_host_register",
    "pinned_use_hip_host_register", "pinned_num_register_threads",
    "pinned_use_background_threads", "pinned_reserve_segment_size_mb",
    "graph_capture_record_stream_reuse", "per_process_memory_fraction",
})
# roundup_power2_divisions buckets: the power-of-two "size intervals between
# 1MB and 64GB" (c10/core/AllocatorConfig.h, torch 2.14), i.e. [1, 2) MiB ..
# [32, 64) GiB = 16, larger sizes in the last one.  The constant itself
# (kRoundUpPowerOfTwoIntervals) is defined in AllocatorConfig.cpp, which is not
# shipped with the wheel, so this follows the header's description; 16 vs 17
# only changes requests of 32 GiB and more.
_ROUNDUP_INTERVALS = 16


class _CudaConf(NamedTuple):
    """The settings of ``CUDACachingAllocator`` the reserve depends on."""

    max_split: float = math.inf         # blocks this large are never split
    non_split_rounding: int = 20 * _MiB  # oversize blocks may exceed a request by this
    divisions: tuple = (0,) * _ROUNDUP_INTERVALS  # roundup_power2_divisions per bucket
    expandable: bool = False
    large_segment: int = 20 * _MiB
    known: bool = True                  # False: a setting the model cannot follow


def _parse_cuda_conf(text: str | None) -> _CudaConf:
    """``_CudaConf`` of a ``PYTORCH_CUDA_ALLOC_CONF`` string (``parseArgs``).

    ``backend:cudaMallocAsync`` and any key this module does not know give
    ``known=False``: the estimate then falls back to the no-reuse bound.
    """
    conf = _CudaConf()
    if not text:
        return conf
    items, depth, cur = [], 0, []
    for ch in text:
        depth += (ch == "[") - (ch == "]")
        if ch == "," and depth == 0:
            items.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    items.append("".join(cur))
    for item in items:
        if not item.strip():
            continue
        key, _, val = item.partition(":")
        key, val = key.strip(), val.strip()
        try:
            if key == "max_split_size_mb":
                conf = conf._replace(max_split=int(val) * _MiB)
            elif key == "max_non_split_rounding_mb":
                conf = conf._replace(non_split_rounding=int(val) * _MiB)
            elif key == "large_segment_size_mb":
                conf = conf._replace(large_segment=int(val) * _MiB)
            elif key == "expandable_segments":
                conf = conf._replace(expandable=val.lower() == "true")
            elif key == "roundup_power2_divisions":
                conf = conf._replace(divisions=_parse_divisions(val))
            elif key == "backend":
                if val.lower() != "native":
                    conf = conf._replace(known=False)
            elif key not in _CUDA_NO_EFFECT:
                conf = conf._replace(known=False)
        except ValueError:
            conf = conf._replace(known=False)
    return conf


def _parse_divisions(val: str) -> tuple:
    """``roundup_power2_divisions``: one value for every size, or
    ``[64:8,256:4,>:1]`` — a value per power-of-two bucket of MiB (the first
    entry also covers every smaller bucket, ``>`` every larger one)."""
    n = _ROUNDUP_INTERVALS
    if not val.startswith("["):
        return (int(val),) * n
    div = [0] * n
    last, first = 0, True
    for entry in val.strip("[]").split(","):
        bound, _, value = entry.partition(":")
        bound, value = bound.strip(), int(value)
        if bound == ">":
            for i in range(last, n):
                div[i] = value
            continue
        b = int(bound)
        if b <= 0 or b & (b - 1):
            raise ValueError(bound)
        index = min(b.bit_length() - 1, n - 1)
        if first:
            for i in range(index):
                div[i] = value
            first = False
        div[index] = value
        last = index
    return tuple(div)


def _round_pow2_division(size: int, divisions: int) -> int:
    """``roundup_power2_next_division``: round up to the next multiple of
    (the power of two below ``size``) / ``divisions``."""
    if size <= 4 or divisions <= 1 or size & (size - 1) == 0:
        return size
    floor = 1 << (size.bit_length() - 1)
    step = floor >> (divisions.bit_length() - 1)
    if step == 0:
        return floor << 1
    base = size & ~(step - 1)
    return size if base == size else base + step


def _cuda_round(s: float, divisions: tuple = _CudaConf().divisions) -> int:
    """``round_size``: 512 B blocks, or power-of-two divisions when set."""
    size = max(512, math.ceil(s))
    bucket = min(max(size.bit_length() - 1 - 20, 0), _ROUNDUP_INTERVALS - 1)
    d = divisions[bucket]
    if d > 1 and size > 512 * d:
        return _round_pow2_division(size, d)
    return math.ceil(size / 512) * 512


def _cuda_policy(conf: _CudaConf) -> _Policy:
    """``_Policy`` of ``CUDACachingAllocator`` under ``conf`` (``get_free_block``,
    ``should_split``, ``get_allocation_size``)."""
    large = conf.large_segment

    def segment(req: int, pool) -> int:
        if pool == "small":
            return 2 * _MiB
        return large if req < 10 * _MiB else math.ceil(req / (2 * _MiB)) * 2 * _MiB

    def split(rest: int, pool, req: int) -> bool:
        if pool == "small" or conf.expandable:
            return rest >= 512
        return req < conf.max_split and rest > _MiB

    def accept(block: int, req: int, pool) -> bool:
        # Only the best-fitting cached block is examined; an oversize block
        # (>= max_split) never serves a smaller request, and an oversize
        # request takes a block at most ``non_split_rounding`` larger.
        if req < conf.max_split and block >= conf.max_split:
            return False
        return not (req >= conf.max_split and block >= req + conf.non_split_rounding)

    return _Policy(
        round=lambda s: _cuda_round(s, conf.divisions),
        pool=lambda req: "small" if req <= _MiB else "large",
        segment=segment,
        split=split,
        accept=accept,
    )


@functools.lru_cache(maxsize=16)
def _cuda_conf(text: str | None) -> _CudaConf:
    return _parse_cuda_conf(text)


# Environment variables torch reads for the caching allocator's settings.
# torch 2.14 (installed headers): ``CUDAAllocatorConfig::instance()``
# (c10/cuda/CUDAAllocatorConfig.h) reads PYTORCH_CUDA_ALLOC_CONF, then —
# on ROCm only — PYTORCH_HIP_ALLOC_CONF, then PYTORCH_ALLOC_CONF; but the
# settings this module models (max_split_size_mb, roundup_power2_divisions,
# expandable_segments, ...) live in ``AcceleratorAllocatorConfig``
# (c10/core/AllocatorConfig.h), documented as "the primary environment
# variable is PYTORCH_ALLOC_CONF; PYTORCH_CUDA_ALLOC_CONF is also supported
# with lower priority" — and its reader (AllocatorConfig.cpp) is not shipped,
# so the precedence between the two cannot be confirmed (with both set,
# torch 2.14's settings getter even fails an internal assert).  Hence no
# precedence is assumed: every set variable is a candidate (``cuda_allocator_confs``).
_ALLOC_CONF_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


def _live_allocator_settings() -> str:
    """The settings string torch last parsed (``getAllocatorSettings``): what
    it read from the environment at start-up, or the last
    ``torch._C._accelerator_setAllocatorSettings`` call.  "" when unavailable."""
    getter = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
    if getter is None:
        return ""
    try:
        return getter() or ""
    except Exception:  # noqa: BLE001  — e.g. torch 2.14 with two variables set
        return ""


def cuda_allocator_confs() -> tuple[str, ...]:
    """Every configuration the CUDA allocator may be running under.

    The estimate prices the *largest* reserve over them, so it can never be
    optimistic about which one is in effect:

    * the default ("") — torch reads the environment once, when the
      allocator is first configured, so a variable set (or changed) after
      that, e.g. from a notebook, is not in effect (#301 review 4);
    * each set variable of ``_ALLOC_CONF_VARS`` (the HIP one on ROCm only) —
      their precedence is not fixed here (see ``_ALLOC_CONF_VARS``);
    * the settings string torch reports (``_live_allocator_settings``).

    ``torch.cuda.memory._set_allocator_settings`` updates are seen only as
    their last string; callers who use it should set the variable too.  The
    backend torch reports is prefixed to every candidate.
    """
    names = [v for v in _ALLOC_CONF_VARS
             if v != "PYTORCH_HIP_ALLOC_CONF" or getattr(torch.version, "hip", None)]
    texts = {""} | {os.environ.get(v, "") for v in names} | {_live_allocator_settings()}
    backend = "native"
    if torch.cuda.is_available():
        try:
            backend = torch.cuda.get_allocator_backend()
        except Exception:  # noqa: BLE001  — no device, driver error
            backend = "native"
    if backend != "native":
        texts = {f"backend:{backend},{t}" if t else f"backend:{backend}" for t in texts}
    return tuple(sorted(texts))


class _Segment:
    __slots__ = ("pool", "blocks", "offs")

    def __init__(self, pool) -> None:
        self.pool = pool
        self.blocks: dict[int, list] = {}   # offset -> [size, is_free]
        self.offs: list[int] = []           # sorted offsets


def _segment_peak(sizes: Sequence[float], sign: Sequence[int], ref: Sequence[int],
                  policy: _Policy) -> float:
    """Peak bytes of segments (CUDA) / heaps (MPS) held while replaying the tape.

    A request takes the smallest free block of its pool that fits (splitting
    off the rest) if the policy accepts it, else a new segment; a freed block
    merges with free neighbours in its segment and stays there — reserved
    memory never shrinks during the call.  This is ``CUDACachingAllocator``
    without expandable segments (``_cuda_policy``), and the behaviour measured for Metal heaps
    under PyTorch's MPS allocator (``_MPS``): freed buffers return their
    space to the heap, holes coalesce, and a request no hole fits opens a new
    heap even when the heap's total free space would suffice.
    """
    pools: dict = {}                       # pool key -> sorted (size, seg, off)
    segs: list[_Segment] = []
    owner: dict[int, tuple[int, int]] = {}
    reserved = peak = 0

    def put_free(seg: int, off: int, size: int) -> None:
        sg = segs[seg]
        if off not in sg.blocks:
            bisect.insort(sg.offs, off)
        sg.blocks[off] = [size, True]
        bisect.insort(pools.setdefault(sg.pool, []), (size, seg, off))

    def take_free(seg: int, off: int) -> int:
        sg = segs[seg]
        size = sg.blocks[off][0]
        pools[sg.pool].remove((size, seg, off))
        return size

    for i, s in enumerate(sizes):
        if sign[i] > 0:
            req = policy.round(s)
            key = policy.pool(req)
            free = pools.setdefault(key, [])
            j = bisect.bisect_left(free, (req, -1, -1))
            if j < len(free) and policy.accept(free[j][0], req, key):
                size, seg, off = free.pop(j)
            else:
                size = policy.segment(req, key)
                seg, off = len(segs), 0
                segs.append(_Segment(key))
                segs[seg].offs.append(0)
                reserved += size
                peak = max(peak, reserved)
            sg = segs[seg]
            rest = size - req
            if rest > 0 and policy.split(rest, key, req):
                put_free(seg, off + req, rest)
                size = req
            sg.blocks[off] = [size, False]
            owner[i] = (seg, off)
        else:
            seg, off = owner.pop(ref[i])
            sg = segs[seg]
            size = sg.blocks[off][0]
            k = bisect.bisect_left(sg.offs, off)
            if k + 1 < len(sg.offs):
                nxt = sg.offs[k + 1]
                if sg.blocks[nxt][1]:
                    size += take_free(seg, nxt)
                    del sg.blocks[nxt]
                    sg.offs.pop(k + 1)
            if k > 0:
                prv = sg.offs[k - 1]
                if sg.blocks[prv][1]:
                    size += take_free(seg, prv)
                    del sg.blocks[off]
                    sg.offs.pop(k)
                    off = prv
            put_free(seg, off, size)
    return float(peak)


def _expandable_peak(sizes: Sequence[float], sign: Sequence[int], ref: Sequence[int],
                     policy: _Policy, large_quantum: int) -> float:
    """Peak mapped bytes of ``CUDACachingAllocator`` with ``expandable_segments``.

    One mapping per pool grows in place in units of 2 MiB (small pool) or
    ``large_quantum`` (large pool, ``kLargeBuffer`` / ``large_segment_size_mb``).
    A request takes the smallest free block that fits among those not at the
    mapped end; failing that, the free block at the end, extended by mapping
    what is missing (``find_expandable_block`` / ``try_allocate_expandable_block``);
    failing that, newly mapped space.  Freed blocks coalesce.  Mapped memory
    is never released during the call.
    """
    pools: dict = {}
    owner: dict[int, tuple] = {}
    reserved = peak = 0

    def pool_of(key):
        if key not in pools:
            pools[key] = {"blocks": {}, "offs": [], "free": [], "mapped": 0}
        return pools[key]

    def put_free(pl, off, size):
        if off not in pl["blocks"]:
            bisect.insort(pl["offs"], off)
        pl["blocks"][off] = [size, True]
        bisect.insort(pl["free"], (size, off))

    def take_free(pl, off):
        size = pl["blocks"][off][0]
        pl["free"].remove((size, off))
        return size

    for i, s in enumerate(sizes):
        if sign[i] > 0:
            req = policy.round(s)
            key = policy.pool(req)
            pl = pool_of(key)
            quantum = 2 * _MiB if key == "small" else large_quantum
            chosen = None
            j = bisect.bisect_left(pl["free"], (req, -1))
            while j < len(pl["free"]):
                size, off = pl["free"][j]
                if off + size != pl["mapped"]:           # not the block at the end
                    chosen = (size, off) if policy.accept(size, req, key) else None
                    break
                j += 1
            if chosen is None:
                tail = pl["offs"][-1] if pl["offs"] else None
                if tail is not None and pl["blocks"][tail][1] and \
                        pl["blocks"][tail][0] + tail == pl["mapped"]:
                    size = take_free(pl, tail)
                    off = tail
                    if size < req:
                        grow = math.ceil((req - size) / quantum) * quantum
                        pl["mapped"] += grow
                        reserved += grow
                        size += grow
                    chosen = (size, off)
                else:
                    grow = math.ceil(req / quantum) * quantum
                    chosen = (grow, pl["mapped"])
                    bisect.insort(pl["offs"], pl["mapped"])
                    pl["mapped"] += grow
                    reserved += grow
                peak = max(peak, reserved)
            else:
                pl["free"].remove(chosen)
            size, off = chosen
            rest = size - req
            if rest > 0 and policy.split(rest, key, req):
                put_free(pl, off + req, rest)
                size = req
            pl["blocks"][off] = [size, False]
            owner[i] = (key, off)
        else:
            key, off = owner.pop(ref[i])
            pl = pools[key]
            size = pl["blocks"][off][0]
            k = bisect.bisect_left(pl["offs"], off)
            if k + 1 < len(pl["offs"]):
                nxt = pl["offs"][k + 1]
                if pl["blocks"][nxt][1]:
                    size += take_free(pl, nxt)
                    del pl["blocks"][nxt]
                    pl["offs"].pop(k + 1)
            if k > 0:
                prv = pl["offs"][k - 1]
                if pl["blocks"][prv][1]:
                    size += take_free(pl, prv)
                    del pl["blocks"][off]
                    pl["offs"].pop(k)
                    off = prv
            put_free(pl, off, size)
    return float(peak)


def _cuda_reserved_peak(sizes: Sequence[float], sign: Sequence[int],
                        ref: Sequence[int], *,
                        conf: str | tuple[str, ...] | None = None) -> float:
    """Peak reserved bytes of ``CUDACachingAllocator`` replaying the tape.

    ``conf`` is an allocator settings string, or several (the largest reserve
    over them is returned); default: every configuration the process may be
    running under (``cuda_allocator_confs``).  Default configuration: a request rounded up to
    512 B takes the smallest cached free block of its pool (<= 1 MiB: small
    pool, else large) that fits, splitting it when the rest is >= 512 B
    (small) or > 1 MiB (large); without one, a new segment of 2 MiB (small),
    20 MiB (< 10 MiB) or the request rounded up to 2 MiB.  ``max_split_size_mb``
    stops splitting and reuse of oversize blocks, ``roundup_power2_divisions``
    rounds requests to power-of-two divisions, ``expandable_segments`` grows
    one mapping per pool (``_expandable_peak``).  Under another backend or an
    unknown setting no free is credited: every allocation is priced as new.
    """
    texts = cuda_allocator_confs() if conf is None else conf
    if not isinstance(texts, str):
        confs = {_cuda_conf(t) for t in texts}
        return max(_cuda_one_peak(sizes, sign, ref, c) for c in confs)
    return _cuda_one_peak(sizes, sign, ref, _cuda_conf(texts))


def _cuda_one_peak(sizes: Sequence[float], sign: Sequence[int], ref: Sequence[int],
                   c: _CudaConf) -> float:
    """``_cuda_reserved_peak`` under one parsed configuration."""
    if not c.known:
        keep = [i for i, g in enumerate(sign) if g > 0]
        return _segment_peak([sizes[i] for i in keep], [1] * len(keep),
                             list(range(len(keep))), _cuda_policy(_CudaConf()))
    if c.expandable:
        return _expandable_peak(sizes, sign, ref, _cuda_policy(c), c.large_segment)
    return _segment_peak(sizes, sign, ref, _cuda_policy(c))


def _mps_heap_peak(sizes: Sequence[float], sign: Sequence[int],
                   ref: Sequence[int], *, credit_frees: bool = False) -> float:
    """Peak bytes of Metal heaps the MPS allocator holds (``_segment_peak``).

    Heap classes measured on Apple unified memory (torch 2.14): 8 MiB heaps
    for buffers <= 1 MiB, 32 MiB up to 10 MiB, 1 GiB up to 512 MiB, an own
    heap beyond; buffer sizes per ``_mps_round``.

    Frees are not credited (``credit_frees=False``, the default): MPS runs
    asynchronously, and a buffer freed while the GPU still has queued work
    on it cannot be reused until that work completes — barley q2 at B=64
    grew the driver by 73 MiB without synchronisation and by 33 MiB with a
    ``synchronize()`` after every op (#301 review A).  When the GPU catches
    up is not known in advance, so every allocation is priced as new.
    """
    if not credit_frees:
        keep = [i for i, g in enumerate(sign) if g > 0]
        sizes, sign, ref = [sizes[i] for i in keep], [1] * len(keep), list(range(len(keep)))
    return _segment_peak(sizes, sign, ref, _MPS)


def _replay(tape: tuple, B: int, dtype_bytes: int, device_type: str,
            conf: str | tuple[str, ...] | None = None) -> tuple[float, float]:
    """``(peak, end)`` bytes the device holds replaying ``tape`` at ``B`` rows.

    CPU: the running sum of live bytes (64 B aligned) — ``end`` is what is
    still allocated afterwards.  MPS / CUDA: the peak of the simulated heaps
    / segments, which are never released during the call (``end == peak``);
    ``conf`` is the CUDA allocator configuration (default: the process's).
    """
    a, r, es, sign, ref = (np.asarray(x) for x in tape)
    if a.size == 0:
        return 0.0, 0.0
    es = np.where(es == _FACTOR, dtype_bytes, es)
    sizes = (a + B * r).astype(np.float64) * es
    if device_type == "cuda":
        peak = _cuda_reserved_peak(sizes.tolist(), sign.tolist(), ref.tolist(), conf=conf)
        return peak, peak
    if device_type == "mps":
        peak = _mps_heap_peak(sizes.tolist(), sign.tolist(), ref.tolist())
        return peak, peak
    running = np.cumsum(sign * _round_up(sizes, 64))
    return float(running[sign > 0].max()), float(running[-1])


# --------------------------------------------------------------------------- #
# The plan's tape
# --------------------------------------------------------------------------- #

class _Scope:
    """Scope-only stand-in for a ``LogFactor`` (variables + cardinalities).

    The peak estimate needs only factor scopes, so it can be computed from
    the DAG before any CPT is tabulated.
    """

    __slots__ = ("variables", "cardinalities")

    def __init__(self, variables, cardinalities) -> None:
        self.variables = list(variables)
        self.cardinalities = dict(cardinalities)


class _PlanCost(NamedTuple):
    """A plan's tape plus the largest single tensor (MPS element cap)."""

    tensor: tuple[int, int]   # (batch_free, per_row) elements of the largest tensor
    tape: tuple               # ``_Tape.frozen()`` of the elimination and the answer
    memo: dict                # (B, dtype, device, embed) -> replayed peak

    def peak(self, B: int, dtype_bytes: int, device_type: str, embed: tuple = ()) -> float:
        """Peak bytes of the plan (and a joint answer's embedding) at ``B``
        rows, memoised: repeated queries of one shape replay it once."""
        conf = cuda_allocator_confs() if device_type == "cuda" else None
        key = (B, dtype_bytes, device_type, embed, conf)
        if key not in self.memo:
            tape = _concat([self.tape, embed]) if embed else self.tape
            self.memo[key] = _replay(tape, B, dtype_bytes, device_type, conf)[0]
        return self.memo[key]

    def peak_rows(self, rows: tuple, dtype_bytes: int, device_type: str,
                  embed: tuple = ()) -> float:
        """Peak bytes of running the plan once per entry of ``rows``, in
        order, on one allocator: what a chunked batch leaves cached for the
        next chunk is replayed, not assumed reusable.  Memoised like ``peak``."""
        conf = cuda_allocator_confs() if device_type == "cuda" else None
        key = (rows, dtype_bytes, device_type, embed, conf)
        if key not in self.memo:
            one = _concat([self.tape, embed]) if embed else self.tape
            tape = _concat([_fold(one, n) for n in rows])
            self.memo[key] = _replay(tape, 1, dtype_bytes, device_type, conf)[0]
        return self.memo[key]


class _Factor(NamedTuple):
    scope: frozenset
    has_b: bool
    cards: Dict[str, int]
    token: int | None         # its allocation on the tape; None: a cached-CPT view


def _elems(scope, has_b: bool, cards) -> tuple[int, int]:
    n = math.prod(cards[v] for v in scope)
    return (0, n) if has_b else (n, 0)


def _plan_cost(
    plan: list[str],
    factors: Mapping[str, object],
    *,
    evidence_keys: tuple[str, ...],
    targets: tuple[str, ...] | None = None,
    target_cards: Mapping[str, int] | None = None,
    joint: bool | None = None,
) -> _PlanCost:
    """Record the allocations of eliminating ``plan`` over ``factors``.

    Mirrors ``query_batch`` step by step (``query`` allocates no more: it
    conditions by views).  ``factors`` maps node -> anything with
    ``variables`` / ``cardinalities`` (``LogFactor`` or ``_Scope``), in the
    order the engine builds them.  ``targets`` (default: whatever the plan
    leaves) are the free query targets; ``joint`` (default
    ``len(targets) != 1``) selects the joint answer.
    """
    ev = set(evidence_keys)
    tape = _Tape()
    biggest = [0, 0]

    def alloc(scope, has_b, cards) -> int:
        a, r = _elems(scope, has_b, cards)
        biggest[0], biggest[1] = max(biggest[0], a), max(biggest[1], r)
        return tape.alloc(a, r)

    pool: list[_Factor] = []
    for f in factors.values():
        variables = list(f.variables)  # type: ignore[attr-defined]
        has_b = any(v in ev for v in variables)
        scope = frozenset(v for v in variables if v not in ev)
        cards = dict(f.cardinalities)  # type: ignore[attr-defined]
        # Evidence makes a new [B, ...] tensor; otherwise a view of the CPT.
        pool.append(_Factor(scope, has_b, cards, alloc(scope, has_b, cards) if has_b else None))

    def release(f: _Factor) -> None:
        if f.token is not None:
            tape.free(f.token)

    def multiply(fs: list[_Factor]) -> _Factor:
        """``_multiply_batched``: each input is released once multiplied in."""
        prod = fs[0]
        for f in fs[1:]:
            scope, has_b = prod.scope | f.scope, prod.has_b or f.has_b
            cards = {**prod.cards, **f.cards}
            new = _Factor(scope, has_b, cards, alloc(scope, has_b, cards))
            release(prod)
            release(f)
            prod = new
        return prod

    def marginalise(prod: _Factor, var: str) -> _Factor:
        """``torch.logsumexp`` (``logsumexp_out_impl``) over ``var``."""
        scope = prod.scope - {var}
        a, r = _elems(scope, prod.has_b, prod.cards)
        maxes = tape.alloc(a, r)               # amax
        absed = tape.alloc(a, r)               # maxes.abs()
        mask = tape.alloc(a, r, _BOOL)         # ... == inf
        tape.free(absed)
        tape.free(mask)
        ta, tr = _elems(prod.scope, prod.has_b, prod.cards)
        biggest[0], biggest[1] = max(biggest[0], ta), max(biggest[1], tr)
        temp = tape.alloc(ta, tr)              # self - maxes, exp_ in place
        result = alloc(scope, prod.has_b, prod.cards)      # sum, log_, add_ in place
        tape.free(temp)
        tape.free(maxes)
        release(prod)
        return _Factor(scope, prod.has_b, prod.cards, result)

    for var in plan:
        rel = [f for f in pool if var in f.scope]
        if not rel:
            continue
        pool = [f for f in pool if var not in f.scope]
        pool.append(marginalise(multiply(rel), var))

    # The answer: multiply what is left, sum out anything that is not a target
    # (defensive in the engine), then normalise into a fresh [B, *targets].
    if pool:
        prod = multiply(pool)
        keep = set(prod.scope if targets is None else targets)
        for var in sorted(prod.scope - keep):
            prod = marginalise(prod, var)
        cards = {**prod.cards, **dict(target_cards or {})}
    else:
        keep, cards = set(targets or ()), dict(target_cards or {})
    if joint is None:
        joint = len(keep) != 1
    out = math.prod(cards[t] for t in keep)
    if joint:
        tape.alloc(0, out)                           # reshape(B, -1) copy
    biggest[1] = max(biggest[1], out)
    tape.alloc(0, out)                               # softmax
    return _PlanCost(tensor=(biggest[0], biggest[1]), tape=tape.frozen(), memo={})


# --------------------------------------------------------------------------- #
# Call-dependent tensors
# --------------------------------------------------------------------------- #

def _same_device(a: torch.device, b: torch.device) -> bool:
    return a.type == b.type and (a.index is None or b.index is None or a.index == b.index)


def _validation(tape: _Tape, n_const: int, n_row: int, float32: bool) -> None:
    """``check_discrete_values`` on one node: a float32 copy (unless already
    float32), its rounding and the mask, all released before the next node."""
    toks = [] if float32 else [tape.alloc(n_const, n_row, 4)]
    toks += [tape.alloc(n_const, n_row, 4), tape.alloc(n_const, n_row, _BOOL)]
    for tok in reversed(toks):
        tape.free(tok)


def evidence_prelude(values: Mapping[str, object], B: int,
                     device) -> tuple[tuple, tuple]:
    """``(persistent, transient)`` tapes for the evidence / do values.

    Persistent: tensors ``query_batch`` materialises from Python lists /
    scalars (``torch.as_tensor``; NumPy arrays share memory) and int64 copies
    of values not already ``long`` on ``device`` — alive for the whole call.
    Transient: the validation temporaries (``_validation``).  A node whose
    value is ``None`` (only its name is known) is priced as a list of ``B``
    floats, the most expensive input: 4 + 8 bytes per row.
    """
    persistent, transient = _Tape(), _Tape()
    device = torch.device(device)
    for v in values.values():
        if v is None:
            raw_es, copy, float32, n_const, n_row = 4, True, True, 0, 1
        else:
            if isinstance(v, torch.Tensor):
                t, raw_es = v, 0
            elif isinstance(v, np.ndarray):
                t, raw_es = torch.as_tensor(v), 0
            else:
                t = torch.as_tensor(v)
                raw_es = t.element_size()
            copy = t.dtype != torch.long or not _same_device(t.device, device)
            float32 = t.dtype == torch.float32
            n = t.numel()
            n_const, n_row = (0, 1) if n == B else (n, 0)
        if raw_es:
            persistent.alloc(n_const, n_row, raw_es)
        _validation(transient, n_const, n_row, float32)
        if copy:
            persistent.alloc(n_const, n_row, _INT64)
    return persistent.frozen(), transient.frozen()


def chunk_validation(n_nodes: int) -> tuple:
    """The validation temporaries of one chunk: int64 row slices."""
    tape = _Tape()
    for _ in range(n_nodes):
        _validation(tape, 0, 1, float32=False)
    return tape.frozen()


def embed_tape(target_cards: Sequence[int], clamped: Sequence[bool]) -> tuple:
    """``_embed_clamped``: per clamped target an int64 one-hot, its float copy
    and the next, wider output (kept: earlier outputs may still be referenced)."""
    tape = _Tape()
    width = math.prod(k for k, c in zip(target_cards, clamped) if not c)
    for k, c in zip(target_cards, clamped):
        if not c:
            continue
        onehot = tape.alloc(0, k, _INT64)
        onehot_f = tape.alloc(0, k)
        tape.free(onehot)
        width *= k
        tape.alloc(0, width)
        tape.free(onehot_f)
    return tape.frozen()


def tables_tape(model) -> tuple:
    """Tabulating every CPT on a cold cache (``_extract_factors``).

    Per node: the mechanism's working memory (``tabulate_working_bytes``:
    its buffers, booked one by one so a device prices each in its own size
    class; all alive at once, released at the end), its output logits
    (unless ``tabulate`` returns a view of stored ones), then the
    log-softmaxed CPT the cache keeps.
    """
    tape = _Tape()
    for node in model.dag.topological_order():
        parents = model.dag.parents(node)
        cards = [model.mechanisms[p].n_classes for p in parents]
        mech = model.mechanisms[node]
        size = math.prod(cards) * mech.n_classes
        view = getattr(mech, "tabulate_returns_view", False)
        work = mech.tabulate_working_bytes(cards)
        buffers = [work] if isinstance(work, (int, float)) else list(work)
        tokens = [tape.alloc(int(b), 0, 1) for b in buffers if b > 0]
        logits = None if view else tape.alloc(size, 0)
        for tok in tokens:
            tape.free(tok)
        tape.alloc(size, 0)
        if logits is not None:
            tape.free(logits)
    return tape.frozen()


# --------------------------------------------------------------------------- #
# The per-call model
# --------------------------------------------------------------------------- #

class _PeakModel:
    """A query's estimated peak bytes as a function of the row count."""

    __slots__ = ("cost", "dtype_bytes", "device_type", "prelude", "transient",
                 "tables", "embed", "out_per_row", "n_nodes", "rows", "_memo")

    def __init__(self, cost: _PlanCost, *, dtype_bytes: int = 4, device_type: str = "cpu",
                 prelude: tuple = (), transient: tuple = (), tables: tuple = (),
                 embed: tuple = (), out_per_row: int = 0, n_nodes: int = 0,
                 rows: int = 0) -> None:
        self.cost = cost
        self.dtype_bytes = dtype_bytes
        self.device_type = device_type
        self.prelude = prelude        # evidence tensors alive for the whole call
        self.transient = transient    # validation temporaries before elimination
        self.tables = tables          # cold-cache tabulation, () when warm
        self.embed = embed            # one-hot axes of clamped joint targets
        self.out_per_row = out_per_row
        self.n_nodes = n_nodes        # evidence + do nodes
        self.rows = rows              # a chunk model: rows of the whole batch (0: one call)
        self._memo: dict[int, int] = {}

    def bytes(self, B: int) -> int:
        """Upper bound of the bytes a ``B``-row call makes the device hold.

        Three phases: the evidence (materialised / copied tensors, alive
        throughout, and the validation temporaries), the tabulation on a cold
        cache (its CPTs stay resident), and the plan (memoised on the plan
        cost, so a repeated query shape replays only its small evidence
        tapes).  On CPU live bytes add exactly; on MPS / CUDA each phase's
        simulated heaps / segments are added, as separate pools — an upper
        bound of sharing them.

        A chunk model on CUDA (``rows`` > ``B``) prices the plan as a
        *sequence*: two ``B``-row chunks and the batch's remainder chunk, on
        one allocator.  Identical chunks reuse each other's cached blocks, but
        the smaller remainder may not — under ``max_split_size_mb`` an
        oversize cached block is neither split for it nor allowed to exceed
        its request by more than ``max_non_split_rounding_mb`` — so a chunk
        priced alone under-estimated the call (#301 CUDA phase).  CPU chunks
        do not interact; MPS credits no free and prices every chunk as new.

        Two chunks plus the remainder is *not* a bound on its own once a
        batch has more than three chunks: every finished chunk's answer stays
        alive until the final concatenation.  The bound holds together with
        the ``held`` reserve the chunk sizer subtracts first (``reserve_bytes``:
        the batch's evidence, the finished answers and their concatenation),
        which ``test_cuda_sequence_of_two_chunks_and_remainder_plus_held_covers_k_chunks``
        checks against an explicit replay of twelve chunks (#301 review 4).
        """
        if B not in self._memo:
            dt, dev = self.dtype_bytes, self.device_type
            conf = cuda_allocator_confs() if dev == "cuda" else None
            if dev == "cuda" and self.rows > B:
                rem = self.rows % B
                seq = (B, B) + ((rem,) if rem else ())
                plan = self.cost.peak_rows(seq, dt, dev, self.embed)
            else:
                plan = self.cost.peak(B, dt, dev, self.embed)
            tab_peak, tab_end = (_replay(self.tables, B, dt, dev, conf) if self.tables
                                 else (0.0, 0.0))
            if dev in ("cuda", "mps"):
                evidence = _replay(_concat([self.prelude, self.transient]), B, dt, dev, conf)[0] \
                    if self.prelude or self.transient else 0.0
                total = evidence + tab_peak + plan
            else:
                held = _replay(self.prelude, B, dt, dev)[1] if self.prelude else 0.0
                trans = _replay(self.transient, B, dt, dev)[0] if self.transient else 0.0
                total = held + max(trans, tab_peak, tab_end + plan)
            self._memo[B] = math.ceil(total * _LIVESET_SAFETY)
        return self._memo[B]

    def tables_bytes(self, *, resident: bool = False) -> int:
        """Upper bound of tabulating every CPT (0 on a warm cache); with
        ``resident``, of the CPTs the cache keeps afterwards."""
        if not self.tables:
            return 0
        peak, end = _replay(self.tables, 1, self.dtype_bytes, self.device_type)
        return math.ceil(_LIVESET_SAFETY * (end if resident else peak))

    def output_bytes(self, B: int) -> int:
        """Bytes of the ``B``-row answer."""
        return B * self.out_per_row * self.dtype_bytes

    def reserve_bytes(self, B: int) -> int:
        """What stays alive next to every chunk of a ``B``-row batch: the
        batch's evidence tensors (chunks index views of them), the finished
        chunks' answers and, at the end, their concatenation."""
        prelude = 0.0
        if self.prelude:
            a, r, es, sign, _ = (np.asarray(x) for x in self.prelude)
            prelude = float(((a + B * r) * es * sign).sum())
        return math.ceil(_LIVESET_SAFETY * (prelude + 2 * self.output_bytes(B)))

    def child(self, *, rows: int = 0) -> _PeakModel:
        """The model of one chunk of a ``rows``-row batch: views of the
        evidence, validated again, CPTs already tabulated by then, no answer
        to concatenate yet; on CUDA the chunks are priced as a sequence."""
        return _PeakModel(self.cost, dtype_bytes=self.dtype_bytes,
                          device_type=self.device_type,
                          transient=chunk_validation(self.n_nodes),
                          embed=self.embed, out_per_row=self.out_per_row,
                          n_nodes=self.n_nodes, rows=rows)


def _max_fitting_rows(peak, limit: int, budget: float) -> int:
    """Largest ``c`` in ``[1, limit]`` with ``peak(c) <= budget`` (0: none).

    Binary search, assuming ``peak`` grows with ``c``.  Allocator granularity
    makes it only nearly monotone; every chunk re-checks its own estimate, so
    a chunk this picks too large is split again rather than over-run.
    """
    if limit < 1 or peak(1) > budget:
        return 0
    lo, hi = 1, limit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if peak(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _factor_dtype_bytes(model) -> int:
    """Bytes per log-CPT element: tabulated CPTs take the parameters' dtype."""
    sizes = [p.element_size() for p in model.parameters() if p.is_floating_point()]
    return max(sizes, default=4)


def _estimate_peak_bytes(
    plan: list[str],
    factors: Mapping[str, object],
    *,
    evidence_keys: tuple[str, ...],
    B: int,
    dtype_bytes: int = 4,
    device_type: str = "cpu",
) -> int:
    """Estimated peak bytes of eliminating ``plan`` over ``factors``, ``B`` rows.

    Plan-level helper (no model at hand): prices the elimination and a joint
    answer over whatever the plan leaves, with the CPTs already tabulated and
    the evidence already int64 on the device.
    ``TensorVariableElimination.estimate_peak_bytes`` is the per-query form.
    """
    cost = _plan_cost(plan, factors, evidence_keys=evidence_keys)
    return _PeakModel(cost, dtype_bytes=dtype_bytes, device_type=device_type).bytes(B)
