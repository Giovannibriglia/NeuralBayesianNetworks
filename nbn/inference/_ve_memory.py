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
     segments included.
3. **A margin**: ``_LIVESET_SAFETY`` (10 %).

``scripts/ve_memory_validation.py`` measures the bound on CPU (exact), MPS
(heap level) and CUDA (reserved); ``docs/ve-memory-model.md`` has the numbers.
"""
from __future__ import annotations

import bisect
import math
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


def _concat(tapes: Sequence[tuple]) -> tuple:
    """Concatenate frozen tapes, re-basing the free -> alloc references."""
    a, r, es, sign, ref = [], [], [], [], []
    for t in tapes:
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


def _mps_charge(s: np.ndarray) -> np.ndarray:
    """Bytes each MPS buffer of size ``s`` costs: its share of its heap."""
    out = np.zeros_like(s)
    for lo, hi, q, heap in ((0, _MiB, 256, 8 * _MiB),
                            (_MiB, 10 * _MiB, 16 * _KiB, 32 * _MiB),
                            (10 * _MiB, 512 * _MiB, 4 * _MiB, _GiB)):
        m = (s > lo) & (s <= hi)
        s1 = _round_up(s[m], q)
        out[m] = heap / np.floor(heap / s1)
    big = s > 512 * _MiB
    out[big] = _round_up(s[big], 16 * _MiB)
    return out


def _mps_heap_reserve(largest: float) -> int:
    """One partly filled heap per pooled size class a buffer reaches."""
    return sum(heap for above, heap in ((0, 8 * _MiB), (_MiB, 32 * _MiB),
                                         (10 * _MiB, _GiB)) if largest > above)


_CUDA_SMALL, _CUDA_MIN_BLOCK = _MiB, 512


class _Segment:
    __slots__ = ("small", "blocks", "offs")

    def __init__(self, small: bool) -> None:
        self.small = small
        self.blocks: dict[int, list] = {}   # offset -> [size, is_free]
        self.offs: list[int] = []           # sorted offsets


def _cuda_reserved_peak(sizes: Sequence[float], sign: Sequence[int],
                        ref: Sequence[int]) -> float:
    """Peak reserved bytes of ``CUDACachingAllocator`` replaying the tape.

    Default configuration (no ``max_split_size``, no garbage collection): a
    request rounded up to 512 B takes the smallest cached free block of its
    pool (<= 1 MiB: small pool, else large) that fits, splitting it when the
    rest is >= 512 B (small) or > 1 MiB (large); without one, a new segment
    of 2 MiB (small), 20 MiB (< 10 MiB) or the request rounded up to 2 MiB.
    Freed blocks merge with free neighbours in their segment and stay cached:
    reserved memory never shrinks during the call.
    """
    pools: dict[bool, list] = {True: [], False: []}   # sorted (size, seg, off)
    segs: list[_Segment] = []
    owner: dict[int, tuple[int, int]] = {}
    reserved = peak = 0

    def put_free(seg: int, off: int, size: int) -> None:
        sg = segs[seg]
        if off not in sg.blocks:
            bisect.insort(sg.offs, off)
        sg.blocks[off] = [size, True]
        bisect.insort(pools[sg.small], (size, seg, off))

    def take_free(seg: int, off: int) -> int:
        sg = segs[seg]
        size = sg.blocks[off][0]
        pools[sg.small].remove((size, seg, off))
        return size

    for i, s in enumerate(sizes):
        if sign[i] > 0:
            req = max(_CUDA_MIN_BLOCK, math.ceil(s / _CUDA_MIN_BLOCK) * _CUDA_MIN_BLOCK)
            small = req <= _CUDA_SMALL
            pool = pools[small]
            j = bisect.bisect_left(pool, (req, -1, -1))
            if j < len(pool):
                size, seg, off = pool.pop(j)
            else:
                size = (2 * _MiB if small else 20 * _MiB if req < 10 * _MiB
                        else math.ceil(req / (2 * _MiB)) * 2 * _MiB)
                seg, off = len(segs), 0
                segs.append(_Segment(small))
                segs[seg].offs.append(0)
                reserved += size
                peak = max(peak, reserved)
            sg = segs[seg]
            rest = size - req
            if (rest >= _CUDA_MIN_BLOCK) if small else (rest > _CUDA_SMALL):
                put_free(seg, off + req, rest)
                size = req
            sg.blocks[off] = [size, False]
            owner[i] = (seg, off)
        else:
            seg, off = owner.pop(ref[i])
            sg = segs[seg]
            size = sg.blocks[off][0]
            k = bisect.bisect_left(sg.offs, off)
            # Merge with the free neighbours in the same segment.
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


def _device_peak(tape: tuple, B: int, dtype_bytes: int, device_type: str) -> float:
    """Peak bytes the device holds for ``tape`` at ``B`` rows (see module doc)."""
    a, r, es, sign, ref = (np.asarray(x) for x in tape)
    if a.size == 0:
        return 0.0
    es = np.where(es == _FACTOR, dtype_bytes, es)
    sizes = (a + B * r).astype(np.float64) * es
    if device_type == "cuda":
        return _cuda_reserved_peak(sizes.tolist(), sign.tolist(), ref.tolist())
    if device_type == "mps":
        charge = _mps_charge(sizes)
        extra = _mps_heap_reserve(float(sizes.max()))
    else:
        charge = _round_up(sizes, 64)
        extra = 0
    running = np.cumsum(sign * charge)
    return float(running[sign > 0].max()) + extra


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
        maxes = tape.alloc(a, r)                     # amax
        absed = tape.alloc(a, r)                     # maxes.abs()
        mask = tape.alloc(a, r, _BOOL)               # ... == inf
        tape.free(absed)
        tape.free(mask)
        temp = alloc(prod.scope, prod.has_b, prod.cards)   # self - maxes (exp_ in place)
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
    return _PlanCost(tensor=(biggest[0], biggest[1]), tape=tape.frozen())


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

    Per node: the mechanism's working memory (``tabulate_working_bytes``,
    released at the end), its output logits (unless ``tabulate`` returns a
    view of stored ones), then the log-softmaxed CPT the cache keeps.
    """
    tape = _Tape()
    for node in model.dag.topological_order():
        parents = model.dag.parents(node)
        cards = [model.mechanisms[p].n_classes for p in parents]
        mech = model.mechanisms[node]
        size = math.prod(cards) * mech.n_classes
        view = getattr(mech, "tabulate_returns_view", False)
        work = tape.alloc(int(mech.tabulate_working_bytes(cards)), 0, 1)
        logits = None if view else tape.alloc(size, 0)
        tape.free(work)
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
                 "tables", "embed", "out_per_row", "n_nodes", "_memo")

    def __init__(self, cost: _PlanCost, *, dtype_bytes: int = 4, device_type: str = "cpu",
                 prelude: tuple = (), transient: tuple = (), tables: tuple = (),
                 embed: tuple = (), out_per_row: int = 0, n_nodes: int = 0) -> None:
        self.cost = cost
        self.dtype_bytes = dtype_bytes
        self.device_type = device_type
        self.prelude = prelude        # evidence tensors alive for the whole call
        self.transient = transient    # validation temporaries before elimination
        self.tables = tables          # cold-cache tabulation, () when warm
        self.embed = embed            # one-hot axes of clamped joint targets
        self.out_per_row = out_per_row
        self.n_nodes = n_nodes        # evidence + do nodes
        self._memo: dict[int, int] = {}

    def bytes(self, B: int) -> int:
        """Upper bound of the bytes a ``B``-row call makes the device hold."""
        if B not in self._memo:
            tapes = [t for t in (self.prelude, self.transient, self.tables,
                                 self.cost.tape, self.embed) if t]
            peak = _device_peak(_concat(tapes), B, self.dtype_bytes, self.device_type)
            self._memo[B] = math.ceil(peak * _LIVESET_SAFETY)
        return self._memo[B]

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

    def child(self) -> _PeakModel:
        """The model of one chunk: views of the evidence, validated again,
        CPTs already tabulated by then, no answer to concatenate yet."""
        return _PeakModel(self.cost, dtype_bytes=self.dtype_bytes,
                          device_type=self.device_type,
                          transient=chunk_validation(self.n_nodes),
                          embed=self.embed, out_per_row=self.out_per_row,
                          n_nodes=self.n_nodes)


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
