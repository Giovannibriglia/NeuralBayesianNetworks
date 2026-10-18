"""Answer many independent queries concurrently (``query_many``).

Why
---
Likelihood weighting and most samplers loop over nodes in topological order,
and each step is a few small tensor operations; intra-op threads cannot split
such small work, so one query gains little from more cores (round-3
benchmark: 1.26x from 8 threads at 10^6 particles).  Independent *queries*,
however, can overlap: PyTorch releases the GIL inside its kernels, so threads
running different queries keep several cores busy.  Measured on an Apple M5
(10 LW queries, 1000-node network, 10^5 particles): 2.2x with 4 workers and
2.4x with 8 when ``torch.set_num_threads(1)``; 1.7x at 4 intra-op threads.

Each worker uses its own deep copy of the engine, so per-engine caches and
``last_*`` attributes are never shared between threads.  The answers have the
same law as sequential calls; with a sampling engine the random streams
interleave, so results are not bit-reproducible across runs (seed per query
with an engine that takes a generator if that matters).

Devices: on CPU and CUDA the threads run as described (CUDA kernels of
different threads share the default stream and run in order).  On MPS the
queries run sequentially: concurrent Metal encoding from several threads
aborts the process ("A command encoder is already encoding to this command
buffer", seen with PyTorch 2.x on Apple silicon).
"""
from __future__ import annotations

import copy
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

__all__ = ["query_many"]

_WARNED: dict = {}


def _device_type(model) -> str:
    dev = getattr(model, "device", "cpu")
    return getattr(dev, "type", str(dev).split(":")[0])


def query_many(
    model,
    queries: Sequence[tuple],
    *,
    engine: Any | None = None,
    workers: int | None = None,
    **kwargs: Any,
) -> list:
    """``[model.query([t], evidence=e, do=d, engine=engine, **kwargs) for ...]``,
    run by ``workers`` threads (default: ``os.cpu_count()``; 1 = sequential).

    ``queries``: tuples ``(targets, evidence)`` or ``(targets, evidence, do)``;
    ``targets`` is a node name or a list of names.  Results keep the order of
    ``queries``.  An exception in any query is raised here.
    """
    qs = []
    for q in queries:
        tg = [q[0]] if isinstance(q[0], str) else list(q[0])
        ev = q[1] if len(q) > 1 else None
        do = q[2] if len(q) > 2 else None
        qs.append((tg, ev, do))
    workers = int(workers or os.cpu_count() or 1)
    if workers > 1 and _device_type(model) == "mps":
        if not _WARNED.get("mps"):
            _WARNED["mps"] = True
            warnings.warn("query_many: MPS is not thread-safe; running the queries "
                          "sequentially.", stacklevel=2)
        workers = 1
    if workers <= 1 or len(qs) <= 1:
        return [model.query(tg, evidence=ev, do=do, engine=engine, **kwargs)
                for tg, ev, do in qs]
    local = threading.local()

    def run(q):
        if not hasattr(local, "engine"):
            local.engine = copy.deepcopy(engine) if engine is not None else None
        tg, ev, do = q
        return model.query(tg, evidence=ev, do=do, engine=local.engine, **kwargs)

    with ThreadPoolExecutor(max_workers=min(workers, len(qs))) as ex:
        return list(ex.map(run, qs))
