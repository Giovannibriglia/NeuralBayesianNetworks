"""Per-model memo helpers shared by the inference engines.

Engines memoise derived data (VE log-CPT factors and elimination plans, LW
sampling layouts) per model.  Keying such a memo on ``id(model)`` alone is
unsound three ways, all of which silently return *wrong numbers*:

1. Mutation.  Refit, ``load_state_dict``, an optimiser step, replacing a
   mechanism or moving the model all change the answer while the id stays.
2. Address reuse.  CPython recycles ``id()`` values, so a dead model can
   hand its address (and its memo entries) to an unrelated new model.
3. Lifetime.  Entries of a dead model are never looked up again; without
   eviction they pin their tensors (possibly GPU memory) forever.

Each entry therefore stores ``(weakref(model), version, payload)``: the
weakref defeats (2) and its death callback evicts the entry (3); the caller
supplies a ``version`` that changes whenever the payload would (1).
"""
from __future__ import annotations

import weakref
from typing import Any, Dict, Hashable

# Soft bound on entries per memo: evidence patterns are open-ended (one plan
# per distinct evidence-key set), so a long-lived engine answering ever-new
# patterns would otherwise grow without limit.  Oldest entries go first;
# a re-planned pattern costs one min-fill pass.
MAX_ENTRIES = 4096


def structure_version(model) -> tuple:
    """Changes whenever the model's *structure* may have changed.

    ``_cache_version`` is bumped by fit / update / set_mechanism (and, on the
    core side, by other CPD mutations); the mechanism ids catch a mechanism
    swapped in directly through ``model.mechanisms[name] = m``; the DAG id
    catches a replaced graph.  Cheap: O(#nodes) ``id`` calls.
    """
    mechs = getattr(model, "mechanisms", None) or {}
    return (
        int(getattr(model, "_cache_version", 0)),
        id(getattr(model, "dag", None)),
        tuple(map(id, mechs.values())),
    )


def tensor_versions(tensors) -> tuple | None:
    """In-place-mutation counters of ``tensors`` (``None`` if unavailable).

    Every in-place write (``copy_`` in ``load_state_dict``, optimiser steps,
    ``with torch.no_grad(): p.add_(…)``) bumps a tensor's ``_version``.
    (Writes through ``p.data`` bypass the counter by design and are not
    seen.)  Inference-mode tensors have no counter; ``None`` then makes the
    memo never validate, which is slow but correct.
    """
    try:
        return tuple([t._version for t in tensors])
    except RuntimeError:
        return None


def cache_get(cache: Dict, key: Hashable, model, version) -> Any | None:
    entry = cache.get(key)
    if entry is None:
        return None
    ref, ver, payload = entry
    if ref() is not model or ver is None or ver != version:
        cache.pop(key, None)
        return None
    return payload


def cache_put(cache: Dict, key: Hashable, model, version, payload):
    def _evict(dead_ref, cache=cache, key=key):
        # Only drop the entry this weakref belongs to: the key may since have
        # been re-filled for a live model that reuses the address.
        entry = cache.get(key)
        if entry is not None and entry[0] is dead_ref:
            cache.pop(key, None)

    cache.pop(key, None)  # re-insert at the end (insertion order = age)
    cache[key] = (weakref.ref(model, _evict), version, payload)
    while len(cache) > MAX_ENTRIES:
        cache.pop(next(iter(cache)))
    return payload
