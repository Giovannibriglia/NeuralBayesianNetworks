from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from nbn.inference._cache import cache_get, cache_put, structure_version


@dataclass(frozen=True)
class InferenceState:
    """Pre-computed, cacheable metadata for a (model, query) pair.

    Computing slices and parent indices once and caching them avoids repeated
    Python-level graph traversal in the hot inference path.
    """

    topo_order: Tuple[str, ...]
    node_to_idx: Dict[str, int]
    parent_idx: Tuple[Tuple[int, ...], ...]   # per node: parent indices in topo order
    evidence_mask: Tuple[bool, ...]
    do_mask: Tuple[bool, ...]
    target_indices: Tuple[int, ...]
    node_slices: Tuple[slice, ...]            # slice into concatenated state vector
    parent_slices: Tuple[Tuple[slice, ...], ...]
    total_dim: int


def get_inference_state(
    model,
    targets: List[str],
    evidence_keys: Tuple[str, ...],
    do_keys: Tuple[str, ...],
    cache: Dict,
) -> InferenceState:
    # Entries are validated against a weakref and the model's structure
    # version (``nbn.inference._cache``), because ``id(model)`` alone is
    # unsound twice over: it does not change when the model is refit or a
    # mechanism is swapped in place, and CPython recycles addresses, so a
    # dead model can hand its id to an unrelated new one.  Both failure modes
    # return stale metadata rather than raising.  (This state holds node
    # slices and parent indices, so a stale hit misreads the sample buffer
    # after a graph-shaped change such as ``intervene()``.)  The version is
    # stored in the entry, not the key, so a refit *replaces* the entry
    # instead of leaving one dead entry per refit; the weakref callback
    # evicts a collected model's entries.
    #
    # Targets are kept in caller order: ``target_indices`` follows it.
    sig = (
        id(model),
        tuple(targets),
        tuple(sorted(evidence_keys)),
        tuple(sorted(do_keys)),
    )
    version = structure_version(model)
    hit = cache_get(cache, sig, model, version)
    if hit is not None:
        return hit

    topo = tuple(model.dag.topological_order())
    node_to_idx = {n: i for i, n in enumerate(topo)}
    evidence_set = set(evidence_keys)
    do_set = set(do_keys)

    parent_idx: List[Tuple[int, ...]] = []
    for node in topo:
        parent_idx.append(tuple(node_to_idx[p] for p in model.dag.parents(node)))

    node_slices: List[slice] = []
    total = 0
    for node in topo:
        dim = model.mechanisms[node].output_dim if node in model.mechanisms else 1
        node_slices.append(slice(total, total + dim))
        total += dim

    parent_slices = tuple(
        tuple(node_slices[p] for p in pi) for pi in parent_idx
    )

    state = InferenceState(
        topo_order=topo,
        node_to_idx=node_to_idx,
        parent_idx=tuple(parent_idx),
        evidence_mask=tuple(n in evidence_set for n in topo),
        do_mask=tuple(n in do_set for n in topo),
        target_indices=tuple(node_to_idx[t] for t in targets),
        node_slices=tuple(node_slices),
        parent_slices=parent_slices,
        total_dim=total,
    )
    return cache_put(cache, sig, model, version, state)
