"""Standard query battery — the VBN-inspired five-kind taxonomy.

Used by every domain's ``load_problem`` to ensure all baselines answer
identical queries on the same problem.
"""
from __future__ import annotations

from typing import Sequence

import torch

from benchmarking.domains.base import Query


def make_query_battery(
    *,
    nodes: Sequence[str],
    discrete_nodes: Sequence[str],
    continuous_nodes: Sequence[str],
    cardinalities: dict[str, int],
    n_marginals: int = -1,
    n_conditional_single: int = 50,
    n_conditional_multi: tuple[int, ...] = (2, 4, 8),
    n_per_multi: int = 50,
    n_map: int = 30,
    n_do: int = 20,
    seed: int = 0,
) -> list[Query]:
    """Generate a fixed, seeded battery of queries.

    Returns a flat list — the runner groups them by ``Query.kind`` for plotting.
    """
    g = torch.Generator().manual_seed(seed)
    qs: list[Query] = []

    # 1. Univariate marginals
    targets = list(nodes) if n_marginals < 0 else list(nodes)[:n_marginals]
    for n in targets:
        qs.append(Query(targets=(n,), evidence={}, kind="marginal"))

    discrete_set = set(discrete_nodes)

    def _rand_node(exclude: set[str]) -> str:
        candidates = [n for n in nodes if n not in exclude]
        if not candidates:
            return next(iter(nodes))
        return candidates[int(torch.randint(0, len(candidates), (1,), generator=g).item())]

    def _rand_value(node: str):
        if node in discrete_set:
            k = cardinalities[node]
            return int(torch.randint(0, k, (1,), generator=g).item())
        return float(torch.randn((1,), generator=g).item())

    # 2. Conditional single
    for _ in range(n_conditional_single):
        tgt = _rand_node(set())
        ev_node = _rand_node({tgt})
        qs.append(Query(targets=(tgt,), evidence={ev_node: _rand_value(ev_node)},
                        kind="conditional"))

    # 3. Conditional multi
    for k in n_conditional_multi:
        for _ in range(n_per_multi):
            tgt = _rand_node(set())
            ev = {}
            taken = {tgt}
            for _ in range(k):
                e = _rand_node(taken)
                taken.add(e)
                ev[e] = _rand_value(e)
            qs.append(Query(targets=(tgt,), evidence=ev, kind="conditional"))

    # 4. MAP
    for _ in range(n_map):
        tgt = _rand_node(set())
        ev_node = _rand_node({tgt})
        qs.append(Query(targets=(tgt,), evidence={ev_node: _rand_value(ev_node)},
                        kind="map"))

    # 5. Do
    for _ in range(n_do):
        tgt = _rand_node(set())
        do_node = _rand_node({tgt})
        qs.append(Query(targets=(tgt,), evidence={do_node: _rand_value(do_node)},
                        kind="do"))

    return qs
