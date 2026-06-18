"""Canonical discrete-CPT extraction from a NeuralBayesianNetwork (#109 PR 2).

Shared by ``ParamLearningMeasurement`` (to read the TRUE model's CPTs) and by
``NBNAdapter.extract_learned_cpts`` (to read the fitted model's CPTs), so the
two tables share an identical, adapter-internal-order-INDEPENDENT layout and
compare cell-by-cell.

Canonical layout (see the BaselineAdapter docstring):
  * one entry per DISCRETE node whose parents are ALL discrete;
  * value is a dense ``[n_parent_configs, K]`` probability tensor;
  * parents sorted lexicographically by name;
  * parent configs in row-major order — FIRST parent varies slowest;
  * each parent ranges over ``0..card-1``; columns are classes ``0..K-1``.

Order safety: a node's mechanism interprets the parent tensor's columns in the
mechanism's OWN fit order (``model.dag.parents(node)``), which may differ from
the canonical lexicographic order. We enumerate configs in canonical order but
PERMUTE the columns fed to ``forward`` into the mechanism's order (by parent
name), so the returned rows are correct AND indexed canonically — regardless of
each model's internal parent ordering. This is what lets true and learned CPTs
align even if they were built with different edge-insertion orders.
"""
from __future__ import annotations

import itertools
from typing import Any

import torch

# Nodes whose parent-config space exceeds this are omitted from recovery
# (enumerating an intractable table is neither feasible nor meaningful). Bounds
# bnlearn giants; never hit by the discrete synthetic configs (cards small,
# in-degree small). Omitted nodes simply don't contribute to the node-mean.
MAX_PARENT_CONFIGS = 1 << 20  # ~1.05M


def extract_discrete_cpts(
    model: Any, variables: dict[str, tuple[str, int]]
) -> dict[str, torch.Tensor]:
    """Return canonical learned/true discrete CPTs from an NBN ``model``.

    ``variables`` maps node -> ``(kind, cardinality)`` (``problem.variables``).
    Returns ``{node: probs[n_parent_configs, K]}`` (on CPU, detached) for every
    discrete node with all-discrete parents whose config space is tractable.
    """
    device = getattr(model, "_device", "cpu")
    out: dict[str, torch.Tensor] = {}

    for node in model.dag.nodes():
        kind, card = variables[node]
        if kind != "discrete":
            continue
        k = int(card)
        parents = sorted(model.dag.parents(node))          # canonical lex order

        parent_cards: list[int] = []
        all_discrete = True
        for p in parents:
            pk, pc = variables[p]
            if pk != "discrete":
                all_discrete = False
                break
            parent_cards.append(int(pc))
        if not all_discrete:
            continue

        mech = model.mechanisms[node]

        if not parents:
            probs = mech.forward(None).probs.reshape(1, k)
            out[node] = probs.detach().to("cpu")
            continue

        n_configs = 1
        for c in parent_cards:
            n_configs *= c
        if n_configs > MAX_PARENT_CONFIGS:
            continue  # intractable -> omit from recovery

        # Canonical configs: row-major over canonical (lex) parents, first
        # parent slowest. itertools.product yields exactly that order.
        configs = list(itertools.product(*(range(c) for c in parent_cards)))
        cfg_canon = torch.tensor(configs, dtype=torch.long)     # [n_configs, P] lex cols

        # Permute columns into the mechanism's own parent order so forward()
        # reads the right value per column; output rows stay canonical-indexed.
        mech_parents = list(model.dag.parents(node))
        canon_pos = {p: i for i, p in enumerate(parents)}
        perm = [canon_pos[p] for p in mech_parents]
        cfg_mech = cfg_canon[:, perm].to(device)                # [n_configs, P] mech cols

        probs = mech.forward(cfg_mech).probs.reshape(n_configs, k)
        out[node] = probs.detach().to("cpu")

    return out
