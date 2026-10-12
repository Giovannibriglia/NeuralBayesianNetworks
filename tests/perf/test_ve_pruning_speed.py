"""VE pruning speed on bnlearn barley (Bug 2 Stage 2b, #127).

Moved from tests/unit/test_pruning.py: it asserts a wall-clock bound and
needs the barley network (2 MB, downloaded on first use), so it runs with
``-m "perf and network"``.  The pruned == unpruned correctness half runs on
small networks in tests/unit/test_pruning.py.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from tests.unit.test_pruning import _all_nodes

pytestmark = [pytest.mark.perf, pytest.mark.network, pytest.mark.slow]


def test_barley_hub_given_mb_correct_and_fast():
    """Regression test for Bug 2 Stage 2b (#127).

    The barley hub (aks_m2) given its full Markov blanket is the case
    from the investigation. Stage 2's first cut made it *26x slower*
    than unpruned because evidence-node CPTs dragged non-relevant
    high-cardinality parents (e.g. `sort`, card 67) into the final
    factor product (a 118M-element joint). Stage 2b expands the
    elimination scope to those free variables so min-fill sums them
    out in good order.

    Verifies (a) pruned == unpruned posterior (correctness preserved)
    and (b) the query is fast (measured ~0.6ms post-fix; assert a
    generous 50ms wall-clock bound to stay robust on slow CI).
    """
    import time
    import networkx as nx
    from nbn.bench.problems.bnlearn import (
        BnlearnConfig, BnlearnProblemSource,
    )
    from nbn.bench.core.config import BaselineSpec, build_adapter
    from nbn.inference.tensor_ve import TensorVariableElimination
    import nbn.inference.tensor_ve as ve_mod

    cfg = BnlearnConfig(networks=["barley"], seeds=[0],
                        n_train=200, n_test=20, n_reference=20)
    problem = next(BnlearnProblemSource().iter_problems(cfg))
    adapter = build_adapter(BaselineSpec(
        library="nbn", mechanism="cat",
        param_method="mle", inference_method="ve",
    ))
    adapter.fit(problem)
    model = adapter.model
    g = model.dag.networkx_graph
    hub = max(g.nodes(), key=lambda n: g.in_degree(n) + g.out_degree(n))
    assert hub == "aks_m2"  # deterministic highest-degree node
    mb = set(g.predecessors(hub)) | set(g.successors(hub))
    for child in g.successors(hub):
        mb |= set(g.predecessors(child)) - {hub}
    evidence = dict.fromkeys(mb, 0)

    eng = TensorVariableElimination()
    pruned = eng.query(model, [hub], evidence)
    with patch.object(ve_mod, "relevant_subnetwork", _all_nodes):
        unpruned = TensorVariableElimination().query(model, [hub], evidence)

    assert torch.allclose(pruned, unpruned, atol=1e-6), (
        f"barley hub-given-MB pruned != unpruned: {pruned} vs {unpruned}"
    )

    eng2 = TensorVariableElimination()
    eng2.query(model, [hub], evidence)  # warm caches
    t0 = time.perf_counter()
    eng2.query(model, [hub], evidence)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50.0, (
        f"barley hub-given-MB took {elapsed_ms:.1f}ms (expected <50ms; "
        f"Stage 2b regression — free-variable elimination scope may be "
        f"broken, re-forming the giant final product)"
    )
