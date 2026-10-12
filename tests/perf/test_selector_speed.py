"""Wall-clock budget of the topological selector's node-role analysis.

Moved from tests/bench/test_selector_topological.py (``-m perf``).
"""
from __future__ import annotations

import time

import networkx as nx
import pytest

from nbn.bench.selectors.topological import compute_node_roles

pytestmark = pytest.mark.perf


@pytest.fixture(scope="module")
def synthetic_n1000() -> nx.DiGraph:
    """Synthetic n=1000 continuous_lg DAG."""
    from nbn.bench.synthetic import make_synthetic_bn
    bn = make_synthetic_bn(n_nodes=1000, family="continuous_lg", seed=0)
    return nx.DiGraph(bn.dag)


def test_compute_time_under_one_second(synthetic_n1000):
    t0 = time.perf_counter()
    compute_node_roles(synthetic_n1000)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"NodeRoles took {elapsed:.2f}s (budget 1s)"
