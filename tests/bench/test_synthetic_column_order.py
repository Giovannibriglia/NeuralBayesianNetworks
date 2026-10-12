"""v0.6a regression tests for the canonical column-order helper.

PR #14 round-4 surfaced a schema bug: ``bn.ground_truth_samples`` is
written in ``nx.topological_sort(dag)`` order, but multiple call sites
in the benchmarking layer indexed it via ``list(bn.dag.nodes()).index``
(insertion order).  These two orders differ on non-trivial DAGs.

This module pins the new contract:

* ``bn.column_order`` always equals ``tuple(nx.topological_sort(dag))``.
* ``bn.column_index(name)`` is the canonical name → column-index map.
* The two are consistent with the actual data tensor.
* The bug class actually existed (insertion order ≠ topological sort
  on real synthetic BNs).
"""
from __future__ import annotations

import networkx as nx
import pytest
import torch

from nbn.bench.synthetic import make_synthetic_bn


@pytest.mark.parametrize(
    "family",
    ["discrete", "continuous_lg", "continuous_nongauss", "hybrid"],
)
@pytest.mark.parametrize("n_nodes", [5, 20, 100])
def test_column_order_equals_topological_sort(family: str, n_nodes: int) -> None:
    """The canonical column order must always be the topological sort
    of the DAG.  This is the invariant the rest of the codebase relies on."""
    bn = make_synthetic_bn(
        family=family, n_nodes=n_nodes, n_train=200,
        n_test=50, n_reference=500, seed=0, device="cpu",
    )
    expected = tuple(nx.topological_sort(bn.dag))
    assert bn.column_order == expected, (
        f"column_order {bn.column_order} != topological_sort {expected} "
        f"for family={family}, n_nodes={n_nodes}"
    )


def test_column_index_returns_correct_column() -> None:
    """``bn.column_index(name)`` must return the index where ancestral
    samples for ``name`` actually land in ``bn.ground_truth_samples``.

    End-to-end check: re-sample independently from the *same* true
    model and verify per-node marginal means agree within 5 SE.  If
    column_index is misaligned, marginal means at the indexed column
    will reflect a different node and the SE check will fire.
    """
    torch.manual_seed(0)
    bn = make_synthetic_bn(
        family="continuous_lg", n_nodes=10, n_train=200,
        n_test=50, n_reference=10_000, seed=0, device="cpu",
    )
    assert bn.ground_truth_samples is not None
    with torch.no_grad():
        independent = bn.true_model.sample(n=10_000)
    for node in bn.dag.nodes:
        col = bn.column_index(node)
        empirical = bn.ground_truth_samples[:, col]
        empirical_mean = empirical.mean().item()
        expected_mean = independent[node].float().reshape(-1).mean().item()
        se = empirical.std().item() / (10_000 ** 0.5)
        assert abs(empirical_mean - expected_mean) < 5 * se, (
            f"column_index({node})={col} but ground_truth_samples[:, {col}] "
            f"has mean {empirical_mean:.3f}, expected ~{expected_mean:.3f} "
            f"({(empirical_mean - expected_mean) / max(se, 1e-12):.1f} SE off "
            f"— likely a column misalignment)"
        )


def test_column_index_raises_on_unknown_node() -> None:
    bn = make_synthetic_bn(
        family="discrete", n_nodes=5, n_train=200, n_test=50,
        n_reference=500, seed=0, device="cpu",
    )
    with pytest.raises(KeyError, match="not in BN"):
        bn.column_index("NOT_A_NODE")


def test_insertion_order_differs_from_topological_for_synthetic_bns() -> None:
    """Sanity meta-test: confirm the bug class actually existed.

    For the synthetic generator's DAGs, ``list(bn.dag.nodes())`` and
    ``tuple(nx.topological_sort(bn.dag))`` must differ on at least one
    family/seed combo — otherwise the schema bug fixed in PR #14 commit
    1bd671e couldn't have manifested, and the v0.6a defensive helper is
    pointless.

    If this ever fails, either the synthetic generator was changed to
    produce DAGs where the two orders always agree (in which case the
    test is fine to remove with a note in the PR description) or there
    is a bug in this test's assumption.
    """
    found_difference = False
    for family in ["discrete", "continuous_lg", "continuous_nongauss", "hybrid"]:
        for seed in [0, 1, 2]:
            bn = make_synthetic_bn(
                family=family, n_nodes=20, n_train=200, n_test=50,
                n_reference=500, seed=seed, device="cpu",
            )
            insertion_order = tuple(bn.dag.nodes())
            topo_order = tuple(nx.topological_sort(bn.dag))
            if insertion_order != topo_order:
                found_difference = True
                break
        if found_difference:
            break
    assert found_difference, (
        "insertion order and topological sort always agree on synthetic BNs — "
        "the schema bug fixed in PR #14 commit 1bd671e couldn't have "
        "manifested.  Either the synthetic generator was changed (in which "
        "case this test is fine to remove) or this test's assumption is wrong."
    )
