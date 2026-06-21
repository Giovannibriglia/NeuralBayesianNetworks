"""HybridRouter engine selection — the narrowed induced_width except (#232).

PR 11 (#231) repaired DAG.induced_width; the bare `except Exception` that had
silently masked those bugs (degrading exact VE to stochastic LW) is narrowed
here to (nx.NetworkXError, ValueError). These tests pin both directions: a
legitimate "can't compute treewidth" failure still falls back to LW, while an
unexpected exception (the shape PR 11 fixed) now propagates instead of hiding.
"""
from __future__ import annotations

import types

import networkx as nx
import pytest

from nbn.inference.hybrid import HybridRouter


class _DiscreteMech:
    is_discrete = True


def _model(induced_width):
    """A minimal all-discrete model stub whose dag.induced_width is supplied."""
    dag = types.SimpleNamespace(induced_width=induced_width)
    return types.SimpleNamespace(mechanisms={"a": _DiscreteMech()}, dag=dag)


def _raises(exc):
    def _f():
        raise exc
    return _f


@pytest.mark.parametrize("exc", [nx.NetworkXError("malformed"), ValueError("empty")])
def test_falls_back_to_lw_on_expected_exception(exc):
    """A legitimate treewidth-computation failure falls back to LW."""
    router = HybridRouter()
    engine = router._select(_model(_raises(exc)))
    assert router._last_engine == "likelihood_weighting"
    assert engine is router._lw


def test_propagates_unexpected_exception():
    """An AttributeError (the PR 11 bug class) propagates, not silently LW."""
    router = HybridRouter()
    with pytest.raises(AttributeError):
        router._select(_model(_raises(AttributeError("missing fn"))))


def test_low_treewidth_still_selects_ve():
    """Happy path unaffected: a small treewidth selects exact VE."""
    router = HybridRouter(treewidth_threshold=25)
    engine = router._select(_model(lambda: 3))
    assert router._last_engine == "tensor_ve"
    assert engine is router._ve
