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


def test_ve_oom_falls_back_to_lw(monkeypatch):
    """A VE out-of-memory error degrades the query to LW, not a failure."""
    import torch

    router = HybridRouter(treewidth_threshold=25)
    model = _model(lambda: 3)

    def _oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("estimated peak exceeds budget")

    monkeypatch.setattr(router._ve, "query", _oom)
    monkeypatch.setattr(
        router._lw, "query", lambda *a, **k: "lw-result", raising=False
    )
    result = router.query(model, ["a"], None)
    assert result == "lw-result"
    assert router._last_engine == "likelihood_weighting"


def test_ve_non_oom_error_propagates(monkeypatch):
    """Only OOM triggers the LW retry — other VE errors still propagate."""
    router = HybridRouter(treewidth_threshold=25)
    model = _model(lambda: 3)
    def _boom(*args, **kwargs):
        raise RuntimeError("real bug")

    monkeypatch.setattr(router._ve, "query", _boom)
    with pytest.raises(RuntimeError, match="real bug"):
        router.query(model, ["a"], None)


def test_lw_oom_propagates(monkeypatch):
    """If LW itself OOMs there is nothing to degrade to — the error surfaces."""
    import torch

    router = HybridRouter(treewidth_threshold=25)
    model = _model(lambda: 30)  # above threshold -> LW selected directly

    def _oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("lw oom")

    monkeypatch.setattr(router._lw, "query", _oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        router.query(model, ["a"], None)


def test_induced_width_is_memoized():
    """DAG.induced_width computes min-fill once; repeat calls hit the cache.

    The router calls induced_width() on every dispatch, so without the memo
    a batched sweep pays the O(n * deg^2) greedy pass per query.
    """
    from unittest import mock

    from nbn.core.dag import DAG

    dag = DAG([("a", "b"), ("b", "c"), ("a", "c")])
    first = dag.induced_width()
    with mock.patch("nbn.core.dag.nx.moral_graph") as moral:
        assert dag.induced_width() == first  # served from cache
        moral.assert_not_called()
