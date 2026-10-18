"""Regressions from the review of the inference certificates."""
import pytest
import torch

from nbn.inference.certified import CertifiedEngine, certified_query
from nbn.inference.hybrid import HybridRouter
from nbn.inference.tensor_ve import TensorVariableElimination
from tests.unit.test_conditioning import _dense_bn


def test_exact_shortcut_covers_the_float64_truth():
    m, names = _dense_bn(3, n=16, k=4, window=6, p=0.7)
    ev = {names[i]: i % 4 for i in range(8, 16)}
    tgt = names[3]
    exact = TensorVariableElimination().query(m, [tgt], ev).double()
    a = certified_query(m, tgt, ev, collapse="auto", memory_budget=2 ** 30)
    assert a.collapsed == () and a.level == 1.0
    assert ((exact >= a.lower) & (exact <= a.upper)).all()
    assert a.evidence_prob_lower < a.evidence_prob_upper


def test_evidence_and_do_on_the_same_node_is_refused():
    m, names = _dense_bn(0, n=10, k=3, window=3, p=0.7)
    with pytest.raises(ValueError, match="both observed and intervened"):
        certified_query(m, names[3], {names[5]: 1}, do={names[5]: 0})


def test_router_certificate_is_cleared_when_ve_answers():
    m, names = _dense_bn(0, n=10, k=3, window=3, p=0.7)
    r = HybridRouter(fallback="certified")
    r._cert = CertifiedEngine(n_samples=4096)
    r._cert.query(m, [names[3]], {names[-1]: 1})
    r._last_engine = "certified"
    assert r.last_certificate is not None
    r._last_engine = "tensor_ve"
    assert r.last_certificate is None
