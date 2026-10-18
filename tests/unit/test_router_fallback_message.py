"""The router's downgrade warning names the engine that actually answers."""
from __future__ import annotations

import logging

from nbn.inference import ConditioningLimitError, HybridRouter
from nbn.inference.conditioning import ConditioningLimitError as _Direct
from tests.unit.test_conditioning import _dense_bn


def _warnings(caplog, fallback):
    m, names = _dense_bn(0, n=8, k=3, window=3, p=0.8)
    router = HybridRouter(treewidth_threshold=0, fallback=fallback, n_samples=2000,
                          certified_samples=4000)
    with caplog.at_level(logging.WARNING, logger="nbn.inference.hybrid"):
        caplog.clear()
        m.query([names[1]], {names[-1]: 0}, engine=router)
    return [r.getMessage() for r in caplog.records], router


def test_certified_fallback_warning_names_the_certified_engine(caplog):
    msgs, router = _warnings(caplog, "certified")
    assert router.last_engine == "certified"
    assert any("certified fallback" in s for s in msgs)
    assert not any("with likelihood weighting (approximate)" in s for s in msgs)


def test_lw_fallback_warning_unchanged(caplog):
    msgs, router = _warnings(caplog, "lw")
    assert router.last_engine == "likelihood_weighting"
    assert any("with likelihood weighting (approximate)" in s for s in msgs)


def test_conditioning_limit_error_is_exported():
    assert ConditioningLimitError is _Direct
