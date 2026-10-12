"""Every shipped benchmark component satisfies its ``runtime_checkable`` Protocol.

The runner only ever sees components through the four Protocols in
``nbn.bench.core.interfaces``; a shipped class that drifts from its Protocol
(a renamed method, a dropped ``query_batch``) would otherwise surface only as
an ``AttributeError`` mid-run.  This pins the real implementations, not
stand-in stubs.
"""
from __future__ import annotations

import pytest

from nbn.bench.adapters import NBNAdapter, PgmpyAdapter, PomegranateAdapter, PyroAdapter
from nbn.bench.core.interfaces import (
    BaselineAdapter,
    Measurement,
    ProblemSource,
    QuerySelector,
)
from nbn.bench.measurements import AccuracyAndTiming, ParamLearningMeasurement, TimingOnly
from nbn.bench.problems import BnlearnProblemSource, SyntheticProblemSource
from nbn.bench.selectors.heaviest import HeaviestQueryByRole
from nbn.bench.selectors.topological import TopologicalAllocator
from nbn.bench.selectors.uniform import UniformRandomSelector

_COMPONENTS = [
    (ProblemSource, SyntheticProblemSource),
    (ProblemSource, BnlearnProblemSource),
    (QuerySelector, UniformRandomSelector),
    (QuerySelector, TopologicalAllocator),
    (QuerySelector, HeaviestQueryByRole),
    (Measurement, AccuracyAndTiming),
    (Measurement, TimingOnly),
    (Measurement, ParamLearningMeasurement),
    (BaselineAdapter, lambda: NBNAdapter(mechanism="cat", engine="ve")),
    (BaselineAdapter, lambda: NBNAdapter(mechanism="lg", engine="lw")),
    (BaselineAdapter, lambda: PgmpyAdapter(param_method="mle", inference_method="ve")),
    (BaselineAdapter, lambda: PomegranateAdapter(device="cpu")),
    (BaselineAdapter, lambda: PyroAdapter(mechanism="empirical",
                                          inference_method="importance")),
]


@pytest.mark.parametrize(
    "protocol,factory", _COMPONENTS,
    ids=[f"{p.__name__}-{i}" for i, (p, _) in enumerate(_COMPONENTS)],
)
def test_shipped_component_satisfies_its_protocol(protocol, factory):
    assert isinstance(factory(), protocol)


def test_an_object_without_query_is_not_an_adapter():
    """The Protocol really constrains: a fit-only object is rejected."""
    class MissingQuery:
        name = "missing"

        def fit(self, problem, **kwargs):
            pass

        def is_applicable(self, problem):
            return True

    assert not isinstance(MissingQuery(), BaselineAdapter)
