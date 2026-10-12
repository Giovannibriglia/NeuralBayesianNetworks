"""Measured VE peak memory never exceeds the estimate, on every device (#178).

Runs the small cases of ``scripts/ve_memory_validation.py`` (peaks up to a
few hundred MiB) on each available device: CPU (exact, profiler allocation
events), CUDA (exact allocated bytes, plus the caching allocator's reserved
growth) and MPS (heap-level growth from an emptied cache, the footprint
Metal's limit applies to).  ``perf``: it allocates real memory and should run
on a quiet runner — ``pytest tests/perf -m perf``.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

from tests.conftest import available_devices

pytestmark = [pytest.mark.perf, pytest.mark.slow]

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                       "ve_memory_validation.py")


def _script():
    spec = importlib.util.spec_from_file_location("ve_memory_validation", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("device", available_devices())
def test_estimate_bounds_measured_peak(device):
    v = _script()
    builders, cases = v.cases()
    models = {}
    for case in cases:
        if case.big:
            continue
        if case.model_key not in models:
            models[case.model_key] = builders[case.model_key]().to(device)
        r = v.run_case(device, models[case.model_key], case)
        assert r["estimate"] >= r["measured"], r
        if device != "cpu":
            # Heap-level growth (MPS heaps, CUDA reserved segments).
            assert r["estimate"] >= r["driver"], r
