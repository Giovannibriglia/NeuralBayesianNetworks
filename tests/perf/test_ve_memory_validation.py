"""Measured VE peak memory never exceeds the estimate, on every device (#178).

Runs the cases of ``scripts/ve_memory_validation.py`` whose peaks stay below
about 1 GiB on each available device, through the script itself: CPU (exact,
profiler allocation events), CUDA (allocated bytes and the caching
allocator's reserved growth) and MPS (heap-level growth).  On CUDA and MPS
the script runs every case in a fresh process — blocks cached by earlier
tests absorbed a case's allocations and let an under-estimate pass when this
test ran inside the full suite (review of #301).  ``perf``: it allocates real
memory and should run on a quiet runner — ``pytest tests/perf -m perf``.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.conftest import available_devices

pytestmark = [pytest.mark.perf, pytest.mark.slow]

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                       "ve_memory_validation.py")


@pytest.mark.parametrize("device", available_devices())
def test_estimate_bounds_measured_peak(device):
    proc = subprocess.run(
        [sys.executable, _SCRIPT, "--device", device, "--quick", "--isolate"],
        capture_output=True, text=True, timeout=3600,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
