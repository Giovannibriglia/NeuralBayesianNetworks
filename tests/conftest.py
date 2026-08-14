"""Pytest fixtures and skip helpers shared across tests."""
from __future__ import annotations

import os

import pytest
import torch

# ---------------------------------------------------------------------------
# Thread pinning under pytest-xdist
# ---------------------------------------------------------------------------
# torch sizes its intra-op thread pool to the machine (14 threads on a
# 20-core host) *per process*.  Under `-n auto` pytest-xdist runs one worker
# process per core, so the suite ran ~20 x 14 = 280 compute threads over 20
# cores and spent most of its time in contention rather than work.
#
# It is not a small effect.  Measured on a 20-core host:
#
#     test_ve_runs_on_neural_categorical_mechanism   300s under -n auto
#                                                    4.3s standalone
#
# — a 70x slowdown, and it is why the "slowest durations" list was topped by
# tests that are individually trivial.  One thread per worker restores the
# intended one-process-per-core layout; the parallelism comes from xdist.
#
# Serial runs (no xdist) keep torch's default, so a developer running a
# single test still gets the full machine.
if os.environ.get("PYTEST_XDIST_WORKER"):
    torch.set_num_threads(1)
    # Inherited by subprocesses the tests launch (the v0.13 cell workers each
    # start their own torch).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def cuda_required():
    """Decorator that skips the test if CUDA is unavailable."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="needs CUDA",
    )


def device_params():
    """Parametrizes a test over ``cpu`` and (if available) ``cuda``."""
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return pytest.mark.parametrize("device", devs)


@pytest.fixture(autouse=True)
def _seed_for_each_test():
    """Reset seed before every test for reproducibility."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    yield
