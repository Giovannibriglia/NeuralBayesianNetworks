"""Pytest fixtures and skip helpers shared across tests."""
from __future__ import annotations

import pytest
import torch


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
