"""Port of tests/unit/test_param_learning_helpers_device_safe.py to v0.13.

Pins device safety of benchmarking.metrics.wasserstein_1d after the in-place
fixes (Phase 3, Decision 3):
  - Both inputs are moved to CPU before quantile computation (PR-B §A.1 fix).
  - Quantile grid length is max(len(x), len(y)) not min (issue #37 fix).
"""
from __future__ import annotations

import pytest
import torch

from benchmarking.metrics import wasserstein_1d


def test_wasserstein_1d_cpu_cpu_returns_nonneg_float() -> None:
    """wasserstein_1d on two CPU tensors returns a non-negative float."""
    a = torch.randn(50)
    b = torch.randn(50)
    result = wasserstein_1d(a, b)
    assert isinstance(result.value, float)
    assert result.value >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_wasserstein_1d_mixed_devices_does_not_raise() -> None:
    """wasserstein_1d(cuda, cpu) must not raise RuntimeError.

    PR-B §A.1 regression: before the .cpu() fix, mixed-device subtraction
    raised RuntimeError: Expected all tensors to be on the same device.
    """
    a_cuda = torch.randn(50, device="cuda")
    b_cpu = torch.randn(50)
    result = wasserstein_1d(a_cuda, b_cpu)
    assert isinstance(result.value, float)
    assert result.value >= 0


def test_wasserstein_1d_unequal_length_uses_max_grid() -> None:
    """wasserstein_1d uses a max(len) quantile grid for unequal-length inputs.

    Issue #37 fix: the old min-length approach compared mismatched quantiles
    for unequal inputs (e.g. n=4000 vs n=2000), inflating W₁.  The max-grid
    approach evaluates both at the same dense grid.

    Regression check: for two samples from the same distribution (N(0,1)),
    W₁ should be small regardless of the length ratio.  With the old min-grid
    approach and a 4:1 length ratio the estimate was biased high.
    """
    torch.manual_seed(42)
    a = torch.randn(4000)  # "predicted" budget
    b = torch.randn(1000)  # "oracle" budget
    result = wasserstein_1d(a, b)
    # Both from N(0,1) → W₁ should be close to 0; allow generous tolerance
    # for finite-sample noise but ensure it's not pathologically inflated.
    assert result.value < 0.5, (
        f"W₁ between two N(0,1) samples (4000 vs 1000) should be < 0.5; "
        f"got {result.value:.4f} — the max-grid fix may have been reverted"
    )
