"""Regression tests for the core review findings (nbn/core, learning, update,
sampling, utils).  Each test names the finding it pins (A1, A2, ... B1, C*).
"""
from __future__ import annotations

import pytest
import torch

from tests.conftest import available_devices

ACCELERATORS = [d for d in available_devices() if d != "cpu"]


# ---------------------------------------------------------------------------
# A1 — to_device must not use non_blocking for device -> CPU copies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ACCELERATORS or ["cpu"])
def test_a1_to_device_accelerator_to_cpu_reads_back_values(device):
    from nbn.utils.device import to_device

    x = torch.randn(20000, 3, device=device) + 5.0
    y = to_device(x, torch.device("cpu"))
    # Read y before anything else synchronises the device: a non_blocking
    # D2H copy is not synchronised and used to come back as zeros / garbage.
    y_snapshot = y.clone()
    torch.testing.assert_close(y_snapshot, x.cpu())
    back = to_device(y, torch.device(device))
    torch.testing.assert_close(back.cpu(), x.cpu())
