"""Moving and casting tensors across devices in the order every backend supports.

On torch 2.14 a fused ``t.to(device="cpu", dtype=torch.float64)`` of an MPS
tensor returns garbled values; ``to_device_dtype`` does the two steps in a safe
order.  These tests run on every available accelerator and are skipped on
CPU-only hosts.
"""
from __future__ import annotations

import pytest
import torch

from nbn.learning.weighting import weighted_moments
from nbn.utils.device import to_device_dtype
from tests.conftest import available_devices

ACCELERATORS = [d for d in available_devices() if d != "cpu"]
ACCEL_PARAMS = ACCELERATORS or [pytest.param("none", marks=pytest.mark.skip(reason="no accelerator"))]


@pytest.mark.parametrize("device", ACCEL_PARAMS)
def test_accelerator_to_cpu_float64_is_exact(device):
    x = torch.randn(4096, 3, device=device)
    reference = x.cpu().double()
    assert torch.equal(to_device_dtype(x, "cpu", torch.float64), reference)


@pytest.mark.parametrize("device", ACCEL_PARAMS)
def test_cpu_float64_to_accelerator_float32(device):
    x = torch.randn(1000, 2, dtype=torch.float64)
    out = to_device_dtype(x, device, torch.float32)
    assert out.device.type == torch.device(device).type and out.dtype == torch.float32
    assert torch.allclose(out.cpu().double(), x, atol=1e-6)


@pytest.mark.parametrize("device", ACCEL_PARAMS)
def test_weights_on_an_accelerator_with_cpu_data(device):
    t = torch.randn(5000, 2)
    w = torch.rand(5000)
    expected = weighted_moments(t, w)
    got = weighted_moments(t, w.to(device))
    for a, b in zip(got, expected):
        assert torch.allclose(a.cpu(), b.cpu(), atol=1e-5)


def test_same_device_cast_is_a_plain_cast():
    x = torch.arange(5)
    assert to_device_dtype(x, "cpu", torch.float32).dtype == torch.float32
