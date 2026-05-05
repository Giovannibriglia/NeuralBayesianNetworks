"""PR-B §A.1 contract: parameter-learning accuracy helpers must not
crash when truth and fit tensors are on different devices.

The pre-fix path subtracted ``true_p - fit_p`` directly without
``.to(...)`` — when ``device='auto'`` resolved to cuda for the fitter
but the truth lived on cpu (or vice-versa), every NBN/pgmpy
param-learning cell hit ``RuntimeError: Expected all tensors to be on
the same device``.
"""
from __future__ import annotations

import pytest
import torch

from benchmarking.crash_test_runner import _w1


def test_w1_handles_mixed_devices_cpu() -> None:
    a = torch.randn(50)
    b = torch.randn(50)
    out = _w1(a, b)
    assert isinstance(out, float)
    assert out >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_w1_handles_mixed_devices_cuda() -> None:
    a_cuda = torch.randn(50, device="cuda")
    b_cpu = torch.randn(50, device="cpu")
    # Pre-fix path raised RuntimeError; post-fix path normalises both
    # tensors to cpu before subtraction.
    out = _w1(a_cuda, b_cpu)
    assert isinstance(out, float)
    assert out >= 0
