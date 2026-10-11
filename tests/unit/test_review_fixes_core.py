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


# ---------------------------------------------------------------------------
# A2 — weighted_moments(unbiased=True) with fractional / normalised weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices())
def test_a2_unbiased_weighted_std_is_scale_invariant_for_fractional_weights(device):
    from nbn.learning.weighting import weighted_moments

    g = torch.Generator().manual_seed(0)
    t = (torch.randn(500, 2, generator=g) * 2.0 + 3.0).to(device)
    ref = t.std(0, unbiased=True)
    # Normalised (EM-responsibility-like) weights: sum(w) == 1, so the old
    # frequency-weight denominator sum(w) - 1 was ~0 and std exploded.
    w = torch.full((500,), 1.0 / 500, device=device)
    _, s = weighted_moments(t, w, unbiased=True)
    torch.testing.assert_close(s, ref, atol=1e-4, rtol=1e-4)
    # Any constant rescaling of fractional weights gives the same answer.
    w2 = torch.rand(500, generator=g).to(device) + 0.1
    _, s_a = weighted_moments(t, w2, unbiased=True)
    _, s_b = weighted_moments(t, w2 * 0.01, unbiased=True)
    torch.testing.assert_close(s_a, s_b, atol=1e-4, rtol=1e-4)
    assert (s_a - ref).abs().max() < 0.3


def test_a2_integer_weights_keep_replication_equivalence():
    from nbn.learning.weighting import weighted_moments

    t = torch.randn(40, 3)
    w = torch.randint(0, 4, (40,)).float()
    w[0] = 2.0
    rep = t.repeat_interleave(w.long(), dim=0)
    m, s = weighted_moments(t, w, unbiased=True)
    torch.testing.assert_close(m, rep.mean(0), atol=1e-5, rtol=0)
    torch.testing.assert_close(s, rep.std(0, unbiased=True), atol=1e-5, rtol=0)


def test_a2_effective_sample_size():
    from nbn.learning.weighting import effective_sample_size

    assert float(effective_sample_size(torch.ones(10))) == pytest.approx(10.0)
    assert float(effective_sample_size(torch.full((10,), 0.1))) == pytest.approx(10.0)
    assert float(effective_sample_size(torch.tensor([1.0, 0.0, 0.0]))) == pytest.approx(1.0)
    assert effective_sample_size(None, n=7) == 7
