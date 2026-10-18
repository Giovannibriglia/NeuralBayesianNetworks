"""Regressions for the guarantee review of October 2026 (each test is a found counterexample)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from nbn.uncertainty.binomial import binomial_interval
from nbn.uncertainty.builders import interval_set_from_counts
from nbn.uncertainty.counts import scatter_counts

MPS = torch.backends.mps.is_available()


@pytest.mark.skipif(not MPS, reason="float32 accumulation is the MPS case")
def test_counts_do_not_saturate_at_2_24_on_mps():
    n = 2 ** 24 + 10
    values = torch.zeros(n, dtype=torch.long, device="mps")
    rows = torch.zeros(n, dtype=torch.long, device="mps")
    counts = scatter_counts(rows, values, 1, 2)
    assert int(counts[0, 0].cpu().long()) == n          # float32 holds this even count
    with pytest.raises(ValueError, match="2\\*\\*24"):
        interval_set_from_counts(counts.cpu().float().to("mps"), delta=0.1)


@pytest.mark.skipif(not MPS, reason="float32 end points are the MPS case")
@pytest.mark.parametrize("n", [10 ** 5, 10 ** 6])
def test_kl_interval_on_mps_is_outward_of_the_float64_one(n):
    k = torch.tensor([0.0, 1.0, 5.0])
    nn = torch.full((3,), float(n))
    lo64, up64 = binomial_interval(k.double(), nn.double(), 0.05)
    lo32, up32 = binomial_interval(k.to("mps"), nn.to("mps"), 0.05)
    assert (lo32.cpu().double() <= lo64).all() and (up32.cpu().double() >= up64).all()


def test_iv_program_does_not_refute_a_valid_point_box():
    pytest.importorskip("scipy")
    from nbn.uncertainty.confounded import RegimeBox, rows_from_box
    p = np.array([[[0.25398490716509553, 0.11470445329851321, 0.01749422634404908],
                   [0.23920294390555, 0.190334093213147, 0.18427937607364514]]])
    lo, hi, used = rows_from_box(RegimeBox.point(p), "lp")
    assert used == "lp" and (lo <= hi).all()


def test_counterfactual_hull_is_outward_in_float32():
    from nbn.scm.categorical import counterfactual_bounds_from_sets
    p = torch.tensor([[1.9212932e-7, 1 - 1.9212932e-7]], dtype=torch.float32)
    lo, _ = counterfactual_bounds_from_sets(
        p, torch.tensor([[0.99999982, 0.0]], dtype=torch.float32),
        torch.tensor([[1.0, 1.7619313e-7]], dtype=torch.float32), torch.tensor([0]))
    # q = (1 - 1.76e-7, 1.76e-7) is in the box and gives 1 - 1.76e-7 / 1.92e-7 = 0.0829
    assert float(lo[0, 0]) <= 1 - 1.7619313e-7 / 1.9212932e-7 + 1e-6


def test_pit_band_lower_end_is_sound_where_the_fitted_cdf_is_flat():
    from nbn.scm.calibration import PITBandSet
    # 90% of the held-out PIT values sit exactly at F_hat = 1 (a saturated
    # Gaussian tail or a class with p_hat ~ 0): P(X <= x) can be as low as
    # the strict count, 0.1, not the inclusive one, 1.0.
    pit = torch.cat([torch.linspace(0.0, 0.5, 100), torch.ones(900)])
    band = PITBandSet(node="x", sorted_pit=pit, epsilon=0.05, level=0.95)
    lo, up = band.bounds_from_cdf(torch.tensor([1.0]))
    assert float(lo) <= 0.1 - 0.05 + 1e-9 and float(up) == 1.0
