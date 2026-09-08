"""NormalizingFlowMechanism.fit_local: early stopping + root learning rate.

Measured motivation (n=50 synthetic, 20480 rows, batch 1024): conditional
flow nodes reach their held-out optimum in ~60-140 steps and then overfit,
while an unconditional (root) flow at lr 5e-4 was still descending after
2000 steps and lost to LinearGaussian.  fit_local now stops on a held-out
slice and scales lr for roots; these tests pin that contract.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("zuko", reason="flow tests need the neural extra")
from nbn.mechanisms.parametric.normalizing_flow import (  # noqa: E402
    NormalizingFlowMechanism,
)


def _small_flow():
    return NormalizingFlowMechanism(num_transforms=1, hidden=(8,), bins=4)


def test_early_stopping_ends_before_the_epoch_cap_and_reports_it():
    torch.manual_seed(0)
    pa = torch.randn(2000, 1)
    x = 0.5 * pa + 0.3 * torch.randn(2000, 1)
    mech = _small_flow()
    metrics = mech.fit_local(x, pa, epochs=200, batch_size=256, lr=5e-3)
    assert metrics["early_stopped"] is True
    assert metrics["epochs_run"] < 200
    assert metrics["n_val"] == 200
    assert metrics["steps"] == metrics["epochs_run"] * 8  # ceil(1800 / 256)
    assert torch.isfinite(torch.tensor(metrics["best_val_nll"]))
    assert mech.is_fitted and not mech.training


def test_early_stopping_is_skipped_when_the_validation_slice_is_too_small():
    torch.manual_seed(0)
    x = torch.randn(100, 1)  # 10% slice = 10 rows < _MIN_VAL_ROWS
    mech = _small_flow()
    metrics = mech.fit_local(x, None, epochs=3, batch_size=50)
    assert metrics["n_val"] == 0
    assert metrics["early_stopped"] is False
    assert metrics["epochs_run"] == 3
    assert metrics["best_val_nll"] is None


def test_early_stopping_can_be_disabled():
    torch.manual_seed(0)
    x = torch.randn(1000, 1)
    mech = _small_flow()
    metrics = mech.fit_local(x, None, epochs=4, batch_size=500, early_stopping=False)
    assert metrics == {
        **metrics, "n_val": 0, "epochs_run": 4, "steps": 8, "early_stopped": False,
    }


def test_epochs_zero_stays_a_no_op_with_early_stopping_on():
    torch.manual_seed(0)
    x = torch.randn(1000, 1)
    mech = _small_flow()
    metrics = mech.fit_local(x, None, epochs=0)
    assert metrics["epochs_run"] == 0 and metrics["steps"] == 0
    assert metrics["n_val"] == 0 and metrics["early_stopped"] is False


def test_restores_best_epoch_parameters(monkeypatch):
    """After stopping, the live parameters equal the best-validation snapshot."""
    torch.manual_seed(0)
    x = torch.randn(1000, 1)
    mech = _small_flow()
    seen: list[float] = []
    real = NormalizingFlowMechanism._val_nll

    def spy(self, *a, **k):
        v = real(self, *a, **k)
        seen.append(v)
        return v

    monkeypatch.setattr(NormalizingFlowMechanism, "_val_nll", spy)
    metrics = mech.fit_local(x, None, epochs=30, batch_size=100, lr=2e-2, patience=2)
    assert seen and min(seen) == pytest.approx(metrics["best_val_nll"])
    # The restored parameters must reproduce the best validation NLL, not
    # the (worse) NLL of the epoch training stopped on.
    assert seen[-1] >= metrics["best_val_nll"]


def test_root_lr_multiplier_scales_only_unconditional_flows():
    captured = {}
    real_adam = torch.optim.Adam

    class SpyAdam(real_adam):
        def __init__(self, params, lr, **kw):
            captured["lr"] = lr
            super().__init__(params, lr=lr, **kw)

    torch.manual_seed(0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.optim, "Adam", SpyAdam)
        _small_flow().fit_local(torch.randn(64, 1), None, epochs=1, lr=1e-3)
        root_lr = captured["lr"]
        _small_flow().fit_local(
            torch.randn(64, 1), torch.randn(64, 1), epochs=1, lr=1e-3,
        )
        cond_lr = captured["lr"]
        _small_flow().fit_local(
            torch.randn(64, 1), None, epochs=1, lr=1e-3, root_lr_multiplier=1.0,
        )
        root_lr_off = captured["lr"]
    assert root_lr == pytest.approx(1e-2)
    assert cond_lr == pytest.approx(1e-3)
    assert root_lr_off == pytest.approx(1e-3)
