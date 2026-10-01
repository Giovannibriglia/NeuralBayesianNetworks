"""macOS / Apple-Silicon (``mps``) compatibility.

The MPS backend differs from CPU and CUDA in two ways NBN's fit paths used
to assume away: it has **no float64 tensors** (creating one raises) and no
LAPACK-backed ``torch.linalg.lstsq``.  There is no Metal device on the Linux
CI hosts, so these tests check the contract two ways:

* the capability helpers in :mod:`nbn.utils.device` answer the right thing
  for an ``mps`` device object (constructing ``torch.device("mps")`` needs no
  hardware), and
* the closed-form fits are run under a *simulated* MPS: ``supports_float64``
  is forced to ``False`` and a ``TorchDispatchMode`` fails the test the moment
  any op produces a float64 tensor.  That is exactly the condition a real MPS
  device enforces, so a pass here means the path cannot hit the
  "Cannot convert a MPS Tensor to float64" error, and the float32 path is
  also asserted to agree with the float64 one.

The macOS CI lane runs the device-parametrised suites on a real ``mps``
device on top of this (``tests/conftest.py::available_devices``).
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import nbn.utils.device as dev
from nbn.bench._env import format_reports, check_environment
from nbn.bench.core._device import resolve_device as bench_resolve_device
from nbn.bench.core.runner import _classify_exception
from nbn.bench.metrics import gpu_peak_mb
from nbn.learning.weighting import (
    validate_weights,
    weighted_mean,
    weighted_moments,
    weighted_quantile,
)
from nbn.mechanisms import (
    CategoricalTableMechanism,
    LinearGaussianMechanism,
    NeuralCategoricalMechanism,
)
from nbn.mechanisms.non_parametric.conditional_kde import ConditionalKDEMechanism
from nbn.update import recursive_gaussian as rg

MPS = torch.device("mps")
CPU = torch.device("cpu")


# --- capability helpers --------------------------------------------------------

def test_accum_dtype_is_float64_except_on_mps():
    assert dev.supports_float64(CPU) and dev.supports_float64("cuda:0")
    assert not dev.supports_float64(MPS)
    assert dev.accum_dtype(CPU) is torch.float64
    assert dev.accum_dtype(MPS) is torch.float32


def test_linalg_device_hops_to_cpu_only_on_mps():
    assert dev.linalg_device(MPS) == CPU
    assert dev.linalg_device(CPU) == CPU
    assert dev.linalg_device(torch.device("cuda", 1)) == torch.device("cuda", 1)


def test_synchronize_and_empty_cache_are_noops_on_cpu():
    dev.synchronize(CPU)
    dev.empty_cache(CPU)
    assert dev.peak_memory_bytes(CPU) is None
    assert gpu_peak_mb(CPU).value == 0.0


def test_accelerator_summary_names_a_backend():
    s = dev.accelerator_summary()
    assert s.startswith(("cuda", "mps", "none"))
    text = format_reports(check_environment({"torch"}))
    assert "accelerator" in text and "platform" in text


def test_resolve_device_auto_prefers_mps_over_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert dev.resolve_device("auto").type == "mps"
    assert dev.mps_available()
    # The benchmark adapters follow the same preference order.
    assert bench_resolve_device("auto") == "mps"
    assert bench_resolve_device(None) == "mps"
    assert bench_resolve_device("cpu") == "cpu"      # explicit wins


def test_resolve_device_auto_prefers_cuda_over_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert dev.resolve_device("auto").type == "cuda"
    assert bench_resolve_device("auto") == "cuda"


def test_mps_oom_is_classified_as_oom():
    exc = RuntimeError(
        "MPS backend out of memory (MPS allocated: 9.01 GB, other allocations: "
        "384.00 KB, max allowed: 9.07 GB). Tried to allocate 64.00 MB on private pool."
    )
    assert _classify_exception(exc) == "oom"


# --- PYTORCH_ENABLE_MPS_FALLBACK -----------------------------------------------

def _reload_mps_env(monkeypatch, platform: str, preset: str | None):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    if preset is not None:
        monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", preset)
    import nbn._mps_env as m
    importlib.reload(m)
    return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")


def test_mps_fallback_env_set_on_darwin_only(monkeypatch):
    assert _reload_mps_env(monkeypatch, "darwin", None) == "1"
    assert _reload_mps_env(monkeypatch, "darwin", "0") == "0"      # user's choice kept
    assert _reload_mps_env(monkeypatch, "linux", None) is None
    assert _reload_mps_env(monkeypatch, "win32", None) is None


# --- simulated MPS: no float64 anywhere in the closed-form fits ---------------

class _NoFloat64(TorchDispatchMode):
    """Fail on the first op whose output is a float64 tensor.

    An MPS device raises on any float64 allocation; this reproduces that
    constraint on CPU so the fit paths can be exercised without the hardware.
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        flat = out if isinstance(out, (list, tuple)) else (out,)
        for t in flat:
            if isinstance(t, torch.Tensor) and t.dtype is torch.float64:
                raise AssertionError(f"{func} produced a float64 tensor (MPS cannot)")
        return out


@pytest.fixture
def simulated_mps(monkeypatch):
    monkeypatch.setattr(dev, "supports_float64", lambda device: False)
    with _NoFloat64():
        yield


def _cat_data(n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (n,), generator=g)
    pa = torch.randint(0, 3, (n, 1), generator=g).float()
    w = torch.rand(n, generator=g) * 3 + 0.1
    return x, pa, w


def _gauss_data(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    pa = torch.randn(n, 2, generator=g)
    x = (pa @ torch.tensor([[1.5], [-0.7]]) + 0.3 + 0.2 * torch.randn(n, 1, generator=g))
    w = torch.rand(n, generator=g) * 3 + 0.1
    return x, pa, w


def test_weight_helpers_without_float64(simulated_mps):
    x, pa, w = _gauss_data()
    v = validate_weights(w, x.shape[0], where="t", device=x.device)
    assert v.dtype is torch.float32 and v.device == x.device
    weighted_mean(x, v)
    weighted_moments(pa, v, unbiased=True)
    weighted_quantile(pa, torch.tensor([0.25, 0.75]), v)


def test_categorical_table_fit_and_update_without_float64(simulated_mps):
    x, pa, w = _cat_data()
    m = CategoricalTableMechanism()
    m.fit_local(x, pa, weights=w, parent_cards=[3], n_classes=2)
    m.update_local(x[:50], pa[:50], parent_cards=[3], n_classes=2)  # update is unweighted
    m2 = CategoricalTableMechanism()
    m2.fit_local(x, pa, parent_cards=[3], n_classes=2)


def test_neural_categorical_root_without_float64(simulated_mps):
    x, _, w = _cat_data()
    m = NeuralCategoricalMechanism(n_classes=2, hidden=(8,))
    m.fit_local(x, None, weights=w, epochs=1)


def test_linear_gaussian_fit_and_update_without_float64(simulated_mps):
    x, pa, w = _gauss_data()
    m = LinearGaussianMechanism()
    m.fit_local(x, pa, weights=w)
    m.update_local(x[:100], pa[:100])  # update_local is unweighted by contract
    r = LinearGaussianMechanism()
    r.fit_local(x, None, weights=w)


def test_recursive_gaussian_without_float64(simulated_mps):
    x, pa, w = _gauss_data()
    st = rg.batch_statistics(pa, x, w)
    assert st.A.dtype is torch.float32
    rg.solve(rg.accumulate(st, rg.batch_statistics(pa[:50], x[:50])))


def test_conditional_kde_fit_and_update_without_float64(simulated_mps):
    x, pa, w = _gauss_data()
    m = ConditionalKDEMechanism()
    m.fit_local(x, pa, weights=w)
    m.update_local(x[:60], pa[:60], weights=w[:60], forgetting=0.9)


# --- the float32 path agrees with the float64 one -----------------------------

def test_float32_accumulation_matches_float64_results(monkeypatch):
    x, pa, w = _cat_data()
    ref = CategoricalTableMechanism()
    ref.fit_local(x, pa, weights=w, parent_cards=[3], n_classes=2)
    gx, gpa, gw = _gauss_data()
    ref_lg = LinearGaussianMechanism()
    ref_lg.fit_local(gx, gpa, weights=gw)

    monkeypatch.setattr(dev, "supports_float64", lambda device: False)
    got = CategoricalTableMechanism()
    got.fit_local(x, pa, weights=w, parent_cards=[3], n_classes=2)
    got_lg = LinearGaussianMechanism()
    got_lg.fit_local(gx, gpa, weights=gw)

    for (_, a), (_, b) in zip(ref.state_dict().items(), got.state_dict().items()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    for (_, a), (_, b) in zip(ref_lg.state_dict().items(), got_lg.state_dict().items()):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
