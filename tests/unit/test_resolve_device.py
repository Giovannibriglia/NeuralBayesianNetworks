"""Device-spec resolution, as one table for both resolvers.

* ``nbn.utils.device.resolve_device`` (library): ``"auto"`` picks
  cuda > mps > cpu; ``None`` means cpu; explicit specs pass through.
* ``nbn.bench.core._device.resolve_device`` (benchmark adapters): ``None``
  and ``"auto"`` both auto-detect (cuda > mps > cpu); ``"gpu"`` means *an
  accelerator* -- mps on Apple Silicon, and cuda (an error row later, never a
  silent cpu fallback) when there is none; explicit specs, including indexed
  ``cuda:N``, pass through without an availability check (that happens at
  allocation time).

Availability is monkeypatched, so every row runs on every host.  Merged from
test_device_propagation.py, test_mps_compat.py and the bench
test_device_autodetect.py::TestResolveDevice.
"""
from __future__ import annotations

import pytest
import torch

import nbn.utils.device as dev
from nbn.bench.core._device import resolve_device as bench_resolve_device

_T, _F = True, False

# (spec, cuda available, mps available, expected device type)
_LIBRARY = [
    ("auto", _T, _T, "cuda"),
    ("auto", _F, _T, "mps"),
    ("auto", _F, _F, "cpu"),
    (None, _T, _T, "cpu"),
    ("cpu", _T, _T, "cpu"),
    (torch.device("cpu"), _T, _T, "cpu"),
]

# (spec, cuda available, mps available, expected string)
_BENCH = [
    (None, _T, _F, "cuda"),
    (None, _F, _T, "mps"),
    (None, _F, _F, "cpu"),
    ("auto", _T, _T, "cuda"),
    ("auto", _F, _T, "mps"),
    ("auto", _F, _F, "cpu"),
    ("cpu", _T, _T, "cpu"),           # explicit wins over availability
    ("cuda", _F, _F, "cuda"),         # no validation here: deferred to alloc time
    ("cuda:0", _F, _F, "cuda:0"),
    ("cuda:1", _F, _F, "cuda:1"),
    ("gpu", _T, _T, "cuda"),
    ("gpu", _F, _T, "mps"),           # Apple Silicon: GPU-only baselines land on Metal
    ("gpu", _F, _F, "cuda"),          # no accelerator: fail into an error row, never cpu
]


def _fake_availability(monkeypatch, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize("spec,cuda,mps,want", _LIBRARY,
                         ids=[f"{s}-cuda{int(c)}-mps{int(m)}" for s, c, m, _ in _LIBRARY])
def test_library_resolve_device(monkeypatch, spec, cuda, mps, want):
    _fake_availability(monkeypatch, cuda, mps)
    assert dev.resolve_device(spec).type == want
    if spec == "auto" and want == "mps":
        assert dev.mps_available()


@pytest.mark.parametrize("spec,cuda,mps,want", _BENCH,
                         ids=[f"{s}-cuda{int(c)}-mps{int(m)}" for s, c, m, _ in _BENCH])
def test_bench_resolve_device(monkeypatch, spec, cuda, mps, want):
    _fake_availability(monkeypatch, cuda, mps)
    assert bench_resolve_device(spec) == want


def test_real_auto_resolution_names_a_known_backend():
    """Unpatched: whatever this host has, ``auto`` lands on a real backend."""
    assert dev.resolve_device("auto").type in {"cpu", "cuda", "mps"}
