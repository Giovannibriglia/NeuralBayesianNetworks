"""Tests for GPU/device auto-detection across the v0.13 framework.

Covers the device-flow fix (investigation report 2026-06-04): every nbn
baseline silently ran on CPU because ``--device auto`` was collapsed to
``None`` at the CLI and then to ``"cpu"`` at ``build_adapter``.

Layers under test:
  1. ``resolve_device()`` — the single shared resolution helper.
  2. Per-adapter resolution — each adapter owns its translation (β).
  3. CLI argument parsing — ``"auto"`` is the default and passes through.
  4. ``CellResult.device`` — the new self-documenting column.
  5. Smoke configs — no longer pin ``device: cpu``.

The cuda-only assertions are skipped when no GPU is present (CI), so the
suite is green on both Giovanni's laptop (RTX 4070) and GPU-less runners.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from benchmarking.adapters import (
    NBNAdapter,
    PgmpyAdapter,
    PomegranateAdapter,
    PyroAdapter,
)
from benchmarking.core._device import resolve_device

_HAS_CUDA = torch.cuda.is_available()
_EXPECTED_AUTO = "cuda" if _HAS_CUDA else "cpu"


# ---------------------------------------------------------------------------
# 1. resolve_device() unit tests
# ---------------------------------------------------------------------------

class TestResolveDevice:
    def test_none_resolves_to_cuda_when_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device(None) == "cuda"

    def test_none_resolves_to_cpu_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device(None) == "cpu"

    def test_auto_resolves_to_cuda_when_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("auto") == "cuda"

    def test_auto_resolves_to_cpu_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device("auto") == "cpu"

    def test_cpu_passes_through(self, monkeypatch):
        # Even with CUDA "available", an explicit "cpu" is honoured.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("cpu") == "cpu"

    def test_cuda_passes_through(self, monkeypatch):
        # No availability validation here — that's deferred to alloc time.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device("cuda") == "cuda"

    def test_cuda_indexed_passes_through(self):
        assert resolve_device("cuda:0") == "cuda:0"
        assert resolve_device("cuda:1") == "cuda:1"


# ---------------------------------------------------------------------------
# 2. Per-adapter device resolution (design choice β: per-adapter ownership)
# ---------------------------------------------------------------------------

def _make(adapter_cls):
    """Construct each adapter with its required args, varying only device."""
    if adapter_cls is NBNAdapter:
        return lambda **kw: NBNAdapter(mechanism="cat", engine="lw", **kw)
    if adapter_cls is PyroAdapter:
        return lambda **kw: PyroAdapter(
            mechanism="empirical", inference_method="importance", **kw,
        )
    if adapter_cls is PomegranateAdapter:
        return lambda **kw: PomegranateAdapter(**kw)
    if adapter_cls is PgmpyAdapter:
        return lambda **kw: PgmpyAdapter(
            param_method="mle", inference_method="ve", **kw,
        )
    raise AssertionError(adapter_cls)


# pgmpy is excluded from the auto/cuda parametrization: it is CPU-only and
# always reports "cpu" regardless of the device arg (tested separately).
_GPU_CAPABLE = [NBNAdapter, PyroAdapter, PomegranateAdapter]


class TestAdapterDeviceResolution:
    @pytest.mark.parametrize("adapter_cls", _GPU_CAPABLE)
    def test_default_none_auto_detects(self, adapter_cls):
        adapter = _make(adapter_cls)(device=None)
        assert str(adapter.device) == resolve_device(None)

    @pytest.mark.parametrize("adapter_cls", _GPU_CAPABLE)
    def test_auto_auto_detects(self, adapter_cls):
        adapter = _make(adapter_cls)(device="auto")
        assert str(adapter.device) == _EXPECTED_AUTO

    @pytest.mark.parametrize("adapter_cls", _GPU_CAPABLE)
    def test_explicit_cpu(self, adapter_cls):
        adapter = _make(adapter_cls)(device="cpu")
        assert str(adapter.device) == "cpu"

    @pytest.mark.parametrize("adapter_cls", _GPU_CAPABLE)
    @pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device available")
    def test_explicit_cuda(self, adapter_cls):
        adapter = _make(adapter_cls)(device="cuda")
        assert str(adapter.device) == "cuda"

    def test_pgmpy_is_cpu_only_for_none(self):
        adapter = _make(PgmpyAdapter)(device=None)
        assert adapter.device == "cpu"

    def test_pgmpy_ignores_cuda_request(self):
        # pgmpy accepts the arg but always runs on CPU (CPU-only library).
        adapter = _make(PgmpyAdapter)(device="cuda")
        assert adapter.device == "cpu"


@pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device available")
class TestAdapterParamsLandOnCuda:
    """When device=cuda, fitted model params actually live on the GPU."""

    def _discrete_problem(self, n=300, seed=0):
        from benchmarking.domains.base import BenchmarkProblem

        torch.manual_seed(seed)
        dag = [("X0", "X1"), ("X1", "X2")]
        td = {
            "X0": torch.randint(0, 2, (n,)),
            "X1": torch.randint(0, 2, (n,)),
            "X2": torch.randint(0, 2, (n,)),
        }
        variables = dict.fromkeys(td, ("discrete", 2))
        return BenchmarkProblem(
            name="d", dag=dag, variables=variables,
            train_data=td, test_data=td, queries=[],
        )

    def test_nbn_params_on_cuda(self):
        adapter = NBNAdapter(mechanism="cat", engine="lw", device="cuda")
        adapter.fit(self._discrete_problem(), epochs=2)
        devs = {p.device.type for p in adapter.model.parameters()}
        assert devs == {"cuda"}

    def test_pomegranate_params_on_cuda(self):
        adapter = PomegranateAdapter(device="cuda")
        adapter.fit(self._discrete_problem())
        devs = {p.device.type for p in adapter.model.parameters()}
        assert devs == {"cuda"}


# ---------------------------------------------------------------------------
# 3. CLI integration: "auto" is the default and parses as a literal string
# ---------------------------------------------------------------------------

class TestCliDeviceParsing:
    def _parse(self, *argv):
        from benchmarking.cli import _build_parser

        return _build_parser().parse_args(list(argv))

    def test_default_is_auto_string(self):
        args = self._parse("inference", "--config", "x.yaml")
        assert args.device == "auto"

    def test_explicit_auto(self):
        args = self._parse("inference", "--config", "x.yaml", "--device", "auto")
        assert args.device == "auto"

    def test_explicit_cuda(self):
        args = self._parse("inference", "--config", "x.yaml", "--device", "cuda")
        assert args.device == "cuda"

    def test_explicit_cpu(self):
        args = self._parse("inference", "--config", "x.yaml", "--device", "cpu")
        assert args.device == "cpu"

    def test_auto_not_collapsed_to_none(self):
        # The bug: cli.py did `None if args.device == "auto" else args.device`.
        # The dispatcher now assigns `device = args.device` verbatim, so the
        # parsed value is never None for the default case.
        args = self._parse("inference", "--config", "x.yaml")
        assert args.device is not None


# ---------------------------------------------------------------------------
# 4. CellResult carries the device column
# ---------------------------------------------------------------------------

class TestCellResultDevice:
    def _row(self, **overrides):
        from benchmarking.core.results import CellResult

        base = dict(
            benchmark="synthetic", family="discrete", problem_id="3",
            seed=0, baseline="nbn-cat-lw", query_role="random",
            metric="tv_per_node", value=0.1, status="ok",
            fit_time_s=1.0, query_time_s=0.5, metrics_time_s=0.2,
        )
        base.update(overrides)
        return CellResult(**base)

    def test_device_field_present(self):
        row = self._row(device="cuda")
        assert row.device == "cuda"
        assert "device" in {f.name for f in dataclasses.fields(row)}

    def test_device_in_asdict(self):
        row = self._row(device="cuda")
        assert dataclasses.asdict(row)["device"] == "cuda"

    def test_device_defaults_to_none(self):
        # Additive column: omitting it must not break construction.
        row = self._row()
        assert row.device is None


# ---------------------------------------------------------------------------
# 5. Smoke configs no longer pin device: cpu
# ---------------------------------------------------------------------------

_SMOKE_CONFIGS = [
    "benchmarking/configs/bnlearn/smoke_tests/inference_smoke.yaml",
    "benchmarking/configs/synthetic/smoke_tests/scalability_smoke.yaml",
]


class TestSmokeConfigsDoNotPinDevice:
    @pytest.mark.parametrize("rel_path", _SMOKE_CONFIGS)
    def test_no_baseline_pins_device(self, rel_path):
        from pathlib import Path

        import yaml

        repo_root = Path(__file__).resolve().parents[3]
        cfg = yaml.safe_load((repo_root / rel_path).read_text())
        pinned = [b for b in cfg["baselines"] if "device" in b]
        assert pinned == [], (
            f"{rel_path} still pins device on baselines: {pinned}; smoke "
            f"configs should auto-detect (GPU locally, CPU on CI)."
        )
