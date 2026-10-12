"""Pytest fixtures, device helpers and marker plumbing shared across tests.

Markers (declared in ``pyproject.toml``; ``--strict-markers`` is on):

* ``slow``        >3 s of CPU; excluded from the fast default selection.
* ``subprocess``  launches cell workers or the ``nbn-bench`` CLI.
* ``network``     may download (bnlearn.com) when the cache is cold.
* ``perf``        wall-clock / speed-ratio assertions (``tests/perf``).
* ``cuda``        needs a CUDA device; ``multi_gpu`` needs two or more.
* ``mps``         needs Apple Metal.
* ``gpu``         legacy alias of ``cuda``.

``cuda`` / ``mps`` are added automatically to any case parametrised with a
``device`` of that type, so a test written as
``@pytest.mark.parametrize("device", available_devices())`` is selected by
``-m mps`` / ``-m cuda`` without further annotation.  Device-marked tests skip
cleanly (with a reason) on a host without the device; they never pass by
silently running on the CPU.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# One GPU per xdist worker on multi-GPU hosts
# ---------------------------------------------------------------------------
# Under ``-n N`` every worker (and every cell subprocess it starts) would
# otherwise auto-select ``cuda:0``, piling N CUDA contexts onto one card.
# Spread workers round-robin over the visible GPUs -- unless the scheduler
# (SLURM, a container runtime) or the user already chose, in which case
# CUDA_VISIBLE_DEVICES is left alone.  This must run before anything touches
# ``torch.cuda``: the visibility mask is read once, at driver initialisation.
# The NVML count does not initialise CUDA.
_XDIST_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "")
if _XDIST_WORKER.startswith("gw") and "CUDA_VISIBLE_DEVICES" not in os.environ:
    import torch.cuda

    _n_gpus = getattr(torch.cuda, "_device_count_nvml", lambda: -1)()
    if _n_gpus and _n_gpus > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(_XDIST_WORKER[2:]) % _n_gpus)

import torch  # noqa: E402  (after the CUDA_VISIBLE_DEVICES decision)

# ---------------------------------------------------------------------------
# Thread pinning under pytest-xdist
# ---------------------------------------------------------------------------
# torch sizes its intra-op thread pool to the machine (14 threads on a
# 20-core host) *per process*.  Under `-n auto` pytest-xdist runs one worker
# process per core, so the suite ran ~20 x 14 = 280 compute threads over 20
# cores and spent most of its time in contention rather than work.
#
# It is not a small effect.  Measured on a 20-core host:
#
#     test_ve_runs_on_neural_categorical_mechanism   300s under -n auto
#                                                    4.3s standalone
#
# — a 70x slowdown, and it is why the "slowest durations" list was topped by
# tests that are individually trivial.  One thread per worker restores the
# intended one-process-per-core layout; the parallelism comes from xdist.
#
# Serial runs (no xdist) keep torch's default, so a developer running a
# single test still gets the full machine.
if _XDIST_WORKER:
    torch.set_num_threads(1)
    # Inherited by subprocesses the tests launch (the bench cell workers each
    # start their own torch).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def cuda_required():
    """Marks a test as needing CUDA (skipped cleanly without it)."""
    return pytest.mark.cuda


def available_devices() -> list[str]:
    """``cpu`` plus whichever accelerators this host has (``cuda``, ``mps``).

    Tests that parametrize over devices use this so the macOS CI lane
    exercises the MPS code paths for real; on a Linux box it is ``["cpu"]``
    or ``["cpu", "cuda"]``.
    """
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if _mps_available():
        devs.append("mps")
    return devs


def available_accelerators() -> list[str]:
    """The non-CPU entries of :func:`available_devices` (possibly empty).

    Parametrise device-*transfer* tests with this, not with
    ``available_accelerators() or ["cpu"]``: a cpu->cpu "transfer" passes
    vacuously.  An empty list is reported by pytest as a skip.
    """
    return [d for d in available_devices() if d != "cpu"]


def device_params():
    """Parametrizes a test over ``cpu`` and the available accelerators."""
    return pytest.mark.parametrize("device", available_devices())


# ---------------------------------------------------------------------------
# Device markers: derive from the ``device`` parameter, skip when absent
# ---------------------------------------------------------------------------

def _param_device_type(item) -> str | None:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None
    dev = callspec.params.get("device")
    if isinstance(dev, (str, torch.device)):
        try:
            return torch.device(dev).type
        except (RuntimeError, TypeError):
            return None
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Runs before ``-m`` deselection so derived markers are selectable."""
    has_cuda = torch.cuda.is_available()
    n_cuda = torch.cuda.device_count() if has_cuda else 0
    has_mps = _mps_available()
    for item in items:
        dtype = _param_device_type(item)
        if dtype == "cuda" or item.get_closest_marker("gpu"):
            item.add_marker(pytest.mark.cuda)
        if dtype == "mps":
            item.add_marker(pytest.mark.mps)
        if item.get_closest_marker("multi_gpu") and n_cuda < 2:
            item.add_marker(pytest.mark.skip(reason=f"needs >=2 CUDA devices (found {n_cuda})"))
        elif item.get_closest_marker("cuda") and not has_cuda:
            item.add_marker(pytest.mark.skip(reason="needs a CUDA device"))
        if item.get_closest_marker("mps") and not has_mps:
            item.add_marker(pytest.mark.skip(reason="needs an Apple MPS device"))


# ---------------------------------------------------------------------------
# bnlearn networks: offline unless the test is marked ``network``
# ---------------------------------------------------------------------------
#: Decompressed copies of bnlearn.com's asia / sachs / alarm ``.bif`` (the same
#: bytes ``nbn.bench.problems.bnlearn._ensure_cached`` would download).
_VENDORED_BIF_DIR = Path(__file__).parent / "data" / "bnlearn"


@pytest.fixture(autouse=True)
def _offline_bnlearn_cache(request, monkeypatch, tmp_path_factory):
    """Serve discrete bnlearn networks from ``tests/data/bnlearn``.

    Tests not marked ``network`` get a private cache seeded with the vendored
    files, and any download attempt fails loudly -- so the default selection
    never touches the network and a test that needs another network says so
    with ``@pytest.mark.network`` (which keeps the real ``~/.cache``).
    """
    if request.node.get_closest_marker("network"):
        yield
        return
    try:
        from nbn.bench.problems import bnlearn
    except ImportError:  # bench extra not installed
        yield
        return
    cache = tmp_path_factory.getbasetemp() / "bnlearn_cache"
    if not cache.exists():
        cache.mkdir()
        for f in _VENDORED_BIF_DIR.glob("*.bif"):
            shutil.copyfile(f, cache / f.name)
    monkeypatch.setattr(bnlearn, "_CACHE_DIR", cache)

    def _refuse_download(url, *args, **kwargs):
        raise RuntimeError(
            f"test tried to download {url!r}; vendor the network under "
            f"tests/data/bnlearn or mark the test @pytest.mark.network")

    monkeypatch.setattr(bnlearn.urllib.request, "urlopen", _refuse_download)
    yield


@pytest.fixture(autouse=True)
def _seed_for_each_test():
    """Reset seed before every test for reproducibility."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    yield
