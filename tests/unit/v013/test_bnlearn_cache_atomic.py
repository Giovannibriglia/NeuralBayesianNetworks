"""bnlearn ``.bif`` cache population is atomic (concurrent xdist workers).

``_ensure_cached`` used to check ``exists()``, then write ``<name>.bif.gz``
and decompress straight into ``<name>.bif``: a second process seeing the
final path mid-write parsed a truncated network, and two writers shared one
``.gz`` path. Both steps now go through private temp files + ``os.replace``.
"""
from __future__ import annotations

import gzip
import io
import threading

from nbn.bench.problems import bnlearn

_BIF = b"network toy {\n}\n" + b"// padding\n" * 5000


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(url, timeout=None, context=None):
    return _Resp(gzip.compress(_BIF))


def test_final_path_never_visible_while_decompressing(tmp_path, monkeypatch):
    monkeypatch.setattr(bnlearn, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(bnlearn.urllib.request, "urlopen", _fake_urlopen)
    seen_during_write = []
    real_open = gzip.open

    class _Spy:
        def __init__(self, f):
            self._f = f

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return self._f.__exit__(*a)

        def read(self, *a):
            # Decompression in progress: is the final path already visible?
            seen_during_write.append((tmp_path / "toy.bif").exists())
            return self._f.read(*a)

    def spying_gzip_open(*a, **k):
        return _Spy(real_open(*a, **k).__enter__())

    monkeypatch.setattr(bnlearn.gzip, "open", spying_gzip_open)
    out = bnlearn._ensure_cached("toy")
    assert seen_during_write == [False]        # old code: created it first
    assert out.read_bytes() == _BIF
    assert sorted(p.name for p in tmp_path.iterdir()) == ["toy.bif"]


def test_concurrent_populators_all_get_the_complete_file(tmp_path, monkeypatch):
    monkeypatch.setattr(bnlearn, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(bnlearn.urllib.request, "urlopen", _fake_urlopen)
    barrier = threading.Barrier(4)
    results, errors = [], []

    def worker():
        barrier.wait()
        try:
            results.append(bnlearn._ensure_cached("toy").read_bytes())
        except Exception as exc:  # pragma: no cover - the failure we guard
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert results == [_BIF] * 4
    assert sorted(p.name for p in tmp_path.iterdir()) == ["toy.bif"]


def test_cached_file_gets_normal_permissions_not_mkstemp_0600(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(bnlearn, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(bnlearn.urllib.request, "urlopen", _fake_urlopen)
    old = os.umask(0o022)
    try:
        out = bnlearn._ensure_cached("toy")
    finally:
        os.umask(old)
    assert (out.stat().st_mode & 0o777) == 0o644
