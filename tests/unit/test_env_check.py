"""Environment check (``nbn/bench/_env.py``, ``nbn-bench check-env``, and the
launch gate of ``nbn-bench inference`` / ``param-learning``)."""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from nbn.bench import _env
from nbn.bench._env import (
    REQUIREMENTS,
    RUN_BASE,
    Requirement,
    check_environment,
    check_one,
    format_reports,
    problems,
    required_for_baselines,
)

_ROOT = Path(__file__).resolve().parents[2]


# --- the table is pinned to pyproject.toml ------------------------------------

def _pyproject_requirements() -> dict[str, tuple[str, str]]:
    tomllib = pytest.importorskip("tomllib")   # 3.11+
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    out: dict[str, tuple[str, str]] = {}
    from packaging.requirements import Requirement as PReq
    for extra, items in [("core", data["project"]["dependencies"])] + list(
            data["project"]["optional-dependencies"].items()):
        if extra in {"dev", "all"}:
            continue
        for item in items:
            r = PReq(item)
            out[r.name.lower()] = (str(r.specifier), extra)
    return out


def test_requirement_table_matches_pyproject():
    expected = _pyproject_requirements()
    table = {r.dist.lower(): (r.spec, r.extra) for r in REQUIREMENTS}
    assert set(table) == set(expected), (
        f"add/remove in nbn/bench/_env.py: {set(table) ^ set(expected)}")
    from packaging.specifiers import SpecifierSet
    for name, (spec, extra) in expected.items():
        t_spec, t_extra = table[name]
        assert (SpecifierSet(t_spec), t_extra) == (SpecifierSet(spec), extra), (
            f"{name}: table {table[name]} vs pyproject {(spec, extra)}")


def test_pgmpy_pin_is_at_least_1_0():
    """DiscreteBayesianNetwork (what the adapter imports) exists from 1.0 on."""
    from packaging.specifiers import SpecifierSet
    spec = SpecifierSet(next(r.spec for r in REQUIREMENTS if r.dist == "pgmpy"))
    assert "0.1.26" not in spec and "1.0.0" in spec


# --- per-baseline requirements -----------------------------------------------

def test_pgmpy_probes_cover_what_the_adapter_imports():
    """The 2026-09-14 run had pgmpy 1.x importable while ``pgmpy.models``
    failed on an old scikit-learn; the check must import the submodules."""
    req = next(r for r in REQUIREMENTS if r.dist == "pgmpy")
    assert {"pgmpy.models", "pgmpy.inference", "pgmpy.estimators"} <= set(req.probes)
    skl = next(r for r in REQUIREMENTS if r.dist == "scikit-learn")
    from packaging.specifiers import SpecifierSet
    assert "1.5.2" not in SpecifierSet(skl.spec) and "1.6.0" in SpecifierSet(skl.spec)


def test_required_for_baselines_maps_libraries_and_flow():
    need = required_for_baselines([
        {"library": "nbn", "mechanism": "cat"},
        {"library": "pgmpy", "mechanism": "discrete"},
    ])
    assert need == RUN_BASE | {"pgmpy", "scikit-learn"}
    assert "matplotlib" not in need and "seaborn" not in need
    assert "pomegranate" not in need and "pgmpy" not in RUN_BASE
    assert "scikit-learn" not in RUN_BASE
    need = required_for_baselines([
        {"library": "pyro", "mechanism": "empirical"},
        {"library": "nbn", "mechanism": "flow"},
        {"library": "pomegranate", "mechanism": "discrete"},
    ])
    assert need == RUN_BASE | {"pyro-ppl", "zuko", "pomegranate"}
    assert required_for_baselines([]) == RUN_BASE
    # objects with attributes work too (BaselineSpec)
    spec = types.SimpleNamespace(library="pgmpy", mechanism="lg")
    assert "pgmpy" in required_for_baselines([spec])


# --- check_one statuses (fake modules / metadata) ------------------------------

def _fake_module(monkeypatch, name, version, file="/venv/site-packages/x/__init__.py"):
    mod = types.ModuleType(name)
    if version is not None:
        mod.__version__ = version
    mod.__file__ = file
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _fake_metadata(monkeypatch, versions: dict):
    def _version(dist):
        if dist in versions:
            return versions[dist]
        raise _env._md.PackageNotFoundError(dist)
    monkeypatch.setattr(_env._md, "version", _version)


def test_check_one_ok_and_too_old(monkeypatch):
    req = Requirement("fakelib", "fakelib_mod", ">=1.0", "bench")
    _fake_metadata(monkeypatch, {"fakelib": "1.2.0"})
    _fake_module(monkeypatch, "fakelib_mod", "1.2.0")
    assert check_one(req).status == "ok"
    _fake_metadata(monkeypatch, {"fakelib": "0.9.5"})
    _fake_module(monkeypatch, "fakelib_mod", "0.9.5")
    r = check_one(req)
    assert r.status == "too-old" and "need >=1.0" in r.detail


def test_check_one_missing_and_broken(monkeypatch):
    req = Requirement("fakelib", "fakelib_mod", ">=1.0", "bench")
    _fake_metadata(monkeypatch, {})
    monkeypatch.delitem(sys.modules, "fakelib_mod", raising=False)
    r = check_one(req)
    assert r.status == "missing" and "pip install" in r.detail
    # metadata present but the import blows up
    _fake_metadata(monkeypatch, {"fakelib": "1.5"})

    def _boom(name, *a, **k):
        if name == "fakelib_mod":
            raise ImportError("cannot import name DiscreteBayesianNetwork")
        return real_import(name, *a, **k)
    real_import = _env.importlib.import_module
    monkeypatch.setattr(_env.importlib, "import_module", _boom)
    r = check_one(req)
    assert r.status == "broken" and "DiscreteBayesianNetwork" in r.detail


def test_check_one_probe_submodule_failure_is_broken(monkeypatch):
    """Top-level import fine, adapter submodule not (pgmpy.models -> sklearn)."""
    req = Requirement("fakelib", "fakelib_mod", ">=1.0", "bench",
                      probes=("fakelib_mod.models",))
    _fake_metadata(monkeypatch, {"fakelib": "1.1.2"})
    _fake_module(monkeypatch, "fakelib_mod", "1.1.2")
    real_import = _env.importlib.import_module

    def _boom(name, *a, **k):
        if name == "fakelib_mod.models":
            raise ImportError("cannot import name 'validate_data' from 'sklearn.utils.validation'")
        return real_import(name, *a, **k)
    monkeypatch.setattr(_env.importlib, "import_module", _boom)
    r = check_one(req)
    assert r.status == "broken"
    assert "fakelib_mod.models" in r.detail and "validate_data" in r.detail
    assert r.installed == "1.1.2" and r.location is not None


def test_check_one_mismatch_between_metadata_and_module(monkeypatch):
    """Two copies on sys.path: metadata from one, module from the other."""
    req = Requirement("fakelib", "fakelib_mod", ">=1.0", "bench")
    _fake_metadata(monkeypatch, {"fakelib": "1.1.2"})
    _fake_module(monkeypatch, "fakelib_mod", "0.1.26", file="/home/u/.local/lib/x/__init__.py")
    r = check_one(req)
    assert r.status == "mismatch" and "0.1.26" in r.detail and "1.1.2" in r.detail


def test_user_site_note(monkeypatch):
    req = Requirement("fakelib", "fakelib_mod", ">=1.0", "bench")
    _fake_metadata(monkeypatch, {"fakelib": "2.0"})
    monkeypatch.setattr(_env.site, "getusersitepackages", lambda: "/home/u/.local/lib/py")
    _fake_module(monkeypatch, "fakelib_mod", "2.0", file="/home/u/.local/lib/py/fakelib_mod/__init__.py")
    r = check_one(req)
    assert r.status == "ok" and "user site-packages" in r.detail


def test_check_environment_filters_and_format():
    reports = check_environment({"numpy", "torch"})
    assert [r.dist for r in reports] == ["torch", "numpy"]     # table order
    text = format_reports(reports)
    assert "torch" in text and "numpy" in text and "python" in text
    assert problems(reports) == []


# --- CLI ----------------------------------------------------------------------

def _cli(*args):
    return subprocess.run([sys.executable, "-m", "nbn.bench.cli", *args],
                          capture_output=True, text=True)


def test_check_env_command_passes_in_this_environment():
    res = _cli("check-env")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "environment OK" in res.stdout and "pgmpy" in res.stdout


def test_check_env_with_config_restricts_to_baselines(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"baselines": [{"library": "nbn", "mechanism": "cat"}]}))
    res = _cli("check-env", "--config", str(cfg))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "pgmpy" not in res.stdout and "torch" in res.stdout
    assert "scikit-learn" not in res.stdout


def test_run_gate_refuses_then_skip_flag_proceeds(monkeypatch, tmp_path, capsys):
    """With a requirement reported missing, inference refuses to start (rc 2)
    unless --skip-env-check is given; then it proceeds to the loader."""
    from nbn.bench import cli

    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"baselines": [{"library": "pgmpy", "mechanism": "discrete"}]}))

    def _fake_check(dists=None):
        return [_env.Report("pgmpy", "pgmpy", ">=1.0", "bench", "0.1.26", "0.1.26",
                            "/home/u/.local", "too-old", "have 0.1.26, need >=1.0")]
    monkeypatch.setattr(_env, "check_environment", _fake_check)

    rc = cli.main(["inference", "--config", str(cfg)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "too-old" in err and "--skip-env-check" in err

    # with the flag the gate only warns and control reaches the config loader.
    import nbn.bench.core.yaml_config as yc

    def _reached(*a, **k):
        raise RuntimeError("reached loader")
    monkeypatch.setattr(yc, "load_runner_config", _reached)
    with pytest.raises(RuntimeError, match="reached loader"):
        cli.main(["inference", "--config", str(cfg), "--skip-env-check"])
