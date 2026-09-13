"""Environment check: are the libraries a benchmark run needs installed, at
the versions the adapters are written against, and imported from where we
think they are?

Why this exists: a run on 2026-09-10 recorded every discrete pgmpy cell as
``not_supported`` because the machine had a pre-1.0 pgmpy in ``~/.local``;
the adapter's import guard turned "cannot import DiscreteBayesianNetwork"
into a bare "pip install pgmpy", and the runner classifies ImportError as
"not supported", so the run finished green with no pgmpy numbers. The
benchmark commands now refuse to start when a library one of their
baselines needs is missing, too old, or shadowed by a second copy, and
``nbn-bench check-env`` prints the same table on demand.

The requirement table below is the single in-code source of truth and is
pinned to ``pyproject.toml`` by ``tests/unit/test_env_check.py`` (installed
package metadata cannot be trusted for this: an editable install keeps the
metadata of whenever it was last re-installed).
"""
from __future__ import annotations

import importlib
import importlib.metadata as _md
import site
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    dist: str          # distribution name as on PyPI / in pyproject
    module: str        # top-level import name
    spec: str          # PEP 440 specifier, e.g. ">=1.0" or ">=1.2,<2.0"
    extra: str         # "core" or the pyproject extra that carries it


# Mirrors [project.dependencies] + the runtime extras of pyproject.toml.
# ``dev`` (test / docs toolchain) is deliberately not here.
REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement("torch", "torch", ">=2.2", "core"),
    Requirement("networkx", "networkx", ">=3.0", "core"),
    Requirement("numpy", "numpy", ">=1.24", "core"),
    Requirement("opt-einsum", "opt_einsum", ">=3.3", "core"),
    Requirement("scipy", "scipy", ">=1.10", "bench"),
    Requirement("pyyaml", "yaml", ">=6.0", "bench"),
    Requirement("tqdm", "tqdm", ">=4.65", "bench"),
    Requirement("pandas", "pandas", ">=2.0", "bench"),
    Requirement("pyarrow", "pyarrow", ">=14.0", "bench"),
    Requirement("pgmpy", "pgmpy", ">=1.0", "bench"),
    Requirement("pomegranate", "pomegranate", ">=1.0", "bench"),
    Requirement("matplotlib", "matplotlib", ">=3.7", "bench"),
    Requirement("seaborn", "seaborn", ">=0.12", "bench"),
    Requirement("psutil", "psutil", ">=5.9", "bench"),
    Requirement("packaging", "packaging", ">=23.0", "bench"),
    Requirement("zuko", "zuko", ">=1.2,<2.0", "neural"),
    Requirement("gpytorch", "gpytorch", ">=1.11", "gp"),
    Requirement("pyro-ppl", "pyro", ">=1.9", "mcmc"),
)

_BY_DIST = {r.dist: r for r in REQUIREMENTS}

# What a benchmark RUN needs regardless of its baselines: the library and
# the runner's own dependencies — core + bench minus the plotting libraries
# (only ``nbn-bench plot`` imports them) and minus the baseline libraries
# (required only when a config lists that baseline, see LIBRARY_DISTS).
_NOT_RUN_BASE = frozenset({"matplotlib", "seaborn", "pgmpy", "pomegranate"})
RUN_BASE: frozenset[str] = frozenset(
    r.dist for r in REQUIREMENTS
    if r.extra in {"core", "bench"} and r.dist not in _NOT_RUN_BASE
)

# Baseline ``library`` (config field) -> distributions its adapter imports.
LIBRARY_DISTS: dict[str, frozenset[str]] = {
    "nbn": frozenset(),
    "pgmpy": frozenset({"pgmpy"}),
    "pomegranate": frozenset({"pomegranate"}),
    "pyro": frozenset({"pyro-ppl"}),
}
# nbn mechanisms with an optional-extra backend.
MECHANISM_DISTS: dict[str, frozenset[str]] = {
    "flow": frozenset({"zuko"}),
}


def required_for_baselines(baselines) -> set[str]:
    """Distributions a run with these baselines needs. ``baselines`` is the
    YAML list (dicts with ``library`` / ``mechanism``) or any objects with
    those attributes. Unknown libraries add nothing (the loader rejects them
    later with its own message)."""
    need = set(RUN_BASE)
    for b in baselines or ():
        lib = b.get("library") if isinstance(b, dict) else getattr(b, "library", None)
        mech = b.get("mechanism") if isinstance(b, dict) else getattr(b, "mechanism", None)
        need |= LIBRARY_DISTS.get(str(lib), frozenset())
        if str(lib) == "nbn":
            need |= MECHANISM_DISTS.get(str(mech), frozenset())
    return need


@dataclass
class Report:
    dist: str
    module: str
    spec: str
    extra: str
    installed: str | None      # version from package metadata, None if absent
    imported: str | None       # module.__version__ when the module exposes one
    location: str | None       # directory the module was imported from
    status: str                # ok | missing | too-old | broken | mismatch
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _satisfies(version: str, spec: str) -> bool:
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version
    try:
        return Version(version) in SpecifierSet(spec, prereleases=True)
    except InvalidVersion:
        return False


def _same_release(a: str, b: str) -> bool:
    """Compare ignoring the local segment: torch reports ``2.11.0+cu130`` from
    ``__version__`` while its wheel metadata says ``2.11.0``."""
    from packaging.version import InvalidVersion, Version
    try:
        return Version(a).public == Version(b).public
    except InvalidVersion:
        return a == b


def _user_site_note(location: str | None) -> str:
    """Flag a module imported from the user site-packages (``~/.local``):
    that copy wins over a venv's only when the venv sees user site, and it is
    how a stale pre-1.0 pgmpy ended up in the 2026-09-10 run."""
    if not location:
        return ""
    try:
        user_site = site.getusersitepackages()
    except Exception:  # pragma: no cover - site misconfiguration
        return ""
    if user_site and location.startswith(user_site):
        return f"imported from user site-packages ({user_site}), not from {sys.prefix}"
    return ""


def check_one(req: Requirement) -> Report:
    installed: str | None
    try:
        installed = _md.version(req.dist)
    except _md.PackageNotFoundError:
        installed = None
    imported: str | None = None
    location: str | None = None
    try:
        mod = importlib.import_module(req.module)
    except Exception as exc:  # ImportError, or a broken install raising anything
        if installed is None:
            return Report(req.dist, req.module, req.spec, req.extra, None, None, None,
                          "missing", f"pip install '{req.dist}{req.spec}'")
        return Report(req.dist, req.module, req.spec, req.extra, installed, None, None,
                      "broken", f"import {req.module} failed: {type(exc).__name__}: {exc}")
    v = getattr(mod, "__version__", None)
    imported = str(v) if v is not None else None
    f = getattr(mod, "__file__", None)
    if f:
        import os
        location = os.path.dirname(os.path.abspath(f))
    if installed is None:
        # importable but no metadata (vendored / PYTHONPATH copy)
        ver = imported
        if ver is None:
            return Report(req.dist, req.module, req.spec, req.extra, None, None, location,
                          "missing", "importable but no package metadata and no __version__")
    else:
        ver = installed
    if (imported is not None and installed is not None
            and not _same_release(imported, installed)):
        # Two copies on sys.path: metadata from one, module from another.
        return Report(req.dist, req.module, req.spec, req.extra, installed, imported,
                      location, "mismatch",
                      f"metadata says {installed} but the imported module is {imported} "
                      f"({location}); remove the extra copy")
    if not _satisfies(ver, req.spec):
        return Report(req.dist, req.module, req.spec, req.extra, installed, imported,
                      location, "too-old",
                      f"have {ver}, need {req.spec}: pip install -U '{req.dist}{req.spec}'")
    return Report(req.dist, req.module, req.spec, req.extra, installed, imported,
                  location, "ok", _user_site_note(location))


def check_environment(dists=None) -> list[Report]:
    """Reports for ``dists`` (default: every requirement), in table order."""
    wanted = set(dists) if dists is not None else set(_BY_DIST)
    return [check_one(r) for r in REQUIREMENTS if r.dist in wanted]


def format_reports(reports) -> str:
    rows = [("package", "extra", "required", "installed", "status", "note")]
    for r in reports:
        rows.append((r.dist, r.extra, r.spec, r.installed or "-", r.status, r.detail))
    widths = [max(len(row[i]) for row in rows) for i in range(5)]
    out = []
    for row in rows:
        cells = [row[i].ljust(widths[i]) for i in range(5)]
        line = "  ".join(cells)
        if row[5]:
            line += "  " + row[5]
        out.append(line.rstrip())
    out.insert(1, "-" * len(out[0]))
    out.append("")
    out.append(f"python {sys.version.split()[0]} at {sys.executable}")
    return "\n".join(out)


def problems(reports) -> list[Report]:
    return [r for r in reports if not r.ok]
