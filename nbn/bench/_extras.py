"""Import-time check that the ``bench`` extra is installed.

``pip install nbn`` deliberately installs only what ``import nbn`` needs; the
benchmark suite's own dependencies (pandas, pyarrow, yaml, scipy, tqdm) come
with ``pip install "nbn[bench]"``. ``nbn/bench/__init__.py`` imports this module
first, so a missing extra surfaces as one actionable ImportError instead of a
bare ModuleNotFoundError from deep inside the runner.
"""
from __future__ import annotations

import importlib.util

_REQUIRED = {"pandas": "pandas", "pyarrow": "pyarrow", "yaml": "pyyaml",
             "scipy": "scipy", "tqdm": "tqdm"}

_missing = [dist for mod, dist in _REQUIRED.items()
            if importlib.util.find_spec(mod) is None]
if _missing:
    raise ImportError(
        "nbn.bench needs the benchmark extras (missing: " + ", ".join(_missing)
        + '). Install them with:  pip install "nbn[bench]"'
    )
