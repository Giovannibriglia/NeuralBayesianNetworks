"""macOS: let torch fall back to the CPU for ops the MPS backend lacks.

Imported first by ``nbn/__init__.py`` so it runs before ``torch`` is.

PyTorch's Metal backend implements most, but not all, of ATen.  An op with
no MPS kernel raises ``NotImplementedError`` unless
``PYTORCH_ENABLE_MPS_FALLBACK=1`` is in the environment, in which case torch
runs that one op on the CPU (with a one-time warning) and moves the result
back.  NBN's own kernels avoid the known gaps (float64, ``linalg.lstsq``; see
``nbn.utils.device``), but the libraries it builds on -- zuko flows,
gpytorch, pyro -- are outside its control, and a hard failure deep inside a
flow's transform is a worse experience on a laptop than a slow op.

``setdefault`` keeps an explicit user setting (including ``0``) intact.
torch reads the variable lazily, on the first op that would need the
fallback, so setting it at ``import nbn`` time is effective even when torch
was imported earlier in the process.  Linux and Windows are untouched: the
variable is meaningless there and never consulted.
"""
from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
