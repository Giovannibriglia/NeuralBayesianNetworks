"""Library-wide constants for NBN.

These are intentionally exposed as module-level globals (not env vars) so
that downstream code can monkey-patch them in tests when needed.
"""
from __future__ import annotations

# Standard deviation of the Dirac-tight-Gaussian used for continuous do(X=v).
# Small enough to be effectively a delta for sampling/inference, large enough
# to keep log-prob and KL computations finite.
DIRAC_GAUSSIAN_SIGMA: float = 1e-4
