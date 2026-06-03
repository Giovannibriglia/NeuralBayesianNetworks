#!/usr/bin/env python3
"""Deprecated entry point — use ``nbn-bench plot`` instead.

The figure-generation logic moved to ``benchmarking/_paper_figures.py`` and is
exposed as the ``nbn-bench plot`` subcommand, mirroring the two-phase
benchmark → figures workflow (``nbn-bench inference`` then ``nbn-bench plot``).

This shim delegates to that module so existing callers — both
``python scripts/make_paper_figures.py --parquet X --output-dir Y`` and
``from scripts.make_paper_figures import run_plot`` / ``n_parameters_lookup`` —
keep working, with a DeprecationWarning recommending migration.
"""
from __future__ import annotations

import argparse
import logging
import warnings

# Re-export the public surface so existing imports continue to resolve.
from benchmarking._paper_figures import (  # noqa: F401
    n_parameters_lookup,
    run_plot,
)

_DEPRECATION = (
    "scripts/make_paper_figures.py is deprecated; use "
    "`nbn-bench plot <parquet-or-dir> --output-dir <dir>` instead."
)


def main(argv: list[str] | None = None) -> int:
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger(__name__).warning(_DEPRECATION)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--aggregation", choices=["iqm_iqr", "mean_std"], default="iqm_iqr")
    ap.add_argument("--benchmark", default=None)
    args = ap.parse_args(argv)

    return run_plot(
        parquet=args.parquet,
        output_dir=args.output_dir,
        aggregation=args.aggregation,
        benchmark=args.benchmark,
    )


if __name__ == "__main__":
    raise SystemExit(main())
