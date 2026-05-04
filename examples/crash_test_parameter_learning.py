"""Parameter-learning crash test entry script.

This is a thin wrapper around :func:`benchmarking.crash_test_runner.run_parameter_learning`
so the v0.4 brief's listed entry-point file exists at the documented path.
The same logic is reachable via ``nbn-bench param-learning --config <yaml>``;
both share the same runner.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Self-bootstrap so the script runs without an editable install.
import sys as _sys
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

from benchmarking.crash_test_runner import run_parameter_learning  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="v0.4 parameter-learning crash test (accuracy only)",
    )
    parser.add_argument(
        "--config", default="benchmarking/configs/crash_test_smoke.yaml",
        help="Crash-test YAML (smoke or full).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    return run_parameter_learning(config_path=args.config, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
