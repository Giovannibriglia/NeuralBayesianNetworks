"""v0.4 inference crash test entry script (synthetic BNs).

Thin wrapper around :func:`benchmarking.crash_test_runner.run_inference`.
Same logic is reachable via ``nbn-bench inference --config <yaml>``.

The v0.2 bnlearn-corpus inference-throughput script lives at
``examples/crash_test_inference_bnlearn.py`` (renamed in v0.4b PR #12
to free the canonical ``crash_test_inference.py`` name for this
synthetic variant).  See ``examples/crash_test.py`` for the v0.3
page-1 figure script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from benchmarking.crash_test_runner import run_inference  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "v0.4 inference crash test on synthetic BNs (accuracy + B=1024 "
            "throughput).  See examples/crash_test_inference_bnlearn.py for "
            "the v0.2 bnlearn-corpus variant."
        ),
    )
    parser.add_argument(
        "--config", default="benchmarking/configs/crash_test_smoke.yaml",
        help="Crash-test YAML (smoke or full).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    return run_inference(config_path=args.config, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
