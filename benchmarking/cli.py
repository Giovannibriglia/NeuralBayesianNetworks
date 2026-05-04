"""Top-level ``nbn-bench`` CLI dispatcher.

Subcommands
-----------
* ``run <config>``         — existing benchmark-suite runner (v0.2/v0.3 path).
* ``param-learning ...``   — parameter-learning crash test (v0.4b — stub).
* ``inference ...``        — inference crash test (v0.4b — stub).

The ``param-learning`` and ``inference`` subcommands are wired here in v0.4a
so the entry point in ``pyproject.toml`` is stable; their actual
implementations land in PR-2 (``feat/v0.4b-crash-tests-and-baselines``).
Calling either today raises ``NotImplementedError`` with a pointer to
PR-2.
"""
from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)

_PR2_REF = "feat/v0.4b-crash-tests-and-baselines"

_PR2_PARAM_LEARNING_MSG = (
    "Parameter-learning crash test is implemented in v0.4b "
    f"(branch '{_PR2_REF}'). PR-1 only ships the synthetic generator + "
    "CLI scaffolding."
)
_PR2_INFERENCE_MSG = (
    "Inference crash test is implemented in v0.4b "
    f"(branch '{_PR2_REF}'). PR-1 only ships the synthetic generator + "
    "CLI scaffolding."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nbn-bench",
        description=(
            "NBN benchmarking command-line entry point. "
            "Dispatches to the legacy YAML-driven runner ('run'), the "
            "parameter-learning crash test ('param-learning'), or the "
            "inference crash test ('inference')."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    p_run = sub.add_parser(
        "run",
        help="Run a benchmark suite from a YAML config (legacy v0.2 path).",
    )
    p_run.add_argument("config", type=str, help="Path to suite YAML.")
    p_run.add_argument("-v", "--verbose", action="store_true")

    p_param = sub.add_parser(
        "param-learning",
        help="Crash test for per-node CPD-learning accuracy (v0.4b).",
    )
    p_param.add_argument(
        "--config", type=str, required=True,
        help="Path to a crash-test YAML "
             "(e.g. benchmarking/configs/crash_test_smoke.yaml).",
    )
    p_param.add_argument("-v", "--verbose", action="store_true")

    p_inf = sub.add_parser(
        "inference",
        help="Crash test for inference accuracy + speed at fixed B (v0.4b).",
    )
    p_inf.add_argument(
        "--config", type=str, required=True,
        help="Path to a crash-test YAML "
             "(e.g. benchmarking/configs/crash_test_smoke.yaml).",
    )
    p_inf.add_argument("-v", "--verbose", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.cmd == "run":
        from benchmarking.runner import run as _run
        _run(args.config)
        return 0
    if args.cmd == "param-learning":
        raise NotImplementedError(_PR2_PARAM_LEARNING_MSG)
    if args.cmd == "inference":
        raise NotImplementedError(_PR2_INFERENCE_MSG)
    parser.error(f"Unhandled subcommand {args.cmd!r}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
