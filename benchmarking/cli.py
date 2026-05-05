"""``nbn-bench`` CLI dispatcher (v0.5).

Two subcommands, both consume a YAML config:

    nbn-bench param-learning --config benchmarking/configs/parameter_learning_smoke.yaml
    nbn-bench inference      --config benchmarking/configs/inference_smoke.yaml

The previous ``nbn-bench run`` subcommand and its YAML-driven legacy
runner were removed in v0.5a.
"""
from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nbn-bench",
        description="NBN benchmarking entry point — parameter-learning and inference crash tests.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    pl = sub.add_parser(
        "param-learning",
        help="Crash test for per-node CPD-learning accuracy.",
    )
    pl.add_argument("--config", required=True,
                    help="Path to a parameter-learning YAML config.")
    pl.add_argument("--device", default="auto",
                    help="'auto' (default), 'cpu', or 'cuda[:i]'.")
    pl.add_argument("-v", "--verbose", action="store_true")

    inf = sub.add_parser(
        "inference",
        help="Crash test for inference total-time + accuracy at fixed B.",
    )
    inf.add_argument("--config", required=True,
                     help="Path to an inference YAML config.")
    inf.add_argument("--device", default="auto")
    inf.add_argument("-v", "--verbose", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if args.cmd == "param-learning":
        from benchmarking.crash_test_runner import run_parameter_learning
        return run_parameter_learning(
            args.config, device=args.device, verbose=args.verbose,
        )
    if args.cmd == "inference":
        from benchmarking.crash_test_runner import run_inference
        return run_inference(
            args.config, device=args.device, verbose=args.verbose,
        )
    raise AssertionError(f"unhandled subcommand {args.cmd!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
