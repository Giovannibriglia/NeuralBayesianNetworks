"""``nbn-bench`` CLI dispatcher (v0.13).

Two subcommands, both consume a YAML config:

    nbn-bench inference     --config benchmarking/configs/inference_smoke.yaml
    nbn-bench param-learning --config benchmarking/configs/parameter_learning_smoke.yaml

``param-learning`` is structurally preserved but stubbed — the
ParamLearningMeasurement is deferred to a later v0.13 phase.
See issue #109 for status.
"""
from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nbn-bench",
        description="NBN benchmarking entry point — inference and parameter-learning.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    pl = sub.add_parser(
        "param-learning",
        help="Parameter-learning benchmark (stubbed in v0.13; use inference).",
    )
    pl.add_argument("--config", required=True,
                    help="Path to a parameter-learning YAML config.")
    pl.add_argument("--device", default="auto",
                    help="'auto' (default), 'cpu', or 'cuda[:i]'.")
    pl.add_argument("-v", "--verbose", action="store_true")

    inf = sub.add_parser(
        "inference",
        help="Inference benchmark: accuracy + timing across baselines and families.",
    )
    inf.add_argument("--config", required=True,
                     help="Path to an inference YAML config.")
    inf.add_argument("--device", default="auto",
                     help="'auto' (default), 'cpu', or 'cuda[:i]'.")
    inf.add_argument("-v", "--verbose", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.cmd == "param-learning":
        print(
            "param-learning is not yet implemented in v0.13.\n"
            "The parameter-learning Measurement is deferred to a later phase.\n"
            "See issue #109 for status. Use `nbn-bench inference` for now.",
            file=sys.stderr,
        )
        return 2

    if args.cmd == "inference":
        from benchmarking.core.yaml_config import load_runner_config
        from benchmarking.core.runner import Runner

        device = None if args.device == "auto" else args.device
        cfg = load_runner_config(args.config, device_override=device)
        runner = Runner(cfg)
        runner.run()
        return 0

    raise AssertionError(f"unhandled subcommand {args.cmd!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
