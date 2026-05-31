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

logger = logging.getLogger(__name__)


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
        return 0

    if args.cmd == "inference":
        from benchmarking.core.yaml_config import load_runner_config
        from benchmarking.core.runner import Runner

        device = None if args.device == "auto" else args.device
        cfg = load_runner_config(args.config, device_override=device)

        # ── cell loop ────────────────────────────────────────────────────────
        for _ in Runner().run(cfg):
            pass

        # ── post-run pipeline ────────────────────────────────────────────────
        # JSONL is already on disk from the runner.  Produce parquet + tables
        # + figures so callers get the same output package as v0.12.
        # Each step is independent: a failure logs the error and sets rc=1
        # but does not prevent the remaining steps from running.
        rc = 0
        results_dir = cfg.jsonl_path.parent
        config_name = cfg.config_name
        parquet_path = results_dir / f"{config_name}_metrics.parquet"

        # Step 1: JSONL → parquet
        try:
            from benchmarking.core.output import jsonl_to_parquet
            jsonl_to_parquet(cfg.jsonl_path, parquet_path)
            logger.info("Wrote parquet: %s", parquet_path)
        except Exception as exc:
            logger.error("Post-run step 1 (jsonl_to_parquet) failed: %s", exc)
            rc = 1

        # Step 2+3: aggregate → tables (independent of figures)
        if parquet_path.exists():
            try:
                from benchmarking._aggregate import aggregate
                from benchmarking._tables import write_all
                aggregated = aggregate(parquet_path)
                table_paths = write_all(
                    aggregated,
                    output_dir=results_dir,
                    output_prefix=config_name,
                )
                logger.info(
                    "Wrote %d table files to: %s",
                    len(table_paths), results_dir / "tables",
                )
            except Exception as exc:
                logger.error(
                    "Post-run step 2 (aggregate/tables) failed: %s", exc,
                )
                rc = 1

            # Step 4: figures
            try:
                from benchmarking._plot_v2 import render_figures
                figure_paths = render_figures(
                    parquet_path=parquet_path,
                    output_dir=results_dir,
                    output_prefix=config_name,
                )
                n_figs = sum(len(v) for v in figure_paths.values())
                logger.info(
                    "Wrote %d figure files to: %s",
                    n_figs, results_dir / "figures",
                )
            except Exception as exc:
                logger.error(
                    "Post-run step 3 (render_figures) failed: %s", exc,
                )
                rc = 1

        return rc

    raise AssertionError(f"unhandled subcommand {args.cmd!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
