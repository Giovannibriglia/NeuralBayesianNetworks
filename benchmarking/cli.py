"""``nbn-bench`` CLI dispatcher (v0.13).

The benchmark → figures workflow is two-phase, both under this one CLI:

    nbn-bench inference --config benchmarking/configs/synthetic/smoke_tests/inference_smoke.yaml
    nbn-bench plot <results-dir-or-parquet> --output-dir <out>

``inference`` runs the benchmark and writes a parquet (+ tables/figures);
``plot`` reads that parquet and renders the paper figures + LaTeX tables
(docs/v0.13-paper-figures.md). ``param-learning`` is structurally preserved
but stubbed — the ParamLearningMeasurement is deferred to a later v0.13 phase.
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

    plot = sub.add_parser(
        "plot",
        help="Generate paper figures + LaTeX tables from a benchmark parquet.",
        description=(
            "Read a benchmark parquet (the output of `nbn-bench inference`) and "
            "produce figures + LaTeX tables per docs/v0.13-paper-figures.md."
        ),
    )
    plot.add_argument("parquet",
                      help="Path to a *_metrics.parquet file, or a directory "
                           "containing one (a results dir from `nbn-bench inference`).")
    plot.add_argument("--output-dir", required=True,
                      help="Directory to write figures + LaTeX tables into.")
    plot.add_argument("--aggregation", choices=["iqm_iqr", "mean_std"],
                      default="iqm_iqr",
                      help="Aggregation statistic (default: iqm_iqr).")
    plot.add_argument("--benchmark", default=None,
                      help="Restrict to one benchmark; default processes all "
                           "present in the parquet.")
    plot.add_argument("-v", "--verbose", action="store_true")

    return parser


def _setup_console_logging(verbose: bool, *, inference: bool) -> None:
    """Configure root + console handler. The inference benchmark is tqdm-led,
    so its console shows WARNING+ unless ``-v`` (the run.log keeps full INFO);
    other commands keep INFO on the console."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):           # idempotent across calls/tests
        root.removeHandler(h)
    console = logging.StreamHandler()
    if inference and not verbose:
        console.setLevel(logging.WARNING)
    else:
        console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(console)


def _attach_run_log(results_dir) -> logging.FileHandler:
    """Attach a FileHandler writing all INFO+ logs to <results_dir>/run.log."""
    from pathlib import Path
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(
        Path(results_dir) / "run.log", mode="w", encoding="utf-8",
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(handler)
    return handler


def _suppress_library_warnings() -> None:
    """Silence known-noisy dependency warnings on the console (the subprocess
    re-emits pgmpy's FutureWarning to its stderr, which is captured to the
    run.log per cell, so nothing is actually lost)."""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"pgmpy.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pyro.*")


def _run_cells(cfg) -> None:
    """Drive the cell loop, with the v0.14 batch_sizes sweep (#148, §5.6).

    No ``batch_sizes`` in the config → single pass, exactly the
    pre-sweep behavior. With ``batch_sizes``, one pass per sweep value:

      * swept baselines — adapter ``supports_batched_queries`` and no
        explicit YAML ``batch_size`` pin — run at every sweep value,
        with their batch_size overridden to it;
      * pinned / non-batchable baselines run once, on the first pass,
        at their own batch_size (typically 1). An explicit pin always
        wins: a batchable baseline pinned to 1 runs once, sequentially.

    All passes append to the same streaming JSONL (JsonlWriter opens in
    append mode), so the final parquet carries every sweep value with
    the batch_size column distinguishing them (§1.4).
    """
    import dataclasses

    from benchmarking.core.config import build_adapter
    from benchmarking.core.runner import Runner

    if not cfg.batch_sizes:
        for _ in Runner().run(cfg):
            pass
        return

    def _swept(spec) -> bool:
        if spec.batch_size_pinned:
            return False
        adapter = build_adapter(spec)
        return bool(getattr(adapter, "supports_batched_queries", False))

    swept_flags = [_swept(spec) for spec in cfg.baselines]
    for i, sweep_value in enumerate(cfg.batch_sizes):
        pass_baselines = []
        for spec, is_swept in zip(cfg.baselines, swept_flags):
            if is_swept:
                pass_baselines.append(
                    dataclasses.replace(spec, batch_size=sweep_value)
                )
            elif i == 0:
                # Pinned / non-batchable: once, at its own batch_size.
                pass_baselines.append(spec)
        if not pass_baselines:
            continue
        logger.info(
            "batch_sizes sweep: pass %d/%d (batch_size=%d, %d baselines)",
            i + 1, len(cfg.batch_sizes), sweep_value, len(pass_baselines),
        )
        pass_cfg = dataclasses.replace(cfg, baselines=pass_baselines)
        for _ in Runner().run(pass_cfg):
            pass


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _setup_console_logging(args.verbose, inference=(args.cmd == "inference"))

    if args.cmd == "param-learning":
        print(
            "param-learning is not yet implemented in v0.13.\n"
            "The parameter-learning Measurement is deferred to a later phase.\n"
            "See issue #109 for status. Use `nbn-bench inference` for now.",
            file=sys.stderr,
        )
        return 0

    if args.cmd == "plot":
        from pathlib import Path

        from benchmarking._paper_figures import run_plot
        return run_plot(
            parquet=Path(args.parquet),
            output_dir=Path(args.output_dir),
            aggregation=args.aggregation,
            benchmark=args.benchmark,
        )

    if args.cmd == "inference":
        from benchmarking.core.yaml_config import load_runner_config

        # "auto" passes through as a literal string; each adapter's
        # resolve_device() turns it into cuda-if-available-else-cpu.
        # (Was: collapsed to None here, which then collapsed to "cpu" at
        # build_adapter — so every nbn baseline silently ran on CPU.)
        device = args.device
        cfg = load_runner_config(args.config, device_override=device)

        # Console noise from dependencies stays out of the terminal; the
        # subprocess re-emits it to stderr -> captured to run.log per cell.
        _suppress_library_warnings()

        # The run.log lives in the results dir alongside the parquet and
        # captures the full INFO stream (incl. per-cell subprocess stderr),
        # regardless of the console level. Detached in `finally` so it closes
        # even on crash.
        results_dir = cfg.jsonl_path.parent
        config_name = cfg.config_name
        parquet_path = results_dir / f"{config_name}_metrics.parquet"
        log_handler = _attach_run_log(results_dir)
        try:
            # ── cell loop (v0.14: sweeps batch_sizes when configured) ──────
            _run_cells(cfg)

            # ── post-run pipeline ──────────────────────────────────────────
            # JSONL is already on disk from the runner.  Produce parquet +
            # tables + figures so callers get the same output package as
            # v0.12.  Each step is independent: a failure logs the error and
            # sets rc=1 but does not prevent the remaining steps from running.
            rc = 0

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
        except Exception:
            # Route any uncaught exception to run.log *before* the finally
            # block detaches the FileHandler.  Python's default excepthook
            # only fires at interpreter top-level — after this finally runs —
            # so a traceback would otherwise miss run.log entirely, leaving it
            # ending on its last INFO line (the "silent stop" seen 2026-06-04).
            # Re-raised so the exit code / stderr traceback are unchanged.
            logger.critical("Unhandled exception during inference run",
                            exc_info=True)
            raise
        finally:
            logging.getLogger().removeHandler(log_handler)
            log_handler.close()

    raise AssertionError(f"unhandled subcommand {args.cmd!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
