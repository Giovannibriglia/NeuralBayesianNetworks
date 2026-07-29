"""``nbn-bench`` CLI dispatcher (v0.13).

The benchmark → figures workflow is two-phase, both under this one CLI:

    nbn-bench inference --config benchmarking/configs/synthetic/smoke_tests/inference_smoke.yaml
    nbn-bench plot <results-dir-or-parquet> --output-dir <out>

``inference`` runs the benchmark and writes the JSONL + parquet (the
canonical run artifact); ``plot`` reads that parquet and renders the paper
figures + LaTeX tables on demand into a chosen output dir
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
    pl.add_argument("--results-dir", default=None,
                    help="Pin the run output directory instead of creating a "
                         "fresh timestamped one. Required for --resume "
                         "(SLURM 24h restart chains).")
    pl.add_argument("--resume", action="store_true",
                    help="Resume an interrupted run in --results-dir: skip "
                         "cells recorded in completed_cells.jsonl and re-run "
                         "only the rest.")
    pl.add_argument("-v", "--verbose", action="store_true")

    inf = sub.add_parser(
        "inference",
        help="Inference benchmark: accuracy + timing across baselines and families.",
    )
    inf.add_argument("--config", required=True,
                     help="Path to an inference YAML config.")
    inf.add_argument("--device", default="auto",
                     help="'auto' (default), 'cpu', or 'cuda[:i]'.")
    inf.add_argument("--results-dir", default=None,
                     help="Pin the run output directory instead of creating a "
                          "fresh timestamped one. Required for --resume "
                          "(SLURM 24h restart chains).")
    inf.add_argument("--resume", action="store_true",
                     help="Resume an interrupted run in --results-dir: skip "
                          "cells recorded in completed_cells.jsonl and re-run "
                          "only the rest.")
    inf.add_argument("-v", "--verbose", action="store_true")

    plot = sub.add_parser(
        "plot",
        help="Generate paper figures + LaTeX tables from a benchmark parquet.",
        description=(
            "Read a benchmark parquet (the output of `nbn-bench inference`) and "
            "produce figures + LaTeX tables per docs/v0.13-paper-figures.md."
        ),
    )
    plot.add_argument("parquet", nargs="+",
                      help="One or more *_metrics.parquet files (or directories "
                           "containing one, a results dir from `nbn-bench "
                           "inference`). Multiple parquets are row-concatenated "
                           "before plotting — e.g. a parameter-learning parquet "
                           "plus an inference parquet for the divergence panel.")
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
    # Append mode: a resumed run (--resume, SLURM restart chain) reuses the
    # same results dir, and each restart's log must not truncate the previous
    # one. Fresh runs get a fresh timestamped dir, so "a" == "w" for them.
    handler = logging.FileHandler(
        Path(results_dir) / "run.log", mode="a", encoding="utf-8",
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


def _resolve_results_dir(args) -> tuple:
    """Turn ``--results-dir`` / ``--resume`` into (jsonl_path, resume).

    Returns ``(None, False)`` when no dir is pinned (the loader then creates
    the usual fresh timestamped dir). A pinned dir that already holds a
    ``metrics.jsonl`` REQUIRES ``--resume`` — silently appending a second run
    to an existing JSONL would duplicate cells in the parquet.
    """
    from pathlib import Path

    if args.resume and not args.results_dir:
        raise SystemExit(
            "--resume requires --results-dir (the pinned directory of the "
            "run to continue).")
    if not args.results_dir:
        return None, False
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / "metrics.jsonl"
    if jsonl_path.exists() and not args.resume:
        raise SystemExit(
            f"{jsonl_path} already exists; pass --resume to continue that "
            f"run, or choose a fresh --results-dir.")
    return jsonl_path, args.resume


def _run_cells(cfg) -> bool:
    """Drive the cell loop.

    v0.14 fit-once query-many (#174, design doc §3.2/§6): a single
    ``Runner().run(cfg)`` pass drives the whole config. The batch_sizes
    sweep is no longer an outer loop here — the runner resolves a
    per-baseline batch_sizes list (``cfg.batch_sizes`` for swept baselines,
    ``[spec.batch_size]`` for pinned / non-batchable ones) and the cell
    worker fits each ``(problem, seed, baseline)`` cell ONCE, then loops
    ``measure()`` over that list. This eliminates the redundant per-sweep-
    value re-fits that the old outer loop incurred (one full fit per sweep
    value); see ``runner._resolve_batch_sizes``.

    With no ``cfg.batch_sizes`` every baseline resolves to a length-1 list,
    so existing non-sweep configs (bnlearn, scalability, smoke) are
    unchanged. The final parquet still carries every sweep value with the
    batch_size column distinguishing them (§1.4), stamped per row by the
    measurement layer (PR #168).
    Returns True when the loop stopped early on a preemption signal
    (SIGUSR1/SIGTERM — SLURM wall-time warning; checkpoint.py).
    """
    from benchmarking.core.runner import Runner

    runner = Runner()
    for _ in runner.run(cfg):
        pass
    return runner.preempted


def _execute_run(cfg, *, what: str = "inference") -> int:
    """Drive a configured run to its parquet artifact.

    Shared by the ``inference`` and ``param-learning`` commands — the only
    difference between them is how ``cfg`` is built (which Measurement it
    carries); everything downstream (warning suppression, run.log, cell loop,
    JSONL → parquet) is identical. Returns the process exit code.

    ``what`` names the command for the crash-log line only, so each command's
    run.log message stays accurate ("Unhandled exception during <what> run").
    """
    # Console noise from dependencies stays out of the terminal; the subprocess
    # re-emits it to stderr -> captured to run.log per cell.
    _suppress_library_warnings()

    # SLURM 24h wall-time support (checkpoint.py): SIGUSR1 (the
    # `--signal=B:USR1@900` early warning) / SIGTERM stop the cell loop on the
    # next cell boundary; the run then exits with PREEMPTED_EXIT_CODE so the
    # next job in the array chain knows to `--resume`.
    from benchmarking.core.checkpoint import (
        PREEMPTED_EXIT_CODE,
        install_preemption_handlers,
    )
    install_preemption_handlers()

    # The run.log lives in the results dir alongside the parquet and captures
    # the full INFO stream (incl. per-cell subprocess stderr), regardless of the
    # console level. Detached in `finally` so it closes even on crash.
    results_dir = cfg.jsonl_path.parent
    parquet_path = results_dir / f"{cfg.config_name}_metrics.parquet"
    log_handler = _attach_run_log(results_dir)
    try:
        # ── cell loop (v0.14: sweeps batch_sizes when configured) ──────────
        preempted = _run_cells(cfg)

        # ── post-run pipeline ──────────────────────────────────────────────
        # JSONL is already on disk from the runner; convert it to the parquet
        # that is the single canonical artifact of a run. Paper figures + LaTeX
        # tables are produced ON DEMAND by the separate `nbn-bench plot
        # <run-dir>` command (benchmarking/_paper_figures), never auto-generated
        # here. On preemption the (partial) parquet is still written — it is
        # simply regenerated from the full JSONL by the final resumed run.
        rc = 0
        try:
            from benchmarking.core.output import jsonl_to_parquet
            jsonl_to_parquet(cfg.jsonl_path, parquet_path)
            logger.info("Wrote parquet: %s", parquet_path)
        except Exception as exc:
            logger.error("Post-run step (jsonl_to_parquet) failed: %s", exc)
            rc = 1
        if preempted and rc == 0:
            logger.warning(
                "run preempted before completion — exiting %d; resume with "
                "--results-dir %s --resume",
                PREEMPTED_EXIT_CODE, results_dir,
            )
            rc = PREEMPTED_EXIT_CODE
        return rc
    except Exception:
        # Route any uncaught exception to run.log *before* the finally block
        # detaches the FileHandler. Python's default excepthook only fires at
        # interpreter top-level — after this finally runs — so a traceback would
        # otherwise miss run.log entirely (the "silent stop" seen 2026-06-04).
        # Re-raised so the exit code / stderr traceback are unchanged.
        logger.critical("Unhandled exception during %s run", what,
                        exc_info=True)
        raise
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _setup_console_logging(args.verbose, inference=(args.cmd == "inference"))

    if args.cmd == "param-learning":
        from benchmarking.core.yaml_config import load_runner_config
        from benchmarking.measurements import ParamLearningMeasurement

        # The param-learning command constructs and injects the PL measurement
        # STRUCTURALLY and un-bypassably (#109): load_runner_config uses this
        # override regardless of the config's `metrics` field, which it instead
        # validates must equal "log_likelihood". The metrics field cannot swap
        # command behavior — inference never passes an override.
        device = args.device
        jsonl_path, resume = _resolve_results_dir(args)
        cfg = load_runner_config(
            args.config,
            device_override=device,
            measurement_override=ParamLearningMeasurement(),
            jsonl_path=jsonl_path,
            resume=resume,
        )
        return _execute_run(cfg, what="param-learning")

    if args.cmd == "plot":
        from pathlib import Path

        from benchmarking._paper_figures import run_plot
        return run_plot(
            parquet=[Path(p) for p in args.parquet],
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
        jsonl_path, resume = _resolve_results_dir(args)
        cfg = load_runner_config(
            args.config, device_override=device,
            jsonl_path=jsonl_path, resume=resume,
        )
        return _execute_run(cfg)

    raise AssertionError(f"unhandled subcommand {args.cmd!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
