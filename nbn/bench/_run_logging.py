"""Structured logging for ``nbn-bench`` runs (v0.6c-A).

For every parameter-learning / inference run, write two artefacts
into ``{output_dir}/raw/`` alongside the metrics parquet so future
diagnostics have full execution context:

* ``{output_dir}/raw/{output_prefix}_{timestamp}.log`` — INFO+ logs
  duplicated from the existing console handler.
* ``{output_dir}/raw/{output_prefix}_{timestamp}.run.json`` —
  metadata dict (git SHA, torch version, cuda device, full config,
  wall time, status summary).

The ``raw/`` subdirectory was introduced in v0.6c-B.  Both files are
gitignored (``*.log`` is already in the repo's root ``.gitignore``;
``*.run.json`` was added in v0.6c-A; ``raw/.gitignore`` lists both
explicitly).  Best-effort: any failure inside this module logs a
warning and lets the run proceed.  We never block a smoke / paper
run on a logging issue.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def setup_run_logging(cfg) -> Dict[str, Any]:
    """Set up file logging + capture run metadata.  Returns the
    metadata dict; pass it to ``finalise_run_logging`` after the run
    completes.

    Never raises: if anything fails (filesystem, git, etc.) we log a
    warning and return a minimal metadata dict so the caller can
    continue.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    meta: Dict[str, Any] = {
        "timestamp": timestamp,
        "started_at": time.time(),
    }

    try:
        # v0.6c-B: logs + run.json land in ``output_dir/raw/`` alongside
        # the metrics parquet so all per-run artefacts are in one place.
        # ``output_dir/figures/`` is for plotter outputs; the layout is
        # enforced in code, not via per-subdir YAML fields.
        out = Path(cfg.output_dir) / "raw"
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / f"{cfg.output_prefix}_{timestamp}.log"
        meta_path = out / f"{cfg.output_prefix}_{timestamp}.run.json"

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"),
        )
        logging.getLogger().addHandler(fh)
        meta["log_path"] = str(log_path)
        meta["meta_path"] = str(meta_path)
        meta["_log_handler_id"] = id(fh)
    except Exception as exc:  # pragma: no cover  (filesystem error)
        logger.warning(
            "_run_logging: could not create file logs (%s); run continues.",
            exc,
        )

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = "unknown"
    meta["git_sha"] = git_sha

    try:
        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
        meta["cuda_device"] = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None
        )
    except Exception:  # pragma: no cover  (torch.cuda probe failed)
        meta["torch_version"] = "unknown"
        meta["cuda_available"] = False
        meta["cuda_device"] = None

    try:
        meta["config"] = dataclasses.asdict(cfg)
    except Exception:
        meta["config"] = repr(cfg)

    return meta


def finalise_run_logging(
    meta: Dict[str, Any],
    status_summary: Dict[str, int] | None = None,
) -> None:
    """Write the metadata + summary to ``.run.json`` after the run
    completes.  Never raises."""
    meta = dict(meta)  # shallow copy so we don't mutate caller state
    meta["ended_at"] = time.time()
    meta["wall_time_s"] = meta["ended_at"] - meta.get("started_at", meta["ended_at"])
    if status_summary is not None:
        meta["status_summary"] = status_summary

    # Drop the file-handler reference before serialising (it was only
    # there to identify the handler we installed).
    meta.pop("_log_handler_id", None)

    if "meta_path" in meta:
        try:
            Path(meta["meta_path"]).write_text(
                json.dumps(meta, indent=2, default=str),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "_run_logging: could not write run.json (%s).", exc,
            )
