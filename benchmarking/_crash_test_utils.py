"""Shared utilities for the v0.4 crash-test scripts.

This module is consumed by ``benchmarking.crash_test_runner`` (and via
``nbn-bench {param-learning, inference}``).  It contains pieces that
both crash tests need:

* ``CrashTestConfig`` — YAML-backed schema (smoke vs full).
* ``cell_iterator`` — yields ``(family, n_nodes, seed)`` cells with
  per-cell timeout + OOM guards.
* ``write_parquet`` / ``read_parquet`` — long-form metric dataframes.
* ``plot_metric_vs_n_nodes`` — the 4-panel figure-renderer used by both
  crash tests.
* ``make_fitter_mechanism`` — family-aware mechanism dispatch shared
  with the synthetic generator.
* ``reproducibility_footer`` — the "NBN v0.4 · seed=… · git=… · torch=…
  · cpu" footer rendered at every figure's bottom-right corner.

Hard rule: the figures committed in PR-2 are **smoke**; full-sweep
figures are reproduced locally via ``nbn-bench``.  Smoke figure
filenames carry an ``_smoke`` suffix to disambiguate from full-sweep
figures committed by the user before merge.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import yaml

logger = logging.getLogger(__name__)

# 4 families × 4 panels.  Used to lay out the page-1 figures.
PANEL_FAMILIES: tuple[str, ...] = (
    "discrete", "continuous_lg", "continuous_nongauss", "hybrid",
)

DEFAULT_PER_CELL_TIMEOUT_S = 30  # smoke soft cap (per pre-Phase-D refinement)


# ---------------------------------------------------------------------- #
# Config schema
# ---------------------------------------------------------------------- #


@dataclass
class CrashTestConfig:
    """v0.5 flat YAML schema.

    Fields match ``benchmarking/configs/{parameter_learning,inference}_{smoke,paper}.yaml``
    exactly.  Generator parameters are top-level (not nested under
    ``generator:``); workload parameters are top-level (not nested under
    ``inference:``).
    """

    mode: str                            # 'parameter_learning' | 'inference'
    families: List[str]
    n_nodes: List[int]
    n_seeds: int
    n_queries_per_cell: int
    n_train: int
    n_test: int                          # parameter_learning only; inference may omit
    n_reference: int
    edge_density: float
    max_in_degree: int
    cardinality: int
    fraction_continuous: float
    baselines: List[str]
    per_cell_timeout_s: int
    output_dir: str
    output_prefix: str

    # Parameter-learning specific
    fit_epochs: int = 50
    batch_size: int = 1024

    # Inference specific
    nbn_batch_size: int = 0              # 0 means 'use n_queries_per_cell'

    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> CrashTestConfig:
        text = Path(path).read_text()
        d = yaml.safe_load(text)
        return cls(
            mode=str(d["mode"]),
            families=list(d["families"]),
            n_nodes=list(d["n_nodes"]),
            n_seeds=int(d["n_seeds"]),
            n_queries_per_cell=int(d["n_queries_per_cell"]),
            n_train=int(d["n_train"]),
            n_test=int(d.get("n_test", 500)),
            n_reference=int(d["n_reference"]),
            edge_density=float(d["edge_density"]),
            max_in_degree=int(d["max_in_degree"]),
            cardinality=int(d["cardinality"]),
            fraction_continuous=float(d["fraction_continuous"]),
            baselines=list(d["baselines"]),
            per_cell_timeout_s=int(d["per_cell_timeout_s"]),
            output_dir=str(d["output_dir"]),
            output_prefix=str(d["output_prefix"]),
            fit_epochs=int(d.get("fit_epochs", 50)),
            batch_size=int(d.get("batch_size", 1024)),
            nbn_batch_size=int(d.get("nbn_batch_size", 0)),
        )

    @property
    def is_smoke(self) -> bool:
        return self.output_prefix.endswith("_smoke")

    @property
    def seeds(self) -> List[int]:
        """Seeds list derived from ``n_seeds`` (for parquet 'seed' column)."""
        return list(range(self.n_seeds))

    def figure_path(self, name: str, ext: str = "png") -> Path:
        """Canonical figure output: ``{output_dir}/{prefix}_{name}.{ext}``."""
        out = Path(self.output_dir) / f"{self.output_prefix}_{name}.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def parquet_path(self) -> Path:
        out = Path(self.output_dir) / f"{self.output_prefix}_metrics.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out


# ---------------------------------------------------------------------- #
# Cell iteration with timeout + OOM guards
# ---------------------------------------------------------------------- #


@dataclass
class CellResult:
    """One row of the long-form crash-test dataframe."""

    family: str
    n_nodes: int
    seed: int
    baseline: str
    metric: str
    value: float
    status: str = "ok"          # ok | timeout | oom | skipped | not_supported
    n_skipped: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class CellTimeout(Exception):
    """Raised when a cell exceeds its soft per-cell timeout."""


def _signal_alarm(seconds: int):
    """Install a SIGALRM-based timeout (POSIX only).  No-op on Windows."""
    @contextmanager
    def _cm():
        if sys.platform == "win32" or seconds <= 0:
            yield
            return

        def _handler(signum, frame):  # noqa: ARG001
            raise CellTimeout(f"cell exceeded {seconds}s soft cap")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    return _cm()


def cell_iterator(
    cfg: CrashTestConfig, *, baselines: Sequence[str],
) -> Iterable[Tuple[str, int, int, str]]:
    """Yield every ``(family, n_nodes, seed, baseline)`` to evaluate."""
    for family in cfg.families:
        for n in cfg.sizes:
            for s in cfg.seeds:
                for b in baselines:
                    yield (family, n, s, b)


def run_with_guard(
    fn: Callable[[], List[CellResult]],
    *,
    family: str,
    n_nodes: int,
    seed: int,
    baseline: str,
    timeout_s: int,
) -> List[CellResult]:
    """Run ``fn`` with timeout / OOM / generic-exception guards.

    Returns whatever rows ``fn`` produced if successful.  On timeout/OOM/
    error, returns a single status row so the figure can render the cell
    as a triangle marker rather than silently dropping it.
    """
    try:
        with _signal_alarm(timeout_s):
            return fn()
    except CellTimeout:
        logger.warning(
            "[%s n=%d s=%d %s] timeout @ %ds",
            family, n_nodes, seed, baseline, timeout_s,
        )
        return [_status_row(family, n_nodes, seed, baseline, "timeout")]
    except (RuntimeError, MemoryError) as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "alloc" in msg or "cuda oom" in msg:
            logger.warning(
                "[%s n=%d s=%d %s] OOM: %s",
                family, n_nodes, seed, baseline, exc,
            )
            return [_status_row(family, n_nodes, seed, baseline, "oom")]
        logger.exception(
            "[%s n=%d s=%d %s] runtime error", family, n_nodes, seed, baseline,
        )
        return [_status_row(family, n_nodes, seed, baseline, "error")]
    except NotImplementedError as exc:
        logger.info(
            "[%s n=%d s=%d %s] not supported: %s",
            family, n_nodes, seed, baseline, exc,
        )
        return [_status_row(family, n_nodes, seed, baseline, "not_supported")]
    except Exception:  # pragma: no cover  (final safety net)
        logger.exception(
            "[%s n=%d s=%d %s] unexpected error", family, n_nodes, seed, baseline,
        )
        return [_status_row(family, n_nodes, seed, baseline, "error")]


def _status_row(family, n_nodes, seed, baseline, status):
    return CellResult(
        family=family, n_nodes=n_nodes, seed=seed, baseline=baseline,
        metric="status", value=float("nan"), status=status,
    )


# ---------------------------------------------------------------------- #
# Parquet I/O
# ---------------------------------------------------------------------- #


def write_parquet(rows: List[CellResult], path: Path) -> None:
    import pandas as pd
    if not rows:
        warnings.warn(f"no rows to write to {path}", stacklevel=2)
        return
    df = pd.DataFrame([
        {
            "family": r.family,
            "n_nodes": r.n_nodes,
            "seed": r.seed,
            "baseline": r.baseline,
            "metric": r.metric,
            "value": r.value,
            "status": r.status,
            "n_skipped": r.n_skipped,
            **r.extra,
        }
        for r in rows
    ])
    df.to_parquet(path, index=False)
    logger.info("wrote %d rows to %s", len(rows), path)


def read_parquet(path: Path):
    import pandas as pd
    return pd.read_parquet(path)


# ---------------------------------------------------------------------- #
# Reproducibility footer
# ---------------------------------------------------------------------- #


def reproducibility_footer(
    *, version: str = "v0.4", seed: int = 0, device: str | torch.device = "cpu",
) -> str:
    sha = _git_short_sha()
    torch_v = torch.__version__.split("+")[0]
    return (
        f"NBN {version} · seed={seed} · git={sha} · torch={torch_v} · {device}"
    )


def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2.0, check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        pass
    return "unknown"


# ---------------------------------------------------------------------- #
# Mechanism dispatch — shared with the synthetic generator's hybrid path
# ---------------------------------------------------------------------- #


def fresh_mechanism_for(
    family: str, kind: str, *, num_components: int = 3,
):
    """Construct a *fresh* (un-fitted) mechanism for the fitter side.

    The synthetic generator hand-sets ground-truth parameters; this
    helper instead returns a brand-new mechanism that the fitter will
    populate via ``fit_local`` from training data.

    Hybrid family keeps continuous→discrete edges through a binner that
    is *fitted from train_data quantiles*, not the generator's truth
    (option (b) per the v0.4b refinement).
    """
    from nbn.mechanisms.categorical_table import CategoricalTableMechanism
    from nbn.mechanisms.linear_gaussian import LinearGaussianMechanism
    from nbn.mechanisms.mdn import MDNMechanism

    if kind == "discrete":
        return CategoricalTableMechanism(alpha=0.0)
    # Continuous
    if family == "continuous_lg" or (family == "hybrid" and kind == "continuous"):
        return LinearGaussianMechanism(ridge=1e-4, learnable=True)
    if family == "continuous_nongauss":
        return MDNMechanism(num_components=num_components, hidden=(32,))
    raise ValueError(f"no fresh mechanism for (family={family}, kind={kind})")


# ---------------------------------------------------------------------- #
# Figure renderer
# ---------------------------------------------------------------------- #


def plot_metric_vs_n_nodes(
    df, *, metric: str, ax_grid, fig,
    metric_label: str, lower_is_better: bool = True,
    log_y: bool = True, log_x: bool = True,
) -> None:
    """4-panel figure: one panel per family in ``PANEL_FAMILIES``.

    Each panel: x = n_nodes (log scale), y = ``metric`` (log scale by
    default).  One line per baseline with mean and ±1 std band over
    seeds, distinguished by both colour and marker shape so the figure
    survives greyscale printing.
    """
    import matplotlib.pyplot as plt
    palette = plt.get_cmap("tab10").colors
    markers = ["o", "s", "D", "^", "v", "P", "X"]

    for ax, family in zip(ax_grid, PANEL_FAMILIES, strict=True):
        sub = df[(df["family"] == family) & (df["metric"] == metric)
                 & (df["status"] == "ok")]
        if sub.empty:
            ax.set_title(f"{family} (no data)")
            ax.text(
                0.5, 0.5, "smoke skipped — see full config",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="gray",
            )
            continue
        baselines = sorted(sub["baseline"].unique())
        for i, b in enumerate(baselines):
            bsub = sub[sub["baseline"] == b]
            agg = bsub.groupby("n_nodes")["value"].agg(["mean", "std"]).reset_index()
            ax.plot(
                agg["n_nodes"], agg["mean"],
                marker=markers[i % len(markers)], color=palette[i % len(palette)],
                label=b, linewidth=1.5,
            )
            if (agg["std"].fillna(0.0) > 0).any():
                ax.fill_between(
                    agg["n_nodes"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                    color=palette[i % len(palette)], alpha=0.18,
                )
        ax.set_title(family)
        ax.set_xlabel("n_nodes")
        ax.set_ylabel(metric_label)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(fontsize=7)

    direction = "lower is better" if lower_is_better else "higher is better"
    fig.suptitle(metric_label + f"  ({direction})", fontsize=11)
