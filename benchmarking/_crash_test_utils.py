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

import pandas as pd
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
    """v0.6c-C-1a (schema v2) flat YAML schema.

    v0.6c-C-1a introduces ``config_schema_version: 2`` and changes
    ``baselines`` from a flat list of strings into a list of structured
    spec dicts ``{library, mechanism, param_method[, inference_method]}``.
    The runner derives canonical labels via
    :func:`benchmarking._baseline_registry._label_from_spec`.

    Schema-v1 configs (flat ``baselines: [nbn_lw, nbn_ve, pgmpy, ...]``)
    are rejected by ``from_yaml`` with a clear migration message.
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
    baselines: List[Dict[str, Any]]      # schema v2: list of spec dicts
    per_cell_timeout_s: int
    output_dir: str
    output_prefix: str

    # Parameter-learning specific
    fit_epochs: int = 50
    batch_size: int = 1024

    # Inference specific
    nbn_batch_size: int = 0              # 0 means 'use n_queries_per_cell'
    nbn_lw_n_samples: int = 512

    device: str = "auto"

    runtime_skip_after_timeout: bool = False

    # Write a JSONL sidecar alongside the parquet so completed cells are
    # durable on crash.  True by default — opt-out via YAML if needed.
    jsonl_sidecar: bool = True

    _SCHEMA_VERSION: int = 2

    _REQUIRED_YAML_FIELDS = frozenset({
        "config_schema_version",         # v0.6c-C-1a: explicit version
        "mode", "families", "n_nodes", "n_seeds", "n_queries_per_cell",
        "n_train", "n_reference", "edge_density", "max_in_degree",
        "cardinality", "fraction_continuous", "baselines",
        "per_cell_timeout_s", "output_dir", "output_prefix",
    })
    _REQUIRED_INFERENCE_FIELDS = frozenset({
        "nbn_lw_n_samples",
    })

    @staticmethod
    def _validate_baseline_spec(b: Any, mode: str) -> Dict[str, Any]:
        """Verify a single ``baselines:`` list entry has the right shape.

        Raises ``ValueError`` on missing/unknown keys.  Returns a clean
        ``dict`` (not the raw YAML node) for downstream consumers.
        """
        if not isinstance(b, dict):
            raise ValueError(
                f"baseline entry {b!r} is not a dict; v0.6c-C-1a schema "
                f"requires {{library, mechanism, param_method"
                f"[, inference_method]}}.  Schema v1 (flat baseline "
                f"list of strings) was retired; see "
                f"benchmarking/configs/inference_smoke.yaml.",
            )
        required = {"library", "mechanism", "param_method"}
        if mode == "inference":
            required |= {"inference_method"}
        missing = required - set(b.keys())
        if missing:
            raise ValueError(
                f"baseline entry {b!r} missing fields: {sorted(missing)}",
            )
        return {
            "library": str(b["library"]),
            "mechanism": str(b["mechanism"]),
            "param_method": str(b["param_method"]),
            "inference_method": (
                str(b["inference_method"])
                if b.get("inference_method") is not None else None
            ),
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> CrashTestConfig:
        text = Path(path).read_text()
        d = yaml.safe_load(text)

        # v0.6c-C-1a: enforce schema v2.  Schema v1 had a flat baseline
        # list of strings; v2 has structured spec dicts.  No silent
        # backward-compat path — v1 configs raise with a migration hint.
        version = d.get("config_schema_version")
        if version != cls._SCHEMA_VERSION:
            raise ValueError(
                f"Config {str(path)!r} has config_schema_version="
                f"{version!r}; v0.6c-C-1a+ requires version "
                f"{cls._SCHEMA_VERSION}.  See "
                f"benchmarking/configs/inference_smoke.yaml for the "
                f"new format: 'baselines' is now a list of spec dicts "
                f"like {{library: nbn, mechanism: cat, param_method: "
                f"mle, inference_method: ve}}.",
            )

        required = set(cls._REQUIRED_YAML_FIELDS)
        if d.get("mode") == "inference":
            required |= cls._REQUIRED_INFERENCE_FIELDS
        missing = required - set(d.keys())
        if missing:
            raise ValueError(
                f"Config {str(path)!r} missing required fields: "
                f"{sorted(missing)}.  As of v0.6c-A round 2,"
                f" nbn_lw_n_samples must be set explicitly on inference"
                f" configs to prevent the silent-defaulting class of bug"
                f" that produced wrong continuous accuracy on paper"
                f" config (W₁ ≈ 1.0 vs smoke's 0.04).  Smoke uses 4000;"
                f" paper uses 4000+.",
            )
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
            baselines=[
                cls._validate_baseline_spec(b, str(d.get("mode", "")))
                for b in d["baselines"]
            ],
            per_cell_timeout_s=int(d["per_cell_timeout_s"]),
            output_dir=str(d["output_dir"]),
            output_prefix=str(d["output_prefix"]),
            fit_epochs=int(d.get("fit_epochs", 50)),
            batch_size=int(d.get("batch_size", 1024)),
            nbn_batch_size=int(d.get("nbn_batch_size", 0)),
            nbn_lw_n_samples=int(d.get("nbn_lw_n_samples", 512)),
            runtime_skip_after_timeout=bool(
                d.get("runtime_skip_after_timeout", False)
            ),
            jsonl_sidecar=bool(d.get("jsonl_sidecar", True)),
        )

    @property
    def is_smoke(self) -> bool:
        return self.output_prefix.endswith("_smoke")

    @property
    def seeds(self) -> List[int]:
        """Seeds list derived from ``n_seeds`` (for parquet 'seed' column)."""
        return list(range(self.n_seeds))

    def figure_path(self, name: str, ext: str = "png") -> Path:
        """Canonical figure output: ``{output_dir}/figures/{prefix}_{name}.{ext}``.

        v0.6c-B: figures live in ``output_dir/figures/``; raw parquets +
        logs + run.json live in ``output_dir/raw/``; tables (placeholder
        for v0.6c-C) live in ``output_dir/tables/``.  The single
        ``output_dir`` field in YAML names the parent; this method and
        :meth:`parquet_path` and :mod:`benchmarking._run_logging`
        enforce the subdirectory split internally.
        """
        out = Path(self.output_dir) / "figures" / f"{self.output_prefix}_{name}.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def parquet_path(self) -> Path:
        """Canonical parquet output: ``{output_dir}/raw/{prefix}_metrics.parquet``."""
        out = Path(self.output_dir) / "raw" / f"{self.output_prefix}_metrics.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def jsonl_path(self) -> Path:
        """Sidecar JSONL output: ``{output_dir}/raw/{prefix}_metrics.jsonl``."""
        out = Path(self.output_dir) / "raw" / f"{self.output_prefix}_metrics.jsonl"
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


class JSONLSidecarWriter:
    """Append-mode JSONL writer that flushes after every row.

    Each line is a flat JSON object matching the schema that
    ``write_parquet`` produces: the eight core ``CellResult`` fields
    plus any ``extra`` dict keys splatted at the top level.  ``NaN``
    values are serialised as JSON ``null`` so the file is valid JSON.

    Usage::

        writer = JSONLSidecarWriter(cfg.jsonl_path())
        try:
            for row in cell_rows:
                writer.append(row)
        finally:
            writer.close()
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a")  # noqa: SIM115 - long-lived handle by design; close() called in finally

    def append(self, row: CellResult) -> None:
        import json
        import math

        d: Dict[str, Any] = {
            "family": row.family,
            "n_nodes": row.n_nodes,
            "seed": row.seed,
            "baseline": row.baseline,
            "metric": row.metric,
            # NaN is not valid JSON; use null so the file is parseable
            # by any JSON consumer.  pd.read_json re-hydrates null → NaN.
            "value": None if (isinstance(row.value, float) and math.isnan(row.value)) else row.value,
            "status": row.status,
            "n_skipped": row.n_skipped,
            **row.extra,
        }
        self._fh.write(json.dumps(d) + "\n")
        self._fh.flush()  # durable on every append — survives a crash

    def close(self) -> None:
        self._fh.close()


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


# PR-B §A.2: structural-limit markers — ValueErrors with these substrings
# come from adapters refusing combinations they don't support and should
# be classified as ``not_supported`` rather than ``error``.
_STRUCTURAL_LIMIT_MARKERS = (
    "only supports",
    "is discrete-only",
    "cannot condition on",
    "not yet wired",
    "non-Gaussian",
    "not applicable to",
    "refused",
)


def _is_structural_limit(exc: BaseException) -> bool:
    if isinstance(exc, NotImplementedError):
        return True
    if isinstance(exc, ValueError):
        msg = str(exc)
        return any(marker in msg for marker in _STRUCTURAL_LIMIT_MARKERS)
    return False


def run_with_guard(
    fn: Callable[[], List[CellResult]],
    *,
    family: str,
    n_nodes: int,
    seed: int,
    baseline: str,
    timeout_s: int,
) -> List[CellResult]:
    """Run ``fn`` with timeout / OOM / not-supported / error guards.

    Returns whatever rows ``fn`` produced if successful.  On a known
    failure mode (timeout / OOM / structural limit), returns a single
    status row so the figure can render the cell as a DNF marker
    rather than silently dropping it.
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
    except torch.cuda.OutOfMemoryError as exc:
        logger.warning(
            "[%s n=%d s=%d %s] cuda OOM: %s",
            family, n_nodes, seed, baseline, exc,
        )
        return [_status_row(family, n_nodes, seed, baseline, "oom",
                            error_msg=str(exc))]
    except MemoryError as exc:
        logger.warning(
            "[%s n=%d s=%d %s] OOM: %s",
            family, n_nodes, seed, baseline, exc,
        )
        return [_status_row(family, n_nodes, seed, baseline, "oom",
                            error_msg=str(exc))]
    except ImportError as exc:
        # Optional dependency missing (gpytorch / pyro / pomegranate not
        # installed in this environment) → structural not_supported.
        logger.info(
            "[%s n=%d s=%d %s] not_supported (import): %s",
            family, n_nodes, seed, baseline, exc,
        )
        return [_status_row(family, n_nodes, seed, baseline, "not_supported",
                            error_msg=str(exc))]
    except (NotImplementedError, ValueError) as exc:
        if _is_structural_limit(exc):
            logger.info(
                "[%s n=%d s=%d %s] not_supported: %s",
                family, n_nodes, seed, baseline, exc,
            )
            return [_status_row(family, n_nodes, seed, baseline, "not_supported",
                                error_msg=str(exc))]
        # ValueError without a structural-limit marker is a real error.
        logger.exception(
            "[%s n=%d s=%d %s] value error",
            family, n_nodes, seed, baseline,
        )
        return [_status_row(family, n_nodes, seed, baseline, "error",
                            error_msg=str(exc))]
    except RuntimeError as exc:
        msg = str(exc).lower()
        # Catch torch's CPU allocator failures (which raise RuntimeError
        # rather than MemoryError) as well as cuda OOMs.
        if (
            "out of memory" in msg or "cuda oom" in msg
            or "alloc_cpu" in msg or "defaultcpuallocator" in msg
            or "cannot allocate memory" in msg or "can't allocate memory" in msg
        ):
            logger.warning(
                "[%s n=%d s=%d %s] OOM via RuntimeError: %s",
                family, n_nodes, seed, baseline, exc,
            )
            return [_status_row(family, n_nodes, seed, baseline, "oom",
                                error_msg=str(exc))]
        logger.exception(
            "[%s n=%d s=%d %s] runtime error",
            family, n_nodes, seed, baseline,
        )
        return [_status_row(family, n_nodes, seed, baseline, "error",
                            error_msg=str(exc))]
    except Exception as exc:  # pragma: no cover  (final safety net)
        logger.exception(
            "[%s n=%d s=%d %s] unexpected error",
            family, n_nodes, seed, baseline,
        )
        return [_status_row(family, n_nodes, seed, baseline, "error",
                            error_msg=str(exc))]


def _status_row(
    family: str, n_nodes: int, seed: int, baseline: str, status: str,
    *, error_msg: str = "",
) -> CellResult:
    return CellResult(
        family=family, n_nodes=n_nodes, seed=seed, baseline=baseline,
        metric="status", value=float("nan"), status=status,
        extra={"error_msg": error_msg} if error_msg else {},
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


def jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> None:
    """Recover a parquet from a JSONL sidecar after a crash.

    Reads the JSONL file line-by-line and writes a parquet with the
    same schema as ``write_parquet``.  The resulting file is intended
    to be a drop-in replacement for the parquet the runner would have
    written on a clean exit.

    Column order and dtypes match ``write_parquet`` because both
    build a ``pd.DataFrame`` from a list of flat dicts with the same
    key insertion order (8 fixed fields first, then ``extra`` keys
    in first-occurrence order).
    """
    import json
    import pandas as pd

    rows = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        warnings.warn(f"no rows in {jsonl_path}; parquet not written", stacklevel=2)
        return
    df = pd.DataFrame(rows)
    # Enforce fixed-schema dtypes so the recovered parquet matches write_parquet
    # regardless of what pandas infers from the JSON values.  In particular,
    # a column where every value is null infers as object, not float64.
    for int_col in ("n_nodes", "seed", "n_skipped"):
        if int_col in df.columns:
            df[int_col] = df[int_col].astype("int64")
    if "value" in df.columns:
        df["value"] = df["value"].astype("float64")
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    logger.info("recovered %d rows from %s → %s", len(rows), jsonl_path, parquet_path)


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
    parent_kinds: list[tuple[str, int]] | None = None,
    cardinality: int = 4,
):
    """Construct a *fresh* (un-fitted) mechanism for the fitter side.

    The synthetic generator hand-sets ground-truth parameters; this
    helper instead returns a brand-new mechanism that the fitter will
    populate via ``fit_local`` from training data.

    Hybrid discrete-target nodes whose ``parent_kinds`` include any
    continuous parents get a :class:`BinningCategoricalTable` whose
    thresholds are learned from train_data quantiles (option (b) per
    the v0.4b refinement).  Pure-discrete-parent discrete targets and
    pure-continuous nodes get the standard mechanism.
    """
    from nbn.mechanisms.binning_categorical import BinningCategoricalTable
    from nbn.mechanisms.categorical_table import CategoricalTableMechanism
    from nbn.mechanisms.linear_gaussian import LinearGaussianMechanism
    from nbn.mechanisms.mdn import MDNMechanism

    if kind == "discrete":
        if (family == "hybrid" and parent_kinds
                and any(pk == "continuous" for (pk, _) in parent_kinds)):
            return BinningCategoricalTable(
                parent_kinds=parent_kinds,
                n_bins=cardinality,
                n_categories=cardinality,
            )
        return CategoricalTableMechanism(alpha=0.0)
    # Continuous
    if family == "continuous_lg" or (family == "hybrid" and kind == "continuous"):
        return LinearGaussianMechanism(ridge=1e-4, learnable=True)
    if family == "continuous_nongauss":
        return MDNMechanism(num_components=num_components, hidden=(32,))
    raise ValueError(f"no fresh mechanism for (family={family}, kind={kind})")


def fresh_mechanism_for_spec(
    spec, family: str, kind: str, *, num_components: int = 3,
    parent_kinds: list[tuple[str, int]] | None = None,
    cardinality: int = 4,
):
    """v0.8 spec-aware mechanism factory (audit v0.7-#43, runtime fix v0.8-#51).

    Replaces the family-keyed :func:`fresh_mechanism_for` for NBN baselines
    on the parameter-learning runner side.  The legacy function collapsed
    every NBN discrete baseline to ``CategoricalTableMechanism`` (so
    ``nbn-cat`` and ``nbn-neuralcat`` produced bit-identical fits despite
    distinct method-keyed labels in the parquet).  This version routes on
    ``spec.mechanism`` instead, so each method-keyed baseline gets the
    mechanism class its label promises.

    NBN-only — pgmpy / gpytorch / pomegranate / pyro have their own
    fitting code paths in the runner.  ``spec.mechanism == "hybrid"``
    falls back to the family-default :func:`fresh_mechanism_for` so the
    HybridRouter case keeps its per-node mechanism dispatch.

    The ``BinningCategoricalTable`` branch (hybrid-family discrete-target
    nodes with at least one continuous parent) is mechanism-independent:
    both ``nbn-cat`` and ``nbn-neuralcat`` resolve to the binner for
    cont→disc edges, so we keep that branch ahead of the mechanism
    dispatch.
    """
    from nbn.mechanisms.binning_categorical import BinningCategoricalTable
    from nbn.mechanisms.categorical_table import CategoricalTableMechanism
    from nbn.mechanisms.linear_gaussian import LinearGaussianMechanism
    from nbn.mechanisms.mdn import MDNMechanism
    from nbn.mechanisms.neural_categorical import NeuralCategoricalMechanism

    if spec.mechanism == "hybrid":
        return fresh_mechanism_for(
            family, kind,
            num_components=num_components,
            parent_kinds=parent_kinds,
            cardinality=cardinality,
        )

    if kind == "discrete":
        if (family == "hybrid" and parent_kinds
                and any(pk == "continuous" for (pk, _) in parent_kinds)):
            return BinningCategoricalTable(
                parent_kinds=parent_kinds,
                n_bins=cardinality,
                n_categories=cardinality,
            )
        if spec.mechanism == "neuralcat":
            return NeuralCategoricalMechanism(n_classes=cardinality)
        # Default for discrete: closed-form counting MLE.
        return CategoricalTableMechanism(alpha=0.0)

    # Continuous nodes — dispatch on spec.mechanism, fallback to
    # family-default for unspecified mechanisms.
    if spec.mechanism == "lg":
        return LinearGaussianMechanism(ridge=1e-4, learnable=True)
    if spec.mechanism == "mdn":
        return MDNMechanism(num_components=num_components, hidden=(32,))
    if spec.mechanism == "flow":
        from nbn.mechanisms.normalizing_flow import NormalizingFlowMechanism
        return NormalizingFlowMechanism()
    return fresh_mechanism_for(
        family, kind,
        num_components=num_components,
        parent_kinds=parent_kinds,
        cardinality=cardinality,
    )


# ---------------------------------------------------------------------- #
# Figure renderer
# ---------------------------------------------------------------------- #


# PR-B §B.1 — DNF marker shapes per status, distinct from data-point shapes.
_DNF_MARKERS = {
    "timeout": "v",         # downward triangle
    "oom": "^",             # upward triangle
    "no_result": "d",       # diamond
    "not_supported": "s",   # square (hollow rendered)
    "error": "x",           # cross
}
_DATA_MARKERS = ["o", "P", "*", "h", "X", "8"]


def plot_metric_vs_n_nodes(
    df, *, metric: str, ax_grid, fig,
    metric_label: str, lower_is_better: bool = True,
    log_y: bool = True, log_x: bool = True,
) -> None:
    """4-panel figure: one panel per family in ``PANEL_FAMILIES``.

    PR-B §B.1 cell-level filtering rules:
    * For each ``(baseline, n_nodes)`` cell, look up ``status`` from the
      parquet.  ``ok`` + finite ``value`` → plot a data point.
    * ``timeout`` / ``oom`` / ``no_result`` / ``not_supported`` /
      ``error`` → DNF marker at ``ymax * 1.1``, with shape distinguishing
      the kind.
    * Connect successive ``ok`` points with a line; do **not** interpolate
      across DNF cells.
    * A baseline that is ``not_supported`` for every n in a family is
      annotated in the legend with ``(not applicable)``.

    Lines + data markers also distinguish baselines for greyscale survival.
    """
    import matplotlib.pyplot as plt
    palette = plt.get_cmap("tab10").colors

    for ax, family in zip(ax_grid, PANEL_FAMILIES, strict=True):
        sub = df[(df["family"] == family) & (df["metric"] == metric)].copy()
        # Pull in status rows whose metric is 'status' (not the requested
        # metric) so cells that DNF-out before the metric was computed
        # still show as DNF triangles on the figure.
        status_rows = df[(df["family"] == family) & (df["metric"] == "status")]
        sub = pd.concat(
            [sub, status_rows.assign(value=float("nan"))], ignore_index=True,
        ) if not status_rows.empty else sub

        if sub.empty:
            ax.set_title(family)
            ax.text(
                0.5, 0.5, "no data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="gray",
            )
            ax.set_xscale("log" if log_x else "linear")
            continue

        baselines = sorted(sub["baseline"].unique())
        # Compute the panel's data-only ymax to position DNF markers AT
        # (PR-B-round-2 §5).  We place DNF markers just inside the data
        # range — never above it — so the panel's autoscaled ylim doesn't
        # leave a large empty band of plot area above the data lines.
        ok_vals = sub[(sub["status"] == "ok") & sub["value"].notna()]["value"]
        if not ok_vals.empty and (ok_vals > 0).all():
            ymax = float(ok_vals.max())
            ymin = float(ok_vals.min())
            # Sit just inside ymax (95%): prominent but doesn't extend axis.
            dnf_y = ymax * 0.95 if log_y else ymin + 0.95 * (ymax - ymin)
        else:
            dnf_y = 1.0

        for i, b in enumerate(baselines):
            colour = palette[i % len(palette)]
            data_marker = _DATA_MARKERS[i % len(_DATA_MARKERS)]
            bsub = sub[sub["baseline"] == b]
            ok_b = bsub[(bsub["status"] == "ok") & bsub["value"].notna()]
            dnf_b = bsub[(bsub["status"] != "ok") | bsub["value"].isna()]

            # All-not-supported annotation
            all_not_sup = (
                not ok_b.shape[0] and
                (dnf_b["status"] == "not_supported").all() and
                dnf_b.shape[0] > 0
            )
            label = f"{b} (not applicable)" if all_not_sup else b

            # Line + data markers for ok cells
            if not ok_b.empty:
                agg = ok_b.groupby("n_nodes")["value"].agg(["mean", "std"]).reset_index()
                ax.plot(
                    agg["n_nodes"], agg["mean"],
                    marker=data_marker, color=colour, label=label,
                    linewidth=1.5, markersize=6,
                )
                if (agg["std"].fillna(0.0) > 0).any():
                    ax.fill_between(
                        agg["n_nodes"],
                        (agg["mean"] - agg["std"]).clip(lower=1e-12) if log_y else agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        color=colour, alpha=0.18,
                    )
            elif all_not_sup:
                # Place a single legend-only entry off-figure so the
                # baseline appears in the legend with its annotation.
                ax.plot([], [], color=colour, marker=data_marker, label=label,
                        linewidth=1.5, markersize=6)

            # DNF markers — one per (n_nodes, status) cell
            for status_kind in dnf_b["status"].unique():
                cells = dnf_b[dnf_b["status"] == status_kind]
                xs = cells["n_nodes"].unique()
                if len(xs) == 0:
                    continue
                ys = [dnf_y] * len(xs)
                # PR-B-round-2 §5: distinguish filled vs hollow markers per
                # status without passing edgecolor='none' on unfilled
                # markers (which warns).
                if status_kind == "not_supported":
                    ax.scatter(xs, ys, marker=_DNF_MARKERS["not_supported"],
                               s=50, zorder=3,
                               facecolors="none", edgecolors=colour, linewidths=1.2)
                else:
                    ax.scatter(xs, ys, marker=_DNF_MARKERS.get(status_kind, "x"),
                               s=50, zorder=3, color=colour)

        ax.set_title(family)
        ax.set_xlabel("n_nodes")
        # PR-B-round-2 §2: single y-axis label per panel; concise so
        # constrained_layout doesn't word-wrap onto two clipped lines.
        ax.set_ylabel(metric_label)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        # PR-B-round-2: only call legend() when there are labelled artists,
        # avoiding the "No artists with labels found" UserWarning.
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="best")

    # NB: caller (the runner) sets ``fig.suptitle`` itself after this
    # returns; we don't set one here to avoid a double-suptitle clash.
