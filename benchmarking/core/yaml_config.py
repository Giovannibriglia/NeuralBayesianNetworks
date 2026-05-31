"""YAML → RunnerConfig loader for v0.13.

Parses the v0.13 native YAML schema (``version: "v0.13"`` required) into
a ``RunnerConfig`` ready for ``Runner.run()``.

Public surface
--------------
``load_runner_config(path, *, device_override=None) -> RunnerConfig``

Schema reference: docs/v0.13-benchmark-redesign.md §1c

Design decisions
----------------
* Hard-coded dispatch tables for ``benchmark`` and ``metrics`` fields.
  There are currently one benchmark type ("synthetic") and two metric
  modes ("all" / "timing"); a registry would be premature.
* ``config_name`` is an explicit required field in the YAML — no
  filename-derived fallback per the v0.13 "no legacy" stance.
* ``seeds`` is an explicit list (e.g. ``seeds: [0, 1, 2, 3, 4]``);
  ``n_seeds: int`` is not supported.
* ``--device`` CLI override sets the default device for any baseline
  whose ``device:`` key is absent/null in YAML.  Baselines with an
  explicit ``device:`` are not overridden.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarking.core.config import BaselineSpec, RunnerConfig
    from benchmarking.problems import SyntheticConfig

_VERSION = "v0.13"

_TOP_LEVEL_KEYS = frozenset({
    "version",
    "benchmark",
    "config_name",
    "metrics",
    "selector",
    "source",
    "baselines",
    "n_queries_per_cell",
    "per_cell_timeout_s",
    "fit_timeout_s_multiplier",
    "fit_timeout_s",
})

_BASELINE_REQUIRED = frozenset({"library", "mechanism", "param_method"})
_BASELINE_KNOWN = frozenset({
    "library", "mechanism", "param_method",
    "inference_method", "device", "extra_kwargs",
})


def load_runner_config(
    path: str | Path,
    *,
    device_override: str | None = None,
    jsonl_path: Path | None = None,
) -> RunnerConfig:
    """Parse a v0.13 YAML config file into a ``RunnerConfig``.

    Parameters
    ----------
    path:
        Path to the ``.yaml`` config file.
    device_override:
        When provided (e.g. from ``--device cpu``), sets the device
        for any baseline whose ``device:`` key is absent or null.
        Baselines with an explicit ``device:`` in YAML are not
        overridden.
    jsonl_path:
        Override the auto-generated output path.  Useful in tests to
        avoid creating output directories as a side effect.

    Raises
    ------
    ValueError
        If ``version`` is missing or not ``"v0.13"``, a required
        field is absent, an unknown field is present, or a dispatch
        target is unrecognised.
    """
    import yaml

    from benchmarking.core.config import RunnerConfig
    from benchmarking.measurements import AccuracyAndTiming, TimingOnly
    from benchmarking.problems import SyntheticProblemSource
    from benchmarking.selectors import UniformRandomSelector

    text = Path(path).read_text()
    d: dict[str, Any] = yaml.safe_load(text)

    # ── version check ────────────────────────────────────────────────────────
    version = d.get("version")
    if version != _VERSION:
        raise ValueError(
            f"Config {str(path)!r} has version={version!r}; "
            f"this runner requires version: {_VERSION!r}. "
            f"See docs/v0.13-benchmark-redesign.md §1c for the new schema."
        )

    # ── unknown top-level keys ───────────────────────────────────────────────
    unknown = set(d.keys()) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"Config {str(path)!r} has unknown top-level fields: "
            f"{sorted(unknown)}. "
            f"Known fields: {sorted(_TOP_LEVEL_KEYS)}."
        )

    # ── required top-level fields ────────────────────────────────────────────
    for req in ("benchmark", "config_name", "source", "baselines",
                "n_queries_per_cell", "per_cell_timeout_s"):
        if req not in d:
            raise ValueError(
                f"Config {str(path)!r} missing required field: {req!r}."
            )

    # ── benchmark dispatch ───────────────────────────────────────────────────
    benchmark = str(d["benchmark"])
    if benchmark == "synthetic":
        problem_source = SyntheticProblemSource()
        source_config = _parse_synthetic_config(d["source"], path)
    else:
        raise ValueError(
            f"Config {str(path)!r}: unknown benchmark={benchmark!r}. "
            f"Supported: 'synthetic'."
        )

    # ── metrics dispatch ─────────────────────────────────────────────────────
    metrics = d.get("metrics", "all")
    if metrics == "all":
        measurement = AccuracyAndTiming()
    elif metrics == "timing":
        measurement = TimingOnly()
    else:
        raise ValueError(
            f"Config {str(path)!r}: unknown metrics={metrics!r}. "
            f"Supported: 'all' (AccuracyAndTiming), 'timing' (TimingOnly)."
        )

    # ── selector dispatch ────────────────────────────────────────────────────
    selector_name = d.get("selector", "uniform_random")
    if selector_name == "uniform_random":
        selector = UniformRandomSelector()
    else:
        raise ValueError(
            f"Config {str(path)!r}: unknown selector={selector_name!r}. "
            f"Supported: 'uniform_random'."
        )

    # ── baselines ────────────────────────────────────────────────────────────
    raw_baselines = d["baselines"]
    if not isinstance(raw_baselines, list) or not raw_baselines:
        raise ValueError(
            f"Config {str(path)!r}: 'baselines' must be a non-empty list."
        )
    baselines = [
        _parse_baseline_spec(b, i, device_override, path)
        for i, b in enumerate(raw_baselines)
    ]

    # ── iteration parameters ─────────────────────────────────────────────────
    n_queries_per_cell = int(d["n_queries_per_cell"])
    per_cell_timeout_s = float(d["per_cell_timeout_s"])
    fit_timeout_s_multiplier = float(d.get("fit_timeout_s_multiplier", 10.0))
    fit_timeout_s = d.get("fit_timeout_s")
    if fit_timeout_s is not None:
        fit_timeout_s = float(fit_timeout_s)

    if jsonl_path is None:
        from benchmarking.core.output import make_results_dir
        jsonl_path = make_results_dir(benchmark, str(d["config_name"])) / "metrics.jsonl"

    return RunnerConfig(
        benchmark=benchmark,
        config_name=str(d["config_name"]),
        problem_source=problem_source,
        source_config=source_config,
        selector=selector,
        measurement=measurement,
        baselines=baselines,
        n_queries_per_cell=n_queries_per_cell,
        per_cell_timeout_s=per_cell_timeout_s,
        fit_timeout_s_multiplier=fit_timeout_s_multiplier,
        fit_timeout_s=fit_timeout_s,
        jsonl_path=jsonl_path,
    )


# ---------------------------------------------------------------------------
# Sub-parsers
# ---------------------------------------------------------------------------

_SYNTHETIC_SOURCE_KEYS = frozenset({
    "families", "n_nodes_list", "seeds",
    "n_train", "n_test", "n_reference",
    "edge_density", "max_in_degree", "cardinality",
    "fraction_continuous", "device",
})
_SYNTHETIC_SOURCE_REQUIRED = frozenset({
    "families", "n_nodes_list", "seeds",
    "n_train", "n_reference",
    "edge_density", "max_in_degree", "cardinality",
    "fraction_continuous",
})


def _parse_synthetic_config(src: Any, path: Any) -> SyntheticConfig:
    from benchmarking.problems import SyntheticConfig

    if not isinstance(src, dict):
        raise ValueError(
            f"Config {str(path)!r}: 'source' must be a mapping, "
            f"got {type(src).__name__}."
        )
    unknown = set(src.keys()) - _SYNTHETIC_SOURCE_KEYS
    if unknown:
        raise ValueError(
            f"Config {str(path)!r}: unknown source fields: "
            f"{sorted(unknown)}. Known: {sorted(_SYNTHETIC_SOURCE_KEYS)}."
        )
    missing = _SYNTHETIC_SOURCE_REQUIRED - set(src.keys())
    if missing:
        raise ValueError(
            f"Config {str(path)!r}: source missing required fields: "
            f"{sorted(missing)}."
        )
    return SyntheticConfig(
        families=list(src["families"]),
        n_nodes_list=[int(n) for n in src["n_nodes_list"]],
        seeds=[int(s) for s in src["seeds"]],
        n_train=int(src["n_train"]),
        n_test=int(src.get("n_test", 2000)),
        n_reference=int(src["n_reference"]),
        edge_density=float(src["edge_density"]),
        max_in_degree=int(src["max_in_degree"]),
        cardinality=int(src["cardinality"]),
        fraction_continuous=float(src["fraction_continuous"]),
        device=str(src.get("device", "cpu")),
    )


def _parse_baseline_spec(
    b: Any,
    idx: int,
    device_override: str | None,
    path: Any,
) -> BaselineSpec:
    from benchmarking.core.config import BaselineSpec

    if not isinstance(b, dict):
        raise ValueError(
            f"Config {str(path)!r}: baselines[{idx}] must be a dict, "
            f"got {type(b).__name__}."
        )
    unknown = set(b.keys()) - _BASELINE_KNOWN
    if unknown:
        raise ValueError(
            f"Config {str(path)!r}: baselines[{idx}] has unknown fields: "
            f"{sorted(unknown)}. Known: {sorted(_BASELINE_KNOWN)}."
        )
    missing = _BASELINE_REQUIRED - set(b.keys())
    if missing:
        raise ValueError(
            f"Config {str(path)!r}: baselines[{idx}] missing required "
            f"fields: {sorted(missing)}."
        )
    # device: per-baseline YAML value takes priority; fall back to override
    raw_device = b.get("device")
    device = str(raw_device) if raw_device is not None else device_override

    return BaselineSpec(
        library=str(b["library"]),
        mechanism=str(b["mechanism"]),
        param_method=str(b["param_method"]),
        inference_method=(
            str(b["inference_method"])
            if b.get("inference_method") is not None else None
        ),
        device=device,
        extra_kwargs=dict(b.get("extra_kwargs") or {}),
    )
