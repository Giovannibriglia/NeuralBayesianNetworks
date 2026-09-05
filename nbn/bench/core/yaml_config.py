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
* ALL-CAPS top-level keys (e.g. ``BATCH_SIZE_FIT: &BATCH_SIZE_FIT 1024``)
  are YAML-anchor constants: a single place to define a value that many
  baselines reference via ``*BATCH_SIZE_FIT``.  They are ignored by the
  unknown-key check and carry no runner semantics of their own.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nbn.bench.core.config import BaselineSpec, RunnerConfig
    from nbn.bench.problems import SyntheticConfig

_VERSION = "v0.13"
# v0.14 (#148): batched-queries configs. Same schema as v0.13 plus the
# optional top-level batch_sizes / n_batch_queries fields below.
_ACCEPTED_VERSIONS = frozenset({"v0.13", "v0.14"})

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
    "fit_timeout_s",
    # v0.14 batched queries (#148, design doc §1.7, §5.3-5.4)
    "batch_sizes",
    "n_batch_queries",
})


def _is_constant_key(key: str) -> bool:
    """True for ALL-CAPS top-level keys, which are YAML-anchor constants.

    Configs may hoist a value shared by many baselines into a single
    top-level definition, e.g.::

        BATCH_SIZE_FIT: &BATCH_SIZE_FIT 1024
        ...
        baselines:
          - {library: nbn, mechanism: mdn, ..., extra_kwargs: {batch_size: *BATCH_SIZE_FIT}}

    PyYAML resolves the alias at parse time; the top-level key itself
    carries no runner semantics and is skipped by the unknown-key check.
    """
    return key.isupper()

_BASELINE_REQUIRED = frozenset({"library", "mechanism", "param_method"})
_BASELINE_KNOWN = frozenset({
    "library", "mechanism", "param_method",
    "inference_method", "device", "extra_kwargs", "batch_size",
})


def load_runner_config(
    path: str | Path,
    *,
    device_override: str | None = None,
    jsonl_path: Path | None = None,
    measurement_override: Any | None = None,
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

    from nbn.bench.core.config import RunnerConfig
    from nbn.bench.measurements import AccuracyAndTiming, TimingOnly
    from nbn.bench.problems import SyntheticProblemSource
    from nbn.bench.selectors import UniformRandomSelector
    from nbn.bench.selectors.heaviest import HeaviestQueryByRole
    from nbn.bench.selectors.topological import TopologicalAllocator

    text = Path(path).read_text()
    d: dict[str, Any] = yaml.safe_load(text)

    # ── version check ────────────────────────────────────────────────────────
    version = d.get("version")
    if version not in _ACCEPTED_VERSIONS:
        raise ValueError(
            f"Config {str(path)!r} has version={version!r}; "
            f"this runner requires version in {sorted(_ACCEPTED_VERSIONS)}. "
            f"See docs/v0.13-benchmark-redesign.md §1c for the new schema."
        )

    # ── unknown top-level keys ───────────────────────────────────────────────
    # ALL-CAPS keys are YAML-anchor constants (see _is_constant_key) and
    # are neither unknown nor consumed below.
    unknown = {k for k in d if not _is_constant_key(k)} - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"Config {str(path)!r} has unknown top-level fields: "
            f"{sorted(unknown)}. "
            f"Known fields: {sorted(_TOP_LEVEL_KEYS)} "
            f"(ALL-CAPS keys are treated as YAML-anchor constants)."
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
    elif benchmark == "bnlearn":
        from nbn.bench.problems import BnlearnProblemSource

        problem_source = BnlearnProblemSource()
        source_config = _parse_bnlearn_config(d["source"], path)
    else:
        raise ValueError(
            f"Config {str(path)!r}: unknown benchmark={benchmark!r}. "
            f"Supported: 'synthetic', 'bnlearn'."
        )

    # ── metrics dispatch ─────────────────────────────────────────────────────
    metrics = d.get("metrics", "all")
    if measurement_override is not None:
        # Parameter-learning command path: the COMMAND supplies the measurement
        # structurally and authoritatively (ParamLearningMeasurement). The
        # config's `metrics` field does NOT select behavior here — it is only
        # validated for hygiene: a PL config must declare `metrics:
        # log_likelihood`. This keeps the metrics field from being able to swap
        # command behavior (the inference command never passes an override, so
        # `metrics: log_likelihood` stays rejected for inference below).
        if metrics != "log_likelihood":
            raise ValueError(
                f"Config {str(path)!r}: parameter-learning requires "
                f"metrics='log_likelihood', got {metrics!r}."
            )
        measurement = measurement_override
    elif metrics == "all":
        measurement = AccuracyAndTiming()
    elif metrics == "timing":
        measurement = TimingOnly()
    else:
        raise ValueError(
            f"Config {str(path)!r}: unknown metrics={metrics!r}. "
            f"Supported: 'all' (AccuracyAndTiming), 'timing' (TimingOnly)."
        )

    # ── selector dispatch ────────────────────────────────────────────────────
    # Accepts both the legacy string form (``selector: uniform_random``) and
    # the Phase 2 dict form (``selector: {type: topological, ...}``).
    selector_cfg = d.get("selector", "uniform_random")

    # Top-level n_batch_queries (v0.14, #148, §5.3): the speed benchmark
    # sets it once for all baselines; a selector-block value overrides.
    top_level_nbq = int(d.get("n_batch_queries", 1))

    def _build_selector(stype: str, block: dict[str, Any]):
        # n_batch_queries (v0.14, #148): evidence-value variants per query
        # position, design doc §1.1, §2.4. Default = top-level value
        # (itself defaulting to 1 = existing behavior).
        n_batch_queries = int(block.get("n_batch_queries", top_level_nbq))
        if stype == "uniform_random":
            return UniformRandomSelector(n_batch_queries=n_batch_queries)
        if stype == "topological":
            return TopologicalAllocator(
                target_allocation=block.get("target_allocation"),
                kind_allocation=block.get("kind_allocation"),
                evidence_allocation=block.get("evidence_allocation"),
                n_evidence=block.get("n_evidence"),
                n_evidence_values=block.get("n_evidence_values"),
                max_retry=block.get("max_retry"),
                n_batch_queries=n_batch_queries,
            )
        if stype == "heaviest_by_role":
            # Phase 3 scalability selector. Deterministic; ignores
            # n_queries_per_cell (emits a fixed count per DAG, spec §2.1).
            return HeaviestQueryByRole(
                n_evidence=block.get("n_evidence"),
                max_retry=block.get("max_retry"),
                n_batch_queries=n_batch_queries,
            )
        raise ValueError(
            f"Config {str(path)!r}: unknown selector type {stype!r}. "
            f"Supported: 'uniform_random', 'topological', 'heaviest_by_role'."
        )

    if isinstance(selector_cfg, str):
        selector = _build_selector(selector_cfg, {})
    elif isinstance(selector_cfg, dict):
        selector = _build_selector(
            selector_cfg.get("type", "uniform_random"), selector_cfg
        )
    else:
        raise ValueError(
            f"Config {str(path)!r}: selector must be a string or mapping, "
            f"got {type(selector_cfg).__name__}."
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
    fit_timeout_s = float(d.get("fit_timeout_s", 1000.0))

    # ── batch_sizes sweep (v0.14, #148, §1.7/§5.4) ───────────────────────────
    raw_sweep = d.get("batch_sizes")
    batch_sizes: list[int] | None = None
    if raw_sweep is not None:
        if not isinstance(raw_sweep, list) or not raw_sweep:
            raise ValueError(
                f"Config {str(path)!r}: batch_sizes must be a non-empty "
                f"list of ints, got {raw_sweep!r}."
            )
        batch_sizes = [int(v) for v in raw_sweep]
        if any(v < 1 for v in batch_sizes):
            raise ValueError(
                f"Config {str(path)!r}: batch_sizes values must be >= 1, "
                f"got {batch_sizes}."
            )

    if jsonl_path is None:
        from nbn.bench.core.output import make_results_dir
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
        fit_timeout_s=fit_timeout_s,
        jsonl_path=jsonl_path,
        batch_sizes=batch_sizes,
    )


# ---------------------------------------------------------------------------
# Sub-parsers
# ---------------------------------------------------------------------------

_SYNTHETIC_SOURCE_KEYS = frozenset({
    "families", "n_nodes_list", "seeds",
    "n_train", "n_train_sweep", "n_test", "n_reference",
    "edge_density", "max_in_degree", "cardinality",
    "fraction_continuous", "device",
})
# n_train is NOT listed required here — exactly one of n_train / n_train_sweep
# must be set (learning-curve sweep, #109 PR 6), validated explicitly below.
_SYNTHETIC_SOURCE_REQUIRED = frozenset({
    "families", "n_nodes_list", "seeds",
    "n_reference",
    "edge_density", "max_in_degree", "cardinality",
    "fraction_continuous",
})


def _parse_synthetic_config(src: Any, path: Any) -> SyntheticConfig:
    from nbn.bench.problems import SyntheticConfig

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

    # n_train vs n_train_sweep: exactly one (learning-curve sweep, #109 PR 6).
    has_scalar = "n_train" in src
    has_sweep = "n_train_sweep" in src
    if has_scalar and has_sweep:
        raise ValueError(
            f"Config {str(path)!r}: source.n_train and source.n_train_sweep "
            f"are mutually exclusive; set exactly one."
        )
    if not has_scalar and not has_sweep:
        raise ValueError(
            f"Config {str(path)!r}: source requires exactly one of n_train "
            f"or n_train_sweep."
        )
    n_train_sweep: list[int] | None = None
    if has_sweep:
        raw = src["n_train_sweep"]
        if not isinstance(raw, (list, tuple)) or len(raw) == 0:
            raise ValueError(
                f"Config {str(path)!r}: source.n_train_sweep must be a "
                f"non-empty list of ints, got {raw!r}."
            )
        n_train_sweep = [int(v) for v in raw]
        if any(v < 1 for v in n_train_sweep):
            raise ValueError(
                f"Config {str(path)!r}: source.n_train_sweep values must be "
                f">= 1, got {n_train_sweep}."
            )

    return SyntheticConfig(
        families=list(src["families"]),
        n_nodes_list=[int(n) for n in src["n_nodes_list"]],
        seeds=[int(s) for s in src["seeds"]],
        n_train=int(src["n_train"]) if has_scalar else 10_000,
        n_train_sweep=n_train_sweep,
        n_test=int(src.get("n_test", 2000)),
        n_reference=int(src["n_reference"]),
        edge_density=float(src["edge_density"]),
        max_in_degree=int(src["max_in_degree"]),
        cardinality=int(src["cardinality"]),
        fraction_continuous=float(src["fraction_continuous"]),
        device=str(src.get("device", "cpu")),
    )


_BNLEARN_SOURCE_KEYS = frozenset({
    "networks", "seeds", "n_train", "n_test", "n_reference",
})
_BNLEARN_SOURCE_REQUIRED = frozenset({"networks", "seeds"})


def _parse_bnlearn_config(src: Any, path: Any):
    from nbn.bench.problems import BnlearnConfig

    if not isinstance(src, dict):
        raise ValueError(
            f"Config {str(path)!r}: 'source' must be a mapping, "
            f"got {type(src).__name__}."
        )
    if "n_train_sweep" in src:
        raise ValueError(
            f"Config {str(path)!r}: n_train_sweep is synthetic-only; bnlearn "
            f"networks use fixed real-world n_train and do not accept this field."
        )
    unknown = set(src.keys()) - _BNLEARN_SOURCE_KEYS
    if unknown:
        raise ValueError(
            f"Config {str(path)!r}: unknown source fields: "
            f"{sorted(unknown)}. Known: {sorted(_BNLEARN_SOURCE_KEYS)}."
        )
    missing = _BNLEARN_SOURCE_REQUIRED - set(src.keys())
    if missing:
        raise ValueError(
            f"Config {str(path)!r}: source missing required fields: "
            f"{sorted(missing)}."
        )
    return BnlearnConfig(
        networks=[str(n) for n in src["networks"]],
        seeds=[int(s) for s in src["seeds"]],
        # n_train / n_reference accept a scalar OR a tiered dict ({"tiers": ...});
        # the dict is passed through and resolved per network from n_parameters
        # in BnlearnProblemSource (_resolve_sample_count). n_test stays scalar.
        n_train=_sample_count_field(src.get("n_train", 5000)),
        n_test=int(src.get("n_test", 1000)),
        n_reference=_sample_count_field(src.get("n_reference", 5000)),
    )


def _sample_count_field(v: Any) -> int | dict:
    """A sample-count source field: an int (coerced) or a tiered dict
    (passed through; validated at resolve time)."""
    return v if isinstance(v, dict) else int(v)


def _parse_baseline_spec(
    b: Any,
    idx: int,
    device_override: str | None,
    path: Any,
) -> BaselineSpec:
    from nbn.bench.core.config import BaselineSpec

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

    # batch_size (v0.14, #148): per-baseline adapter consumption, §1.5.
    # An explicit YAML value pins the baseline (§5.6): it runs once at
    # that value in a batch_sizes sweep instead of following the sweep.
    batch_size_pinned = "batch_size" in b
    batch_size = int(b.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError(
            f"Config {str(path)!r}: baselines[{idx}] batch_size must be "
            f">= 1, got {batch_size}."
        )

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
        batch_size=batch_size,
        batch_size_pinned=batch_size_pinned,
    )
