"""v0.13 runner configuration dataclasses and adapter dispatch helper.

``BaselineSpec`` is the structured baseline identifier for the v0.13 runner.
``RunnerConfig`` wires the four composition axes (ProblemSource, QuerySelector,
Measurement, list[BaselineSpec]) plus iteration parameters into a single config
object consumed by ``Runner.run()``.

``build_adapter(spec)`` dispatches a BaselineSpec to the appropriate v0.13
stateful adapter instance (NBNAdapter, PgmpyAdapter, PomegranateAdapter,
PyroAdapter).

Reference: docs/v0.13-benchmark-redesign.md §3, §4.1, §6
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nbn.bench.core.interfaces import Measurement, ProblemSource, QuerySelector


@dataclass
class BaselineSpec:
    """Structured baseline identifier for the v0.13 runner.

    Fields match the v0.12 BaselineSpec (nbn/bench/_baseline_registry.py)
    plus ``extra_kwargs`` for adapter-specific construction knobs (e.g.
    ``n_samples`` for LW-based adapters).

    Dispatch table (``build_adapter`` uses ``library`` to route):
        ``"nbn"``         → NBNAdapter(mechanism, engine=inference_method)
        ``"pgmpy"``       → PgmpyAdapter(param_method, inference_method)
        ``"pomegranate"`` → PomegranateAdapter()
        ``"pyro"``        → PyroAdapter(mechanism, inference_method)

    ``device`` is optional and passed through verbatim (``None`` |
    ``"auto"`` | a concrete string). Each adapter calls
    ``resolve_device()`` in its ``__init__`` to translate ``None`` /
    ``"auto"`` into cuda-if-available-else-cpu.
    """

    library: str                          # 'nbn' | 'pgmpy' | 'pomegranate' | 'pyro'
    mechanism: str                        # 'cat' | 'neuralcat' | 'lg' | 'mdn' | 'flow'
                                          # | 'hybrid' | 'discrete' | 'empirical'
    param_method: str                     # 'mle' | 'bayes' | 'lg' | 'empirical'
    inference_method: str | None = None   # 've' | 'lw' | 'router' | 'predict'
                                          # | 'importance' | None
    device: str | None = None
    extra_kwargs: dict = field(default_factory=dict)
    # Batched-queries (v0.14, #148): how many queries this baseline's
    # adapter consumes per query_batch call. 1 = sequential (default;
    # pyro/pgmpy stay pinned here). Set per-baseline in YAML (§1.5).
    batch_size: int = 1
    # True iff the YAML explicitly set batch_size for this baseline
    # (§5.6 sweep dispatch): a pinned baseline runs once at its pinned
    # value; an unpinned baseline whose adapter supports batching
    # follows the batch_sizes sweep. Explicit pin always wins — a
    # batchable baseline pinned to 1 runs once, sequentially.
    batch_size_pinned: bool = False


@dataclass
class RunnerConfig:
    """Configuration for the v0.13 runner orchestrator.

    Holds the composition of ProblemSource, QuerySelector, Measurement, and a
    list of BaselineSpecs.  The runner calls
    ``problem_source.iter_problems(source_config)`` to iterate over problems.

    Attributes
    ----------
    benchmark:
        Name for the v3 parquet schema ("synthetic" | "scalability" | "bnlearn").
    config_name:
        Human-readable label for this run configuration (e.g., "paper",
        "smoke").  Used in the run output directory name produced by
        ``make_results_dir(benchmark, config_name)``.
    problem_source:
        ProblemSource instance (e.g. SyntheticProblemSource).
    source_config:
        Source-specific config passed to ``problem_source.iter_problems()``.
        For SyntheticProblemSource: a SyntheticConfig.
    selector:
        QuerySelector instance.
    measurement:
        Measurement instance (AccuracyAndTiming or TimingOnly).
    baselines:
        List of BaselineSpec.  Each is instantiated fresh per cell via
        ``build_adapter(spec)``.
    n_queries_per_cell:
        Number of queries to select per (problem, baseline) cell.
    per_cell_timeout_s:
        Soft cumulative budget on query_time_s per cell.  When the
        cumulative total of adapter.query() wall-clock for a cell exceeds
        this value, remaining queries receive status="timeout" rows.
        Does NOT gate fit() or metrics computation.
    fit_timeout_s:
        fit() is allowed up to this many seconds.  If
        ``fit_time_s > fit_timeout_s``, the cell is emitted with
        status="timeout" and remaining queries are skipped.  Default
        1000s is a generous safety net without letting a runaway fit
        block the full run.
    jsonl_path:
        Path to the JSONL output file (streaming, line-buffered).

    Reference: docs/v0.13-benchmark-redesign.md §3, §4.1, §6
    """

    benchmark: str
    config_name: str
    problem_source: ProblemSource
    source_config: Any
    selector: QuerySelector
    measurement: Measurement
    baselines: list[BaselineSpec]
    n_queries_per_cell: int
    per_cell_timeout_s: float
    fit_timeout_s: float = 1000.0  # fit() safety budget (seconds)
    jsonl_path: Path = field(default_factory=lambda: Path("output.jsonl"))
    # Batched-queries speed benchmark (v0.14, #148, design doc §1.7/§5.6):
    # when set, the CLI iterates these batch_size values — baselines whose
    # adapters support batching (and are not YAML-pinned) run once per
    # value; pinned / non-batchable baselines run once on the first pass.
    # None = no sweep (every existing benchmark).
    batch_sizes: list[int] | None = None


def build_adapter(spec: BaselineSpec, *, require_engine: bool = False) -> Any:
    """Dispatch BaselineSpec → v0.13 adapter instance.

    Instantiates a fresh adapter for each cell.  Adapters are stateful
    (``fit()`` populates ``self.model``); do not reuse across cells.

    Parameters
    ----------
    spec:
        BaselineSpec with at minimum ``library``, ``mechanism``,
        ``param_method``, and (for query/inference) ``inference_method``.
    require_engine:
        Whether ``inference_method`` is mandatory. ``True`` on the
        inference (query) path: a spec missing ``inference_method`` is a
        misconfiguration and is rejected early, here, before any fit (the
        long-standing safety). ``False`` (default) on the parameter-learning
        / fit-only path (#109): the adapter is queried by nobody — it is fit
        and scored via ``score_data`` — so it is constructed WITHOUT an
        inference engine and carries an engine-less name (``"nbn-cat"``,
        ``"pgmpy-mle"`` …, the keys the applicability table already uses).
        ``False`` is also correct for the runner's name/probe call sites,
        which never query.

    Returns
    -------
    A v0.13 BaselineAdapter instance (NBNAdapter, PgmpyAdapter,
    PomegranateAdapter, or PyroAdapter).

    Raises
    ------
    ValueError
        Unknown ``library``, or ``inference_method`` is None while
        ``require_engine`` is True.
    """
    from nbn.bench.adapters import NBNAdapter, PgmpyAdapter, PomegranateAdapter, PyroAdapter

    lib = spec.library
    # Pass the raw spec value through (None | "auto" | concrete); each
    # adapter calls resolve_device() to translate. Collapsing to "cpu"
    # here was the bug that forced every baseline onto CPU.
    device = spec.device
    kw = spec.extra_kwargs

    def _need_engine(adapter_name: str) -> None:
        if require_engine and spec.inference_method is None:
            raise ValueError(
                f"{adapter_name} requires inference_method on the inference "
                f"path; got None in spec {spec!r}"
            )

    if lib == "nbn":
        _need_engine("NBNAdapter")
        return NBNAdapter(
            mechanism=spec.mechanism,
            engine=spec.inference_method,   # None -> fit-only, engine-less
            device=device,
            **kw,
        )

    if lib == "pgmpy":
        _need_engine("PgmpyAdapter")
        return PgmpyAdapter(
            param_method=spec.param_method,
            inference_method=spec.inference_method,   # None -> fit-only
            device=device,
            **kw,
        )

    if lib == "pomegranate":
        return PomegranateAdapter(device=device, **kw)

    if lib == "pyro":
        _need_engine("PyroAdapter")
        return PyroAdapter(
            mechanism=spec.mechanism,
            inference_method=spec.inference_method,   # None -> fit-only
            device=device,
            **kw,
        )

    raise ValueError(
        f"Unknown library {lib!r} in spec {spec!r}. "
        f"Valid: 'nbn', 'pgmpy', 'pomegranate', 'pyro'"
    )
