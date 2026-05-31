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

from benchmarking.core.interfaces import Measurement, ProblemSource, QuerySelector


@dataclass
class BaselineSpec:
    """Structured baseline identifier for the v0.13 runner.

    Fields match the v0.12 BaselineSpec (benchmarking/_baseline_registry.py)
    plus ``extra_kwargs`` for adapter-specific construction knobs (e.g.
    ``n_samples`` for LW-based adapters).

    Dispatch table (``build_adapter`` uses ``library`` to route):
        ``"nbn"``         → NBNAdapter(mechanism, engine=inference_method)
        ``"pgmpy"``       → PgmpyAdapter(param_method, inference_method)
        ``"pomegranate"`` → PomegranateAdapter()
        ``"pyro"``        → PyroAdapter(mechanism, inference_method)

    ``device`` is optional; if None, adapters use their own default ("cpu").
    """

    library: str                          # 'nbn' | 'pgmpy' | 'pomegranate' | 'pyro'
    mechanism: str                        # 'cat' | 'neuralcat' | 'lg' | 'mdn' | 'flow'
                                          # | 'hybrid' | 'discrete' | 'empirical'
    param_method: str                     # 'mle' | 'bayes' | 'lg' | 'empirical'
    inference_method: str | None = None   # 've' | 'lw' | 'router' | 'predict'
                                          # | 'importance' | None
    device: str | None = None
    extra_kwargs: dict = field(default_factory=dict)


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
    fit_timeout_s_multiplier:
        fit() is allowed up to this multiple of ``per_cell_timeout_s``.
        If ``fit_time_s > fit_timeout_s_multiplier * per_cell_timeout_s``,
        the cell is emitted with status="timeout" and remaining queries
        are skipped.  Default 10x provides a generous safety net without
        letting a runaway fit block the full run.
    jsonl_path:
        Path to the JSONL output file (streaming, line-buffered).

    Reference: docs/v0.13-benchmark-redesign.md §3, §4.1, §6
    """

    benchmark: str
    problem_source: ProblemSource
    source_config: Any
    selector: QuerySelector
    measurement: Measurement
    baselines: list[BaselineSpec]
    n_queries_per_cell: int
    per_cell_timeout_s: float
    fit_timeout_s_multiplier: float = 10.0
    fit_timeout_s: float | None = None  # explicit override; if set, ignores multiplier
    jsonl_path: Path = field(default_factory=lambda: Path("output.jsonl"))


def build_adapter(spec: BaselineSpec) -> Any:
    """Dispatch BaselineSpec → v0.13 adapter instance.

    Instantiates a fresh adapter for each cell.  Adapters are stateful
    (``fit()`` populates ``self.model``); do not reuse across cells.

    Parameters
    ----------
    spec:
        BaselineSpec with at minimum ``library``, ``mechanism``,
        ``param_method``, and (for most libraries) ``inference_method``.

    Returns
    -------
    A v0.13 BaselineAdapter instance (NBNAdapter, PgmpyAdapter,
    PomegranateAdapter, or PyroAdapter).

    Raises
    ------
    ValueError
        Unknown ``library`` or required field is None.
    """
    from benchmarking.adapters import NBNAdapter, PgmpyAdapter, PomegranateAdapter, PyroAdapter

    lib = spec.library
    device = spec.device or "cpu"
    kw = spec.extra_kwargs

    if lib == "nbn":
        if spec.inference_method is None:
            raise ValueError(
                f"NBNAdapter requires inference_method; got None in spec {spec!r}"
            )
        return NBNAdapter(
            mechanism=spec.mechanism,
            engine=spec.inference_method,
            device=device,
            **kw,
        )

    if lib == "pgmpy":
        if spec.inference_method is None:
            raise ValueError(
                f"PgmpyAdapter requires inference_method; got None in spec {spec!r}"
            )
        return PgmpyAdapter(
            param_method=spec.param_method,
            inference_method=spec.inference_method,
            **kw,
        )

    if lib == "pomegranate":
        return PomegranateAdapter(**kw)

    if lib == "pyro":
        if spec.inference_method is None:
            raise ValueError(
                f"PyroAdapter requires inference_method; got None in spec {spec!r}"
            )
        return PyroAdapter(
            mechanism=spec.mechanism,
            inference_method=spec.inference_method,
            device=device,
            **kw,
        )

    raise ValueError(
        f"Unknown library {lib!r} in spec {spec!r}. "
        f"Valid: 'nbn', 'pgmpy', 'pomegranate', 'pyro'"
    )
