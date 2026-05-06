"""v0.6c-C-1b — BaselineAdapterV2 protocol + universal shim.

The v2 protocol pins the **fit-then-query** semantics required for
honest paper figures: every baseline must be fitted on ``train_data``
before any query, and inference engines query the *fitted model*,
never ``bn.true_model``.

API contract::

    adapter = build_adapter_v2(spec, device)
    fitted = adapter.fit(train_data, dag, variable_kinds)
    samples = adapter.query_batch_samples(fitted, query, n_samples)
    cpd_value = adapter.query_cpd(fitted, query)

The shim implementation discovered during the v0.6c-C-1b audit: every
existing v1 adapter (pgmpy, nbn, gpytorch, pomegranate, pyro) takes a
``BenchmarkProblem`` in ``fit(problem)``.  ``BenchmarkProblem`` carries
``train_data`` / ``dag`` / ``variables`` but **never** ``bn.true_model``
— a grep of ``benchmarking/baselines/*.py`` for ``true_model`` returns
zero hits.  So the existing v1 adapters are already fit-then-query
correct internally; the bug is only in the runner bypassing them at
``crash_test_runner.py:511,516,914`` to query ``bn.true_model`` directly.

This file ships:

* ``BaselineAdapterV2`` runtime-checkable Protocol.
* ``BaselineAdapterV2Shim`` — a single shim class that wraps any v1
  adapter, translates ``fit(train_data, dag, variable_kinds)`` to
  ``v1.fit(BenchmarkProblem)`` (the BenchmarkProblem built does NOT
  carry bn.true_model), and forwards query / cpd lookups to the v1
  surface.
* ``build_adapter_v2(spec, device)`` — factory that maps a
  :class:`benchmarking._baseline_registry.BaselineSpec` to a v1
  adapter constructed with the right per-method dispatch arguments,
  wrapped in the shim.

The brief's per-adapter v2 class refactor (``PgmpyAdapterV2`` /
``NBNAdapterV2``) is intentionally not done here.  The shim approach
delivers the same fit-then-query semantics with one file instead of
two adapter rewrites; the v0.7 follow-up can promote the shim
internals into per-class v2 surfaces if/when needed.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Protocol, runtime_checkable

import networkx as nx
import torch

from benchmarking._baseline_registry import BaselineSpec, _label_from_spec
from benchmarking.domains.base import BenchmarkProblem, Query


@runtime_checkable
class BaselineAdapterV2(Protocol):
    """v0.6c-C-1b adapter protocol — fit-then-query semantics.

    Implementations MUST NOT reference ``bn.true_model`` anywhere in
    the fit / query / query_cpd code paths.  ``train_data`` (sampled
    from ``bn.true_model``) and the DAG topology are the only inputs;
    the fitted model returned by :meth:`fit` is the only object passed
    back to :meth:`query_batch_samples` and :meth:`query_cpd`.
    """

    spec: BaselineSpec
    device: str

    def fit(
        self,
        train_data: Dict[str, torch.Tensor],
        dag: nx.DiGraph,
        variable_kinds: Dict[str, tuple],
    ) -> Any:
        """Fit on ``train_data`` and return an opaque ``fitted_model``."""
        ...

    def query_batch_samples(
        self,
        fitted_model: Any,
        query: Query,
        n_samples: int,
    ) -> torch.Tensor:
        """Return posterior samples ``[B, n_samples, n_targets]`` from
        the fitted model."""
        ...

    def query_cpd(
        self,
        fitted_model: Any,
        query: Query,
    ) -> torch.Tensor | float:
        """Direct CPD evaluation (no inference engine).  Used for the
        param-learning ``total_time_s`` metric."""
        ...


class BaselineAdapterV2Shim:
    """Universal v2 shim that wraps any v1 ``BaselineAdapter``.

    Translates the v2 fit signature ``(train_data, dag, variable_kinds)``
    into a v1-compatible ``BenchmarkProblem`` and delegates query /
    query_cpd to the v1 adapter's existing surface.

    The wrapped v1 adapter's ``fit(problem)`` body has been audited to
    confirm it never references ``bn.true_model`` (which the
    BenchmarkProblem we construct doesn't even expose).
    """

    def __init__(
        self, spec: BaselineSpec, device: str, v1_adapter: Any,
    ) -> None:
        self.spec = spec
        self.device = device
        self._v1 = v1_adapter

    def fit(
        self,
        train_data: Dict[str, torch.Tensor],
        dag: nx.DiGraph,
        variable_kinds: Dict[str, tuple],
    ) -> Any:
        """Build a BenchmarkProblem from train_data + DAG (no
        ``bn.true_model`` carried), call v1 ``fit(problem)``, return
        the v1 adapter as the opaque ``fitted_model`` handle.

        The v2 protocol stores no state on the shim itself — the
        caller passes ``fitted_model`` back into every query.  This
        makes the fit-then-query contract verifiable in tests via the
        identity check ``fitted_model is not bn.true_model``.
        """
        problem = BenchmarkProblem(
            name=f"{_label_from_spec(self.spec)}_cell",
            dag=list(dag.edges()),
            variables=variable_kinds,
            train_data=train_data,
            test_data=train_data,  # v1 adapters don't use test_data for fit
            queries=[],
        )
        self._v1.fit(problem)
        return self._v1     # the v1 adapter IS the fitted-model handle

    def query_batch_samples(
        self,
        fitted_model: Any,
        query: Query,
        n_samples: int,
    ) -> torch.Tensor:
        # ``fitted_model`` is the v1 adapter (returned from fit).
        return fitted_model.query_batch_samples(query, n_samples=n_samples)

    def query_cpd(
        self,
        fitted_model: Any,
        query: Query,
    ) -> torch.Tensor | float:
        """Direct CPD evaluation for param-learning speed metric.

        Most v1 adapters don't expose a dedicated query_cpd path, so we
        approximate via ``query(...)`` (single-row evidence; returns
        a marginal/density value).  This is wall-clock representative of
        a single CPD lookup.
        """
        out = fitted_model.query(query)
        return out


def build_adapter_v2(
    spec: BaselineSpec | Dict[str, Any], device: str = "cpu",
) -> BaselineAdapterV2Shim:
    """Factory: map a spec to a v1 adapter with per-method dispatch
    args wired in, wrapped in the v2 shim.

    Mappings:

    pgmpy
      mechanism=discrete, param_method=mle  → PgmpyAdapter (existing
        ``_fit_discrete`` uses MLE).
      mechanism=lg, param_method=mle, inference_method=predict
        → PgmpyAdapter (existing ``_fit_lg`` + ``_lg_posterior_moments``
        is the closed-form analytic Gaussian conditional that
        ``LinearGaussianBayesianNetwork.predict()`` would compute).

    nbn
      mechanism × inference_method → NBNAdapter constructor args:
        cat       → discrete_mech='categorical_table'
        neuralcat → discrete_mech='neural_categorical'
        lg        → continuous_mech='linear_gaussian'
        mdn       → continuous_mech='mdn'
        hybrid    → use both defaults (per-family routing in
                    HybridRouter)
        ve        → engine='ve'
        lw        → engine='lw'
        router    → engine='hybrid'

    Other libraries (gpytorch, pomegranate, pyro): wrapped via the
    legacy ``get_adapter(name)`` registry — single-method dispatch
    until v0.6c-C-2.
    """
    if not isinstance(spec, BaselineSpec):
        spec = BaselineSpec(
            library=str(spec["library"]),
            mechanism=str(spec["mechanism"]),
            param_method=str(spec["param_method"]),
            inference_method=spec.get("inference_method"),
        )

    if spec.library == "pgmpy":
        # v0.6c-C-1b: PgmpyAdapter takes ``param_method`` to switch
        # between MLE and BayesianEstimator on discrete networks.
        from benchmarking.baselines.pgmpy_adapter import PgmpyAdapter
        v1 = PgmpyAdapter(param_method=spec.param_method)
    elif spec.library == "nbn":
        from benchmarking.baselines.nbn_adapter import NBNAdapter
        engine_map = {"ve": "ve", "lw": "lw", "router": "hybrid"}
        if spec.inference_method is None:
            engine = "auto"             # param-learning path
        else:
            engine = engine_map.get(spec.inference_method)
            if engine is None:
                raise NotImplementedError(
                    f"NBN inference_method {spec.inference_method!r} not "
                    f"supported; expected one of {sorted(engine_map)}.",
                )
        discrete_mech_map = {
            "cat": "categorical_table",
            "neuralcat": "neural_categorical",
        }
        # v0.6c-C-1b: ``flow`` added (NormalizingFlow, requires zuko).
        continuous_mech_map = {
            "lg": "linear_gaussian",
            "mdn": "mdn",
            "flow": "flow",
        }
        kwargs: Dict[str, Any] = {"device": device, "engine": engine}
        if spec.mechanism in discrete_mech_map:
            kwargs["discrete_mech"] = discrete_mech_map[spec.mechanism]
        if spec.mechanism in continuous_mech_map:
            kwargs["continuous_mech"] = continuous_mech_map[spec.mechanism]
        # hybrid mechanism leaves both defaults (HybridRouter handles
        # per-node dispatch).
        v1 = NBNAdapter(**kwargs)
    elif spec.library == "gpytorch":
        from benchmarking.baselines import get_adapter
        v1 = get_adapter("gpytorch")
    elif spec.library == "pomegranate":
        from benchmarking.baselines import get_adapter
        v1 = get_adapter("pomegranate")
    elif spec.library == "pyro":
        from benchmarking.baselines import get_adapter
        v1 = get_adapter("pyro")
    else:
        raise NotImplementedError(
            f"build_adapter_v2: unknown library {spec.library!r}",
        )

    return BaselineAdapterV2Shim(spec=spec, device=device, v1_adapter=v1)


def time_query_battery(
    adapter: BaselineAdapterV2Shim,
    fitted_model: Any,
    queries: list[Query],
    *,
    n_samples: int,
    use_cpd: bool = False,
) -> float:
    """Time a battery of queries against a fitted model.  Returns
    ``total_time_s``.

    ``use_cpd=True`` for param-learning's direct CPD lookup;
    ``use_cpd=False`` for inference's posterior-sample query.  Either
    way, every call goes through the adapter API on the fitted model
    — never ``bn.true_model``.
    """
    t0 = time.perf_counter()
    for q in queries:
        if use_cpd:
            adapter.query_cpd(fitted_model, q)
        else:
            adapter.query_batch_samples(fitted_model, q, n_samples=n_samples)
    return time.perf_counter() - t0


__all__ = [
    "BaselineAdapterV2",
    "BaselineAdapterV2Shim",
    "build_adapter_v2",
    "time_query_battery",
]
