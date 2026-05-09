"""Pin the fit-then-query semantics introduced in v0.6c-C-1b.

These tests ensure the runner / adapters / engines never query
``bn.true_model`` during inference.  The honest paper-figure
contract is: every baseline fits parameters on ``train_data`` first,
then the inference engine queries the *fitted model*.  Any future
PR that regresses this contract should fail one of these tests.
"""
from __future__ import annotations

import inspect
import time
from typing import Any

import networkx as nx
import pytest
import torch

from benchmarking._baseline_registry import (
    BaselineSpec,
    is_applicable,
    known_labels,
)
from benchmarking.baselines._adapter_v2 import (
    BaselineAdapterV2,
    BaselineAdapterV2Shim,
    build_adapter_v2,
    time_query_battery,
)
from benchmarking.synthetic import make_synthetic_bn


# ---------------------------------------------------------------------- #
# Test 1 — V2 protocol contract
# ---------------------------------------------------------------------- #


def test_adapter_v2_protocol_runtime_checkable() -> None:
    """``BaselineAdapterV2`` is a runtime-checkable Protocol.  The shim
    satisfies it.  Future per-class adapter rewrites must satisfy it
    too."""
    spec = BaselineSpec(library="pgmpy", mechanism="discrete",
                        param_method="mle", inference_method="ve")
    adapter = build_adapter_v2(spec, device="cpu")
    assert isinstance(adapter, BaselineAdapterV2)
    assert isinstance(adapter, BaselineAdapterV2Shim)


# ---------------------------------------------------------------------- #
# Test 2 — Static check: bn.true_model removed from inference query path
# ---------------------------------------------------------------------- #


def test_inference_runner_does_not_query_bn_true_model() -> None:
    """Walk the runner source and assert NO line passes ``bn.true_model``
    into an engine ``.query()``/`.query_batch()`/etc.  Lines containing
    ``bn.true_model`` are allowed only in:

    * `_avg_*` parameter-learning truth comparisons (CPD diff).
    * `_forward_with_clamp_posterior_samples` ground-truth oracle.
    * ``bn.true_model.sample(...)`` ground-truth pool synthesis.
    * ``bn.true_model.device`` device extraction (read-only attribute).
    * Comments and docstrings.

    This is the audit pin for v0.6c-C-1b: if a future PR re-introduces
    ``eng.query(bn.true_model, ...)`` in the inference codepath, this
    test fires.
    """
    import benchmarking.crash_test_runner as runner
    src = inspect.getsource(runner)
    bad_lines = []
    for i, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') \
                or stripped.startswith("'''"):
            continue
        if "bn.true_model" not in stripped:
            continue
        # Backtick-quoted references inside docstrings (e.g.,
        # ``eng.query_batch(bn.true_model, ...)`` quoted as a
        # description of pre-PR behaviour) are not real code paths.
        if "``" in stripped or "`eng" in stripped:
            continue
        # Allowed forms.
        is_oracle_sample = (
            ".sample(" in stripped and "bn.true_model.sample(" in stripped
        )
        is_device_attr = "bn.true_model.device" in stripped
        is_param_truth = (
            "true_logits" in stripped or "true_mech" in stripped
        )
        if is_oracle_sample or is_device_attr or is_param_truth:
            continue
        # The forbidden pattern: passing bn.true_model into a query
        # call (any *.query* method).
        if any(qm in stripped for qm in (
            "query_batch(bn.true_model", "query(bn.true_model",
            "query_batch_samples(bn.true_model", ".infer(bn.true_model",
        )):
            bad_lines.append(f"line {i}: {stripped}")
    assert not bad_lines, (
        "Inference path passes bn.true_model to a query method:\n  "
        + "\n  ".join(bad_lines)
    )


# ---------------------------------------------------------------------- #
# Test 3 — End-to-end fit-then-query via the v2 shim
# ---------------------------------------------------------------------- #


def test_v2_shim_fit_then_query_pgmpy_discrete_ve() -> None:
    """The v2 shim's fit-then-query path returns finite samples and
    operates on a fitted model (not bn.true_model)."""
    bn = make_synthetic_bn(
        family="discrete", n_nodes=5, n_train=200, n_test=50,
        n_reference=200, seed=0, device="cpu",
    )
    spec = BaselineSpec(library="pgmpy", mechanism="discrete",
                        param_method="mle", inference_method="ve")
    adapter = build_adapter_v2(spec, device="cpu")
    fitted = adapter.fit(bn.train_data, bn.dag, bn.variable_specs)
    # The fitted model is the v1 adapter object — emphatically not
    # ``bn.true_model``.
    assert fitted is not bn.true_model
    # Build a small query and verify samples are finite.
    from benchmarking.domains.base import Query
    target = "X1"
    ev_node = "X0"
    B = 4
    evidence = {ev_node: torch.zeros(B, dtype=torch.long)}
    q = Query(targets=(target,), evidence=evidence, kind="marginal")
    samples = adapter.query_batch_samples(fitted, q, n_samples=200)
    assert torch.isfinite(samples.float()).all()


def test_v2_shim_fit_then_query_nbn_lg_lw() -> None:
    """NBN continuous_lg + LW: the fitted NBN's mechanisms must have
    populated ``_logits`` (categorical) or ``_weight``/``_bias``
    (LinearGaussian) — proof that fit() actually trained, vs the
    pre-PR bypass that queried truth without fitting."""
    bn = make_synthetic_bn(
        family="continuous_lg", n_nodes=5, n_train=200, n_test=50,
        n_reference=200, seed=0, device="cpu",
    )
    spec = BaselineSpec(library="nbn", mechanism="lg",
                        param_method="mle", inference_method="lw")
    adapter = build_adapter_v2(spec, device="cpu")
    fitted = adapter.fit(bn.train_data, bn.dag, bn.variable_specs)
    fitted_model = adapter._v1.model
    assert fitted_model is not bn.true_model
    # Every node has a fitted LinearGaussian mechanism.
    for node in bn.dag.nodes:
        mech = fitted_model.mechanisms[node]
        assert hasattr(mech, "_weight") and mech._weight is not None
        assert hasattr(mech, "_bias") and mech._bias is not None


# ---------------------------------------------------------------------- #
# Test 4 — pgmpy lg path uses analytic Gaussian conditional
# ---------------------------------------------------------------------- #


def test_pgmpy_lg_predict_returns_finite_samples() -> None:
    """``pgmpy-lg-predict`` (analytic Gaussian conditional via
    ``_lg_posterior_moments``) must return finite posterior samples
    on a continuous_lg cell."""
    bn = make_synthetic_bn(
        family="continuous_lg", n_nodes=5, n_train=200, n_test=50,
        n_reference=200, seed=0, device="cpu",
    )
    spec = BaselineSpec(library="pgmpy", mechanism="lg",
                        param_method="mle", inference_method="predict")
    adapter = build_adapter_v2(spec, device="cpu")
    fitted = adapter.fit(bn.train_data, bn.dag, bn.variable_specs)
    from benchmarking.domains.base import Query
    target = "X2"
    ev_node = "X0"
    B = 1
    q = Query(
        targets=(target,),
        evidence={ev_node: torch.zeros(B, dtype=torch.float32)},
        kind="marginal",
    )
    samples = adapter.query_batch_samples(fitted, q, n_samples=200)
    assert torch.isfinite(samples.float()).all()
    assert samples.shape[1] == 200


# ---------------------------------------------------------------------- #
# Test 5 — Param-learning emits total_time_s
# ---------------------------------------------------------------------- #


def test_param_learning_cell_emits_total_time_s() -> None:
    """v0.6c-C-1b: parameter-learning crash test reports BOTH per-CPD
    accuracy AND ``total_time_s``.

    v0.8-#51: ``_param_learning_cell`` now takes a ``BaselineSpec``
    (method-keyed dispatch) rather than a legacy adapter string.
    """
    from benchmarking._crash_test_utils import CrashTestConfig
    from benchmarking._baseline_registry import BaselineSpec
    from benchmarking.crash_test_runner import _param_learning_cell

    cfg = CrashTestConfig.from_yaml(
        "benchmarking/configs/parameter_learning_smoke.yaml",
    )
    spec = BaselineSpec(
        library="nbn", mechanism="cat", param_method="mle",
    )
    rows = _param_learning_cell(cfg, "discrete", 5, 0, spec)
    metrics = {r.metric for r in rows}
    assert "total_time_s" in metrics, (
        f"expected 'total_time_s' in metrics, got {metrics}"
    )
    speed_rows = [r for r in rows if r.metric == "total_time_s"]
    assert len(speed_rows) == 1
    assert speed_rows[0].status == "ok"
    assert speed_rows[0].value > 0


# ---------------------------------------------------------------------- #
# Test 6 — Tracking adapter pins the call sequence (fit → query)
# ---------------------------------------------------------------------- #


class _TrackingV2Adapter:
    """A minimal v2 adapter that records the sequence of calls and
    asserts the runner passes the fitted-model object back into every
    query (i.e., the runner doesn't substitute ``bn.true_model``)."""

    def __init__(self, spec: BaselineSpec, device: str) -> None:
        self.spec = spec
        self.device = device
        self.sequence: list[str] = []
        self.fitted_token = "<<FITTED_MODEL>>"

    def fit(self, train_data, dag, variable_kinds):
        self.sequence.append("fit")
        return self.fitted_token

    def query_batch_samples(self, fitted_model, query, n_samples):
        self.sequence.append("query_batch_samples")
        # Pin: runner must pass the same opaque fitted_model token back.
        assert fitted_model == self.fitted_token, (
            f"query_batch_samples received {fitted_model!r}; expected "
            f"the fitted-model token from .fit()"
        )
        B = next(iter(query.evidence.values())).shape[0]
        return torch.zeros(B, n_samples, len(query.targets))

    def query_cpd(self, fitted_model, query):
        self.sequence.append("query_cpd")
        assert fitted_model == self.fitted_token
        return 0.0


def test_tracking_adapter_satisfies_v2_protocol() -> None:
    """Sanity-check that a minimal tracking adapter satisfies the
    runtime-checkable Protocol — pins the v2 surface contract."""
    spec = BaselineSpec(library="pgmpy", mechanism="discrete",
                        param_method="mle", inference_method="ve")
    adapter = _TrackingV2Adapter(spec, device="cpu")
    assert isinstance(adapter, BaselineAdapterV2)


def test_time_query_battery_records_calls_in_fit_then_query_order() -> None:
    """``time_query_battery`` exercises the v2 surface end-to-end.  A
    tracking adapter records the call sequence; the assertion pins
    that every call is to ``query_batch_samples`` (or ``query_cpd``)
    with the fitted token, NEVER something else."""
    spec = BaselineSpec(library="pgmpy", mechanism="discrete",
                        param_method="mle", inference_method="ve")
    adapter = _TrackingV2Adapter(spec, device="cpu")
    fitted = adapter.fit(
        train_data={"X0": torch.zeros(10)},
        dag=nx.DiGraph([("X0", "X1")]),
        variable_kinds={"X0": ("discrete", 2), "X1": ("discrete", 2)},
    )
    from benchmarking.domains.base import Query
    queries = [
        Query(
            targets=("X1",),
            evidence={"X0": torch.zeros(2, dtype=torch.long)},
            kind="marginal",
        )
    ]
    # Inference path (n_samples-based queries).
    t = time_query_battery(adapter, fitted, queries, n_samples=10)
    assert t >= 0
    # Param-learning path (cpd lookups).
    t2 = time_query_battery(
        adapter, fitted, queries, n_samples=10, use_cpd=True,
    )
    assert t2 >= 0
    # Sequence: fit, then queries.  All queries received the fitted
    # token (asserted inline in the tracking adapter).
    assert adapter.sequence == [
        "fit", "query_batch_samples", "query_cpd",
    ]


# ---------------------------------------------------------------------- #
# Test 7 — NeuralCategorical-VE deferral pin (same as C-1a's pin,
# duplicated here so the v0.6c-C-1b test file is self-contained on the
# label rules it depends on).
# ---------------------------------------------------------------------- #


def test_inference_runner_skips_non_applicable_cells(tmp_path) -> None:
    """The runner MUST consult ``is_applicable(label, family)`` BEFORE
    adapter dispatch and emit ``status='not_supported'`` for non-
    applicable cells.

    Regression for the v0.6c-C-1b post-merge bug: pre-fix the runner
    consulted only the legacy ``_NOT_APPLICABLE`` table inside
    ``_inference_cell``, which is keyed by polymorphic legacy adapter
    strings ("pgmpy", "nbn_lw") that pre-date the C-1a per-mechanism
    label scheme.  As a result, ``pgmpy-mle-ve`` ran on continuous_lg
    (via pgmpy's lg path), ``nbn-cat-lw`` ran on continuous_nongauss
    (via NBN's mdn default), and ``nbn-cat-lw`` on hybrid hit a cuda
    device-side assert.  The fix routes the registry-based applicability
    gate at the runner-loop level, BEFORE any adapter dispatch.

    This test exercises the gate end-to-end via a tiny YAML config
    that includes one non-applicable spec; ``adapter.fit`` must NEVER
    be called for that cell.
    """
    import yaml
    from benchmarking._crash_test_utils import CrashTestConfig
    from benchmarking.crash_test_runner import run_inference

    # Build a tiny inference config with one non-applicable spec.
    config = {
        "config_schema_version": 2,
        "mode": "inference",
        "families": ["continuous_lg"],   # ← deliberate
        "n_nodes": [5],
        "n_seeds": 1,
        "n_queries_per_cell": 4,
        "nbn_batch_size": 4,
        "n_train": 200,
        "n_reference": 500,
        "edge_density": 0.20,
        "max_in_degree": 4,
        "cardinality": 4,
        "fraction_continuous": 0.5,
        "nbn_lw_n_samples": 200,
        # Non-applicable: pgmpy-mle-ve is discrete-only per registry.
        "baselines": [
            {"library": "pgmpy", "mechanism": "discrete",
             "param_method": "mle", "inference_method": "ve"},
        ],
        "per_cell_timeout_s": 60,
        "output_dir": str(tmp_path),
        "output_prefix": "applicability_test",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    # Sanity: build_adapter_v2 would succeed for this spec — gating
    # must happen at the runner level, not inside the adapter.
    spec = BaselineSpec(library="pgmpy", mechanism="discrete",
                        param_method="mle", inference_method="ve")
    _ = build_adapter_v2(spec, device="cpu")

    # Run the inference crash test.
    rc = run_inference(str(cfg_path), device="cpu")
    assert rc == 0

    # Read the parquet and confirm the cell is status='not_supported',
    # not status='ok' (which would mean we ran inference on a
    # non-applicable family).
    import pandas as pd
    parquet = tmp_path / "raw" / "applicability_test_metrics.parquet"
    df = pd.read_parquet(parquet)
    assert len(df) == 1, f"expected 1 row, got {len(df)}: {df}"
    assert df.iloc[0]["status"] == "not_supported", (
        f"continuous_lg + pgmpy-mle-ve must be not_supported; "
        f"got {df.iloc[0]['status']!r}.  This indicates the "
        f"registry-based applicability gate is missing from the "
        f"runner loop."
    )
    assert df.iloc[0]["baseline"] == "pgmpy-mle-ve"
    # ``write_parquet`` flattens ``CellResult.extra`` into top-level
    # columns; the not_applicable row sets extra={"error_msg": "..."}.
    error_msg = str(df.iloc[0].get("error_msg", "") or "")
    assert "not applicable" in error_msg.lower(), (
        f"expected 'not applicable' in error_msg, got {error_msg!r}"
    )


def test_nbn_inference_path_records_total_time_s(tmp_path) -> None:
    """Issue #35 regression test: NBN inference baselines must populate
    ``total_time_s`` with a finite positive value, not NaN.

    Pre-fix, ``_time_nbn_inference`` called
    ``adapter.query_batch_samples(...)`` which ``NBNAdapter`` did not
    implement (the base class raised ``NotImplementedError``); the
    function silently swallowed the exception and returned ``float('nan')``.
    Cells then landed in the parquet as ``status='ok'`` with NaN value,
    which disappeared from the rendered figures.

    After the v0.7-#35 fix, ``NBNAdapter.query_batch_samples`` exposes
    the engine's batched path (single torch call over [B, ...] evidence,
    sample post-processing per-row) and the runner no longer swallows
    exceptions silently.  This test pins both.
    """
    import yaml
    import pandas as pd
    from benchmarking.crash_test_runner import run_inference

    config = {
        "config_schema_version": 2,
        "mode": "inference",
        "families": ["discrete"],
        "n_nodes": [5],
        "n_seeds": 1,
        "n_queries_per_cell": 4,
        "nbn_batch_size": 4,
        "n_train": 200,
        "n_reference": 500,
        "edge_density": 0.20,
        "max_in_degree": 4,
        "cardinality": 4,
        "fraction_continuous": 0.0,
        "nbn_lw_n_samples": 200,
        "baselines": [
            {"library": "nbn", "mechanism": "cat",
             "param_method": "mle", "inference_method": "ve"},
        ],
        "per_cell_timeout_s": 60,
        "output_dir": str(tmp_path),
        "output_prefix": "nbn_timing_test",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    rc = run_inference(str(cfg_path), device="cpu")
    assert rc == 0

    parquet = tmp_path / "raw" / "nbn_timing_test_metrics.parquet"
    df = pd.read_parquet(parquet)

    time_rows = df[
        (df["baseline"] == "nbn-cat-ve")
        & (df["metric"] == "total_time_s")
        & (df["status"] == "ok")
    ]
    assert len(time_rows) >= 1, (
        f"NBN inference must emit total_time_s rows; got "
        f"{df[df['metric']=='total_time_s'].to_dict('records')}"
    )

    finite_values = time_rows["value"].dropna()
    assert len(finite_values) >= 1, (
        f"NBN inference total_time_s must have at least one finite value; "
        f"got {time_rows['value'].tolist()}"
    )
    assert all(v > 0 for v in finite_values), (
        f"NBN inference total_time_s values must be positive; "
        f"got {finite_values.tolist()}"
    )


def test_pgmpy_lg_predict_w1_in_expected_range(tmp_path) -> None:
    """Issue #37 regression test: pgmpy-lg-predict's W₁ on continuous_lg
    should be in the same range as nbn-lg-lw post-fix.

    Pre-fix, ``_w1`` did ``min(len(a), len(b))`` truncation post-sort —
    a mismatched-quantile comparison rather than a Wasserstein estimator.
    The continuous_lg accuracy cell hit this because pgmpy's predicted
    samples (budget=max(n_samples, n_lw_samples)) differed in length
    from the oracle (always 2000), so ``_w1`` compared the lower half
    of sorted pgmpy against all of sorted oracle.

    Post-fix, ``_w1`` uses quantile-interpolation at evenly-spaced
    quantiles (length ``max(len)``).  Both pgmpy-lg-predict and
    nbn-lg-lw should now produce small W₁ values close to each other,
    because they're computing W₁ on essentially the same predicted
    distribution (analytic LG vs LW on fitted LG) against the same
    oracle.
    """
    import yaml
    import pandas as pd
    from benchmarking.crash_test_runner import run_inference

    config = {
        "config_schema_version": 2,
        "mode": "inference",
        "families": ["continuous_lg"],
        "n_nodes": [10],
        "n_seeds": 1,
        "n_queries_per_cell": 16,
        "nbn_batch_size": 16,
        "n_train": 2000,
        "n_reference": 2000,
        "edge_density": 0.20,
        "max_in_degree": 4,
        "cardinality": 4,
        "fraction_continuous": 1.0,
        "nbn_lw_n_samples": 2000,
        "baselines": [
            {"library": "pgmpy", "mechanism": "lg",
             "param_method": "mle", "inference_method": "predict"},
            {"library": "nbn", "mechanism": "lg",
             "param_method": "mle", "inference_method": "lw"},
        ],
        "per_cell_timeout_s": 120,
        "output_dir": str(tmp_path),
        "output_prefix": "issue37_test",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    rc = run_inference(str(cfg_path), device="cpu")
    assert rc == 0

    parquet = tmp_path / "raw" / "issue37_test_metrics.parquet"
    df = pd.read_parquet(parquet)

    pgmpy_rows = df[
        (df["baseline"] == "pgmpy-lg-predict")
        & (df["metric"] == "accuracy")
        & (df["status"] == "ok")
    ]
    nbn_rows = df[
        (df["baseline"] == "nbn-lg-lw")
        & (df["metric"] == "accuracy")
        & (df["status"] == "ok")
    ]
    assert len(pgmpy_rows) >= 1, (
        f"pgmpy-lg-predict must emit an accuracy ok row; got {df.to_dict('records')}"
    )
    assert len(nbn_rows) >= 1, (
        f"nbn-lg-lw must emit an accuracy ok row; got {df.to_dict('records')}"
    )

    pgmpy_w1 = float(pgmpy_rows["value"].iloc[0])
    nbn_w1 = float(nbn_rows["value"].iloc[0])

    # Post-fix expected: both small, both close.
    assert pgmpy_w1 < 0.15, (
        f"pgmpy-lg-predict W₁ should be small post-fix (got {pgmpy_w1:.4f})"
    )
    assert nbn_w1 < 0.15, (
        f"nbn-lg-lw W₁ should be small post-fix (got {nbn_w1:.4f})"
    )
    assert abs(pgmpy_w1 - nbn_w1) < 0.10, (
        f"pgmpy-lg-predict and nbn-lg-lw W₁ should be similar post-fix; "
        f"gap {abs(pgmpy_w1 - nbn_w1):.4f} suggests metric still has bias "
        f"(pgmpy={pgmpy_w1:.4f}, nbn={nbn_w1:.4f})"
    )


def test_nbn_neuralcat_ve_remains_deferred_in_c1b() -> None:
    """C-1a deferred ``nbn-neuralcat-ve`` to v0.7.  C-1b's adapter
    factory must not silently start supporting it (which would route
    a label that no engine can actually evaluate).  Issue #26."""
    assert "nbn-neuralcat-ve" not in known_labels()
    for family in ("discrete", "continuous_lg", "continuous_nongauss",
                   "hybrid"):
        assert not is_applicable("nbn-neuralcat-ve", family)
    spec = BaselineSpec(library="nbn", mechanism="neuralcat",
                        param_method="mle", inference_method="ve")
    # The factory should produce *some* adapter (the spec is well-
    # formed) but the underlying engine call would fail.  We only
    # check that the registry refuses to mark it applicable.
    # Building should not crash.
    _ = build_adapter_v2(spec, device="cpu")
