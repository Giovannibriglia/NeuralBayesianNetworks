"""ParamLearningMeasurement calibration emission (#109 PR 7, Stage C).

Gating taxonomy (continuous-only, complementary to discrete-only recovery):
  - NBN-lg on continuous   -> calibration_pit_ks + calibration_sd_ratio ok, finite;
  - NBN-cat on discrete    -> calibration_* not_applicable (no continuous nodes);
  - no supports_calibration -> calibration_* not_supported;
  - determinism            -> two measure() calls give bit-identical values;
  - oracle cache           -> the oracle draw fires once per problem across baselines.
"""
from __future__ import annotations

import math

import pytest
import torch

from benchmarking.adapters import NBNAdapter
from benchmarking.domains.base import BenchmarkProblem
from benchmarking.measurements import ParamLearningMeasurement


def _problem(family: str, *, seed: int = 1, n_nodes: int = 5):
    from benchmarking.synthetic import make_synthetic_bn

    bn = make_synthetic_bn(
        n_nodes=n_nodes, family=family, cardinality=3, edge_density=0.5,
        max_in_degree=2, n_train=500, n_test=150, n_reference=300,
        seed=seed, device="cpu",
    )
    return BenchmarkProblem(
        name=bn.name, dag=list(bn.dag.edges()), variables=bn.variable_specs,
        train_data=bn.train_data, test_data=bn.test_data, queries=[],
        true_model=bn.true_model, family=family, problem_id=str(n_nodes), seed=seed,
    )


def _fitted(problem, mechanism):
    a = NBNAdapter(mechanism=mechanism, engine=None, device="cpu")
    a.fit(problem, epochs=8)
    return a


def _measure(m, problem, adapter):
    return m.measure(
        problem, adapter, [], fit_time_s=0.0, benchmark="synthetic",
        seed=problem.seed,
    )


def _by_metric(rows):
    return {r.metric: r for r in rows}


def _assert_calibration_ok(prob, adapter):
    by = _by_metric(_measure(ParamLearningMeasurement(), prob, adapter))
    pit, sd = by["calibration_pit_ks"], by["calibration_sd_ratio"]
    assert pit.status == "ok" and math.isfinite(pit.value) and pit.value >= 0.0
    assert sd.status == "ok" and math.isfinite(sd.value) and sd.value > 0.0
    # recovery is the complement: not_applicable on a continuous problem.
    assert by["param_recovery_tv"].status == "not_applicable"


@pytest.mark.slow
@pytest.mark.parametrize("mechanism", ["lg", "mdn", "kde", "knn"])
def test_calibration_ok_on_continuous(mechanism):
    # Parametric (lg/mdn) + non-parametric (kde/knn, #223) — all CPU-tractable.
    prob = _problem("continuous_lg")
    _assert_calibration_ok(prob, _fitted(prob, mechanism))


@pytest.mark.gpu
def test_calibration_ok_flexcode_gpu():
    # FlexCode is gpu-only at benchmark scale (#223); CI skips via `not gpu`.
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the flexcode baseline")
    prob = _problem("continuous_lg")
    adapter = NBNAdapter(mechanism="flexcode", engine=None, device="cuda")
    adapter.fit(prob, epochs=8)
    _assert_calibration_ok(prob, adapter)


@pytest.mark.slow
def test_calibration_not_applicable_on_discrete():
    prob = _problem("discrete")
    rows = _measure(ParamLearningMeasurement(), prob, _fitted(prob, "cat"))
    by = _by_metric(rows)
    for metric in ("calibration_pit_ks", "calibration_sd_ratio"):
        assert by[metric].status == "not_applicable"
        assert math.isnan(by[metric].value)
        assert "continuous node" in (by[metric].error_msg or "")
    # recovery is the complement: ok on a fully-discrete problem.
    assert by["param_recovery_tv"].status == "ok"


def test_calibration_not_supported_without_flag():
    # An adapter that doesn't set supports_calibration -> not_supported, gate
    # short-circuits before any sampling.
    class _NoCalib:
        name = "pgmpy-lg-predict"

        def predictive_samples(self, td):  # pragma: no cover - must not run
            raise AssertionError("predictive_samples must not be called when gated off")

    # Minimal discrete problem fixture (no true_model needed — gate 1 fires first).
    prob = BenchmarkProblem(
        name="x", dag=[], variables={"A": ("discrete", 2)},
        train_data={"A": torch.zeros(4, dtype=torch.long)},
        test_data={"A": torch.zeros(4, dtype=torch.long)}, queries=[],
        family="discrete", problem_id="1", seed=0,
    )
    rows = ParamLearningMeasurement().measure(
        prob, _NoCalib(), [], fit_time_s=0.0, benchmark="synthetic", seed=0,
    )
    by = _by_metric(rows)
    for metric in ("calibration_pit_ks", "calibration_sd_ratio"):
        assert by[metric].status == "not_supported"

    # The real pgmpy-lg adapter indeed leaves the flag unset (future PR territory).
    from benchmarking.adapters import PgmpyAdapter
    assert not getattr(
        PgmpyAdapter(param_method="lg", inference_method="predict"),
        "supports_calibration", False,
    )


@pytest.mark.slow
def test_calibration_deterministic():
    prob = _problem("continuous_lg")
    adapter = _fitted(prob, "lg")
    m = ParamLearningMeasurement()
    first = _by_metric(_measure(m, prob, adapter))
    second = _by_metric(_measure(m, prob, adapter))
    for metric in ("calibration_pit_ks", "calibration_sd_ratio"):
        assert first[metric].value == second[metric].value, metric


@pytest.mark.slow
def test_oracle_cache_hits_across_baselines():
    # In-process cache scope: one measurement instance reused across baselines
    # builds the oracle once (it's baseline-independent). NOTE the production
    # runner is subprocess-per-cell, so this perf optimization spans cells only
    # in-process; results are identical either way because the draw is seeded.
    prob = _problem("continuous_lg")
    m = ParamLearningMeasurement()
    calls = {"n": 0}
    orig = m._build_calibration_context

    def _counting(problem):
        calls["n"] += 1
        return orig(problem)

    m._build_calibration_context = _counting
    # Two distinct calibration-capable baselines on the SAME problem instance.
    _measure(m, prob, _fitted(prob, "lg"))
    _measure(m, prob, _fitted(prob, "mdn"))
    assert calls["n"] == 1
