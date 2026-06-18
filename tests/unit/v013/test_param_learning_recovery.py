"""ParamLearningMeasurement parameter-recovery rows (#109 PR 2, Stage 3).

These tests exercise the measurement-side gating taxonomy and the per-problem
weight cache WITHOUT any real adapter extraction (NBN.extract_learned_cpts
lands in Stage 4). Stubs stand in for adapters:
  * an adapter without ``supports_param_recovery`` -> not_supported rows;
  * a supported adapter on a continuous-family cell -> not_applicable;
  * a malformed true CPT (monkeypatched extraction) -> error;
  * the weight draw happens once per problem across baselines.
"""
from __future__ import annotations

import math

import pytest
import torch

from benchmarking.domains.base import BenchmarkProblem
from benchmarking.measurements import ParamLearningMeasurement


# ---- fixtures ---------------------------------------------------------------

def _bn_problem(family: str, *, seed: int = 0, n_nodes: int = 3) -> BenchmarkProblem:
    from benchmarking.synthetic import make_synthetic_bn

    bn = make_synthetic_bn(
        n_nodes=n_nodes, family=family, cardinality=3,
        edge_density=0.5, max_in_degree=2,
        n_train=200, n_test=100, n_reference=200,
        seed=seed, device="cpu",
    )
    return BenchmarkProblem(
        name=bn.name,
        dag=list(bn.dag.edges()),
        variables=bn.variable_specs,
        train_data=bn.train_data,
        test_data=bn.test_data,
        queries=[],
        true_model=bn.true_model,
        family=family,
        problem_id=str(n_nodes),
        seed=seed,
    )


def _zeros_score(test_data):
    n = next(iter(test_data.values())).shape[0]
    return torch.zeros(n)


class _NoRecovery:
    """Supports scoring but NOT parameter recovery."""
    name = "stub-noreco"
    supports_scoring = True

    def score_data(self, test_data):
        return _zeros_score(test_data)


class _SupportsRecoveryNoExtract:
    """Supports recovery; extract_learned_cpts must NOT be called (gated off)."""
    name = "stub-reco"
    supports_scoring = True
    supports_param_recovery = True

    def score_data(self, test_data):
        return _zeros_score(test_data)

    def extract_learned_cpts(self):
        raise AssertionError("extract_learned_cpts must not be called here")


class _PerfectRecovery:
    """Supports recovery; learned CPTs == true CPTs (perfect recovery)."""
    supports_scoring = True
    supports_param_recovery = True

    def __init__(self, name, true_model, variables):
        self.name = name
        self._true_model = true_model
        self._variables = variables

    def score_data(self, test_data):
        return _zeros_score(test_data)

    def extract_learned_cpts(self):
        from benchmarking.core.cpt_extraction import extract_discrete_cpts
        return extract_discrete_cpts(self._true_model, self._variables)


def _rows_by_metric(rows):
    return {r.metric: r for r in rows}


# ---- tests ------------------------------------------------------------------

def test_recovery_not_supported_without_flag():
    prob = _bn_problem("discrete")
    rows = ParamLearningMeasurement().measure(
        prob, _NoRecovery(), [], seed=prob.seed
    )
    by = _rows_by_metric(rows)
    assert by["log_likelihood"].status == "ok"          # LL unaffected
    for m in ("param_recovery_tv", "param_recovery_kl"):
        assert by[m].status == "not_supported"
        assert math.isnan(by[m].value)


def test_recovery_not_applicable_on_continuous_family():
    prob = _bn_problem("continuous_lg")
    # Supported adapter, but the cell is non-fully-discrete -> not_applicable,
    # and extract_learned_cpts must never run (it raises if it does).
    rows = ParamLearningMeasurement().measure(
        prob, _SupportsRecoveryNoExtract(), [], seed=prob.seed
    )
    by = _rows_by_metric(rows)
    assert by["log_likelihood"].status == "ok"          # LL still emits
    for m in ("param_recovery_tv", "param_recovery_kl"):
        assert by[m].status == "not_applicable"
        assert math.isnan(by[m].value)


def test_recovery_error_on_malformed_true_cpt(monkeypatch):
    prob = _bn_problem("discrete")
    node = next(iter(prob.variables))

    def _bad_extract(model, variables):
        # A row that sums to 0.9 — a buggy generator/loader signal.
        return {node: torch.tensor([[0.5, 0.4]])}

    monkeypatch.setattr(
        "benchmarking.measurements.param_learning.extract_discrete_cpts",
        _bad_extract,
    )
    rows = ParamLearningMeasurement().measure(
        prob, _SupportsRecoveryNoExtract(), [], seed=prob.seed
    )
    by = _rows_by_metric(rows)
    assert by["log_likelihood"].status == "ok"
    for m in ("param_recovery_tv", "param_recovery_kl"):
        assert by[m].status == "error"
        assert math.isnan(by[m].value)
        assert node in by[m].error_msg
        assert "0.9" in by[m].error_msg          # the offending row sum


def test_perfect_recovery_is_zero_and_ok():
    prob = _bn_problem("discrete")
    adapter = _PerfectRecovery("nbn-cat", prob.true_model, prob.variables)
    rows = ParamLearningMeasurement().measure(prob, adapter, [], seed=prob.seed)
    by = _rows_by_metric(rows)
    assert by["param_recovery_tv"].status == "ok"
    assert by["param_recovery_kl"].status == "ok"
    # learned == true -> TV and KL are exactly 0.
    assert math.isclose(by["param_recovery_tv"].value, 0.0, abs_tol=1e-6)
    assert math.isclose(by["param_recovery_kl"].value, 0.0, abs_tol=1e-6)


def test_weight_sample_drawn_once_per_problem():
    prob = _bn_problem("discrete")

    calls = {"n": 0}
    orig_sample = prob.true_model.sample

    def _counting_sample(*args, **kwargs):
        calls["n"] += 1
        return orig_sample(*args, **kwargs)

    prob.true_model.sample = _counting_sample

    m = ParamLearningMeasurement()
    a = _PerfectRecovery("nbn-cat", prob.true_model, prob.variables)
    b = _PerfectRecovery("nbn-neuralcat", prob.true_model, prob.variables)

    rows_a = m.measure(prob, a, [], seed=prob.seed)
    rows_b = m.measure(prob, b, [], seed=prob.seed)

    # Two baselines on the SAME problem -> the 20k draw happens once (cached).
    assert calls["n"] == 1
    # Both still produce ok recovery rows.
    for rows in (rows_a, rows_b):
        by = _rows_by_metric(rows)
        assert by["param_recovery_tv"].status == "ok"
        assert by["param_recovery_kl"].status == "ok"
