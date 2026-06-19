"""NBNAdapter.predictive_samples — calibration capability (#109 PR 7, Stage A).

Verifies the adapter-side predictive sampling that the calibration metrics
consume: shape per continuous node, empty dict on discrete-only problems, and
determinism within a seeded scope (the measurement seeds it for reproducibility).
"""
from __future__ import annotations

import pytest
import torch

from benchmarking.adapters import NBNAdapter
from benchmarking.domains.base import BenchmarkProblem


def _problem(family: str, mechanism: str, *, seed: int = 1, n_nodes: int = 4):
    from benchmarking.synthetic import make_synthetic_bn

    bn = make_synthetic_bn(
        n_nodes=n_nodes, family=family, cardinality=3, edge_density=0.5,
        max_in_degree=2, n_train=400, n_test=120, n_reference=200,
        seed=seed, device="cpu",
    )
    prob = BenchmarkProblem(
        name=bn.name, dag=list(bn.dag.edges()), variables=bn.variable_specs,
        train_data=bn.train_data, test_data=bn.test_data, queries=[],
        true_model=bn.true_model, family=family, problem_id=str(n_nodes), seed=seed,
    )
    a = NBNAdapter(mechanism=mechanism, engine=None, device="cpu")
    a.fit(prob, epochs=8)
    return prob, a


@pytest.mark.slow
@pytest.mark.parametrize("mechanism", ["lg", "mdn"])
def test_predictive_samples_shape_continuous(mechanism):
    prob, a = _problem("continuous_lg", mechanism)
    out = a.predictive_samples(prob.test_data)
    cont_nodes = [n for n, (k, _) in prob.variables.items() if k == "continuous"]
    assert set(out) == set(cont_nodes) and len(out) > 0
    n_test = next(iter(prob.test_data.values())).shape[0]
    for node, samp in out.items():
        assert tuple(samp.shape) == (n_test, a.N_CALIBRATION_SAMPLES)
        assert torch.isfinite(samp).all()


@pytest.mark.slow
def test_predictive_samples_empty_on_discrete():
    prob, a = _problem("discrete", "cat")
    # No continuous nodes -> empty dict (the measurement gates this not_applicable).
    assert a.predictive_samples(prob.test_data) == {}


@pytest.mark.slow
def test_predictive_samples_deterministic_within_seed():
    prob, a = _problem("continuous_lg", "lg")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        first = a.predictive_samples(prob.test_data)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        second = a.predictive_samples(prob.test_data)
    assert set(first) == set(second)
    for node in first:
        assert torch.equal(first[node], second[node]), node
