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


def _problem(family: str, mechanism: str, *, seed: int = 1, n_nodes: int = 4,
             device: str = "cpu"):
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
    a = NBNAdapter(mechanism=mechanism, engine=None, device=device)
    a.fit(prob, epochs=8)
    return prob, a


def _assert_predictive_shape(prob, a):
    out = a.predictive_samples(prob.test_data)
    cont_nodes = [n for n, (k, _) in prob.variables.items() if k == "continuous"]
    assert set(out) == set(cont_nodes) and len(out) > 0
    n_test = next(iter(prob.test_data.values())).shape[0]
    for node, samp in out.items():
        assert tuple(samp.shape) == (n_test, a.N_CALIBRATION_SAMPLES)
        assert torch.isfinite(samp).all()


@pytest.mark.parametrize("mechanism", ["kde", "knn", "flexcode"])
def test_nonparametric_construction_no_keyerror(mechanism):
    # #223 / PR 8: the three non-parametric mechanisms are registered as valid
    # adapter mechanism strings (construction must not KeyError/ValueError).
    NBNAdapter(mechanism=mechanism, engine=None, device="cpu")


@pytest.mark.slow
@pytest.mark.parametrize("mechanism", ["lg", "mdn", "flow", "kde", "knn"])
def test_predictive_samples_shape_continuous(mechanism):
    # CPU-tractable mechanisms (kde/knn sub-second; flexcode is gpu-marked below).
    prob, a = _problem("continuous_lg", mechanism)
    _assert_predictive_shape(prob, a)


@pytest.mark.gpu
def test_predictive_samples_shape_flexcode_gpu():
    # FlexCode trains a 200-epoch MLP — CPU-intractable at benchmark scale
    # (#223); exercised on GPU only. CI skips via `-m "not gpu and not slow"`.
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the flexcode baseline")
    prob, a = _problem("continuous_lg", "flexcode", device="cuda")
    _assert_predictive_shape(prob, a)


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
