"""query_many: concurrent independent queries keep order and answers."""
import pytest
import torch

from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
from nbn.inference.parallel import query_many
from nbn.inference.tensor_ve import TensorVariableElimination
from tests.unit.test_conditioning import _dense_bn


@pytest.mark.parametrize("workers", [1, 4])
def test_query_many_matches_sequential_exact_answers(workers):
    m, names = _dense_bn(0, n=10, k=3, window=3, p=0.7)
    qs = [(names[i], {names[-1]: i % 3}) for i in range(6)] + [(names[2], {}, {names[0]: 1})]
    seq = [m.query([q[0]], evidence=q[1] or None, do=(q[2] if len(q) > 2 else None),
                   engine=TensorVariableElimination()) for q in qs]
    par = query_many(m, qs, engine=TensorVariableElimination(), workers=workers)
    for a, b in zip(seq, par):
        assert torch.allclose(a, b, atol=1e-6)


def test_query_many_sampling_engine_is_close_to_exact():
    m, names = _dense_bn(1, n=10, k=3, window=3, p=0.7)
    qs = [(names[i], {names[-1]: 1}) for i in range(5)]
    exact = [m.query([t], evidence=e, engine=TensorVariableElimination()) for t, e in qs]
    out = query_many(m, qs, engine=LikelihoodWeightingEngine(n_samples=50_000), workers=3)
    assert max(float((a - b).abs().max()) for a, b in zip(out, exact)) < 0.05


def test_query_many_raises_the_error_of_a_query():
    m, names = _dense_bn(0, n=6, k=3, window=2, p=0.7)
    with pytest.raises(KeyError):
        query_many(m, [(names[1], {}), ("not_a_node", {})], workers=2)
