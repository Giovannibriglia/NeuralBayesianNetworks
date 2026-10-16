"""SparseCategoricalTableMechanism answers exactly what the dense table answers."""
from __future__ import annotations

import pytest
import torch

from nbn.core.network import NeuralBayesianNetwork
from nbn.inference.tensor_ve import TensorVariableElimination
from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism
from nbn.mechanisms.parametric.sparse_categorical_table import (
    SparseCategoricalTableMechanism,
)


def _data(n=300, p=4, k=3, seed=0, unseen_class=False):
    g = torch.Generator().manual_seed(seed)
    pa = torch.randint(0, k, (n, p), generator=g)
    x = (pa.sum(-1) + torch.randint(0, 2, (n,), generator=g)) % k
    if unseen_class:
        x = x.clamp(max=k - 2)  # class k-1 never observed
    return x, pa


@pytest.mark.parametrize("alpha", [0.0, 1.0])
@pytest.mark.parametrize("unseen_class", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_sparse_equals_dense(alpha, unseen_class, weighted):
    k, p = 3, 6
    x, pa = _data(p=p, k=k, unseen_class=unseen_class)
    w = (torch.arange(x.shape[0]) % 3).double() if weighted else None
    d, s = CategoricalTableMechanism(alpha), SparseCategoricalTableMechanism(alpha)
    d.fit_local(x, pa, parent_cards=[k] * p, n_classes=k, weights=w)
    info = s.fit_local(x, pa, parent_cards=[k] * p, n_classes=k, weights=w)
    assert info["n_stored_rows"] <= x.shape[0] and s.n_stored_rows < k ** p
    # every parent configuration, seen or not
    grid = torch.cartesian_prod(*[torch.arange(k)] * p)
    for c in range(k):
        xc = torch.full((grid.shape[0],), c)
        assert torch.allclose(d.log_prob(xc, grid), s.log_prob(xc, grid), atol=1e-6)
    assert torch.allclose(torch.softmax(d.tabulate(), -1), torch.softmax(s.tabulate(), -1),
                          atol=1e-6)
    assert torch.allclose(d.cpt, s.cpt, atol=1e-6)


def test_sparse_root_and_sampling_law():
    k = 3
    x, _ = _data(n=2000, p=1, k=k)
    s = SparseCategoricalTableMechanism()
    s.fit_local(x, None, n_classes=k)
    torch.manual_seed(0)
    draws = s.sample(None, n=50_000).reshape(-1).long()
    freq = torch.bincount(draws, minlength=k).float() / draws.numel()
    assert (freq - s.cpt[0]).abs().max() < 0.01
    # unseen parent rows sample uniformly
    x2, pa2 = _data(n=50, p=6, k=k, seed=3)
    s2 = SparseCategoricalTableMechanism()
    s2.fit_local(x2, pa2, parent_cards=[k] * 6, n_classes=k)
    q = torch.full((30_000, 6), 2)
    q[:, 0] = 0
    keyset = set(s2._keys.tolist())
    assert int((q[0] * torch.tensor(s2._parent_strides)).sum()) not in keyset
    v = s2.sample(q, 1).reshape(-1).long()
    assert ((torch.bincount(v, minlength=k).float() / v.numel()) - 1 / k).abs().max() < 0.02


def test_sparse_in_a_network_matches_dense_queries():
    k, n = 3, 6
    names = [f"V{i}" for i in range(n)]
    edges = [(names[i], names[j]) for j in range(n) for i in range(j) if (i + j) % 2 == 0 or j - i == 1]
    g = torch.Generator().manual_seed(1)
    data = {}
    for j, v in enumerate(names):
        pa = [a for a, b in edges if b == v]
        base = sum(data[a] for a in pa) if pa else torch.zeros(400, dtype=torch.long)
        data[v] = (base + torch.randint(0, 2, (400,), generator=g)) % k

    def model(cls):
        m = NeuralBayesianNetwork(edges, variables={v: ("discrete", k) for v in names})
        for v in names:
            m.set_mechanism(v, cls())
        m.fit(data)
        return m

    md, ms = model(CategoricalTableMechanism), model(SparseCategoricalTableMechanism)
    eng = TensorVariableElimination()
    ev = {"V1": 1, "V4": 0}
    a = md.query(["V5"], evidence=ev, engine=eng)
    b = ms.query(["V5"], evidence=ev, engine=TensorVariableElimination())
    assert torch.allclose(a, b, atol=1e-6)
    test = {k_: v[:50].reshape(-1, 1) for k_, v in data.items()}
    assert torch.allclose(md.log_prob(test), ms.log_prob(test), atol=1e-5)


def test_sparse_unsupported_paths_raise():
    s = SparseCategoricalTableMechanism()
    with pytest.raises(NotImplementedError):
        s.update_local(torch.zeros(3), None)
    with pytest.raises(NotImplementedError):
        s.fit_from_counts(torch.zeros(1, 2), [])
