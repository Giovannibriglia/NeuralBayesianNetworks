"""kNN neighbour search: backends agree, ties follow ``tie_break``, dedup is exact."""
from __future__ import annotations

import io

import pytest
import torch

from nbn.mechanisms.non_parametric.knn_conditional import (
    KNNConditionalMechanism,
    _select_lowest_index,
)

pytest.importorskip("scipy")


def _fit(parents, y, **kw):
    m = KNNConditionalMechanism(**kw)
    m.fit_local(y, parents)
    return m


def _sets(idx):
    return [tuple(sorted(r.tolist())) for r in idx]


def test_select_lowest_index_rule():
    d = torch.tensor([[3.0, 1.0, 2.0, 2.0, 2.0, 0.5]])
    # k=3: 0.5 and 1.0 are in, then one of the three 2.0s -> the lowest index (2)
    assert _select_lowest_index(d, 3).tolist() == [[1, 2, 5]]
    assert _select_lowest_index(d, 4).tolist() == [[1, 2, 3, 5]]


@pytest.mark.parametrize("d_pa", [1, 2, 3])
def test_backends_agree_continuous_parents(d_pa):
    g = torch.Generator().manual_seed(d_pa)
    n = 5000
    pa = torch.randn(n, d_pa, generator=g)
    y = pa.sum(-1, keepdim=True) + 0.1 * torch.randn(n, 1, generator=g)
    q = torch.randn(700, d_pa, generator=g)
    brute = _fit(pa, y, search="brute")
    tree = _fit(pa, y, search="kdtree")
    old = _fit(pa, y, tie_break="any")
    xs = brute._std_parents(q)
    a = brute._knn_search(xs)
    b = tree._knn_search(xs)
    c = old._knn_search(xs)
    assert torch.equal(a, b)
    # exact float64 reference: the k nearest by a full sort
    d64 = torch.cdist(xs.double(), brute._train_pa.double(),
                      compute_mode="donot_use_mm_for_euclid_dist")
    ref = torch.sort(d64, dim=1, stable=True).indices[:, :brute._k_eff].sort(1).values
    assert torch.equal(a, ref)
    # the historical float32 search agrees except on rare near-ties (its
    # matmul-form distances lose a few digits)
    agree = sum(x == y for x, y in zip(_sets(a), _sets(c))) / len(_sets(a))
    assert agree > 0.95
    with torch.no_grad():
        yq = torch.randn(700, 1, generator=g)
        assert torch.allclose(brute.log_prob(yq, q), tree.log_prob(yq, q))
        same = torch.tensor([x == y for x, y in zip(_sets(a), _sets(c))])
        assert torch.allclose(brute.log_prob(yq, q)[same], old.log_prob(yq, q)[same],
                              atol=1e-5)


def test_ties_discrete_parents_backends_agree_and_rule_holds():
    g = torch.Generator().manual_seed(0)
    n = 6000
    pa = torch.randint(0, 3, (n, 2), generator=g).float()       # heavy ties
    y = pa.sum(-1, keepdim=True) + torch.randn(n, 1, generator=g)
    q = torch.randint(0, 3, (400, 2), generator=g).float()
    brute = _fit(pa, y, search="brute", k=50)
    tree = _fit(pa, y, search="kdtree", k=50)
    xs = brute._std_parents(q)
    a = brute._knn_search(xs)
    b = tree._knn_search(xs)
    assert torch.equal(a, b)
    # rule: all 50 neighbours are training rows equal to the query, lowest indices first
    for r in range(5):
        same = (pa == q[r]).all(-1).nonzero().squeeze(1)
        assert a[r].tolist() == same[:50].tolist()


def test_dedup_equals_rowwise_search():
    g = torch.Generator().manual_seed(1)
    pa = torch.randn(3000, 2, generator=g)
    y = torch.randn(3000, 1, generator=g)
    m = _fit(pa, y, search="brute")
    base = torch.randn(50, 2, generator=g)
    q = base[torch.randint(0, 50, (900,), generator=g)]          # many repeats
    xs = m._std_parents(q)
    a = m._knn_search(xs)
    b = torch.cat([m._knn_search_unique(xs[i:i + 1]) for i in range(xs.shape[0])])
    assert torch.equal(a, b)


def test_discrete_child_and_sampling_unchanged_by_backend():
    g = torch.Generator().manual_seed(2)
    pa = torch.randn(5000, 1, generator=g)
    y = (pa[:, 0] > 0).long()
    a = KNNConditionalMechanism(discrete_child=True, search="brute")
    b = KNNConditionalMechanism(discrete_child=True, search="kdtree")
    a.fit_local(y, pa)
    b.fit_local(y, pa)
    q = torch.randn(300, 1, generator=g)
    assert torch.allclose(a._categorical_probs(q, 300), b._categorical_probs(q, 300))


def test_pickle_after_tree_built():
    g = torch.Generator().manual_seed(3)
    pa = torch.randn(5000, 1, generator=g)
    m = _fit(pa, pa.clone(), search="kdtree")
    m._knn_search(m._std_parents(pa[:10]))
    assert m._kdtree is not None
    buf = io.BytesIO()
    torch.save(m, buf)
    buf.seek(0)
    m2 = torch.load(buf, weights_only=False)
    assert m2._kdtree is None
    assert torch.equal(m2._knn_search(m2._std_parents(pa[:10])),
                       m._knn_search(m._std_parents(pa[:10])))


def test_bad_options_rejected():
    with pytest.raises(ValueError):
        KNNConditionalMechanism(tie_break="random")
    with pytest.raises(ValueError):
        KNNConditionalMechanism(search="faiss")


@pytest.mark.parametrize("seed", range(6))
def test_two_stage_brute_equals_exact_rule(seed):
    g = torch.Generator().manual_seed(seed)
    n, d = 2500, 1 + seed % 3
    pa = torch.randn(n, d, generator=g)
    if seed % 2:
        pa = (pa * 4).round() / 4          # exact ties and many near-ties
    m = KNNConditionalMechanism(search="brute", k=37 + seed)
    m.fit_local(torch.randn(n, 1, generator=g), pa)
    xs = m._std_parents(torch.randn(300, d, generator=g) if seed % 2 == 0
                        else pa[torch.randint(0, n, (300,), generator=g)])
    assert torch.equal(m._brute_lowest_index(xs), m._exact_rows(xs))
