"""Credal query bounds (nbn.uncertainty.credal): soundness, exactness, PAC."""
from __future__ import annotations

import copy

import pytest
import torch

from nbn.inference.tensor_ve import TensorVariableElimination
from nbn.uncertainty.credal import (
    CredalSets, _interval_opt, _l1_opt, credal_query, pac_query, pac_sets)
from nbn.uncertainty.sets import IntervalSet, L1BallSet
from tests.unit.test_conditioning import _dense_bn


def _cpt(m, n):
    return torch.softmax(m.mechanisms[n]._logits.detach().double(), -1)


def _exact(m, tgt, ev, do=None):
    return TensorVariableElimination().query(m, [tgt], ev, do=do).double()


def _box(m, names, w, gen):
    out = {}
    for n in names:
        t = _cpt(m, n)
        lo = (t - w * torch.rand(t.shape, generator=gen, dtype=torch.float64)).clamp(0, 1)
        up = (t + w * torch.rand(t.shape, generator=gen, dtype=torch.float64)).clamp(0, 1)
        out[n] = IntervalSet(level=0.99, guarantee="pac",
                             unseen=torch.zeros(t.shape[0], dtype=torch.bool),
                             lower=lo, upper=up)
    return out


def _vertex(lo, up, gen):
    """A random vertex of {lo <= p <= up, sum p = 1} per row."""
    order = torch.argsort(torch.rand(lo.shape, generator=gen), -1)
    cap = (up - lo).gather(-1, order)
    budget = 1 - lo.sum(-1, keepdim=True)
    extra = torch.minimum(cap, (budget - (torch.cumsum(cap, -1) - cap)).clamp_min(0))
    return lo.clone().scatter_add_(-1, order, extra)


@pytest.mark.parametrize("use_do", [False, True])
def test_singleton_sets_give_the_exact_answer(use_do):
    m, names = _dense_bn(0, n=10, k=3, window=3, p=0.7)
    tgt, ev = names[3], {names[-1]: 1, names[7]: 0}
    do = {names[1]: 2} if use_do else None
    sets = {n: IntervalSet(level=1.0, guarantee="pac",
                           unseen=torch.zeros(_cpt(m, n).shape[0], dtype=torch.bool),
                           lower=_cpt(m, n), upper=_cpt(m, n)) for n in names}
    b = credal_query(m, sets, tgt, ev, do=do)
    ex = _exact(m, tgt, ev, do)
    assert torch.allclose(b.lower, ex, atol=1e-6) and torch.allclose(b.upper, ex, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bounds_contain_every_network_in_the_sets(seed):
    m, names = _dense_bn(seed, n=8, k=3, window=2, p=0.8)
    tgt, ev = names[2], {names[-1]: 1}
    gen = torch.Generator().manual_seed(seed)
    sets = _box(m, names, 0.1, gen)
    for do in (None, {names[1]: 0}):
        b = credal_query(m, sets, tgt, ev, do=do, point=False)
        for _ in range(15):
            m2 = copy.deepcopy(m)
            for n in names:
                lo, up = sets[n].bounds()
                m2.mechanisms[n]._logits.data = torch.log(
                    _vertex(lo, up, gen).clamp_min(1e-300)).float()
            p = _exact(m2, tgt, ev, do)
            assert ((p >= b.lower - 1e-6) & (p <= b.upper + 1e-6)).all()


def test_generalized_bayes_rule_never_widens():
    m, names = _dense_bn(1, n=8, k=3, window=2, p=0.8)
    sets = _box(m, names, 0.1, torch.Generator().manual_seed(0))
    tgt, ev = names[2], {names[-1]: 1}
    tight = credal_query(m, sets, tgt, ev, point=False)
    loose = credal_query(m, sets, tgt, ev, point=False, tight=False)
    assert (tight.lower >= loose.lower - 1e-12).all()
    assert (tight.upper <= loose.upper + 1e-12).all()


def test_inner_problems_match_brute_force():
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        k = 4
        g = torch.randn(k, generator=gen, dtype=torch.float64)
        c = torch.distributions.Dirichlet(torch.ones(k, dtype=torch.float64)).sample()
        lo, up = (c - 0.1).clamp(0, 1), (c + 0.15).clamp(0, 1)
        r = torch.rand((), generator=gen, dtype=torch.float64) * 0.8
        # brute force over a fine grid of the simplex
        pts = torch.distributions.Dirichlet(torch.ones(k, dtype=torch.float64) * 0.3) \
            .sample((200000,))
        box = ((pts >= lo) & (pts <= up)).all(-1)
        ball = (pts - c).abs().sum(-1) <= r
        vals = pts @ g
        for sel, opt in ((box, lambda mx: _interval_opt(g, lo, up, mx)),
                         (ball, lambda mx: _l1_opt(g, c, r.reshape(1), mx))):
            if sel.sum() == 0:
                continue
            assert float(opt(True)) >= float(vals[sel].max()) - 1e-9
            assert float(opt(False)) <= float(vals[sel].min()) + 1e-9
    # vertices attain the optimum: the greedy value is reached by a feasible point
    g = torch.tensor([0.3, -1.0, 2.0], dtype=torch.float64)
    lo = torch.tensor([0.1, 0.2, 0.0], dtype=torch.float64)
    up = torch.tensor([0.5, 0.6, 0.4], dtype=torch.float64)
    assert abs(float(_interval_opt(g, lo, up, True)) - (0.4 * 0.3 + 0.2 * -1 + 0.4 * 2)) < 1e-12


def test_l1_sets_are_sound():
    m, names = _dense_bn(2, n=7, k=3, window=2, p=0.8)
    tgt, ev = names[2], {names[-1]: 0}
    sets = {n: L1BallSet(level=0.99, guarantee="pac",
                         unseen=torch.zeros(_cpt(m, n).shape[0], dtype=torch.bool),
                         center=_cpt(m, n), radius=torch.full((_cpt(m, n).shape[0],), 0.1,
                                                              dtype=torch.float64))
            for n in names}
    b = credal_query(m, sets, tgt, ev)
    ex = _exact(m, tgt, ev)
    assert ((ex >= b.lower - 1e-9) & (ex <= b.upper + 1e-9)).all()
    assert b.max_width > 0


def test_level_is_a_union_bound_over_the_sets_used():
    m, names = _dense_bn(0, n=8, k=3, window=2, p=0.8)
    sets = _box(m, names, 0.05, torch.Generator().manual_seed(1))
    tgt, ev = names[2], {names[4]: 1}
    b = credal_query(m, sets, tgt, ev, point=False)
    assert b.level == pytest.approx(1 - 0.01 * len(b.nodes))
    assert b.guarantee == "pac_under_model"
    joint = CredalSets(sets, level=0.95, guarantee="pac_under_model", delta=0.05)
    assert credal_query(m, joint, tgt, ev, point=False).level == 0.95


def test_pac_query_covers_the_truth_and_shrinks_with_data():
    m, names = _dense_bn(3, n=8, k=3, window=2, p=0.8)
    tgt, ev, do = names[2], {names[-1]: 1}, {names[1]: 0}
    truth = _exact(m, tgt, ev)
    truth_do = _exact(m, tgt, ev, do)
    widths = []
    for n_rows in (2000, 50000):
        for r in range(4):
            torch.manual_seed(100 + r)
            data = {k: v.reshape(-1).long() for k, v in m.sample(n_rows).items()}
            b = pac_query(m, data, tgt, ev, delta=0.1)
            bd = pac_query(m, data, tgt, ev, do=do, delta=0.1)
            assert ((truth >= b.lower) & (truth <= b.upper)).all()
            assert ((truth_do >= bd.lower) & (truth_do <= bd.upper)).all()
            assert b.level == pytest.approx(0.9)
        widths.append(b.max_width)
    assert widths[1] < widths[0]


def test_pac_sets_hold_jointly_for_many_queries():
    m, names = _dense_bn(3, n=8, k=3, window=2, p=0.8)
    torch.manual_seed(0)
    data = {k: v.reshape(-1).long() for k, v in m.sample(20000).items()}
    sets = pac_sets(m, data, delta=0.1)
    assert sets.level == pytest.approx(0.9)
    for tgt, ev in ((names[2], {names[-1]: 1}), (names[5], {names[0]: 2}), (names[7], {})):
        b = credal_query(m, sets, tgt, ev, point=False)
        ex = _exact(m, tgt, ev or None)
        assert ((ex >= b.lower) & (ex <= b.upper)).all()
