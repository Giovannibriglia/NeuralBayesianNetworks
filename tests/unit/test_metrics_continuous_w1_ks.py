"""Continuous posterior accuracy is W1 + KS at equal sample counts (D2, D3).

The former 256-bin histogram TV/JSD over ``[min, max]`` of both samples was
noise-dominated (two draws of the *same* distribution scored TV ~ 0.2) and a
single outlier stretched the bins until everything else shared a few of them.
Continuous scores also depended on how many posterior draws each method
returned (pyro 50, pgmpy 1024, nbn 2048).
"""
from __future__ import annotations

import torch

from nbn.bench.domains.base import Query
from nbn.bench.domains.posterior import Posterior
from nbn.bench.measurements import accuracy_timing as at
from nbn.bench.metrics import _compute_metrics_per_node, ks_distance
from nbn.bench.problems import SyntheticConfig, SyntheticProblemSource


def test_same_distribution_scores_near_zero():
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(2000, generator=g), torch.randn(2000, generator=g)
    res = _compute_metrics_per_node(sample_pairs=[(x, y)])
    # Histogram TV of these two samples was ~0.2; KS / W1 sit at their
    # O(1/sqrt(n)) floor.
    assert res["ks_per_node"] < 0.05
    assert res["w1_per_node"] < 0.08
    assert res["tv_per_node"] is None and res["jsd_per_node"] is None


def test_single_outlier_cannot_game_the_score():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2000, generator=g)
    y = torch.randn(2000, generator=g)
    y_out = y.clone()
    y_out[0] = 1e6
    base = _compute_metrics_per_node(sample_pairs=[(x, y)])["ks_per_node"]
    moved = _compute_metrics_per_node(sample_pairs=[(x, y_out)])["ks_per_node"]
    assert abs(moved - base) <= 1.0 / 2000 + 1e-12


def test_ks_distance_closed_form_disjoint_and_equal():
    assert ks_distance(torch.zeros(10), torch.ones(10)).value == 1.0
    a = torch.arange(100.0)
    assert ks_distance(a, a.flip(0)).value == 0.0


class _SamplesAdapter:
    name = "fixed-samples"

    def __init__(self, n):
        self.n = n

    def query(self, q):
        g = torch.Generator().manual_seed(7)
        return Posterior(samples=torch.randn(self.n, generator=g))


def test_scoring_uses_equal_sample_counts(monkeypatch):
    cfg = SyntheticConfig(families=["continuous_lg"], n_nodes_list=[4], seeds=[0],
                          n_train=100, n_test=10, n_reference=100)
    problem = next(iter(SyntheticProblemSource().iter_problems(cfg)))
    nodes = list(problem.variables)
    q = Query(targets=(nodes[-1],), evidence={nodes[0]: torch.tensor(0.3)},
              kind="marginal")
    seen = []
    real = at._compute_metrics_per_node

    def spy(**kw):
        seen.extend((a.numel(), b.numel()) for a, b in kw.get("sample_pairs", ()))
        return real(**kw)

    monkeypatch.setattr(at, "_compute_metrics_per_node", spy)
    m = at.AccuracyAndTiming(n_oracle_samples=300)
    for n in (50, 300, 2048):
        m.measure(problem, _SamplesAdapter(n), [q])
    assert seen == [(300, 300)] * 3
    # Deterministic: a rerun scores identically (seeded resample + oracle).
    r1 = [r.value for r in m.measure(problem, _SamplesAdapter(50), [q])
          if r.metric == "ks_per_node"]
    r2 = [r.value for r in m.measure(problem, _SamplesAdapter(50), [q])
          if r.metric == "ks_per_node"]
    assert r1 == r2 and r1[0] == r1[0]
