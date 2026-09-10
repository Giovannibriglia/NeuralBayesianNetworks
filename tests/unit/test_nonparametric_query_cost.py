"""Query-cost paths of the non-parametric mechanisms (#251).

Three opt-in / transparent accelerations, each pinned against the reference
path it must reproduce:

* ConditionalKDE ``n_neighbors``: the truncated Nadaraya--Watson sums.
  ``k >= N`` takes the exact code path (byte-identical); ``k < N`` must be a
  close approximation when ``k`` covers the parent kernel's support, and a
  finite, shape-correct density otherwise.
* KNNConditional neighbour memoisation: identical answers to a cache-free
  instance, one search per distinct parent batch, cache invalidated by
  fit / update_local / load_state_dict.
* FlexCode ``dedup_parents``: the coefficient + normaliser work done once per
  distinct parent row must equal the per-row reference.
"""
from __future__ import annotations

import torch

from nbn.mechanisms.non_parametric.conditional_kde import ConditionalKDEMechanism
from nbn.mechanisms.non_parametric.flexcode import FlexCodeMechanism
from nbn.mechanisms.non_parametric.knn_conditional import KNNConditionalMechanism


def _gen(n, *, seed, d_pa=2):
    g = torch.Generator().manual_seed(seed)
    pa = torch.randn(n, d_pa, generator=g)
    y = 0.8 * pa[:, :1] - 0.3 * pa[:, 1:2] + 0.2 * torch.randn(n, 1, generator=g)
    label = (y[:, 0] > 0).long() + (pa[:, 0] > 1.0).long()
    return pa, y, label


def _query(n=64, seed=9, d_pa=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d_pa, generator=g), torch.randn(n, 1, generator=g)


# ── ConditionalKDE: neighbour truncation ─────────────────────────────────────

class TestKDETruncation:
    def test_rejects_non_positive_k(self):
        import pytest
        with pytest.raises(ValueError):
            ConditionalKDEMechanism(n_neighbors=0)

    def test_k_at_least_n_is_byte_identical(self):
        pa, y, _ = _gen(300, seed=1)
        qpa, qy = _query(50)
        exact = ConditionalKDEMechanism(train_chunk=97)
        exact.fit_local(y, pa)
        for k in (300, 1000):
            trunc = ConditionalKDEMechanism(train_chunk=97, n_neighbors=k)
            trunc.fit_local(y, pa)
            assert trunc._truncation_k() is None
            assert torch.equal(trunc.log_prob(qy, qpa), exact.log_prob(qy, qpa))

    def test_truncated_close_to_exact_when_k_covers_kernel(self):
        # With k = N-1 exactly one row (the farthest, kernel weight ~0) is
        # dropped per query, so even tail queries agree to 1e-5.  With
        # k = N/2 the dropped half sits beyond ~1.3 parent std, i.e. several
        # bandwidths out: in-distribution queries agree to ~1e-3 in log
        # density (an approximation bound, not exactness — far-tail queries
        # where the omitted rows carry a visible share can differ by ~0.04).
        pa, y, _ = _gen(400, seed=2)
        qpa, qy = _query(64)
        exact = ConditionalKDEMechanism(train_chunk=64, query_chunk=7)
        exact.fit_local(y, pa)
        ref = exact.log_prob(qy, qpa)
        near = ConditionalKDEMechanism(train_chunk=64, query_chunk=7, n_neighbors=399)
        near.fit_local(y, pa)
        assert torch.allclose(near.log_prob(qy, qpa), ref, atol=1e-5, rtol=1e-5)
        half = ConditionalKDEMechanism(train_chunk=64, query_chunk=7, n_neighbors=200)
        half.fit_local(y, pa)
        in_pa, in_y = pa[:64], y[:64] + 0.1        # queries from the data bulk
        assert torch.allclose(
            half.log_prob(in_y, in_pa), exact.log_prob(in_y, in_pa), atol=2e-3, rtol=1e-3,
        )

    def test_neighbour_search_matches_brute_force(self):
        pa, y, _ = _gen(500, seed=3)
        qpa, _ = _query(40)
        mech = ConditionalKDEMechanism(train_chunk=33, query_chunk=11, n_neighbors=16)
        mech.fit_local(y, pa)
        xs = mech._std_parents(qpa)
        idx, d2 = mech._topk_neighbours(xs, 16)
        # Brute force in the same bandwidth-scaled space.
        h = mech._h
        full = ((xs / h).unsqueeze(1) - (mech._train_pa / h).unsqueeze(0)).pow(2).sum(-1)
        ref_d2, ref_idx = torch.topk(full, 16, largest=False, dim=1, sorted=True)
        assert torch.allclose(d2, ref_d2, atol=1e-6)
        # Index sets agree (ties could permute equal distances).
        assert torch.equal(idx.sort(dim=1).values, ref_idx.sort(dim=1).values)

    def test_small_k_finite_normalised_and_sample_shapes(self):
        pa, y, _ = _gen(600, seed=4)
        mech = ConditionalKDEMechanism(n_neighbors=8)
        mech.fit_local(y, pa)
        qpa, qy = _query(32)
        lp = mech.log_prob(qy, qpa)
        assert lp.shape == (32,) and torch.isfinite(lp).all()
        # Grid integral of the conditional at one parent row ≈ 1.
        xs = torch.linspace(-4, 4, 4001).unsqueeze(-1)
        dens = mech.log_prob(xs, qpa[:1].expand(4001, -1)).exp()
        assert abs(torch.trapz(dens, xs.squeeze(-1)).item() - 1.0) < 2e-2
        # [B, S, D] log_prob and sampling shapes.
        lp3 = mech.log_prob(torch.randn(5, 7, 1), torch.randn(5, 7, 2))
        assert lp3.shape == (5, 7)
        s = mech.sample(qpa, n=13)
        assert s.shape == (32, 13, 1) and torch.isfinite(s).all()
        # The truncated sampler draws from the truncated density: its sample
        # mean matches the mean of the truncated log_prob at the same parent
        # row (integrated on the grid), not merely something finite.
        torch.manual_seed(0)
        for row in range(3):
            grid_mean = (xs.squeeze(-1) * mech.log_prob(
                xs, qpa[row:row + 1].expand(4001, -1)).exp()).sum() * (8.0 / 4000)
            samp_mean = mech.sample(qpa[row:row + 1], n=8000).mean()
            assert abs(samp_mean.item() - grid_mean.item()) < 0.05

    def test_truncation_respects_zero_weight_rows(self):
        pa, y, _ = _gen(200, seed=5)
        w = torch.ones(200)
        w[:100] = 0.0                       # first half is deleted by weight
        mech = ConditionalKDEMechanism(n_neighbors=199)
        mech.fit_local(y, pa, weights=w)
        ref = ConditionalKDEMechanism()
        ref.fit_local(y[100:], pa[100:], weights=w[100:])
        qpa, qy = pa[100:120], y[100:120] - 0.05
        # The truncated weighted sums carry log w_i on the gathered rows, so a
        # zero-weight neighbour drops out exactly (-inf in both logsumexps),
        # as in the exact estimator; only the farthest row is truncated.
        assert torch.allclose(mech.log_prob(qy, qpa), ref.log_prob(qy, qpa), atol=1e-4)

    def test_root_node_always_exact(self):
        _, y, _ = _gen(100, seed=6)
        mech = ConditionalKDEMechanism(n_neighbors=5)
        mech.fit_local(y, None)
        assert mech._truncation_k() is None
        assert torch.isfinite(mech.log_prob(torch.randn(9, 1), None)).all()

    def test_update_local_after_query_matches_pooled_fit(self):
        pa_a, y_a, _ = _gen(300, seed=1)
        pa_b, y_b, _ = _gen(200, seed=2)
        qpa, qy = _query(30)
        chunked = ConditionalKDEMechanism(n_neighbors=64, train_chunk=97)
        chunked.fit_local(y_a, pa_a)
        chunked.log_prob(qy, qpa)              # a query before the update
        chunked.update_local(y_b, pa_b)
        pooled = ConditionalKDEMechanism(n_neighbors=64, train_chunk=97)
        pooled.fit_local(torch.cat([y_a, y_b]), torch.cat([pa_a, pa_b]))
        assert torch.equal(chunked.log_prob(qy, qpa), pooled.log_prob(qy, qpa))


# ── KNNConditional: neighbour memoisation ────────────────────────────────────

class TestKNNMemoisation:
    def _pair(self, **kw):
        pa, y, label = _gen(400, seed=7)
        cached = KNNConditionalMechanism(**kw)
        fresh_kw = {k: v for k, v in kw.items() if k != "neighbour_cache_size"}
        fresh = KNNConditionalMechanism(neighbour_cache_size=0, **fresh_kw)
        return pa, y, label, cached, fresh

    def test_continuous_matches_cache_free_instance(self, monkeypatch):
        pa, y, _, cached, fresh = self._pair(k=12, query_chunk=37)
        cached.fit_local(y, pa)
        fresh.fit_local(y, pa)
        qpa, qy = _query(64)
        calls = {"n": 0}
        orig = KNNConditionalMechanism._knn_search

        def _spy(self, xs):
            calls["n"] += 1
            return orig(self, xs)
        monkeypatch.setattr(KNNConditionalMechanism, "_knn_search", _spy)

        lp = cached.log_prob(qy, qpa)
        torch.manual_seed(0)
        s1 = cached.sample(qpa, n=5)
        assert calls["n"] == 1, "second call on the same parents must hit the cache"
        assert torch.equal(lp, fresh.log_prob(qy, qpa))
        torch.manual_seed(0)
        assert torch.equal(s1, fresh.sample(qpa, n=5))
        assert calls["n"] == 3                  # the fresh instance searched twice

        # forward(pa) on NEW parents, then log_prob + sample: one search.
        calls["n"] = 0
        q2, y2 = _query(64, seed=21)
        d = cached.forward(q2)
        d.log_prob(y2)
        d.sample()
        assert calls["n"] == 1

    def test_discrete_matches_cache_free_instance(self, monkeypatch):
        pa, _, label, cached, fresh = self._pair(k=9, discrete_child=True)
        cached.fit_local(label.float(), pa, n_classes=3)
        fresh.fit_local(label.float(), pa, n_classes=3)
        qpa, _ = _query(48)
        calls = {"n": 0}
        orig = KNNConditionalMechanism._knn_search

        def _spy(self, xs):
            calls["n"] += 1
            return orig(self, xs)
        monkeypatch.setattr(KNNConditionalMechanism, "_knn_search", _spy)
        p1 = cached.forward(qpa).probs
        p2 = cached.forward(qpa).probs
        assert calls["n"] == 1
        assert torch.equal(p1, p2)
        assert torch.equal(p1, fresh.forward(qpa).probs)

    def test_different_parents_do_not_collide(self):
        pa, y, _, cached, fresh = self._pair(k=10)
        cached.fit_local(y, pa)
        fresh.fit_local(y, pa)
        q1, y1 = _query(20, seed=1)
        q2, y2 = _query(20, seed=2)
        q3 = q1.clone()
        q3[0, 0] += 1e-3                        # one entry differs
        cached.log_prob(y1, q1)
        for q, yy in ((q2, y2), (q3, y1), (q1, y1)):
            assert torch.equal(cached.log_prob(yy, q), fresh.log_prob(yy, q))

    def test_cache_is_bounded(self):
        pa, y, _, cached, _ = self._pair(k=5, neighbour_cache_size=2)
        cached.fit_local(y, pa)
        for seed in range(5):
            qpa, qy = _query(8, seed=seed)
            cached.log_prob(qy, qpa)
        assert len(cached._nbr_cache) == 2

    def test_invalidated_by_fit_update_and_state_dict(self):
        pa_a, y_a, _ = _gen(300, seed=1)
        pa_b, y_b, _ = _gen(200, seed=2)
        qpa, qy = _query(30)
        m = KNNConditionalMechanism(k=7)
        m.fit_local(y_a, pa_a)
        m.log_prob(qy, qpa)
        assert len(m._nbr_cache) == 1
        m.update_local(y_b, pa_b)
        assert m._nbr_cache == []
        pooled = KNNConditionalMechanism(k=7, neighbour_cache_size=0)
        pooled.fit_local(torch.cat([y_a, y_b]), torch.cat([pa_a, pa_b]))
        assert torch.equal(m.log_prob(qy, qpa), pooled.log_prob(qy, qpa))
        # Loading a different sample of the same shape must not reuse the
        # neighbours of the old one.
        other = KNNConditionalMechanism(k=7, neighbour_cache_size=0)
        pa_c, y_c, _ = _gen(500, seed=3)
        other.fit_local(y_c, pa_c)
        m.load_state_dict(other.state_dict())
        assert m._nbr_cache == []
        assert torch.equal(m.log_prob(qy, qpa), other.log_prob(qy, qpa))
        # refit clears too
        m.fit_local(y_a, pa_a)
        assert m._nbr_cache == []

    def test_root_continuous_chunked_marginal_matches_dense(self):
        _, y, _ = _gen(230, seed=8)
        chunked = KNNConditionalMechanism(query_chunk=17)
        chunked.fit_local(y, None)
        dense = KNNConditionalMechanism(query_chunk=100000)
        dense.fit_local(y, None)
        qy = torch.randn(41, 1)
        assert torch.allclose(chunked.log_prob(qy, None), dense.log_prob(qy, None), atol=1e-6)
        assert torch.allclose(dense.log_prob(qy, None), _dense_root_kde(dense, qy), atol=1e-5)


def _dense_root_kde(mech, y):
    import math
    ty, b = mech._train_y, mech._b_global
    dy = (y.unsqueeze(1) - ty.unsqueeze(0)) / b
    logk = (-0.5 * dy.pow(2)).sum(-1) - (torch.log(b) + 0.5 * math.log(2 * math.pi)).sum()
    return torch.logsumexp(logk, dim=1) - math.log(ty.shape[0])


# ── FlexCode: normaliser per distinct parent row ─────────────────────────────

class TestFlexCodeDedup:
    def _fit(self, **kw):
        torch.manual_seed(0)
        pa, y, _ = _gen(300, seed=11)
        m = FlexCodeMechanism(epochs=3, **kw)
        m.fit_local(y, pa)
        return m

    def test_expanded_parents_match_reference(self):
        m = self._fit()
        x = torch.randn(6, 9, 1)
        pa = torch.randn(6, 2)
        lp = m.log_prob(x, pa)
        m.dedup_parents = False
        ref = m.log_prob(x, pa)
        assert lp.shape == (6, 9)
        assert torch.allclose(lp, ref, atol=1e-5, rtol=1e-5)

    def test_flat_duplicate_rows_match_reference(self):
        m = self._fit()
        base = torch.randn(5, 2)
        pa = base.repeat_interleave(4, dim=0)[torch.randperm(20)]   # 20 rows, 5 distinct
        x = torch.randn(20, 1)
        lp = m.log_prob(x, pa)
        m.dedup_parents = False
        assert torch.allclose(lp, m.log_prob(x, pa), atol=1e-5, rtol=1e-5)
        # 3-D parents with duplicates across particles.
        pa3 = base.unsqueeze(1).expand(5, 4, 2).clone()
        pa3[2, 1] = base[0]
        x3 = torch.randn(5, 4, 1)
        m.dedup_parents = True
        lp3 = m.log_prob(x3, pa3)
        m.dedup_parents = False
        assert torch.allclose(lp3, m.log_prob(x3, pa3), atol=1e-5, rtol=1e-5)

    def test_nan_parent_rows_dedup_like_sanitised(self):
        m = self._fit()
        pa = torch.zeros(4, 2)
        pa[1, 0] = float("nan")          # sanitised to 0 → equal to row 0
        x = torch.randn(4, 1)
        lp = m.log_prob(x, pa)
        m.dedup_parents = False
        assert torch.allclose(lp, m.log_prob(x, pa), atol=1e-6)
        assert torch.isfinite(lp).all()

    def test_root_and_single_particle_unchanged(self):
        torch.manual_seed(0)
        _, y, _ = _gen(200, seed=12)
        root = FlexCodeMechanism()
        root.fit_local(y, None)
        q = torch.randn(7, 1)
        a = root.log_prob(q, None)
        root.dedup_parents = False
        assert torch.equal(a, root.log_prob(q, None))
        m = self._fit()
        pa = torch.randn(7, 2)
        a = m.log_prob(q, pa)              # s == 1: no dedup path
        m.dedup_parents = False
        assert torch.equal(a, m.log_prob(q, pa))

    def test_grad_flows_through_dedup(self):
        m = self._fit()
        x = torch.randn(3, 5, 1)
        pa = torch.randn(3, 2)
        m.log_prob(x, pa).sum().backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in m.net.parameters())
