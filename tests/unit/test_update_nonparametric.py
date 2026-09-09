"""Exact-updater equivalence for the non-parametric mechanisms.

A KDE or kNN estimator *is* its sample, so ``fit_local(A)`` followed by
``update_local(B)`` must reproduce ``fit_local(A | B)``: same buffers, same
``log_prob``.  These tests pin that equivalence for the mechanisms directly
and through the public ``model.fit`` / ``model.update`` API, plus the KDE's
forgetting semantics (a faded row equals a down-weighted row) and the kNN's
refusals.
"""
import pytest
import torch

from nbn import NeuralBayesianNetwork as NBN
from nbn.mechanisms.non_parametric.conditional_kde import ConditionalKDEMechanism
from nbn.mechanisms.non_parametric.knn_conditional import KNNConditionalMechanism
from nbn.mechanisms.parametric.linear_gaussian import LinearGaussianMechanism


def _gen(n, *, seed, d_pa=2):
    g = torch.Generator().manual_seed(seed)
    pa = torch.randn(n, d_pa, generator=g)
    y = 0.8 * pa[:, :1] - 0.3 * pa[:, 1:2] + 0.2 * torch.randn(n, 1, generator=g)
    label = (y[:, 0] > 0).long() + (pa[:, 0] > 1.0).long()      # classes 0..2
    return pa, y, label


def _query(n=64, seed=9):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 2, generator=g), torch.randn(n, 1, generator=g)


def _kde(**kw):
    return ConditionalKDEMechanism(bw_factor=0.5, train_chunk=97, query_chunk=13, **kw)


class TestKDEAppend:
    def test_fit_then_update_equals_pooled_fit(self):
        pa_a, y_a, _ = _gen(300, seed=1)
        pa_b, y_b, _ = _gen(200, seed=2)
        chunked = _kde()
        chunked.fit_local(y_a, pa_a)
        info = chunked.update_local(y_b, pa_b)
        pooled = _kde()
        pooled.fit_local(torch.cat([y_a, y_b]), torch.cat([pa_a, pa_b]))
        assert info["method"] == "kde_append" and info["n_train"] == 500 and info["n_new"] == 200
        for name in ("_train_y", "_train_pa", "_h", "_b", "_pa_mean", "_pa_std"):
            assert torch.allclose(getattr(chunked, name), getattr(pooled, name), atol=1e-5), name
        assert chunked._train_logw is None
        qpa, qy = _query()
        assert torch.allclose(chunked.log_prob(qy, qpa), pooled.log_prob(qy, qpa), atol=1e-4)

    def test_three_chunks_equal_pooled(self):
        parts = [_gen(n, seed=s) for n, s in ((120, 3), (80, 4), (150, 5))]
        chunked = _kde()
        chunked.fit_local(parts[0][1], parts[0][0])
        for pa, y, _ in parts[1:]:
            chunked.update_local(y, pa)
        pooled = _kde()
        pooled.fit_local(torch.cat([p[1] for p in parts]), torch.cat([p[0] for p in parts]))
        qpa, qy = _query()
        assert torch.allclose(chunked.log_prob(qy, qpa), pooled.log_prob(qy, qpa), atol=1e-4)

    def test_root_node_update(self):
        _, y_a, _ = _gen(100, seed=6)
        _, y_b, _ = _gen(60, seed=7)
        chunked = _kde()
        chunked.fit_local(y_a, None)
        chunked.update_local(y_b, None)
        pooled = _kde()
        pooled.fit_local(torch.cat([y_a, y_b]), None)
        _, qy = _query()
        assert torch.allclose(chunked.log_prob(qy, None), pooled.log_prob(qy, None), atol=1e-5)

    def test_forgetting_equals_weighted_pooled_fit(self):
        pa_a, y_a, _ = _gen(300, seed=1)
        pa_b, y_b, _ = _gen(200, seed=2)
        chunked = _kde()
        chunked.fit_local(y_a, pa_a)
        chunked.update_local(y_b, pa_b, forgetting=0.5)
        weighted = _kde()
        w = torch.cat([torch.full((300,), 0.5), torch.ones(200)])
        weighted.fit_local(torch.cat([y_a, y_b]), torch.cat([pa_a, pa_b]), weights=w)
        qpa, qy = _query()
        assert torch.allclose(chunked.log_prob(qy, qpa), weighted.log_prob(qy, qpa), atol=1e-4)
        # a second faded update compounds: old rows at 0.25, middle at 0.5, new at 1
        pa_c, y_c, _ = _gen(100, seed=8)
        chunked.update_local(y_c, pa_c, forgetting=0.5)
        w = torch.cat([torch.full((300,), 0.25), torch.full((200,), 0.5), torch.ones(100)])
        weighted = _kde()
        weighted.fit_local(torch.cat([y_a, y_b, y_c]), torch.cat([pa_a, pa_b, pa_c]), weights=w)
        assert torch.allclose(chunked.log_prob(qy, qpa), weighted.log_prob(qy, qpa), atol=1e-4)

    def test_new_row_weights_are_multiplicities(self):
        """Integer weights on the update rows equal appending replicated rows."""
        pa_a, y_a, _ = _gen(100, seed=1)
        pa_b, y_b, _ = _gen(40, seed=2)
        chunked = _kde()
        chunked.fit_local(y_a, pa_a)
        chunked.update_local(y_b, pa_b, weights=torch.full((40,), 2.0))
        replicated = _kde()
        replicated.fit_local(torch.cat([y_a, y_b, y_b]), torch.cat([pa_a, pa_b, pa_b]))
        qpa, qy = _query()
        assert torch.allclose(chunked.log_prob(qy, qpa), replicated.log_prob(qy, qpa), atol=1e-4)

    def test_new_row_weights_equal_weighted_pooled_fit(self):
        pa_a, y_a, _ = _gen(100, seed=1)
        pa_b, y_b, _ = _gen(40, seed=2)
        chunked = _kde()
        chunked.fit_local(y_a, pa_a)
        chunked.update_local(y_b, pa_b, weights=torch.full((40,), 2.0))
        weighted = _kde()
        weighted.fit_local(torch.cat([y_a, y_b]), torch.cat([pa_a, pa_b]),
                           weights=torch.cat([torch.ones(100), torch.full((40,), 2.0)]))
        qpa, qy = _query()
        assert torch.allclose(chunked.log_prob(qy, qpa), weighted.log_prob(qy, qpa), atol=1e-4)

    def test_guards(self):
        pa_a, y_a, _ = _gen(50, seed=1)
        m = _kde()
        with pytest.raises(AssertionError):
            m.update_local(y_a, pa_a)
        m.fit_local(y_a, pa_a)
        with pytest.raises(ValueError):
            m.update_local(y_a, pa_a[:, :1])          # parent dim changed
        with pytest.raises(ValueError):
            m.update_local(y_a, None)                 # became a root
        with pytest.raises(ValueError):
            m.update_local(y_a, pa_a, forgetting=0.0)


class TestKNNAppend:
    def test_discrete_child_equals_pooled(self):
        pa_a, _, c_a = _gen(300, seed=1)
        pa_b, _, c_b = _gen(200, seed=2)
        chunked = KNNConditionalMechanism(discrete_child=True, query_chunk=17)
        chunked.fit_local(c_a, pa_a, n_classes=3)
        info = chunked.update_local(c_b, pa_b, n_classes=3)
        pooled = KNNConditionalMechanism(discrete_child=True, query_chunk=17)
        pooled.fit_local(torch.cat([c_a, c_b]), torch.cat([pa_a, pa_b]), n_classes=3)
        assert info["method"] == "knn_append" and info["k"] == pooled._k_eff == round(500 ** 0.5)
        qpa, _ = _query()
        assert torch.allclose(chunked._categorical_probs(qpa, 64), pooled._categorical_probs(qpa, 64))

    def test_class_first_seen_in_update_grows_the_table(self):
        pa_a, _, c_a = _gen(200, seed=1)
        pa_b, _, c_b = _gen(100, seed=2)
        m = KNNConditionalMechanism(discrete_child=True)
        m.fit_local(c_a.clamp_max(1), pa_a)          # classes {0, 1} only, nothing declared
        assert m._n_classes == 2
        m.update_local(c_b, pa_b)                    # class 2 appears
        assert m._n_classes == 3
        assert m.tabulate is not None

    def test_continuous_child_equals_pooled(self):
        pa_a, y_a, _ = _gen(300, seed=1)
        pa_b, y_b, _ = _gen(200, seed=2)
        chunked = KNNConditionalMechanism(k=15)
        chunked.fit_local(y_a, pa_a)
        chunked.update_local(y_b, pa_b)
        pooled = KNNConditionalMechanism(k=15)
        pooled.fit_local(torch.cat([y_a, y_b]), torch.cat([pa_a, pa_b]))
        qpa, qy = _query()
        assert torch.allclose(chunked.log_prob(qy, qpa), pooled.log_prob(qy, qpa), atol=1e-4)

    def test_refusals(self):
        pa_a, y_a, c_a = _gen(50, seed=1)
        m = KNNConditionalMechanism(discrete_child=True)
        m.fit_local(c_a, pa_a)
        with pytest.raises(NotImplementedError):
            m.update_local(c_a, pa_a, forgetting=0.9)
        with pytest.raises(NotImplementedError):
            m.update_local(c_a, pa_a, weights=torch.ones(50))
        with pytest.raises(ValueError):
            m.update_local(c_a, pa_a[:, :1])


class TestThroughModelAPI:
    """A DBN-like network: continuous roots, a KDE child, a kNN classifier."""

    @staticmethod
    def _model():
        model = NBN(
            [("X", "Y"), ("Z", "Y"), ("X", "C"), ("Z", "C")],
            {"X": ("continuous", 1), "Z": ("continuous", 1),
             "Y": ("continuous", 1), "C": ("discrete", 3)},
            device="cpu",
        )
        model.set_mechanism("X", LinearGaussianMechanism())
        model.set_mechanism("Z", LinearGaussianMechanism())
        model.set_mechanism("Y", _kde())
        model.set_mechanism("C", KNNConditionalMechanism(discrete_child=True))
        return model

    @staticmethod
    def _data(n, seed):
        pa, y, c = _gen(n, seed=seed)
        return {"X": pa[:, 0], "Z": pa[:, 1], "Y": y[:, 0], "C": c}

    def test_update_is_not_skipped_and_matches_pooled_fit(self):
        a, b = self._data(300, 1), self._data(200, 2)
        chunked = self._model()
        chunked.fit(a)
        hist = chunked.update(b)
        assert hist.skipped == []
        assert hist.node_methods["Y"] == "kde_append"
        assert hist.node_methods["C"] == "knn_append"
        pooled = self._model()
        pooled.fit({k: torch.cat([a[k], b[k]]) for k in a})
        qpa, qy = _query()
        assert torch.allclose(
            chunked.mechanisms["Y"].log_prob(qy, qpa), pooled.mechanisms["Y"].log_prob(qy, qpa),
            atol=1e-4)
        assert torch.allclose(
            chunked.mechanisms["C"]._categorical_probs(qpa, 64),
            pooled.mechanisms["C"]._categorical_probs(qpa, 64))
