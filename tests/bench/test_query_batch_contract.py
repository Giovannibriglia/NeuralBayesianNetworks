"""``query_batch`` is a faithful batched ``query``, for every adapter (#148).

Shared contract, parametrised over the adapters' batching paths:

* ``nbn-cat-ve``   library-batched VE (exact, atol 1e-6);
* ``pomegranate``  one ``predict_proba`` call over ``[B, n]`` (loopy BP;
  atol 1e-5 between the ``[B, n]`` and ``[1, n]`` message passes);
* ``pgmpy-mle-ve`` the protocol's default sequential helper (bitwise).

Each must: match looped ``query()``; follow the input order; answer B=1
exactly as ``query``; return ``[]`` for ``[]``; fall back to sequential for
heterogeneous targets, evidence keys or mixed None/concrete values; keep
rows independent; and marginalise an evidence key that is None in every row.

Engine-specific checks follow: the LW statistical equivalences and particle
independence (nbn), the single-``predict_proba`` spy (pomegranate) and the
answered-once empty-mode batch (nbn, #294 / PR 5 speed smoke).

Merged from test_nbn_query_batch.py, test_pomegranate_query_batch.py and the
real-adapter half of test_adapter_protocol_query_batch.py.
"""
from __future__ import annotations

import pytest
import torch

from nbn.bench.adapters import NBNAdapter, PgmpyAdapter, PomegranateAdapter
from nbn.bench.core.interfaces import default_query_batch
from nbn.bench.domains.base import BenchmarkProblem, Query
from nbn.bench.domains.posterior import Posterior

# ---- problems / queries -------------------------------------------------------


def _discrete_problem(n_samples: int = 500, seed: int = 0) -> BenchmarkProblem:
    """4-node binary BN: X0 → X1 → X2, X0 → X3 (correlated, non-degenerate)."""
    g = torch.Generator().manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2"), ("X0", "X3")]
    x0 = torch.randint(0, 2, (n_samples,), generator=g)
    flip = lambda x, p: torch.where(  # noqa: E731 — local helper
        torch.rand(n_samples, generator=g) < p, 1 - x, x
    )
    train_data = {"X0": x0, "X1": flip(x0, 0.3), "X2": flip(flip(x0, 0.3), 0.3),
                  "X3": flip(x0, 0.2)}
    return BenchmarkProblem(
        name="test_discrete", dag=dag,
        variables=dict.fromkeys(train_data, ("discrete", 2)),
        train_data=train_data, test_data=train_data, queries=[],
    )


def _continuous_lg_problem(n_samples: int = 400, seed: int = 0) -> BenchmarkProblem:
    """3-node linear-Gaussian chain: X0 → X1 → X2."""
    torch.manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2")]
    x0 = torch.randn(n_samples)
    x1 = 0.5 * x0 + 0.5 * torch.randn(n_samples)
    x2 = 0.5 * x1 + 0.5 * torch.randn(n_samples)
    train_data = {"X0": x0, "X1": x1, "X2": x2}
    return BenchmarkProblem(
        name="test_continuous_lg", dag=dag,
        variables=dict.fromkeys(train_data, ("continuous", None)),
        train_data=train_data, test_data=train_data, queries=[],
    )


def _discrete_queries(b: int = 4) -> list[Query]:
    """B queries sharing (target=X2, evidence_keys={X0}); X0 alternates 0, 1."""
    return [Query(targets=("X2",), evidence={"X0": torch.tensor(v % 2)}, kind="marginal")
            for v in range(b)]


def _continuous_queries(b: int = 4) -> list[Query]:
    return [Query(targets=("X2",), evidence={"X0": torch.tensor(-1.0 + 0.5 * v)},
                  kind="marginal")
            for v in range(b)]


def _q(target: str, **evidence) -> Query:
    ev = {k: (None if v is None else torch.tensor(v)) for k, v in evidence.items()}
    return Query(targets=(target,), evidence=ev, kind="marginal")


# ---- adapters -----------------------------------------------------------------

class _DefaultHelper:
    """pgmpy answered through the protocol's default sequential helper."""

    def __init__(self, adapter):
        self._a = adapter

    def query(self, q):
        return self._a.query(q)

    def query_batch(self, queries):
        return default_query_batch(self._a, queries)


def _fit_nbn_cat_ve():
    a = NBNAdapter(mechanism="cat", engine="ve", device="cpu")
    a.fit(_discrete_problem(), epochs=5)
    return a


def _fit_pomegranate():
    a = PomegranateAdapter(device="cpu")
    a.fit(_discrete_problem())
    return a


def _fit_pgmpy_default():
    a = PgmpyAdapter(param_method="mle", inference_method="ve")
    a.fit(_discrete_problem())
    return _DefaultHelper(a)


_BATCHERS = {
    "nbn-cat-ve": (_fit_nbn_cat_ve, 1e-6),
    "pomegranate": (_fit_pomegranate, 1e-5),
    "pgmpy-mle-ve": (_fit_pgmpy_default, 0.0),
}


@pytest.fixture(scope="module", params=sorted(_BATCHERS))
def batcher(request):
    fit, atol = _BATCHERS[request.param]
    return fit(), atol


def _assert_same(batched, sequential, atol):
    assert len(batched) == len(sequential)
    for bp, sp in zip(batched, sequential):
        assert isinstance(bp, Posterior)
        assert bp.probs is not None and sp.probs is not None
        assert bp.probs.shape == sp.probs.shape
        torch.testing.assert_close(bp.probs, sp.probs, atol=atol, rtol=0)


# ---- shared contract ----------------------------------------------------------


def test_batch_matches_sequential(batcher):
    adapter, atol = batcher
    queries = _discrete_queries(b=4)
    _assert_same(adapter.query_batch(queries), [adapter.query(q) for q in queries], atol)


def test_batch_order_follows_input(batcher):
    adapter, _ = batcher
    queries = _discrete_queries(b=4)
    fwd = adapter.query_batch(queries)
    rev = adapter.query_batch(list(reversed(queries)))
    for f, r in zip(fwd, reversed(rev)):
        torch.testing.assert_close(f.probs, r.probs, atol=1e-6, rtol=0)


def test_b1_is_identical_to_query(batcher):
    adapter, _ = batcher
    q = _discrete_queries(b=1)
    [batched] = adapter.query_batch(q)
    assert torch.equal(batched.probs, adapter.query(q[0]).probs)


def test_empty_batch_returns_empty_list(batcher):
    adapter, _ = batcher
    assert adapter.query_batch([]) == []


@pytest.mark.parametrize("queries", [
    pytest.param([_q("X2", X0=0), _q("X3", X0=1)], id="heterogeneous-targets"),
    pytest.param([_q("X2", X0=0), _q("X2", X1=1)], id="heterogeneous-evidence-keys"),
    pytest.param([_q("X2", X0=0), _q("X2", X0=None)], id="mixed-none-evidence"),
])
def test_heterogeneous_batches_fall_back_to_sequential(batcher, queries):
    adapter, _ = batcher
    _assert_same(adapter.query_batch(queries), [adapter.query(q) for q in queries], 1e-6)


def test_all_none_evidence_key_is_marginalised(batcher):
    """A key that is None in every row is dropped (empty mode), exactly as
    ``query`` leaves it unobserved."""
    adapter, atol = batcher
    queries = [_q("X2", X0=None) for _ in range(3)]
    _assert_same(adapter.query_batch(queries), [adapter.query(q) for q in queries],
                 max(atol, 1e-5))


def test_rows_are_independent(batcher):
    """X2 depends on X0, so rows with different evidence differ and rows with
    equal evidence agree -- guards rows accidentally sharing one evidence."""
    adapter, _ = batcher
    batched = adapter.query_batch(_discrete_queries(b=4))   # X0 = 0, 1, 0, 1
    assert not torch.allclose(batched[0].probs, batched[1].probs, atol=1e-3)
    torch.testing.assert_close(batched[0].probs, batched[2].probs, atol=1e-6, rtol=0)
    torch.testing.assert_close(batched[1].probs, batched[3].probs, atol=1e-6, rtol=0)


# ---- nbn: likelihood weighting (statistical equivalence) -------------------------


def _ks_statistic(a: torch.Tensor, b: torch.Tensor) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (sup |ECDF_a − ECDF_b|)."""
    combined = torch.cat([a, b]).sort().values
    cdf_a = torch.searchsorted(a.sort().values, combined, right=True) / len(a)
    cdf_b = torch.searchsorted(b.sort().values, combined, right=True) / len(b)
    return float((cdf_a - cdf_b).abs().max())


@pytest.fixture(scope="module")
def cat_lw_adapter() -> NBNAdapter:
    a = NBNAdapter(mechanism="cat", engine="lw", device="cpu", n_samples=4096)
    a.fit(_discrete_problem(), epochs=5)
    return a


@pytest.fixture(scope="module")
def lg_lw_adapter() -> NBNAdapter:
    a = NBNAdapter(mechanism="lg", engine="lw", device="cpu", n_samples=4096)
    a.fit(_continuous_lg_problem(), epochs=5)
    return a


def test_lw_discrete_batch_matches_sequential_in_tv(cat_lw_adapter):
    queries = _discrete_queries(b=4)
    torch.manual_seed(7)
    batched = cat_lw_adapter.query_batch(queries)
    torch.manual_seed(7)
    sequential = [cat_lw_adapter.query(q) for q in queries]
    for bp, sp in zip(batched, sequential):
        tv = 0.5 * (bp.probs - sp.probs).abs().sum()
        assert tv <= 0.05, f"TV {tv:.4f} > 0.05"


def test_lw_continuous_batch_matches_sequential_in_ks(lg_lw_adapter):
    """The batched-vs-looped equivalence the V1 investigation flagged as missing."""
    queries = _continuous_queries(b=4)
    torch.manual_seed(7)
    batched = lg_lw_adapter.query_batch(queries)
    torch.manual_seed(7)
    sequential = [lg_lw_adapter.query(q) for q in queries]
    for i, (bp, sp) in enumerate(zip(batched, sequential)):
        assert bp.samples.shape == sp.samples.shape == (4096,)
        ks = _ks_statistic(bp.samples, sp.samples)
        assert ks <= 0.10, f"query {i}: KS {ks:.4f} > 0.10"


def test_lw_continuous_rows_have_independent_particle_pools(lg_lw_adapter):
    """V1 finding: a shared pool conditioned on one row's evidence would
    collapse the posterior means across rows (X2 | X0=v has mean 0.25·v)."""
    torch.manual_seed(7)
    batched = lg_lw_adapter.query_batch(_continuous_queries(b=4))
    means = torch.tensor([p.samples.mean() for p in batched])
    assert means[-1] - means[0] > 0.15, f"posterior means collapsed: {means}"


# ---- engine-path spies ------------------------------------------------------------


def test_pomegranate_homogeneous_batch_is_one_predict_proba_call():
    """Without this the equivalence test could pass while silently falling
    back to sequential."""
    adapter = _fit_pomegranate()
    calls = []
    orig = adapter.model.predict_proba

    def spy(x):
        calls.append(tuple(x.shape))
        return orig(x)

    adapter.model.predict_proba = spy
    try:
        out = adapter.query_batch(_discrete_queries(b=4))
    finally:
        adapter.model.predict_proba = orig
    assert len(out) == 4
    assert calls == [(4, 4)]  # one call, [B=4, n=4]


def test_nbn_all_empty_mode_batch_is_answered_once(monkeypatch):
    """All-None evidence (empty mode, e.g. heaviest V2 queries): the B queries
    are one marginal, answered by a single ``query`` call and returned B times
    (it used to be B sequential calls, which made the per-query time flat in
    B).  Still B posteriors of shape (K,): regression for the PR 5 speed-smoke
    failure (shape (1, K) for B=4)."""
    adapter = _fit_nbn_cat_ve()
    queries = [_q("X2", X0=None) for _ in range(4)]
    expected = adapter.query(queries[0])
    calls = []
    real_query = adapter.query
    monkeypatch.setattr(adapter, "query", lambda q: calls.append(q) or real_query(q))
    batched = adapter.query_batch(queries)
    assert len(calls) == 1
    assert len(batched) == 4
    for bp in batched:
        assert bp.probs.shape == expected.probs.shape
        torch.testing.assert_close(bp.probs, expected.probs, atol=1e-6, rtol=0)
