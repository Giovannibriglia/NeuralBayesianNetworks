"""Scoring-status semantics of AccuracyAndTiming (D5, #198, #217).

* a probability vector of the wrong length is an ``error`` (it was truncated
  with ``[:k]`` or silently turned into ``not_supported``);
* a NaN posterior is an ``error``, never an ``ok`` row with a NaN value;
* a query the oracle cannot produce a reference for is ``no_result``;
* every row carries its query's position (``query_idx``).
"""
from __future__ import annotations

import torch

from nbn.bench.core import oracle as oracle_mod
from nbn.bench.domains.base import Query
from nbn.bench.domains.posterior import Posterior
from nbn.bench.measurements import accuracy_timing as at
from nbn.bench.problems import SyntheticConfig, SyntheticProblemSource


def _problem():
    cfg = SyntheticConfig(families=["discrete"], n_nodes_list=[4], seeds=[0],
                          n_train=200, n_test=20, n_reference=200, cardinality=4)
    p = next(iter(SyntheticProblemSource().iter_problems(cfg)))
    nodes = list(p.variables)
    q = Query(targets=(nodes[-1],), evidence={nodes[0]: torch.tensor(1.0)},
              kind="marginal")
    return p, q


class _Fixed:
    name = "fixed"

    def __init__(self, probs):
        self.probs = probs

    def query(self, q):
        return Posterior(probs=self.probs)


def _acc_rows(rows):
    return [r for r in rows if r.metric in ("tv_per_node", "jsd_per_node")]


def test_short_probability_vector_is_an_error_not_not_supported():
    p, q = _problem()
    rows = at.AccuracyAndTiming().measure(p, _Fixed(torch.tensor([0.5, 0.5])), [q])
    acc = _acc_rows(rows)
    assert {r.status for r in acc} == {"error"}
    assert "length 2" in acc[0].error_msg


def test_unconfirmed_nan_or_inf_posterior_is_an_error_198():
    """N2: a NaN / inf / all-zero posterior is a numerical failure (error)
    unless the adapter CONFIRMED zero-probability evidence; labelling every
    NaN 'undefined' hid real failures and let same-query averaging score the
    other methods on an easier subset."""
    p, q = _problem()
    for bad in (torch.full((4,), float("nan")), torch.tensor([float("inf"), 0.0, 0.0, 0.0]),
                torch.zeros(4)):
        rows = at.AccuracyAndTiming().measure(p, _Fixed(bad), [q])
        acc = _acc_rows(rows)
        assert {r.status for r in acc} == {"error"}, bad
        assert "without a confirmed zero-probability evidence" in acc[0].error_msg


def test_confirmed_zero_probability_evidence_is_no_result():
    p, q = _problem()

    class _Confirmed(_Fixed):
        def query(self, q):
            return Posterior(probs=self.probs, undefined_reason="checked")

    rows = at.AccuracyAndTiming().measure(p, _Confirmed(torch.full((4,), float("nan"))), [q])
    acc = _acc_rows(rows)
    assert {r.status for r in acc} == {"no_result"}
    assert at.ZERO_PROB_EVIDENCE in acc[0].error_msg


def test_oracle_failure_is_no_result(monkeypatch):
    p, q = _problem()
    monkeypatch.setattr(at, "discrete_oracle", lambda *a, **k: (None, oracle_mod.ORACLE_LW))
    rows = at.AccuracyAndTiming().measure(p, _Fixed(torch.full((4,), 0.25)), [q])
    assert {r.status for r in _acc_rows(rows)} == {"no_result"}
    # The query itself ran: its timing rows stay ok.
    assert {r.status for r in rows if r.metric == "query_time_s"} == {"ok"}


def test_rows_carry_query_idx():
    p, q = _problem()
    rows = at.AccuracyAndTiming().measure(p, _Fixed(torch.full((4,), 0.25)), [q, q, q])
    assert sorted({r.query_idx for r in rows}) == [0, 1, 2]


def test_pgmpy_declared_state_grid_217():
    """A state never seen in training keeps its slot: the posterior has K
    entries in declared order and evidence on the unseen state works."""
    import pytest

    pytest.importorskip("pgmpy")
    from nbn.bench.adapters.pgmpy_adapter import PgmpyAdapter
    from nbn.bench.domains.base import BenchmarkProblem

    data = {"A": torch.tensor([1, 2, 1, 2, 2, 1]), "B": torch.tensor([0, 1, 0, 1, 0, 1])}
    prob = BenchmarkProblem(name="t", dag=[("A", "B")],
                            variables={"A": ("discrete", 3), "B": ("discrete", 3)},
                            train_data=data, test_data=data, queries=[])
    for method in ("mle", "bayes"):
        a = PgmpyAdapter(method, "ve")
        a.fit(prob)
        marg = a.query(Query(targets=("A",), evidence={}, kind="marginal")).probs
        assert marg.shape == (3,)
        if method == "mle":
            # state 0 unseen -> probability 0 at index 0 (not shifted).
            assert marg[0] == 0 and torch.allclose(marg[1:], torch.tensor([0.5, 0.5]))
        cond = a.query(Query(targets=("B",), evidence={"A": torch.tensor(0.0)},
                             kind="marginal")).probs
        assert cond.shape == (3,) and torch.isfinite(cond).all()


def test_zero_probability_evidence_reason_and_pgmpy_vs_pomegranate():
    """D10 (rest): evidence on a state the MLE never saw has P(e) = 0 under
    the fitted model. pomegranate's loopy BP then returns NaN -> an error row
    whose reason says so. pgmpy prunes the network before VE: when the
    zero-probability evidence only enters through the target's CPT row it
    returns that row (a defined conditional), so the two libraries do NOT
    fail on the same queries -- the shared error_msg lets reports count
    such queries per baseline."""
    import pytest

    pytest.importorskip("pgmpy")
    pytest.importorskip("pomegranate")
    from nbn.bench.adapters.pgmpy_adapter import PgmpyAdapter
    from nbn.bench.adapters.pomegranate_adapter import PomegranateAdapter
    from nbn.bench.domains.base import BenchmarkProblem

    d = {"A": torch.tensor([0, 1, 0, 1, 1, 0]), "B": torch.tensor([0, 1, 1, 1, 0, 0])}
    prob = BenchmarkProblem(name="t", dag=[("A", "B")],
                            variables={"A": ("discrete", 3), "B": ("discrete", 2)},
                            train_data=d, test_data=d, queries=[],
                            family="discrete", problem_id="t", seed=0)
    q = Query(targets=("B",), evidence={"A": torch.tensor(2.0)}, kind="marginal")
    pom = PomegranateAdapter(device="cpu")
    pom.fit(prob)
    with pytest.raises(at.UndefinedPosterior, match=at.ZERO_PROB_EVIDENCE):
        at.validated_probs(pom.query(q), "B", 2)
    pg = PgmpyAdapter("mle", "ve")
    pg.fit(prob)
    # defined: the pruned network's answer is the CPT row P(B | A=2) = uniform
    assert torch.allclose(at.validated_probs(pg.query(q), "B", 2).float(),
                          torch.tensor([0.5, 0.5]))


def test_pomegranate_flags_only_genuinely_impossible_evidence():
    """The pomegranate adapter confirms P(e)=0 by exact elimination on its
    fitted CPTs before flagging; a NaN with P(e) > 0 stays unflagged."""
    import pytest

    pytest.importorskip("pomegranate")
    from nbn.bench.adapters.pomegranate_adapter import PomegranateAdapter
    from nbn.bench.domains.base import BenchmarkProblem

    d = {"A": torch.tensor([0, 1, 0, 1, 1, 0]), "B": torch.tensor([0, 1, 1, 1, 0, 0])}
    prob = BenchmarkProblem(name="t", dag=[("A", "B")],
                            variables={"A": ("discrete", 3), "B": ("discrete", 2)},
                            train_data=d, test_data=d, queries=[])
    a = PomegranateAdapter(device="cpu")
    a.fit(prob)
    impossible = Query(targets=("B",), evidence={"A": torch.tensor(2.0)}, kind="marginal")
    possible = Query(targets=("B",), evidence={"A": torch.tensor(1.0)}, kind="marginal")
    assert a.query(impossible).undefined_reason
    assert a.query(possible).undefined_reason is None
    nan = torch.tensor([float("nan"), float("nan")])
    assert a._undefined_reason(possible, nan) is None      # NaN but P(e) > 0
