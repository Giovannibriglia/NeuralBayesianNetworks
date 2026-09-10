"""AIS per-query ESS fallback + ESS-plateau early stopping (#250).

The fit-time ESS gate judges the proposal on ONE synthetic evidence pattern
and, at paper scale, collapsed to all-or-nothing (learned 100% at n=10,
0% at n≥100) after spending the full 600 s training cap.  Two changes:

1. per-query fallback: rows whose learned-proposal ESS fraction is below the
   query threshold are re-run with the prior (LW) proposal and the row with
   the higher ESS is kept — never worse than LW on any single query;
2. early stopping on a fixed held-out ESS set (best checkpoint restored) so a
   proposal that plateaus stops paying for training it will not benefit from.
"""
from __future__ import annotations

import copy
import dataclasses

import pytest
import torch

from nbn.bench.adapters import NBNAdapter
from nbn.bench.core.results import CellResult
from nbn.bench.domains.base import BenchmarkProblem, Query
from nbn.inference.amortized_is import AmortizedISEngine, _ess_fraction_rows
from tests.unit.test_amortized_is_engine import (
    _fit_adapter,
    _make_small_discrete_problem,
)


@pytest.fixture(scope="module")
def discrete_problem() -> BenchmarkProblem:
    return _make_small_discrete_problem()


def _batched_evidence(b: int) -> dict:
    return {"X0": torch.tensor([v % 2 for v in range(b)])}


def _fit_model(problem: BenchmarkProblem):
    return _fit_adapter("cat", "ve", problem, epochs=5).model


def _trained_engine(model, **kw) -> AmortizedISEngine:
    eng = AmortizedISEngine(n_samples=256, **kw)
    torch.manual_seed(0)
    metrics = eng.train_proposal(model, device="cpu")
    assert metrics["proposal_used"] == "learned", metrics
    return eng


def _spoil_rows(eng: AmortizedISEngine, rows: slice):
    """Make the LEARNED proposal's weights degenerate on ``rows`` (one
    dominant particle → ESS fraction ≈ 1/S) without touching the rest."""
    orig = eng._run_learned

    def spoiled(*a, **k):
        log_w, buf = orig(*a, **k)
        log_w[rows, 0] += 50.0
        return log_w, buf

    eng._run_learned = spoiled  # type: ignore[assignment]


# ---- per-query fallback -------------------------------------------------------

@pytest.mark.slow
def test_low_ess_rows_are_answered_by_lw(discrete_problem):
    model = _fit_model(discrete_problem)
    eng = _trained_engine(model)
    _spoil_rows(eng, slice(None))          # every row degenerate
    p = eng.query(model, ["X2"], _batched_evidence(8))
    assert torch.isfinite(p).all() and p.shape == (8, 2)
    assert eng.last_fallback_mask is not None and bool(eng.last_fallback_mask.all())
    assert eng.n_queries_seen == 8 and eng.n_queries_fallback == 8
    assert eng.query_fallback_frac == 1.0


@pytest.mark.slow
def test_fallback_is_row_selective_and_leaves_good_rows_untouched(discrete_problem):
    """Only the low-ESS rows are re-run; the others keep the learned result."""
    model = _fit_model(discrete_problem)
    eng = _trained_engine(model, query_ess_threshold=0.05)
    ev = _batched_evidence(8)

    torch.manual_seed(1)
    eng.per_query_fallback = False
    p_learned = eng.query(model, ["X2"], ev)

    _spoil_rows(eng, slice(1, None, 2))     # odd rows degenerate
    eng.per_query_fallback = True
    torch.manual_seed(1)
    p_mixed = eng.query(model, ["X2"], ev)

    mask = eng.last_fallback_mask
    assert bool(mask[1::2].all()), "spoiled rows must fall back to LW"
    assert not bool(mask[0::2].any()), "healthy rows must keep the learned proposal"
    # Same seed → the learned pass is identical; the even rows are spliced
    # through untouched.
    assert torch.allclose(p_mixed[0::2], p_learned[0::2])
    assert eng.query_fallback_frac == 0.5


@pytest.mark.slow
def test_fallback_keeps_the_higher_ess_row(discrete_problem):
    """LW replaces a row only where its ESS beats the learned proposal's."""
    model = _fit_model(discrete_problem)
    eng = _trained_engine(model)
    _spoil_rows(eng, slice(None))
    lw_calls = {"n": 0}
    orig_lw = eng.__class__.__mro__[1]._run

    def worse_lw(self_, *a, **k):
        lw_calls["n"] += 1
        log_w, buf = orig_lw(self_, *a, **k)
        log_w[:, 0] += 100.0                 # even more degenerate than learned
        return log_w, buf

    eng.__class__.__mro__[1]._run = worse_lw  # type: ignore[assignment]
    try:
        eng.query(model, ["X2"], _batched_evidence(4))
    finally:
        eng.__class__.__mro__[1]._run = orig_lw  # type: ignore[assignment]
    assert lw_calls["n"] == 1
    assert not bool(eng.last_fallback_mask.any())
    assert eng.query_fallback_frac == 0.0


@pytest.mark.slow
def test_fallback_can_be_disabled(discrete_problem):
    model = _fit_model(discrete_problem)
    eng = _trained_engine(model, per_query_fallback=False)
    _spoil_rows(eng, slice(None))
    p = eng.query(model, ["X2"], _batched_evidence(4))
    assert torch.isfinite(p).all()
    assert eng.last_fallback_mask is None and eng.query_fallback_frac is None


@pytest.mark.slow
def test_broadcast_evidence_row_subset(discrete_problem):
    """A [1]-shaped evidence tensor broadcast across a wider ``do`` batch is
    kept whole when the fallback subsets rows (LW's own broadcast rule)."""
    model = _fit_model(discrete_problem)
    eng = _trained_engine(model)
    _spoil_rows(eng, slice(0, 1))
    p = eng.query(
        model, ["X2"], {"X0": torch.tensor([1])}, do={"X1": torch.tensor([0, 1, 0])},
    )
    assert p.shape == (3, 2) and torch.isfinite(p).all()
    assert bool(eng.last_fallback_mask[0]) and eng.n_queries_seen == 3


def test_ess_fraction_rows_matches_definition():
    log_w = torch.tensor([[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]])
    ess = _ess_fraction_rows(log_w)
    assert torch.allclose(ess[0], torch.tensor(1.0))
    assert ess[1] < 0.3


# ---- early stopping on the held-out ESS --------------------------------------

@pytest.mark.slow
def test_ess_plateau_stops_training_early(discrete_problem):
    model = _fit_model(discrete_problem)
    eng = AmortizedISEngine(n_samples=256)
    eng._estimate_ess_fraction = lambda *a, **k: 0.01  # type: ignore[assignment]
    metrics = eng.train_proposal(
        model, n_training_samples=500, steps_per_node=1, device="cpu",
        ess_check_every=25, ess_patience=3,
    )
    # checks at 25 (best), 50, 75, 100 (3 bad) → stop at 100 of 500 steps
    assert metrics["early_stopped"] is True
    assert metrics["grad_steps"] == 100 and metrics["target_steps"] == 500
    assert metrics["proposal_used"] == "lw_fallback"
    assert [s for s, _ in metrics["ess_history"]] == [25, 50, 75, 100]


@pytest.mark.slow
def test_early_stop_restores_best_checkpoint(discrete_problem):
    model = _fit_model(discrete_problem)
    eng = AmortizedISEngine(n_samples=256)
    seq = iter([0.5, 0.3, 0.3, 0.3, 0.3])   # 4 checkpoints + final gate
    snapshot = {}

    def fake_ess(*a, **k):
        v = next(seq)
        if v == 0.5:
            snapshot["state"] = copy.deepcopy(eng.recognition_net.state_dict())
        return v

    eng._estimate_ess_fraction = fake_ess  # type: ignore[assignment]
    metrics = eng.train_proposal(
        model, n_training_samples=500, steps_per_node=1, device="cpu",
        ess_check_every=25, ess_patience=3,
    )
    assert metrics["early_stopped"] and metrics["best_step"] == 25
    assert metrics["ess_fraction"] == 0.5 and metrics["proposal_used"] == "learned"
    for name, param in eng.recognition_net.state_dict().items():
        assert torch.equal(param, snapshot["state"][name]), name


@pytest.mark.slow
def test_early_stop_can_be_disabled(discrete_problem):
    model = _fit_model(discrete_problem)
    eng = AmortizedISEngine(n_samples=256)
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return 0.01

    eng._estimate_ess_fraction = counting  # type: ignore[assignment]
    metrics = eng.train_proposal(
        model, n_training_samples=500, steps_per_node=1, device="cpu",
        early_stop=False,
    )
    assert metrics["early_stopped"] is False and metrics["grad_steps"] == 500
    assert calls["n"] == 1                    # final gate only


# ---- benchmark plumbing: query_fallback_frac column ---------------------------

@pytest.mark.slow
def test_adapter_exposes_query_fallback_frac(discrete_problem):
    adapter = _fit_adapter("cat", "ais", discrete_problem, n_samples=512, epochs=5)
    assert adapter.proposal_used == "learned"
    assert adapter.query_fallback_frac is None      # no query yet
    adapter.query_batch([Query(targets=("X2",), evidence={"X0": torch.tensor(v % 2)},
                               kind="marginal") for v in range(6)])
    frac = adapter.query_fallback_frac
    assert frac is not None and 0.0 <= frac <= 1.0

    lw = _fit_adapter("cat", "lw", discrete_problem, n_samples=512, epochs=5)
    assert lw.query_fallback_frac is None


def test_cellresult_has_additive_query_fallback_column():
    names = {f.name for f in dataclasses.fields(CellResult)}
    assert "query_fallback_frac" in names
    assert CellResult.__dataclass_fields__["query_fallback_frac"].default is None
