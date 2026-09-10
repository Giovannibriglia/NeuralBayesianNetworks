"""AVI fit-time quality gate with LW fallback + held-out-ELBO early stopping (#249).

Before: the only check after training was a log-warning at ``elbo_gap > 5``
that ran on ≤12-node all-discrete nets only — on every benchmark-scale network
the variational posterior shipped untested.  Now the trained ``q``'s TARGET
MARGINALS are compared to a likelihood-weighting reference on held-out random
evidence patterns; a ``q`` no closer to it than the naive clamped prior is
unset and the engine answers with LW (``proposal_used = "lw_fallback"``),
mirroring the AIS ESS gate.  Training also stops early when the held-out ELBO
plateaus (best checkpoint restored).
"""
from __future__ import annotations

import copy
import logging

import pytest
import torch

from nbn.bench.adapters import NBNAdapter
from nbn.bench.domains.base import BenchmarkProblem, Query
from nbn.inference.amortized_vi import AmortizedVIEngine, _weighted_quantiles
from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
from tests.unit.test_amortized_vi_engine import (
    _chain_discrete,
    _fit_adapter,
    _make_small_continuous_problem,
    _make_small_discrete_problem,
)


@pytest.fixture(scope="module")
def discrete_problem() -> BenchmarkProblem:
    return _make_small_discrete_problem()


@pytest.fixture(scope="module")
def continuous_problem() -> BenchmarkProblem:
    return _make_small_continuous_problem()


_BAD = {"d_q": 0.4, "d_prior": 0.1, "gain": -0.3, "lw_ess": 0.9}


# ---- gate ---------------------------------------------------------------------

@pytest.mark.slow
def test_trained_q_clears_the_gate(discrete_problem):
    model = _fit_adapter("cat", "ve", discrete_problem, epochs=5).model
    eng = AmortizedVIEngine(n_samples=256)
    torch.manual_seed(0)
    m = eng.train_proposal(model, device="cpu")
    assert m["proposal_used"] == "learned" and eng.recognition_net is not None
    assert m["gain"] is not None and m["gain"] > 0.0 and m["d_q"] < m["d_prior"]
    assert eng.proposal_used == "learned" and eng._fallback is None


@pytest.mark.slow
def test_untrained_q_is_rejected(discrete_problem):
    """Two epochs on 500 samples is not a posterior — the gate must say so."""
    model = _fit_adapter("cat", "ve", discrete_problem, epochs=5).model
    eng = AmortizedVIEngine(n_samples=256)
    torch.manual_seed(0)
    m = eng.train_proposal(model, n_training_samples=500, n_epochs=2, device="cpu")
    assert m["proposal_used"] == "lw_fallback" and m["gain"] < 0.0
    assert eng.recognition_net is None and isinstance(eng._fallback, LikelihoodWeightingEngine)


@pytest.mark.slow
def test_rejected_q_answers_with_lw(discrete_problem, caplog):
    model = _fit_adapter("cat", "ve", discrete_problem, epochs=5).model
    eng = AmortizedVIEngine(n_samples=256)
    eng._estimate_quality = lambda *a, **k: dict(_BAD)  # type: ignore[assignment]
    with caplog.at_level(logging.WARNING):
        m = eng.train_proposal(model, n_training_samples=500, n_epochs=2, device="cpu")
    assert m["proposal_used"] == "lw_fallback"
    msgs = [r.getMessage() for r in caplog.records]
    assert any("no closer to the LW reference" in x for x in msgs), "expected diagnostic"
    assert any("falling back to LW" in x for x in msgs), "expected fallback action"

    torch.manual_seed(0)
    p = eng.query(model, ["X2"], {"X0": torch.tensor([0])})
    assert p.shape == (2,) and torch.isfinite(p).all() and abs(float(p.sum()) - 1) < 1e-5
    pb = eng.query_batch(model, ["X2"], {"X0": torch.tensor([0, 1, 0, 1])})
    assert pb.shape == (4, 2) and torch.isfinite(pb).all()
    # LW's diagnostics pass through the fallback path unchanged.
    out, ess, khat = eng.query_batch(model, ["X2"], {"X0": torch.tensor([0, 1])},
                                     return_ess=True, return_psis_k=True)
    assert out.shape == (2, 2) and ess.shape == (2,) and khat.shape == (2,)


@pytest.mark.slow
def test_rejected_q_continuous_matches_lw_contract(continuous_problem):
    model = _fit_adapter("lg", "lw", continuous_problem, epochs=5).model
    eng = AmortizedVIEngine(n_samples=128)
    eng._estimate_quality = lambda *a, **k: dict(_BAD)  # type: ignore[assignment]
    eng.train_proposal(model, n_training_samples=500, n_epochs=2, device="cpu")
    tgt = list(model.dag.nodes())[-1] if hasattr(model.dag, "nodes") else "X3"
    w, s = eng.query(model, [tgt], {"X0": torch.tensor([[0.3]])})
    assert w.shape == (1, 128) and s.shape[:2] == (1, 128)
    assert torch.isfinite(s).all() and abs(float(w.sum()) - 1) < 1e-4


@pytest.mark.slow
def test_gate_can_be_disabled(discrete_problem):
    model = _fit_adapter("cat", "ve", discrete_problem, epochs=5).model
    eng = AmortizedVIEngine(n_samples=256)
    eng._estimate_quality = lambda *a, **k: dict(_BAD)  # type: ignore[assignment]
    m = eng.train_proposal(model, n_training_samples=500, n_epochs=2, device="cpu",
                           quality_gate=False)
    assert m["proposal_used"] == "learned" and m["gain"] is None
    assert eng.recognition_net is not None


@pytest.mark.slow
def test_gate_prefers_q_on_downstream_evidence():
    """Diagnostic chain: q (post-#267) recovers P(X0 | X1) while the naive
    clamped prior returns the prior — the gate must see q win clearly."""
    model = _fit_adapter("cat", "ve", _chain_discrete()).model
    eng = AmortizedVIEngine(n_samples=256)
    torch.manual_seed(0)
    m = eng.train_proposal(model, device="cpu")
    assert m["proposal_used"] == "learned" and m["gain"] > 0.1, m


# ---- early stopping on the held-out ELBO ---------------------------------------

@pytest.mark.slow
def test_elbo_plateau_stops_training_early(discrete_problem):
    model = _fit_adapter("cat", "ve", discrete_problem, epochs=5).model
    eng = AmortizedVIEngine(n_samples=256)
    eng._heldout_elbo_per_latent = lambda *a, **k: -1.0  # type: ignore[assignment]
    m = eng.train_proposal(model, device="cpu", min_steps=500, steps_per_node=1,
                           check_every=25, patience=3, quality_gate=False)
    # checks at 25 (best), 50, 75, 100 (3 bad) -> stop at 100 of 500
    assert m["early_stopped"] is True and m["grad_steps"] == 100
    assert [s for s, _ in m["elbo_history"]] == [25, 50, 75, 100]


@pytest.mark.slow
def test_early_stop_restores_best_checkpoint(discrete_problem):
    model = _fit_adapter("cat", "ve", discrete_problem, epochs=5).model
    eng = AmortizedVIEngine(n_samples=256)
    seq = iter([-1.0, -2.0, -2.0, -2.0, -2.0])   # 4 checkpoints + final
    snap = {}

    def fake(*a, **k):
        v = next(seq)
        if v == -1.0:
            snap["state"] = copy.deepcopy(eng.recognition_net.state_dict())
        return v

    eng._heldout_elbo_per_latent = fake  # type: ignore[assignment]
    m = eng.train_proposal(model, device="cpu", min_steps=500, steps_per_node=1,
                           check_every=25, patience=3, quality_gate=False)
    assert m["early_stopped"] and m["best_step"] == 25
    for name, param in eng.recognition_net.state_dict().items():
        assert torch.equal(param, snap["state"][name]), name


@pytest.mark.slow
def test_early_stop_can_be_disabled(discrete_problem):
    model = _fit_adapter("cat", "ve", discrete_problem, epochs=5).model
    eng = AmortizedVIEngine(n_samples=256)
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return -1.0

    eng._heldout_elbo_per_latent = counting  # type: ignore[assignment]
    m = eng.train_proposal(model, device="cpu", min_steps=500, steps_per_node=1,
                           early_stop=False, quality_gate=False)
    assert m["early_stopped"] is False and m["grad_steps"] == 500 and calls["n"] == 0


# ---- helpers / plumbing ----------------------------------------------------------

def test_weighted_quantiles_uniform_matches_torch_quantile():
    x = torch.randn(3, 200)
    w = torch.full_like(x, 1.0 / 200)
    u = torch.tensor([0.1, 0.5, 0.9])
    got = _weighted_quantiles(x, w, u)
    ref = torch.quantile(x, u, dim=-1).T
    assert (got - ref).abs().max() < 0.1


@pytest.mark.slow
def test_eval_set_has_downstream_evidence_and_latent_targets(discrete_problem):
    model = _fit_adapter("cat", "ve", discrete_problem, epochs=5).model
    eng = AmortizedVIEngine(n_samples=64)
    eng.train_proposal(model, device="cpu", n_epochs=1, quality_gate=False, early_stop=False)
    pats = eng._make_eval_set(model, n_patterns=8, rows_per_pattern=4)
    assert len(pats) == 8
    for p in pats:
        assert p["target"] not in p["evidence"] and p["evidence"]
        assert all(v.shape == (4, 1) for v in p["evidence"].values())
    # Same seed -> same patterns (fixed set across checkpoints).
    again = eng._make_eval_set(model, n_patterns=8, rows_per_pattern=4)
    assert [p["target"] for p in again] == [p["target"] for p in pats]


@pytest.mark.slow
def test_adapter_records_proposal_used_for_avi(discrete_problem, monkeypatch):
    adapter = _fit_adapter("cat", "avi", discrete_problem, n_samples=256, epochs=5)
    assert adapter.proposal_used == "learned"

    monkeypatch.setattr(AmortizedVIEngine, "_estimate_quality",
                        lambda self, *a, **k: dict(_BAD))
    bad = _fit_adapter("cat", "avi", discrete_problem, n_samples=256, epochs=5)
    assert bad.proposal_used == "lw_fallback"
    post = bad.query(Query(targets=("X2",), evidence={"X0": torch.tensor(1)}, kind="marginal"))
    assert post.probs is not None and torch.isfinite(post.probs).all()
    batch = bad.query_batch([Query(targets=("X2",), evidence={"X0": torch.tensor(v % 2)},
                                   kind="marginal") for v in range(4)])
    assert len(batch) == 4 and all(torch.isfinite(b.probs).all() for b in batch)


def test_engine_is_untrained_until_fit():
    eng = AmortizedVIEngine()
    assert eng.proposal_used is None and eng._fallback is None
