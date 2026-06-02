"""Phase 3 Stage 3 — per-adapter None-evidence handling.

Each test fits a real model and issues an empty-mode query (evidence value
is ``None``) plus, where relevant, a partial-None query. The contract is
minimal: the adapter must NOT crash on ``None`` and must return a valid
Posterior (marginalizing the None-valued variables). Rigorous V1-vs-V2
comparison happens in the end-to-end smoke.

These fit real models, so — like the behavioral tests in the per-adapter
suites — they are marked ``slow`` and excluded from the fast gate.

Reference: docs/phase3-design-draft.md §4.
"""
from __future__ import annotations

import pytest
import torch

from benchmarking.adapters import (
    NBNAdapter,
    PgmpyAdapter,
    PomegranateAdapter,
    PyroAdapter,
)
from benchmarking.domains.base import BenchmarkProblem, Query
from benchmarking.domains.posterior import Posterior


# ---- Helpers ----------------------------------------------------------------

def _discrete_problem(n_samples: int = 400, seed: int = 0) -> BenchmarkProblem:
    """3-node binary BN: X0 → X1 → X2."""
    torch.manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2")]
    train_data = {
        "X0": torch.randint(0, 2, (n_samples,)),
        "X1": torch.randint(0, 2, (n_samples,)),
        "X2": torch.randint(0, 2, (n_samples,)),
    }
    variables = dict.fromkeys(train_data, ("discrete", 2))
    return BenchmarkProblem(
        name="disc", dag=dag, variables=variables,
        train_data=train_data, test_data=train_data, queries=[],
    )


def _lg_problem(n_samples: int = 400, seed: int = 0) -> BenchmarkProblem:
    """3-node linear-Gaussian chain: X0 → X1 → X2."""
    torch.manual_seed(seed)
    dag = [("X0", "X1"), ("X1", "X2")]
    x0 = torch.randn(n_samples)
    x1 = 0.5 * x0 + 0.5 * torch.randn(n_samples)
    x2 = 0.5 * x1 + 0.5 * torch.randn(n_samples)
    train_data = {"X0": x0, "X1": x1, "X2": x2}
    variables = dict.fromkeys(train_data, ("continuous", None))
    return BenchmarkProblem(
        name="lg", dag=dag, variables=variables,
        train_data=train_data, test_data=train_data, queries=[],
    )


def _assert_valid_discrete(posterior: Posterior) -> None:
    assert isinstance(posterior, Posterior)
    assert posterior.probs is not None
    assert posterior.probs.shape == (2,)
    assert torch.isfinite(posterior.probs).all()
    assert torch.isclose(posterior.probs.sum(), torch.tensor(1.0), atol=1e-3)


def _assert_valid_samples(posterior: Posterior) -> None:
    assert isinstance(posterior, Posterior)
    assert posterior.samples is not None
    assert posterior.samples.ndim == 1
    assert torch.isfinite(posterior.samples).all()


# ---- Tests ------------------------------------------------------------------

@pytest.mark.slow
class TestAdapterNoneEvidence:
    """Empty-mode (None-valued) evidence runs without crashing on each adapter."""

    def test_pgmpy_mle_ve_none_evidence(self):
        adapter = PgmpyAdapter(param_method="mle", inference_method="ve")
        adapter.fit(_discrete_problem(seed=1))
        q = Query(targets=("X2",), evidence={"X0": None}, kind="marginal")
        _assert_valid_discrete(adapter.query(q))

    def test_pgmpy_bayes_ve_none_evidence(self):
        adapter = PgmpyAdapter(param_method="bayes", inference_method="ve")
        adapter.fit(_discrete_problem(seed=2))
        q = Query(targets=("X2",), evidence={"X0": None, "X1": None}, kind="marginal")
        _assert_valid_discrete(adapter.query(q))

    def test_pgmpy_lg_predict_none_evidence(self):
        adapter = PgmpyAdapter(
            param_method="lg", inference_method="predict", n_samples=256,
        )
        adapter.fit(_lg_problem(seed=3))
        q = Query(targets=("X2",), evidence={"X0": None}, kind="marginal")
        _assert_valid_samples(adapter.query(q))

    def test_pgmpy_lg_predict_partial_none_evidence(self):
        """Mixed: X0 observed, X1 None — uses X0, marginalizes X1 (alignment fix)."""
        adapter = PgmpyAdapter(
            param_method="lg", inference_method="predict", n_samples=256,
        )
        adapter.fit(_lg_problem(seed=4))
        q = Query(
            targets=("X2",),
            evidence={"X0": torch.tensor(1.0), "X1": None},
            kind="marginal",
        )
        _assert_valid_samples(adapter.query(q))

    def test_pomegranate_none_evidence(self):
        adapter = PomegranateAdapter()
        adapter.fit(_discrete_problem(seed=5))
        q = Query(targets=("X2",), evidence={"X0": None}, kind="marginal")
        _assert_valid_discrete(adapter.query(q))

    def test_nbn_cat_ve_none_evidence(self):
        adapter = NBNAdapter(mechanism="cat", engine="ve")
        adapter.fit(_discrete_problem(seed=6))
        q = Query(targets=("X2",), evidence={"X0": None}, kind="marginal")
        _assert_valid_discrete(adapter.query(q))

    def test_pyro_none_evidence(self):
        adapter = PyroAdapter(mechanism="empirical", inference_method="importance")
        adapter.fit(_discrete_problem(seed=7))
        q = Query(targets=("X2",), evidence={"X0": None}, kind="marginal")
        posterior = adapter.query(q)
        # Pyro returns a probs histogram for discrete targets; accept either
        # contract form defensively.
        if posterior.probs is not None:
            assert torch.isfinite(posterior.probs).all()
        else:
            _assert_valid_samples(posterior)
