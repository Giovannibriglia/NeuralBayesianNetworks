"""Tests for Phase 3 Stage 2 — data model + oracle None-handling.

Covers:
- Query.evidence accepting None values (type signature change).
- CellResult.evidence_mode field + backward-compat default.
- _evidence_mode_for helper classification.
- Oracle (filter_ground_truth + forward_with_clamp) skipping None evidence.
- evidence_mode propagation Query → Measurement → CellResult.

Adapter None-handling lands in Stage 3.

Reference: docs/phase3-design-draft.md §3, §5.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from benchmarking.core.oracle import (
    filter_ground_truth,
    forward_with_clamp_posterior_samples,
)
from benchmarking.core.results import VALID_EVIDENCE_MODES, CellResult
from benchmarking.core.runner import _evidence_mode_for
from benchmarking.domains.base import BenchmarkProblem, GroundTruth, Query
from benchmarking.measurements import TimingOnly


# --- Fixtures ---


def _make_problem_with_gt() -> BenchmarkProblem:
    """3-node discrete problem (Y→X, Z→X) with a ground-truth sample pool."""
    edges = [("Y", "X"), ("Z", "X")]
    variables = dict.fromkeys(("X", "Y", "Z"), ("discrete", 2))
    # Column order is the topological sort; build samples in that order.
    from benchmarking.core.oracle import _column_order

    test_data = {n: torch.zeros(8) for n in ("X", "Y", "Z")}
    problem = BenchmarkProblem(
        name="gt3",
        dag=edges,
        variables=variables,
        train_data={},
        test_data=test_data,
        queries=[],
        seed=0,
        family="discrete",
        problem_id="gt3",
        true_model=None,
    )
    col_order = _column_order(problem)
    gen = torch.Generator().manual_seed(0)
    n_rows = 40
    cols = [torch.randint(0, 2, (n_rows,), generator=gen) for _ in col_order]
    problem.ground_truth = GroundTruth(samples=torch.stack(cols, dim=1).float())
    return problem


def _make_minimal_problem() -> BenchmarkProblem:
    """Minimal problem for Measurement-level propagation tests."""
    variables = dict.fromkeys(("X", "Y"), ("discrete", 2))
    return BenchmarkProblem(
        name="m",
        dag=[("Y", "X")],
        variables=variables,
        train_data={},
        test_data={"X": torch.zeros(4), "Y": torch.zeros(4)},
        queries=[],
        seed=0,
        family="discrete",
        problem_id="m",
        true_model=None,
    )


class TestEvidenceModeDataModel:
    """Stage 2: Query None values + CellResult.evidence_mode."""

    def test_query_accepts_none_evidence_values(self):
        """Type signature change: Query can be constructed with None values."""
        q = Query(
            targets=("X",),
            evidence={"Y": None, "Z": None},
            kind="marginal",
        )
        assert q.evidence["Y"] is None

    def test_cell_result_default_evidence_mode_is_full(self):
        """Backward-compat: existing CellResult construction → 'full'."""
        cr = CellResult(
            benchmark="test", family="discrete", problem_id="t",
            seed=0, baseline="x", query_role="random",
            metric="tv_per_node", value=0.1, status="ok",
            fit_time_s=0.0, query_time_s=0.0, metrics_time_s=0.0,
        )
        assert cr.evidence_mode == "full"

    def test_cell_result_accepts_empty_mode(self):
        """CellResult can be constructed with evidence_mode='empty'."""
        cr = CellResult(
            benchmark="test", family="discrete", problem_id="t",
            seed=0, baseline="x", query_role="random",
            metric="tv_per_node", value=0.1, status="ok",
            fit_time_s=0.0, query_time_s=0.0, metrics_time_s=0.0,
            evidence_mode="empty",
        )
        assert cr.evidence_mode == "empty"

    def test_valid_evidence_modes_set(self):
        """VALID_EVIDENCE_MODES enumerates the two modes."""
        assert frozenset({"full", "empty"}) == VALID_EVIDENCE_MODES

    def test_evidence_mode_helper(self):
        """_evidence_mode_for correctly classifies queries."""
        q_full = Query(targets=("X",), evidence={"Y": 1.0, "Z": 2.0}, kind="marginal")
        q_empty = Query(targets=("X",), evidence={"Y": None, "Z": None}, kind="marginal")
        assert _evidence_mode_for(q_full) == "full"
        assert _evidence_mode_for(q_empty) == "empty"

    def test_evidence_mode_helper_mixed_is_empty(self):
        """Defensive: any None → empty mode."""
        q_mixed = Query(targets=("X",), evidence={"Y": 1.0, "Z": None}, kind="marginal")
        assert _evidence_mode_for(q_mixed) == "empty"

    def test_evidence_mode_helper_no_evidence_is_full(self):
        """Empty evidence dict has no None values → full mode."""
        q_none = Query(targets=("X",), evidence={}, kind="marginal")
        assert _evidence_mode_for(q_none) == "full"


class TestOracleNoneHandling:
    """Stage 2: oracle functions skip None-valued evidence."""

    def test_filter_ground_truth_skips_none(self):
        """None-valued evidence is marginalized (no filtering), not a crash."""
        problem = _make_problem_with_gt()
        # All-None evidence → no filtering → returns the full target column.
        out = filter_ground_truth(problem, {"Y": None, "Z": None}, target="X")
        assert out is not None
        assert out.shape[0] == 40  # no rows filtered out

    def test_filter_ground_truth_concrete_still_works(self):
        """Concrete evidence still filters the pool (backward-compat)."""
        problem = _make_problem_with_gt()
        out = filter_ground_truth(
            problem, {"Y": torch.tensor(1.0)}, target="X"
        )
        # Either a filtered tensor or None (if too few survive); must not crash.
        assert out is None or out.dim() == 1

    def test_filter_ground_truth_mixed_skips_only_none(self):
        """A mix: concrete Y filters, None Z is skipped — no crash."""
        problem = _make_problem_with_gt()
        out = filter_ground_truth(
            problem, {"Y": torch.tensor(0.0), "Z": None}, target="X"
        )
        assert out is None or out.dim() == 1

    def test_forward_with_clamp_skips_none(self):
        """None evidence values are dropped before reaching true_model.sample."""
        problem = _make_minimal_problem()
        captured = {}

        class _FakeModel:
            def sample(self, n, evidence):
                captured["evidence"] = evidence
                return {"X": torch.zeros(n, 1)}

        problem.true_model = _FakeModel()
        out = forward_with_clamp_posterior_samples(
            problem, targets=["X"], evidence={"Y": None}, n_samples=5
        )
        assert out is not None
        # None-valued Y must NOT have been passed to sample().
        assert "Y" not in captured["evidence"]


class TestEvidenceModePropagation:
    """Stage 2: evidence_mode threads Query → Measurement → CellResult.

    Uses TimingOnly with a mock adapter to avoid real inference (Stage 3).
    """

    def _mock_adapter(self):
        adapter = MagicMock()
        adapter.name = "mock-baseline"
        adapter.query.return_value = None
        return adapter

    def test_full_mode_query_produces_full_cellresult(self):
        problem = _make_minimal_problem()
        q = Query(targets=("X",), evidence={"Y": 1.0}, kind="marginal")
        rows = TimingOnly().measure(
            problem, self._mock_adapter(), [q],
            evidence_modes=["full"],
        )
        assert rows
        assert all(r.evidence_mode == "full" for r in rows)

    def test_empty_mode_query_produces_empty_cellresult(self):
        problem = _make_minimal_problem()
        q = Query(targets=("X",), evidence={"Y": None}, kind="marginal")
        # Derive evidence_modes the way the runner does.
        modes = [_evidence_mode_for(q)]
        rows = TimingOnly().measure(
            problem, self._mock_adapter(), [q],
            evidence_modes=modes,
        )
        assert rows
        assert all(r.evidence_mode == "empty" for r in rows)

    def test_evidence_modes_defaults_to_full(self):
        """Omitting evidence_modes → all rows 'full' (backward-compat)."""
        problem = _make_minimal_problem()
        q = Query(targets=("X",), evidence={"Y": 1.0}, kind="marginal")
        rows = TimingOnly().measure(problem, self._mock_adapter(), [q])
        assert rows
        assert all(r.evidence_mode == "full" for r in rows)

    def test_evidence_modes_length_validation(self):
        """measure() rejects evidence_modes length mismatch."""
        problem = _make_minimal_problem()
        q = Query(targets=("X",), evidence={"Y": 1.0}, kind="marginal")
        with pytest.raises(ValueError, match="evidence_modes length"):
            TimingOnly().measure(
                problem, self._mock_adapter(), [q],
                evidence_modes=["full", "empty"],  # len 2 != 1
            )
