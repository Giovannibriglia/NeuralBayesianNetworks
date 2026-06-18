"""Tests for NBNAdapter.score_data — parameter-learning scoring (#109 PR 1).

Covers the held-out joint log-likelihood path:
  - numeric correctness vs an independent numpy recomputation from the fitted
    categorical _logits (the joint = sum over nodes of log P(x | parents));
  - the metric cell value == mean over test rows;
  - score_data == sum of independent single-node mechanism.log_prob calls;
  - out-of-support discrete test rows surface as status="error" (not an opaque
    IndexError) through ParamLearningMeasurement;
  - the supports_scoring gate: adapters without the flag stay not_supported.

The fit-based tests are @pytest.mark.slow (they train a tiny net), matching the
NBNAdapter behavioral-test convention.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from benchmarking.adapters import NBNAdapter
from benchmarking.domains.base import BenchmarkProblem
from benchmarking.measurements import ParamLearningMeasurement


# ---------------------------------------------------------------------------
# Fixture: a tiny known categorical chain X0 -> X1 -> X2 (all binary).
# ---------------------------------------------------------------------------

def _make_chain_problem(seed: int = 0) -> BenchmarkProblem:
    """3-node binary chain with a held-out test split distinct from train."""
    import networkx as nx
    g = torch.Generator().manual_seed(seed)

    dag = nx.DiGraph()
    dag.add_edges_from([("X0", "X1"), ("X1", "X2")])
    variables = {"X0": ("discrete", 2), "X1": ("discrete", 2), "X2": ("discrete", 2)}

    def _draw(n: int) -> dict[str, torch.Tensor]:
        # Correlated chain so the fitted CPTs are non-uniform (a stronger check
        # than independent uniform draws). Each child copies its parent w.p. ~0.8.
        x0 = torch.randint(0, 2, (n,), generator=g)
        flip1 = (torch.rand(n, generator=g) < 0.2).long()
        x1 = x0 ^ flip1
        flip2 = (torch.rand(n, generator=g) < 0.2).long()
        x2 = x1 ^ flip2
        return {"X0": x0, "X1": x1, "X2": x2}

    return BenchmarkProblem(
        name="chain",
        dag=dag,
        variables=variables,
        train_data=_draw(400),
        test_data=_draw(64),
        queries=[],
        family="discrete",
        problem_id="3",
        seed=seed,
    )


def _fit_cat_adapter(problem: BenchmarkProblem) -> NBNAdapter:
    adapter = NBNAdapter(mechanism="cat", engine="ve", device="cpu")
    adapter.fit(problem, epochs=5)
    return adapter


# ---- independent numpy-from-_logits recomputation helpers ----

def _cls_to_idx(mech, value: int) -> int:
    """Map a raw class value to its contiguous CPT column index."""
    cv = getattr(mech, "_class_values", None)
    if cv is None:
        return int(value)
    return cv.long().tolist().index(int(value))


def _parent_row(mech, parent_values: list[int]) -> int:
    """Replicate CategoricalTableMechanism._parent_to_row_idx (mixed radix)."""
    strides = getattr(mech, "_parent_strides", None)
    if not strides:
        return 0
    return int(sum(int(v) * int(s) for v, s in zip(parent_values, strides)))


def _reference_joint_logprobs(adapter: NBNAdapter,
                              test_data: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-row joint log-prob recomputed independently of score_data().

    Reads the fitted categorical ``_logits`` directly, normalizes with
    log_softmax, and indexes each test row's (parent-config row, class column)
    by hand, summing over nodes. No use of score_data's vectorized assembly.
    """
    model = adapter.model
    order = model.dag.topological_order()
    b = next(iter(test_data.values())).shape[0]
    ref = torch.zeros(b, dtype=torch.float64)
    for node in order:
        mech = model.mechanisms[node]
        table = F.log_softmax(mech._logits.detach().double(), dim=-1)  # [rows, K]
        parents = model.dag.parents(node)
        for i in range(b):
            pvals = [int(test_data[p][i]) for p in parents]
            row = _parent_row(mech, pvals)
            col = _cls_to_idx(mech, int(test_data[node][i]))
            ref[i] += table[row, col]
    return ref


# ---- Tests ----

@pytest.mark.slow
def test_score_data_matches_independent_logits_recomputation():
    problem = _make_chain_problem(seed=1)
    adapter = _fit_cat_adapter(problem)

    got = adapter.score_data(problem.test_data).double()
    ref = _reference_joint_logprobs(adapter, problem.test_data)

    assert got.shape == (problem.test_data["X0"].shape[0],)
    assert torch.allclose(got, ref, atol=1e-5), (
        f"score_data joints diverge from the from-_logits recomputation\n"
        f"max abs diff: {(got - ref).abs().max().item():.3e}"
    )

    # The metric cell value must equal the mean over test rows.
    rows = ParamLearningMeasurement().measure(
        problem, adapter, [], fit_time_s=0.0, benchmark="synthetic",
        seed=problem.seed,
    )
    # The measurement now emits log_likelihood + the two recovery rows (#109
    # PR 2); this test concerns the log_likelihood scoring only.
    cell = next(r for r in rows if r.metric == "log_likelihood")
    assert cell.status == "ok"
    assert math.isclose(cell.value, float(ref.mean()), rel_tol=1e-5, abs_tol=1e-5)


@pytest.mark.slow
def test_score_data_equals_sum_of_per_node_logprob():
    from nbn.utils.batching import pack_parents

    problem = _make_chain_problem(seed=2)
    adapter = _fit_cat_adapter(problem)
    model = adapter.model

    data = {k: v.to(adapter.device) for k, v in problem.test_data.items()}
    manual = torch.zeros(data["X0"].shape[0])
    for node in model.dag.topological_order():
        mech = model.mechanisms[node]
        pa = pack_parents(data, model.dag.parents(node))
        manual = manual + mech.log_prob(data[node], pa).reshape(-1).detach().cpu()

    got = adapter.score_data(problem.test_data)
    assert torch.allclose(got, manual, atol=1e-6)


@pytest.mark.slow
def test_out_of_range_test_value_surfaces_as_error():
    problem = _make_chain_problem(seed=3)
    adapter = _fit_cat_adapter(problem)

    # Corrupt one held-out row with a class value outside [0, 2).
    bad = {k: v.clone() for k, v in problem.test_data.items()}
    bad["X1"][0] = 5
    bad_problem = BenchmarkProblem(
        name=problem.name, dag=problem.dag, variables=problem.variables,
        train_data=problem.train_data, test_data=bad, queries=[],
        family="discrete", problem_id="3", seed=problem.seed,
    )

    rows = ParamLearningMeasurement().measure(
        bad_problem, adapter, [], fit_time_s=0.0, benchmark="synthetic",
        seed=problem.seed,
    )
    # The out-of-range value corrupts held-out scoring, so the log_likelihood
    # row is the error row (recovery rows are independent — bad_problem has no
    # true_model, so they are not_applicable).
    cell = next(r for r in rows if r.metric == "log_likelihood")
    # Classified as a real error (not "not_supported", not an uncaught crash).
    assert cell.status == "error", cell
    assert math.isnan(cell.value)
    assert "out of range" in (cell.error_msg or "")

    # And calling score_data directly raises a clean ValueError, not IndexError.
    with pytest.raises(ValueError, match="out of range"):
        adapter.score_data(bad)


def test_not_supported_gate_holds_for_non_scoring_adapter():
    """An adapter without supports_scoring is never scored (#109)."""
    problem = _make_chain_problem(seed=4)

    class _NoScore:
        name = "pgmpy-mle-ve"

        def score_data(self, test_data):  # pragma: no cover - must never run
            raise AssertionError("score_data must not be called when gated off")

    rows = ParamLearningMeasurement().measure(
        problem, _NoScore(), [], fit_time_s=0.0, benchmark="synthetic",
        seed=problem.seed,
    )
    # The measurement emits the log_likelihood row plus the two
    # parameter-recovery rows (#109 PR 2); _NoScore opts into none, so every
    # row is not_supported with a NaN value.
    by_metric = {r.metric: r for r in rows}
    assert set(by_metric) == {
        "log_likelihood", "param_recovery_tv", "param_recovery_kl"
    }
    for r in rows:
        assert r.status == "not_supported"
        assert math.isnan(r.value)
