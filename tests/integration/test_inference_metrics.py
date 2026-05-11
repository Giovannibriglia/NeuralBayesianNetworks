"""Integration tests for Pass 9 Phase 3 — inference pipeline emits
{tv, jsd, w1}_per_node per cell via _compute_inference_metrics, and the
family-dispatched accuracy row aliases the appropriate per-metric value.

Property-based assertions only — no hardcoded TV/JSD/W1 values, since
those are machine-dependent.  The meaningful regression catch for
accuracy-aliasing correctness is the math.isclose check in
test_inference_discrete_accuracy_aliases_tv and
test_inference_continuous_accuracy_aliases_w1.
"""
from __future__ import annotations

import math

import pytest
import torch

from benchmarking._crash_test_utils import CrashTestConfig
from benchmarking.crash_test_runner import _inference_cell


@pytest.fixture
def smoke_cfg() -> CrashTestConfig:
    return CrashTestConfig.from_yaml(
        "benchmarking/configs/inference_smoke.yaml",
    )


def _by_metric(rows):
    return {r.metric: r for r in rows}


def test_inference_discrete_accuracy_aliases_tv(smoke_cfg):
    """Discrete (pgmpy, n=5): accuracy row is the family-dispatched alias of tv_per_node.

    Property assertions:
    - accuracy: status='ok', value == tv_per_node.value (bit-exact alias)
    - tv_per_node: status='ok', finite, in [0, 1]
    - jsd_per_node: status='ok', Pinsker bound (JSD/log2 ≤ TV)
    - w1_per_node: status='not_supported', NaN value
    - total_time_s: present
    """
    pytest.importorskip("pgmpy")
    torch.manual_seed(0)
    rows = _inference_cell(smoke_cfg, "discrete", 5, 0, "pgmpy", "pgmpy-mle-ve")
    m = _by_metric(rows)

    for name in ("accuracy", "tv_per_node", "jsd_per_node", "w1_per_node", "total_time_s"):
        assert name in m, f"missing required metric row: {name}"

    assert m["accuracy"].status == "ok"
    assert m["tv_per_node"].status == "ok"
    assert math.isclose(m["accuracy"].value, m["tv_per_node"].value, abs_tol=1e-9), (
        "family-dispatched accuracy must equal tv_per_node value on discrete; "
        f"accuracy={m['accuracy'].value}, tv_per_node={m['tv_per_node'].value}"
    )

    assert 0 <= m["tv_per_node"].value <= 1
    assert math.isfinite(m["tv_per_node"].value)

    assert m["jsd_per_node"].status == "ok"
    assert m["jsd_per_node"].value <= m["tv_per_node"].value + 1e-6, (
        f"Pinsker bound violated: JSD={m['jsd_per_node'].value}, "
        f"TV={m['tv_per_node'].value}"
    )

    assert m["w1_per_node"].status == "not_supported"
    assert math.isnan(m["w1_per_node"].value)


def test_inference_continuous_accuracy_aliases_w1(smoke_cfg):
    """Continuous_lg (nbn_lw, n=5): accuracy row is the family-dispatched alias of w1_per_node.

    Property assertions:
    - accuracy: status='ok', value == w1_per_node.value (bit-exact alias)
    - All three per-metric rows: status='ok', finite, non-negative
    - total_time_s: present
    """
    torch.manual_seed(0)
    rows = _inference_cell(smoke_cfg, "continuous_lg", 5, 0, "nbn_lw", "nbn-lg-lw")
    m = _by_metric(rows)

    for name in ("accuracy", "tv_per_node", "jsd_per_node", "w1_per_node", "total_time_s"):
        assert name in m, f"missing required metric row: {name}"

    assert m["accuracy"].status == "ok"
    assert m["w1_per_node"].status == "ok"
    assert math.isclose(m["accuracy"].value, m["w1_per_node"].value, abs_tol=1e-9), (
        "family-dispatched accuracy must equal w1_per_node value on continuous; "
        f"accuracy={m['accuracy'].value}, w1_per_node={m['w1_per_node'].value}"
    )

    for name in ("tv_per_node", "jsd_per_node", "w1_per_node"):
        assert m[name].status == "ok"
        assert math.isfinite(m[name].value)
        assert m[name].value >= 0


def test_inference_hybrid_emits_all_rows(smoke_cfg):
    """Hybrid (nbn_lw, n=5): all five metric-name keys present.

    Mixed discrete+continuous nodes → all three per-metric rows should
    appear alongside accuracy and total_time_s.  No value constraints —
    machine-dependent hybrid mix.
    """
    torch.manual_seed(0)
    rows = _inference_cell(smoke_cfg, "hybrid", 5, 0, "nbn_lw", "nbn-mdn-lw")
    m = _by_metric(rows)

    for name in ("accuracy", "tv_per_node", "jsd_per_node", "w1_per_node", "total_time_s"):
        assert name in m, f"missing required metric row: {name}"


def test_inference_accuracy_not_applicable_short_circuit(smoke_cfg):
    """accuracy_supported=False short-circuit: only the accuracy row emitted (status='not_supported').

    gpytorch on continuous_lg hits the short-circuit that skips
    _compute_inference_metrics entirely.  The cell must emit exactly one
    accuracy row with status='not_supported' and NO explicit per-metric rows.
    """
    pytest.importorskip("gpytorch")
    torch.manual_seed(0)
    rows = _inference_cell(smoke_cfg, "continuous_lg", 5, 0, "gpytorch", "gpytorch-gp-predict")

    accuracy_rows = [r for r in rows if r.metric == "accuracy"]
    explicit_rows = [r for r in rows if r.metric in ("tv_per_node", "jsd_per_node", "w1_per_node")]

    assert len(accuracy_rows) == 1
    assert accuracy_rows[0].status == "not_supported"
    assert len(explicit_rows) == 0, (
        "accuracy_supported=False short-circuit must emit ONLY the accuracy row, "
        f"not explicit per-metric rows; got {[r.metric for r in explicit_rows]}"
    )
