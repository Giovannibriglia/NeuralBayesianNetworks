"""Integration tests for Pass 9 Phase 2 — param-learning pipeline emits
{tv, jsd, w1}_per_node per cell via _compute_metrics_per_node.

Property-based assertions only — no hardcoded TV/JSD/W1 values, since
those are machine-dependent (sandbox vs laptop produce different
floats with identical code).  The meaningful regression catch for
TV value distinctness is already in
test_param_learning_dispatch.py::test_param_learning_dispatch_triplet
assertion 5 (nbn-cat vs nbn-neuralcat differ measurably); this module
pins the row-shape contract instead.
"""
from __future__ import annotations

import math

import pytest
import torch

from benchmarking._baseline_registry import BaselineSpec
from benchmarking._crash_test_utils import CrashTestConfig
from benchmarking.crash_test_runner import _param_learning_cell


@pytest.fixture
def smoke_cfg() -> CrashTestConfig:
    return CrashTestConfig.from_yaml(
        "benchmarking/configs/parameter_learning_smoke.yaml",
    )


def _by_metric(rows):
    return {r.metric: r for r in rows}


def test_param_learning_discrete_emits_tv_jsd_w1_skip(smoke_cfg):
    """Discrete (nbn-cat n=5): TV + JSD + W1-as-not_supported + total_time_s.

    Property assertions:
    - TV / JSD: status='ok', value finite and in [0, 1]
    - W1: status='not_supported', NaN value, error_msg cites the
      Hamming-collapse rationale
    - Pinsker bound: JSD/log(2) ≤ TV (Lin 1991)
    - No 'accuracy' row (param-learning emits explicit names only;
      'accuracy' is the inference-side family-dispatched alias)
    """
    torch.manual_seed(0)
    spec = BaselineSpec(library="nbn", mechanism="cat", param_method="mle")
    rows = _param_learning_cell(smoke_cfg, "discrete", 5, 0, spec)
    m = _by_metric(rows)

    assert "accuracy" not in m, (
        "param-learning pipeline must not emit 'accuracy' rows; "
        "the family-dispatched alias is inference-only"
    )
    for name in ("tv_per_node", "jsd_per_node", "w1_per_node", "total_time_s"):
        assert name in m, f"missing required metric row: {name}"

    assert m["tv_per_node"].status == "ok"
    assert 0 <= m["tv_per_node"].value <= 1
    assert math.isfinite(m["tv_per_node"].value)

    assert m["jsd_per_node"].status == "ok"
    assert 0 <= m["jsd_per_node"].value <= 1
    assert math.isfinite(m["jsd_per_node"].value)

    # Pinsker / Lin 1991: JSD/log(2) ≤ TV on discrete distributions.
    assert m["jsd_per_node"].value <= m["tv_per_node"].value + 1e-6, (
        f"Pinsker bound violated: JSD={m['jsd_per_node'].value}, "
        f"TV={m['tv_per_node'].value}"
    )

    assert m["w1_per_node"].status == "not_supported"
    assert math.isnan(m["w1_per_node"].value)
    err = m["w1_per_node"].extra.get("error_msg", "")
    assert "Hamming" in err or "1/2 * TV" in err, (
        f"W1 skip reason should cite the Hamming-collapse rationale; "
        f"got error_msg={err!r}"
    )


def test_param_learning_continuous_lg_emits_all_three_ok(smoke_cfg):
    """Continuous_lg (nbn-lg n=5): TV (binned) + JSD (binned) + W1 (sample).

    All three metrics 'ok' with finite, non-negative values.  No
    hardcoded magnitudes (machine-dependent).  No 'accuracy' row.
    """
    torch.manual_seed(0)
    spec = BaselineSpec(library="nbn", mechanism="lg", param_method="mle")
    rows = _param_learning_cell(smoke_cfg, "continuous_lg", 5, 0, spec)
    m = _by_metric(rows)

    assert "accuracy" not in m
    for name in ("tv_per_node", "jsd_per_node", "w1_per_node", "total_time_s"):
        assert name in m, f"missing required metric row: {name}"
        assert m[name].status == "ok"
        assert math.isfinite(m[name].value)
        assert m[name].value >= 0


def test_param_learning_pgmpy_discrete_same_row_shape(smoke_cfg):
    """pgmpy-fitted discrete cell follows the same emit shape as nbn-fitted.

    Catches a regression where _score_param_learning_metrics_pgmpy
    diverges from _score_param_learning_metrics_nbn in row count or
    metric naming.  Property: the SET of metric names is identical
    between the two paths on the same family.
    """
    pytest.importorskip("pgmpy")
    torch.manual_seed(0)
    nbn_spec = BaselineSpec(library="nbn", mechanism="cat", param_method="mle")
    pgmpy_spec = BaselineSpec(
        library="pgmpy", mechanism="discrete", param_method="mle",
    )
    nbn_rows = _param_learning_cell(smoke_cfg, "discrete", 5, 0, nbn_spec)
    torch.manual_seed(0)
    pgmpy_rows = _param_learning_cell(smoke_cfg, "discrete", 5, 0, pgmpy_spec)
    assert {r.metric for r in nbn_rows} == {r.metric for r in pgmpy_rows}, (
        "nbn-fitted and pgmpy-fitted discrete cells must emit the same "
        "metric-name set; divergence indicates a Phase 2 wiring drift"
    )
