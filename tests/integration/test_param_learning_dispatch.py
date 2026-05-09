"""v0.8 regression: parameter-learning dispatch + mechanism + adapter triplet.

Pins the runtime behaviour after the v0.7-#43 audit's three runtime fixes
(#51 runner dispatch, #52 NeuralCategoricalMechanism root-node training,
#53 _fit_and_score_pgmpy param_method plumbing + pgmpy 1.x API adapt).

Pre-fix the v0.6c-d parameter-learning parquet showed bit-identical
``tv_per_node`` rows for ``(nbn-cat, nbn-neuralcat)`` and
``(pgmpy-mle, pgmpy-bayes)`` (max abs diff = 0.0 across 24-25 cells per
pair) because the runner dispatched on a polymorphic legacy adapter
string and silently relabeled rows post-hoc.  The audit at
``docs/audits/v0.7-43-fit-path-audit.md`` documents the full diagnosis;
this test locks in the runtime fix.

Five assertions:

1. ``pgmpy-mle`` and ``pgmpy-bayes`` produce distinct TV on a fixture
   where the fits genuinely differ (~0.5% relative margin at
   ``n_train=500``; catches #51 dispatch + #53 plumbing/API regressions).

2. ``fresh_mechanism_for_spec`` returns distinct mechanism *classes*
   for ``nbn-cat`` vs ``nbn-neuralcat`` specs (catches #51 mechanism-factory
   regression even if downstream TV happens to agree by numerical
   coincidence on a different fixture).

3. ``_param_learning_cell`` signature accepts ``BaselineSpec`` (source
   inspection — catches a revert to the legacy-string signature).

4. ``NeuralCategoricalMechanism`` root-node logits reflect empirical
   log-frequency, not uniform (catches a #52-only regression even if
   downstream TV happens to differ for unrelated reasons).

5. ``nbn-cat`` and ``nbn-neuralcat`` produce distinct TV on the same
   fixture — the most direct echo of the v0.6c-d bit-identical pattern
   for the NBN pair.  This assertion was originally dropped because
   ``_avg_tv_per_node`` read ``mech._logits`` directly and crashed on
   ``NeuralCategoricalMechanism`` (no flat ``_logits`` attribute).
   v0.8-#59 + v0.8-#26 introduced ``Mechanism.tabulate(parent_cards)``
   on the discrete mechanisms and rewired the metric site to use it,
   so the NBN-pair TV comparison can now run.
"""
from __future__ import annotations

import inspect
import warnings

import pytest
import torch

from benchmarking._baseline_registry import BaselineSpec
from benchmarking._crash_test_utils import CrashTestConfig, fresh_mechanism_for_spec
from benchmarking.crash_test_runner import (
    _fit_and_score_pgmpy,
    _param_learning_cell,
)
from benchmarking.synthetic import make_synthetic_bn
from nbn.mechanisms.categorical_table import CategoricalTableMechanism
from nbn.mechanisms.neural_categorical import NeuralCategoricalMechanism


def _build_fixture_bn():
    """v0.8 dispatch-test fixture.

    n_train=500 chosen empirically: at this scale the BDeu prior at
    ESS=5 contributes ~0.5% of posterior mass (vs ~0.05% at n_train=10000),
    giving the pgmpy-mle vs pgmpy-bayes assertion comfortable margin
    above CI numerical noise.  cardinality=4 matches the canonical
    paper configuration.
    """
    return make_synthetic_bn(
        family="discrete",
        n_nodes=5,
        edge_density=0.20,
        max_in_degree=2,
        cardinality=4,
        fraction_continuous=0.0,
        n_train=500,
        n_test=200,
        n_reference=200,
        seed=0,
        device="cpu",
    )


@pytest.fixture
def fixture_bn():
    return _build_fixture_bn()


@pytest.fixture
def smoke_cfg() -> CrashTestConfig:
    return CrashTestConfig.from_yaml(
        "benchmarking/configs/parameter_learning_smoke.yaml",
    )


def test_param_learning_dispatch_triplet(
    fixture_bn, smoke_cfg: CrashTestConfig,
) -> None:
    """v0.8-#51/#52/#53/#59/#26 regression: pre-fix the v0.6c-d parquet
    showed bit-identical TV across ``(nbn-cat, nbn-neuralcat)`` and
    ``(pgmpy-mle, pgmpy-bayes)``; post-fix each must be measurably
    distinct or structurally caught.  Five assertions; see module
    docstring for the per-assertion rationale.
    """
    bn = fixture_bn
    cat_spec       = BaselineSpec(library="nbn",   mechanism="cat",       param_method="mle")
    neuralcat_spec = BaselineSpec(library="nbn",   mechanism="neuralcat", param_method="mle")
    mle_spec       = BaselineSpec(library="pgmpy", mechanism="discrete",  param_method="mle")
    bayes_spec     = BaselineSpec(library="pgmpy", mechanism="discrete",  param_method="bayes")

    # ── Assertion 1: pgmpy-mle vs pgmpy-bayes distinct TV ─────────────
    # v0.8 Pass-9: _fit_and_score_pgmpy now returns
    # list[tuple[name, value, status, error_msg]] — one row per metric
    # in {tv_per_node, jsd_per_node, w1_per_node}.  Pluck tv_per_node
    # via name lookup; the assertion semantics (bit-identical TV
    # detection) are unchanged.
    def _tv_value(rows):
        for name, value, status, _err in rows:
            if name == "tv_per_node" and status == "ok":
                return value
        raise AssertionError(f"no tv_per_node row in {rows!r}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mle_rows   = _fit_and_score_pgmpy(bn, "discrete", mle_spec)
        bayes_rows = _fit_and_score_pgmpy(bn, "discrete", bayes_spec)
    mle_value = _tv_value(mle_rows)
    bayes_value = _tv_value(bayes_rows)
    mle_bayes_diff = abs(mle_value - bayes_value)
    assert mle_bayes_diff > 1e-6, (
        f"pgmpy-mle ({mle_value:.6e}) and pgmpy-bayes ({bayes_value:.6e}) "
        f"are bit-identical (diff={mle_bayes_diff:.6e}); the v0.6c-d "
        f"runner-dispatch pattern has regressed (#51) or "
        f"_fit_and_score_pgmpy is no longer plumbing param_method (#53)"
    )

    # ── Assertion 2: fresh_mechanism_for_spec returns distinct classes ─
    cat_mech       = fresh_mechanism_for_spec(cat_spec,       "discrete", "discrete", cardinality=4)
    neuralcat_mech = fresh_mechanism_for_spec(neuralcat_spec, "discrete", "discrete", cardinality=4)
    assert isinstance(cat_mech,       CategoricalTableMechanism), (
        f"nbn-cat spec must return CategoricalTableMechanism; got {type(cat_mech).__name__}"
    )
    assert isinstance(neuralcat_mech, NeuralCategoricalMechanism), (
        f"nbn-neuralcat spec must return NeuralCategoricalMechanism; got {type(neuralcat_mech).__name__}"
    )
    assert type(cat_mech) is not type(neuralcat_mech), (
        "nbn-cat and nbn-neuralcat must resolve to different mechanism classes"
    )

    # ── Assertion 3: _param_learning_cell signature accepts BaselineSpec ─
    sig = inspect.signature(_param_learning_cell)
    spec_param = sig.parameters.get("spec")
    assert spec_param is not None, (
        f"_param_learning_cell must accept a 'spec' parameter; "
        f"current signature: {sig}"
    )
    annotation = spec_param.annotation
    annotation_str = (
        annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    )
    assert "BaselineSpec" in annotation_str, (
        f"_param_learning_cell 'spec' parameter must be annotated as "
        f"BaselineSpec (signals method-keyed dispatch); got {annotation_str!r}"
    )

    # ── Assertion 4: NeuralCategoricalMechanism root-node uses empirical frequency ─
    # Biased training data: 80% class 0, 15% class 1, 5% class 2.
    biased_x = torch.cat([
        torch.zeros(800, dtype=torch.long),
        torch.ones(150, dtype=torch.long),
        torch.full((50,), 2, dtype=torch.long),
    ])
    expected_freq = torch.tensor([0.8, 0.15, 0.05])
    mech = NeuralCategoricalMechanism(n_classes=3)
    mech.fit_local(biased_x, parents=None)
    fitted_probs = torch.softmax(mech._root_logits.detach(), dim=-1)
    max_diff_from_expected = (expected_freq - fitted_probs).abs().max().item()
    max_diff_from_uniform = (torch.full((3,), 1 / 3) - fitted_probs).abs().max().item()
    assert max_diff_from_expected < 1e-4, (
        f"NeuralCategoricalMechanism root-node logits must reflect empirical "
        f"log-frequency (#52); got fitted={fitted_probs.tolist()} vs "
        f"expected={expected_freq.tolist()} (max abs diff={max_diff_from_expected:.6e})"
    )
    assert max_diff_from_uniform > 0.1, (
        f"NeuralCategoricalMechanism root-node logits must NOT be uniform "
        f"(uniform-init was the v0.7-#43 audit's #52 finding); got "
        f"fitted={fitted_probs.tolist()} (max diff from uniform={max_diff_from_uniform:.6e})"
    )

    # ── Assertion 5: nbn-cat vs nbn-neuralcat distinct TV ──────────────
    # The most direct echo of the v0.6c-d bit-identical pattern for the
    # NBN pair.  Originally dropped because _avg_tv_per_node crashed on
    # NeuralCategoricalMechanism (no _logits attribute); v0.8-#59 +
    # v0.8-#26 introduced Mechanism.tabulate() which makes this work.
    # v0.8 Pass-9: _fit_and_score_nbn returns list of metric rows;
    # pluck tv_per_node via _tv_value (defined above).
    from benchmarking.crash_test_runner import _fit_and_score_nbn
    cat_rows = _fit_and_score_nbn(smoke_cfg, bn, "discrete", cat_spec)
    neuralcat_rows = _fit_and_score_nbn(smoke_cfg, bn, "discrete", neuralcat_spec)
    cat_value = _tv_value(cat_rows)
    neuralcat_value = _tv_value(neuralcat_rows)
    cat_neural_diff = abs(cat_value - neuralcat_value)
    assert cat_neural_diff > 1e-4, (
        f"nbn-cat ({cat_value:.6e}) and nbn-neuralcat ({neuralcat_value:.6e}) "
        f"are bit-identical (diff={cat_neural_diff:.6e}); the v0.6c-d "
        f"runner-dispatch pattern has regressed for the NBN pair (#51) or "
        f"_avg_tv_per_node has reverted to reading mech._logits directly "
        f"(#59) — both mechanisms would then be measured via the same "
        f"path and produce the same value."
    )
