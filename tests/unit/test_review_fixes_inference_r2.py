"""Round-2 regression tests for the inference review fixes (R5-R7, R14b, CUDA-by-reading).

Ground truth is brute-force joint enumeration (``brute_posterior`` from the
round-1 module).
"""
from __future__ import annotations

import pytest
import torch

from tests.unit.test_review_fixes_inference import _random_discrete, brute_posterior

# --------------------------------------------------------------------------- #
# R7 — discrete evidence / do must be valid integral states on every engine
# --------------------------------------------------------------------------- #


def _engines():
    from nbn import TensorVariableElimination
    from nbn.inference.amortized_is import AmortizedISEngine
    from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine

    return {
        "ve": TensorVariableElimination(),
        "lw": LikelihoodWeightingEngine(64),
        "ais": AmortizedISEngine(n_samples=64),
    }


@pytest.mark.parametrize("engine", ["ve", "lw", "ais"])
@pytest.mark.parametrize("bad", [-1, 3, 1.5])
@pytest.mark.parametrize("kind", ["evidence", "do"])
def test_r7_invalid_discrete_values_are_rejected(engine, bad, kind):
    eng = _engines()[engine]
    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    single = {nodes[2]: torch.tensor(bad)}
    batch = {nodes[2]: torch.tensor([1, bad])}
    for method, spec in (("query", single), ("query_batch", batch)):
        ev, kw = ({}, {"do": spec}) if kind == "do" else (spec, {})
        with pytest.raises(ValueError, match="integer state|outside"):
            getattr(eng, method)(model, [nodes[-1]], ev, **kw)


def test_r7_integral_float_evidence_still_accepted():
    """1.0 as a float is a valid state (bench evidence arrives as float)."""
    eng = _engines()["ve"]
    model = _random_discrete()
    nodes = list(model.dag.topological_order())
    out = eng.query_batch(model, [nodes[-1]], {nodes[0]: torch.tensor([[1.0], [2.0]])})
    ref = torch.stack([brute_posterior(model, nodes[-1], {nodes[0]: v}) for v in (1, 2)])
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
