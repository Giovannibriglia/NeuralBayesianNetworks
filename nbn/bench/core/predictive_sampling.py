"""Per-continuous-node predictive sampling for calibration (#109 PR 7).

A single helper shared by BOTH sides of the calibration comparison so they are
sampled by byte-identical logic (same parent assembly, same root-node handling,
same S):

  * ``NBNAdapter.predictive_samples`` — draws from the FITTED model (PIT-KS,
    sharpness numerator);
  * ``ParamLearningMeasurement`` oracle draw — draws from ``problem.true_model``
    (sd_ratio denominator).

If these two diverged, sd_ratio (predictive-SD / true-SD) would compare
apples to oranges, so the logic lives in one place.
"""
from __future__ import annotations

from typing import Any

import torch


def continuous_predictive_samples(
    model: Any,
    variables: dict[str, tuple],
    test_data: dict[str, torch.Tensor],
    n_samples: int,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """``{node: samples[N_test, S]}`` predictive draws per CONTINUOUS node.

    For each continuous node, draw ``S = n_samples`` samples from ``model``'s
    mechanism conditioned on each test row's parent values
    (``mech.sample(pack_parents(test_data, parents), n=S)``). Discrete nodes are
    omitted (calibration is a continuous-node metric).

    Root continuous nodes (no parents) have one unconditional predictive
    distribution; ``mech.sample(parents=None, n=S)`` returns ``[1, S, D]`` and the
    returned ``[N_test, S]`` is that same row expanded across all test rows.
    Non-root nodes get ``N_test`` independent predictive distributions, one per
    test row's parent assignment. Both are correct: root predictives are
    inherently row-invariant; non-root predictives are row-conditional.

    ``model`` must expose ``dag`` (with ``topological_order`` / ``parents``) and
    ``mechanisms`` — satisfied by both the fitted NBN and ``true_model``.
    INTENTIONALLY STOCHASTIC: ``mech.sample`` draws from the global RNG; the
    caller seeds it (``fork_rng`` + ``manual_seed``) for reproducible values.
    """
    from nbn.utils.batching import pack_parents

    data = {k: torch.as_tensor(v).to(device) for k, v in test_data.items()}
    n_test = int(next(iter(data.values())).shape[0])
    out: dict[str, torch.Tensor] = {}
    for node in model.dag.topological_order():
        kind, _ = variables[node]
        if kind != "continuous":
            continue
        mech = model.mechanisms[node]
        pa = pack_parents(data, model.dag.parents(node))   # [N, D_pa] or None
        samples = mech.sample(pa, n=n_samples)[..., 0]     # [B, S] (B=N or 1)
        if samples.shape[0] == 1 and n_test > 1:
            samples = samples.expand(n_test, -1)
        out[node] = samples.detach().cpu()                 # [N_test, S]
    return out
