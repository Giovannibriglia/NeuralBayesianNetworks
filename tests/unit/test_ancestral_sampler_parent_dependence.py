"""Regression test for the [n,n,D]-cube + samp[0] bug fixed in PR #12.

Prior to commit ``a980aab``, ``nbn/sampling/ancestral.py``'s non-root
branch materialised ``mech.sample(pa_tensor, n=n)`` of shape
``[n, n, D]`` and then took ``samp[0]``, which is *n* samples all
conditioned on ``pa_tensor[0]`` — silently dropping per-row parent
variation.  The bug was invisible to v0.3-era tests because they used
external data rather than ``model.sample(...)`` output; the synthetic-BN
harness in v0.4 exposed it because every metric is defined relative to
data drawn from a known generative process.

These two tests would have flagged the bug pre-merge in v0.3 if they
had existed; they are added now so the bug class stays regressable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nbn.core.network import NeuralBayesianNetwork
from nbn.mechanisms.categorical_table import CategoricalTableMechanism
from nbn.mechanisms.linear_gaussian import LinearGaussianMechanism


def test_non_root_lg_sampler_recovers_parent_slope() -> None:
    """A 2-node LG chain X1 -> X2 with slope=2.0 must show the slope
    when we OLS-regress X2 on X1 from sampled data.

    Pre-fix, the recovered slope was ≈ 0 because every X2 sample was
    drawn at parent value X1[0], yielding ``cov(X1, X2) ≈ 0``.
    """
    torch.manual_seed(0)
    model = NeuralBayesianNetwork(
        dag=[("X1", "X2")],
        variables={"X1": ("continuous", 1), "X2": ("continuous", 1)},
        device="cpu",
    )

    # X1 ~ N(0, 1) — root LG
    x1_mech = LinearGaussianMechanism(ridge=0.0, learnable=True)
    x1_mech.output_dim = 1
    x1_mech._input_dim = 0
    x1_mech._weight = nn.Parameter(torch.zeros(0, 1))
    x1_mech._bias = nn.Parameter(torch.zeros(1))
    x1_mech._log_scale = nn.Parameter(torch.log(torch.tensor([1.0])))
    model.set_mechanism("X1", x1_mech)

    # X2 = 2.0 * X1 + N(0, 0.1) — non-root LG with explicit slope
    true_slope = 2.0
    x2_mech = LinearGaussianMechanism(ridge=0.0, learnable=True)
    x2_mech.output_dim = 1
    x2_mech._input_dim = 1
    x2_mech._weight = nn.Parameter(torch.tensor([[true_slope]]))
    x2_mech._bias = nn.Parameter(torch.zeros(1))
    x2_mech._log_scale = nn.Parameter(torch.log(torch.tensor([0.1])))
    model.set_mechanism("X2", x2_mech)

    samples = model.sample(n=2000)
    x1 = samples["X1"].reshape(-1).float()
    x2 = samples["X2"].reshape(-1).float()
    cov = ((x1 - x1.mean()) * (x2 - x2.mean())).sum()
    var = ((x1 - x1.mean()) ** 2).sum()
    recovered = (cov / var).item()
    assert abs(recovered - true_slope) < 0.10, (
        f"Recovered slope {recovered:.3f} differs from truth {true_slope} by "
        f"more than 5% — non-root sampler may not be using per-row parents."
    )


def test_non_root_categorical_sampler_uses_per_row_parent() -> None:
    """A 2-node discrete chain X1 -> X2 where X2 deterministically copies
    X1: empirical ``P(X2=k | X1=k)`` must be ≈ 1 across all k.

    Pre-fix, agreement collapsed to ``P(X1 == X1[0]) ≈ 1/K`` because
    every X2 sample was drawn at parent value X1[0], not the per-row
    parent value.
    """
    torch.manual_seed(0)
    K = 4
    model = NeuralBayesianNetwork(
        dag=[("X1", "X2")],
        variables={"X1": ("discrete", K), "X2": ("discrete", K)},
        device="cpu",
    )

    # X1 uniform Categorical over K classes
    x1_mech = CategoricalTableMechanism(alpha=0.0)
    x1_mech._logits = nn.Parameter(torch.zeros(1, K))
    x1_mech._n_classes = K
    x1_mech._parent_cards = []
    x1_mech._parent_strides = []
    x1_mech._class_values = torch.arange(K, dtype=torch.float)
    x1_mech.output_dim = 1
    model.set_mechanism("X1", x1_mech)

    # X2 | X1=k = delta(k) — copy the parent class
    cpt = torch.full((K, K), -1e9)
    for k in range(K):
        cpt[k, k] = 0.0
    x2_mech = CategoricalTableMechanism(alpha=0.0)
    x2_mech._logits = nn.Parameter(cpt)
    x2_mech._n_classes = K
    x2_mech._parent_cards = [K]
    x2_mech._parent_strides = [1]
    x2_mech._class_values = torch.arange(K, dtype=torch.float)
    x2_mech.output_dim = 1
    model.set_mechanism("X2", x2_mech)

    samples = model.sample(n=4000)
    x1 = samples["X1"].reshape(-1).long()
    x2 = samples["X2"].reshape(-1).long()
    agreement = (x1 == x2).float().mean().item()
    assert agreement > 0.95, (
        f"X2 agrees with X1 only {agreement:.3f} of the time on a "
        f"copy-parent mechanism — non-root sampler may be using parent row 0 "
        f"for every output row (the v0.4b PR #12 / v0.5 #11 bug class)."
    )
