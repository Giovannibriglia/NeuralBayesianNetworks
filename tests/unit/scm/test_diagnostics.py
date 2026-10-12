"""Diagnostics: size under the null, power under the alternative."""
from __future__ import annotations

import pytest
import torch

from nbn.scm import (
    AdditiveNoiseMechanism,
    DiscreteAdditiveNoiseMechanism,
    StructuralCausalModel,
    benjamini_hochberg,
    context_invariance_test,
    freshness_test,
    noise_sharing_test,
    noise_tying_test,
    residual_dependence_test,
    residuals,
)
from tests.unit.scm.simulators import confounded_regimes, rehab_iid, rehab_model


def _fitted_rehab(n, seed, q=None):
    kwargs = {} if q is None else {"q": q}
    data = rehab_iid(n, generator=torch.Generator().manual_seed(seed), **kwargs).data
    model = rehab_model()
    model.fit(data)
    return model, data


# ---------------------------------------------------------------------------
# Noise sharing (discrete, parametric bootstrap)
# ---------------------------------------------------------------------------


def test_sharing_test_accepts_shared_noise():
    model, data = _fitted_rehab(3000, seed=1)
    assert noise_sharing_test(model, data, "h'", n_boot=100).p_value > 0.01


def test_sharing_test_rejects_noise_that_depends_on_the_parents():
    gen = torch.Generator().manual_seed(2)
    data = rehab_iid(3000, generator=gen).data
    # Tired patients get a much noisier outcome: one law no longer fits all rows.
    tired = data["f"] == 1
    extra = (torch.randint(-1, 2, (3000,), generator=gen) * tired).float()
    data["h'"] = (data["h'"] + extra).clamp(0, 3)
    model = rehab_model(max_offset=2)  # the extra noise reaches offsets of +-2
    model.fit(data)
    assert noise_sharing_test(model, data, "h'", n_boot=100).p_value < 0.01


@pytest.mark.slow
def test_sharing_test_size_is_at_most_alpha():
    replicates, alpha = 60, 0.05
    rejections = sum(
        noise_sharing_test(*_fitted_rehab(1500, seed=100 + r), "h'", n_boot=100).p_value < alpha
        for r in range(replicates)
    )
    assert rejections / replicates <= alpha + 3 * (alpha * (1 - alpha) / replicates) ** 0.5


# ---------------------------------------------------------------------------
# Noise sharing (continuous) and tying
# ---------------------------------------------------------------------------


def _continuous(n=3000, seed=0, heteroscedastic=False):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, generator=gen)
    scale = 0.2 + 0.5 * x.abs() if heteroscedastic else torch.full_like(x, 0.4)
    data = {"x": x, "y": x + scale * torch.randn(n, 1, generator=gen),
            "z": -x + 0.4 * torch.randn(n, 1, generator=gen)}
    model = StructuralCausalModel([("x", "y"), ("x", "z")],
                                  dict.fromkeys(data, ("continuous", 1)), device="cpu")
    model.set_mechanism("y", AdditiveNoiseMechanism("linear"))
    model.set_mechanism("z", AdditiveNoiseMechanism("linear"))
    model.auto_mechanisms(default_continuous="lg")
    model.fit(data)
    return model, data


def test_continuous_sharing_test_detects_a_scale_that_varies_with_the_parents():
    assert noise_sharing_test(*_continuous(), "y").p_value > 0.01
    assert noise_sharing_test(*_continuous(heteroscedastic=True), "y").p_value < 1e-4


def test_continuous_tying_test():
    model, data = _continuous()
    assert noise_tying_test(model, data, ["y", "z"]).p_value > 0.01
    model, data = _continuous(heteroscedastic=True)
    assert noise_tying_test(model, data, ["y", "z"]).p_value < 1e-3


def test_discrete_tying_test():
    gen = torch.Generator().manual_seed(0)
    n = 4000
    x = torch.randint(0, 4, (n,), generator=gen)

    def build(q_z):
        u1 = torch.multinomial(torch.tensor([0.15, 0.7, 0.15]), n, replacement=True, generator=gen) - 1
        u2 = torch.multinomial(q_z, n, replacement=True, generator=gen) - 1
        data = {"x": x.float(), "y": (x + u1).clamp(0, 3).float(), "z": (x + u2).clamp(0, 3).float()}
        model = StructuralCausalModel([("x", "y"), ("x", "z")],
                                      dict.fromkeys(data, ("discrete", 4)), device="cpu")
        model.set_mechanism("y", DiscreteAdditiveNoiseMechanism(max_offset=1))
        model.set_mechanism("z", DiscreteAdditiveNoiseMechanism(max_offset=1))
        model.auto_mechanisms()
        model.fit(data)
        return model, data

    same = build(torch.tensor([0.15, 0.7, 0.15]))
    different = build(torch.tensor([0.4, 0.2, 0.4]))
    assert noise_tying_test(*same, ["y", "z"], n_boot=100).p_value > 0.01
    assert noise_tying_test(*different, ["y", "z"], n_boot=100).p_value < 0.01


# ---------------------------------------------------------------------------
# Freshness and cross-node dependence
# ---------------------------------------------------------------------------


def _ar_noise(rho, n_episodes=200, horizon=20, seed=0):
    gen = torch.Generator().manual_seed(seed)
    u = torch.zeros(n_episodes, horizon)
    u[:, 0] = torch.randn(n_episodes, generator=gen)
    for t in range(1, horizon):
        u[:, t] = rho * u[:, t - 1] + (1 - rho**2) ** 0.5 * torch.randn(n_episodes, generator=gen)
    episode = torch.arange(n_episodes).repeat_interleave(horizon)
    time = torch.arange(horizon).repeat(n_episodes)
    return u.reshape(-1), episode, time


def test_freshness_test():
    assert freshness_test(*_ar_noise(0.0)).p_value > 0.01
    assert freshness_test(*_ar_noise(0.4)).p_value < 1e-6


def test_freshness_ignores_row_order_and_episode_boundaries():
    u, e, t = _ar_noise(0.0)
    perm = torch.randperm(u.numel())
    result = freshness_test(u[perm], e[perm], t[perm])
    assert result.details["pairs"] == 200 * 19


def test_residual_dependence_pearson_and_hsic():
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(400, generator=gen)
    independent = torch.randn(400, generator=gen)
    quadratic = a**2 - 1 + 0.1 * torch.randn(400, generator=gen)  # uncorrelated, dependent
    assert residual_dependence_test(a, independent).p_value > 0.01
    assert residual_dependence_test(a, a + 0.5 * independent).p_value < 1e-6
    assert residual_dependence_test(a, independent, method="hsic", n_permutations=100).p_value > 0.01
    assert residual_dependence_test(a, quadratic, method="hsic", n_permutations=100).p_value < 0.02


def test_discrete_residuals_are_posterior_mean_offsets():
    model, data = _fitted_rehab(2000, seed=0)
    r = residuals(model, data, "h'")
    assert r.shape == (2000,)
    assert float(r.min()) >= -1.0 and float(r.max()) <= 1.0


# ---------------------------------------------------------------------------
# Context (policy) invariance
# ---------------------------------------------------------------------------


def _regime_model(data):
    model = StructuralCausalModel(
        [("h", "h'"), ("a", "h'")],
        {"h": ("discrete", 4), "a": ("discrete", 2), "h'": ("discrete", 4), "c": ("discrete", 3)},
        context="c", device="cpu",
    )
    model.auto_mechanisms()
    model.fit(data)
    return model


def test_context_test_flags_the_confounded_logger_only():
    data = confounded_regimes(3000, generator=torch.Generator().manual_seed(0))
    clean = {k: v[data["c"] < 2] for k, v in data.items()}
    assert context_invariance_test(_regime_model(clean), clean, "h'").p_value > 0.01
    assert context_invariance_test(_regime_model(data), data, "h'").p_value < 1e-6


def test_context_test_on_a_continuous_mechanism():
    gen = torch.Generator().manual_seed(0)
    n = 2000
    c = torch.arange(2).repeat_interleave(n)
    x = torch.randn(2 * n, 1, generator=gen)
    shift = 0.5 * c.unsqueeze(-1).float()
    data = {"x": x, "y": x + 0.3 * torch.randn(2 * n, 1, generator=gen), "c": c.float()}
    drifted = dict(data, y=data["y"] + shift)
    for d, small in ((data, False), (drifted, True)):
        model = StructuralCausalModel([("x", "y")], {"x": ("continuous", 1), "y": ("continuous", 1),
                                                     "c": ("discrete", 2)}, context="c", device="cpu")
        model.set_mechanism("y", AdditiveNoiseMechanism("linear"))
        model.auto_mechanisms(default_continuous="lg")
        model.fit(d)
        p = context_invariance_test(model, d, "y").p_value
        assert (p < 1e-6) if small else (p > 0.01)


def test_benjamini_hochberg_matches_a_hand_computed_example():
    p = torch.tensor([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    # thresholds 0.05*i/8: only the first two p-values pass, and no later one does.
    assert benjamini_hochberg(p, alpha=0.05).tolist() == [True, True] + [False] * 6
