"""StructuralCausalModel: tying, counterfactuals, declared confounding, episodes."""
from __future__ import annotations

import pytest
import torch

from nbn.scm import (
    AdditiveNoiseMechanism,
    DiscreteAdditiveNoiseMechanism,
    StructuralCausalModel,
    transitions_from_episodes,
)
from tests.unit.scm.simulators import TRUE_Q


def _two_counters(n=20_000, seed=0):
    """Two discrete nodes driven by noise with the same law (tying is true)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 5, (n,), generator=gen)
    u1 = torch.multinomial(TRUE_Q, n, replacement=True, generator=gen) - 1
    u2 = torch.multinomial(TRUE_Q, n, replacement=True, generator=gen) - 1
    y = (x + u1).clamp(0, 4)
    z = (4 - x + u2).clamp(0, 4)
    model = StructuralCausalModel(
        [("x", "y"), ("x", "z")],
        {"x": ("discrete", 5), "y": ("discrete", 5), "z": ("discrete", 5)}, device="cpu",
    )
    model.set_mechanism("y", DiscreteAdditiveNoiseMechanism(max_offset=1))
    model.set_mechanism("z", DiscreteAdditiveNoiseMechanism(max_offset=1))
    model.auto_mechanisms()
    return model, {"x": x.float(), "y": y.float(), "z": z.float()}


def test_tied_discrete_nodes_share_one_law_and_recover_it():
    model, data = _two_counters()
    model.tie_noise(["y", "z"], name="day")
    model.fit(data)
    y, z = model.mechanisms["y"], model.mechanisms["z"]
    assert y.noise is z.noise
    assert torch.allclose(y.noise.probs.detach(), TRUE_Q, atol=0.01)
    assert model.noise_groups == {"day": ["y", "z"]}


def test_tied_continuous_nodes_pool_their_residuals():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(4000, 1, generator=gen)
    data = {"x": x, "y": 2 * x + 0.5 * torch.randn(4000, 1, generator=gen),
            "z": -x + 0.5 * torch.randn(4000, 1, generator=gen)}
    model = StructuralCausalModel([("x", "y"), ("x", "z")],
                                  dict.fromkeys(data, ("continuous", 1)), device="cpu")
    model.set_mechanism("y", AdditiveNoiseMechanism("linear"))
    model.set_mechanism("z", AdditiveNoiseMechanism("linear"))
    model.auto_mechanisms(default_continuous="lg")
    model.tie_noise(["y", "z"])
    model.fit(data)
    assert model.mechanisms["y"].noise is model.mechanisms["z"].noise
    assert float(model.mechanisms["y"].noise.scale) == pytest.approx(0.5, abs=0.02)


def test_tie_noise_rejects_bad_requests():
    model, _ = _two_counters()
    with pytest.raises(ValueError, match="at least two"):
        model.tie_noise(["y"])
    with pytest.raises(TypeError, match="structural"):
        model.tie_noise(["x", "y"])
    model.tie_noise(["y", "z"])
    with pytest.raises(ValueError, match="already tied"):
        model.tie_noise(["y", "z"])


def test_unknown_names_are_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown node"):
        StructuralCausalModel([("x", "y")], {"x": ("discrete", 2), "y": ("discrete", 2)},
                              bidirected=[("x", "w")], device="cpu")
    with pytest.raises(ValueError, match="context"):
        StructuralCausalModel([("x", "y")], {"x": ("discrete", 2), "y": ("discrete", 2)},
                              context="c", device="cpu")


def test_linear_counterfactual_is_exact():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(3000, 1, generator=gen)
    y = 1.5 * x + 0.4 * torch.randn(3000, 1, generator=gen)
    model = StructuralCausalModel([("x", "y")], {"x": ("continuous", 1), "y": ("continuous", 1)},
                                  device="cpu")
    model.set_mechanism("x", AdditiveNoiseMechanism("linear"))
    model.set_mechanism("y", AdditiveNoiseMechanism("linear"))
    model.fit({"x": x, "y": y})
    slope = float(model.mechanisms["y"].net.weight)
    world = model.counterfactual_sample({"x": x[:10], "y": y[:10]}, do={"x": 2.0})
    expected = y[:10] + slope * (2.0 - x[:10])
    assert world["y"].shape == (1, 10, 1)
    assert torch.allclose(world["y"][0], expected, atol=1e-5)


def test_discrete_counterfactual_is_certain_when_the_noise_is_identified():
    model, data = _two_counters()
    model.fit(data)
    # y = 2 from location x = 2: the offset is 0 for sure (no clipping) ...
    world = model.counterfactual_sample(
        {"x": torch.tensor([2.0]), "y": torch.tensor([2.0]), "z": torch.tensor([2.0])},
        do={"x": 3.0}, n_samples=50,
    )
    # ... so under do(x = 3) the unit's y is 3 in every draw.
    assert bool((world["y"] == 3).all())


def test_counterfactuals_refuse_declared_hidden_common_causes():
    model = StructuralCausalModel([("x", "y")], {"x": ("continuous", 1), "y": ("continuous", 1)},
                                  bidirected=[("x", "y")], device="cpu")
    with pytest.raises(NotImplementedError, match="hidden common causes"):
        model.counterfactual_sample({"x": torch.zeros(1, 1), "y": torch.zeros(1, 1)}, do={"x": 1.0})


def test_transitions_from_episodes_flattens_two_slices():
    states = {"s": torch.arange(12.0).reshape(2, 6)}
    actions = {"a": torch.ones(2, 5)}
    rows = transitions_from_episodes(states, actions)
    assert rows["s"].tolist()[:5] == [0, 1, 2, 3, 4]
    assert rows["s'"].tolist()[:5] == [1, 2, 3, 4, 5]
    assert rows["episode"].tolist() == [0] * 5 + [1] * 5
    assert rows["time"].tolist() == list(range(5)) * 2
    with pytest.raises(ValueError, match="expected"):
        transitions_from_episodes(states, {"a": torch.ones(2, 6)})
