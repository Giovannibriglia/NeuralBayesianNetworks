"""Regression tests for the issues found in code review (one test per finding)."""
from __future__ import annotations

import warnings

import pytest
import torch

from nbn.mechanisms import CategoricalTableMechanism
from nbn.scm import (
    AdditiveNoiseMechanism,
    DiscreteAdditiveNoiseMechanism,
    StructuralCausalModel,
    context_invariance_test,
    noise_parameter_set,
    noise_sharing_test,
    residuals,
)
from nbn.scm._stats import kolmogorov_sf
from nbn.uncertainty import SampleSet, count_table, l1_ball_set
from tests.conftest import available_devices
from tests.unit.scm.simulators import rehab_iid, rehab_model

ACCELERATORS = [d for d in available_devices() if d != "cpu"]


def _xy(n=500, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, generator=gen)
    return {"x": x, "y": 1.5 * x + 0.3 * torch.randn(n, 1, generator=gen)}


def _continuous_scm(device="cpu", **mech_kwargs):
    model = StructuralCausalModel([("x", "y")], {"x": ("continuous", 1), "y": ("continuous", 1)},
                                  device=device)
    model.set_mechanism("y", AdditiveNoiseMechanism(**mech_kwargs))
    model.auto_mechanisms(default_continuous="linear")
    return model


def _discrete_root_scm(device="cpu", root_mechanism=None):
    """A discrete structural root ``x`` and a structural child ``y = clip(x + U)``."""
    model = StructuralCausalModel([("x", "y")], {"x": ("discrete", 4), "y": ("discrete", 4)},
                                  device=device)
    model.set_mechanism("x", root_mechanism or DiscreteAdditiveNoiseMechanism())
    model.set_mechanism("y", DiscreteAdditiveNoiseMechanism(max_offset=1))
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(0, 4, (2000,), generator=gen)
    y = (x + torch.randint(-1, 2, (2000,), generator=gen)).clamp(0, 3)
    data = {"x": x.float(), "y": y.float()}
    model.fit(data)
    return model, data


# 1 ---------------------------------------------------------------------------
def test_continuous_log_prob_accepts_flat_input():
    model = _continuous_scm()
    data = _xy()
    model.fit(data)
    mech = model.mechanisms["y"]
    flat = mech.log_prob(data["y"].reshape(-1), data["x"])
    assert flat.shape == (500,)
    assert torch.allclose(flat, mech.log_prob(data["y"], data["x"]))
    x3 = data["y"][:4].unsqueeze(1).expand(-1, 3, -1)
    assert mech.log_prob(x3, data["x"][:4]).shape == (4, 3)


# 2 ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", ACCELERATORS or ["cpu"])
def test_discrete_structural_root_fits_on_every_device(device):
    model, _ = _discrete_root_scm(device=device)
    assert model.mechanisms["x"].table().device.type == torch.device(device).type


# 3 + 10 ------------------------------------------------------------------------
def test_counterfactual_with_a_discrete_structural_root():
    model, data = _discrete_root_scm()
    world = model.counterfactual_sample({k: v[:5] for k, v in data.items()}, do={"x": 0.0}, n_samples=4)
    assert world["y"].shape == (4, 5, 1)
    assert bool((world["y"] <= 1).all())  # y = clip(0 + u), u in {-1, 0, 1}


def test_counterfactual_copies_non_descendants_and_accepts_any_root_mechanism():
    model, data = _discrete_root_scm(root_mechanism=CategoricalTableMechanism())
    world = model.counterfactual_sample({k: v[:5] for k, v in data.items()}, do={"y": 2.0})
    assert torch.equal(world["x"][0, :, 0], data["x"][:5])
    assert bool((world["y"] == 2).all())


def test_counterfactual_accepts_one_do_value_per_unit():
    model = _continuous_scm()
    model.set_mechanism("x", AdditiveNoiseMechanism())
    data = _xy()
    model.fit(data)
    per_unit = torch.tensor([0.0, 1.0, 2.0])
    world = model.counterfactual_sample({k: v[:3] for k, v in data.items()}, do={"x": per_unit})
    assert torch.equal(world["x"][0, :, 0], per_unit)
    slope = float(model.mechanisms["y"].net.weight)
    expected = data["y"][:3, 0] + slope * (per_unit - data["x"][:3, 0])
    assert torch.allclose(world["y"][0, :, 0], expected, atol=1e-5)


# 4 ---------------------------------------------------------------------------
def test_mlp_respects_the_epoch_budget_and_tied_history_is_post_refit():
    data = _xy(n=300)
    data["z"] = -data["x"] + 0.3 * torch.randn(300, 1)
    model = StructuralCausalModel([("x", "y"), ("x", "z")],
                                  dict.fromkeys(data, ("continuous", 1)), device="cpu")
    model.set_mechanism("y", AdditiveNoiseMechanism("mlp", epochs=1))
    model.set_mechanism("z", AdditiveNoiseMechanism("mlp", epochs=1))
    model.auto_mechanisms(default_continuous="linear")
    model.tie_noise(["y", "z"])
    history = model.fit(data, epochs=0)  # zero epochs: the MLPs must stay at initialisation
    for node in ("y", "z"):
        lp = model.mechanisms[node].log_prob(data[node], data["x"]).mean()
        assert history.node_log_likelihoods[node][-1] == pytest.approx(float(lp), rel=1e-5)


def test_tying_a_fitted_mechanism_is_refused():
    model = _continuous_scm()
    model.set_mechanism("x", AdditiveNoiseMechanism())
    model.fit(_xy())
    with pytest.raises(ValueError, match="already fitted"):
        model.tie_noise(["x", "y"])


# 5 + 6 -----------------------------------------------------------------------
@pytest.mark.parametrize("device", ACCELERATORS or ["cpu"])
def test_entry_points_move_cpu_data_to_the_model_device(device):
    data = rehab_iid(2000, generator=torch.Generator().manual_seed(0)).data
    model = rehab_model(device=device)
    model.fit(data)  # data stay on the CPU throughout
    noise_parameter_set(model, data, ["h'"])
    noise_sharing_test(model, data, "h'", n_boot=20)
    assert residuals(model, data, "h'").device.type == torch.device(device).type
    regimes = dict(data, c=(torch.arange(2000) % 2).float())
    regime_model = StructuralCausalModel(
        [("h", "h'"), ("a", "h'")],
        {"h": ("discrete", 4), "a": ("discrete", 2), "h'": ("discrete", 4), "c": ("discrete", 2)},
        context="c", device=device,
    )
    regime_model.auto_mechanisms()
    regime_model.fit(regimes)
    assert 0.0 <= context_invariance_test(regime_model, regimes, "h'").p_value <= 1.0


# 7 ---------------------------------------------------------------------------
def test_noise_polytope_gives_unseen_rows_the_whole_simplex():
    data = rehab_iid(2000, generator=torch.Generator().manual_seed(0)).data
    data = {k: v[data["a"] == 0] for k, v in data.items()}
    model = rehab_model()
    model.fit(data)
    box = noise_parameter_set(model, data, ["h'"]).table_set("h'")
    lower, upper = box.bounds()
    assert int(box.unseen.sum()) == 8
    assert bool((lower[box.unseen] == 0).all()) and bool((upper[box.unseen] == 1).all())


# 8 ---------------------------------------------------------------------------
def test_mixture_noise_is_replication_equivalent():
    data = _xy(n=400)
    data["y"] = data["y"] + torch.where(torch.rand(400, 1) < 0.5, -1.0, 1.0)
    weights = torch.randint(0, 3, (400,)).float()
    replicated = {k: v.repeat_interleave(weights.long(), dim=0) for k, v in data.items()}
    a = _continuous_scm(noise="mixture", n_components=2)
    b = _continuous_scm(noise="mixture", n_components=2)
    a.fit(data, weights=weights)
    b.fit(replicated)
    na, nb = a.mechanisms["y"].noise, b.mechanisms["y"].noise
    assert torch.allclose(na.means, nb.means, atol=1e-4)
    assert torch.allclose(na.mix_logits.softmax(-1), nb.mix_logits.softmax(-1), atol=1e-4)


def test_mlp_standardisation_is_weighted():
    data = _xy(n=200)
    weights = torch.randint(0, 3, (200,)).float()
    replicated = {k: v.repeat_interleave(weights.long(), dim=0) for k, v in data.items()}
    a = AdditiveNoiseMechanism("mlp", epochs=0)
    b = AdditiveNoiseMechanism("mlp", epochs=0)
    a.fit_local(data["y"], data["x"], weights=weights)
    b.fit_local(replicated["y"], replicated["x"])
    assert torch.allclose(a._pa_mean, b._pa_mean, atol=1e-5)
    assert torch.allclose(a._pa_std, b._pa_std, atol=1e-5)


# 9 ---------------------------------------------------------------------------
def test_save_and_load_keep_tying_bidirected_and_context(tmp_path):
    model = StructuralCausalModel(
        [("x", "y"), ("x", "z")], {"x": ("discrete", 4), "y": ("discrete", 4),
                                   "z": ("discrete", 4), "c": ("discrete", 2)},
        bidirected=[("y", "z")], context="c", device="cpu",
    )
    model.set_mechanism("y", DiscreteAdditiveNoiseMechanism(max_offset=1))
    model.set_mechanism("z", DiscreteAdditiveNoiseMechanism(max_offset=1))
    model.auto_mechanisms()
    model.tie_noise(["y", "z"], name="g")
    x = torch.randint(0, 4, (500,))
    data = {"x": x.float(), "y": x.float(), "z": x.float(), "c": (x % 2).float()}
    model.fit(data)
    path = tmp_path / "scm.pt"
    model.save(str(path))
    loaded = StructuralCausalModel.load(str(path))
    assert loaded.noise_groups == {"g": ["y", "z"]}
    assert loaded.bidirected == [("y", "z")]
    assert loaded.context == "c"
    assert loaded.mechanisms["y"].noise is loaded.mechanisms["z"].noise


# 13 --------------------------------------------------------------------------
def test_sample_set_warns_when_too_few_draws_support_the_level():
    s = SampleSet(level=0.95, guarantee="bayesian", unseen=torch.zeros(10, dtype=torch.bool),
                  samples=torch.rand(100, 10, 4))
    with pytest.warns(UserWarning, match="draws expected"):
        s.bounds()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SampleSet(level=0.95, guarantee="bayesian", unseen=torch.zeros(1, dtype=torch.bool),
                  samples=torch.rand(1000, 1, 1), simultaneous=False).bounds()


# 14 --------------------------------------------------------------------------
def test_count_table_rejects_out_of_range_values():
    data = rehab_iid(100, generator=torch.Generator().manual_seed(0)).data
    model = rehab_model()
    bad_child = dict(data, **{"h'": data["h'"].clone().index_fill_(0, torch.tensor([0]), 7.0)})
    with pytest.raises(ValueError, match="declared states"):
        count_table(model, bad_child, "h'")
    bad_parent = dict(data, h=data["h"].clone().index_fill_(0, torch.tensor([0]), 9.0))
    with pytest.raises(ValueError, match="declared states"):
        count_table(model, bad_parent, "h'")


def test_l1_ball_handles_a_single_value():
    from nbn import NeuralBayesianNetwork

    model = NeuralBayesianNetwork([("x", "y")], {"x": ("discrete", 2), "y": ("discrete", 1)}, device="cpu")
    data = {"x": torch.tensor([0.0, 1.0, 1.0]), "y": torch.zeros(3)}
    ball = l1_ball_set(model, data, "y")
    assert torch.equal(ball.radius, torch.zeros(2))


# 15 --------------------------------------------------------------------------
def test_kolmogorov_tail_matches_scipy():
    special = pytest.importorskip("scipy.special")
    for lam in (0.01, 0.3, 0.6, 0.99, 1.0, 1.36, 2.5):
        assert kolmogorov_sf(lam) == pytest.approx(float(special.kolmogorov(lam)), abs=1e-10)


# ===========================================================================
# Round 2
# ===========================================================================


def _brute_force_offset_fit(transfers, counts):
    """Best mean log-likelihood over every joint location assignment (batched EM for q)."""
    import itertools

    from nbn.scm.fitting import _em_step, _objective

    ranges = [range(a.shape[0]) for a, c in zip(transfers, counts) for _ in range(c.shape[0])]
    grid = torch.tensor(list(itertools.product(*ranges)))
    locations, start = [], 0
    for c in counts:
        locations.append(grid[:, start:start + c.shape[0]])
        start += c.shape[0]
    n_assign, n_offsets = grid.shape[0], transfers[0].shape[-1]
    stacked = [c.unsqueeze(0).expand(n_assign, *c.shape) for c in counts]
    q = torch.full((n_assign, n_offsets), 1.0 / n_offsets, dtype=torch.float64)
    for _ in range(400):
        q = _em_step(transfers, stacked, locations, q)
    total = sum(float(c.sum()) for c in counts)
    return float(_objective(transfers, stacked, locations, q).max()) / total


@pytest.mark.parametrize("members,law", [(1, "unimodal"), (1, "dirichlet"), (2, "unimodal")])
def test_offset_fit_reaches_the_global_optimum(members, law):
    """Locations and law must move together to escape local optima; the staged
    search (multi-start, shift polish, row-move polish) must find the optimum."""
    from nbn.scm.fitting import child_law, fit_offset_law, offset_transfer

    torch.manual_seed(1)
    transfer = offset_transfer(3, torch.arange(-2, 3)).double()          # G = 7 locations
    for _ in range(6):
        q = (torch.distributions.Dirichlet(torch.ones(5)).sample().double() if law == "dirichlet"
             else torch.tensor([.05, .2, .5, .2, .05], dtype=torch.float64))
        n_rows = 3 if members == 1 else 2   # keeps the brute force at <= 7^4 assignments
        counts = []
        for _ in range(members):
            g = torch.randint(0, transfer.shape[0], (n_rows,))
            counts.append(torch.stack([
                torch.distributions.Multinomial(150, probs=child_law(transfer, q)[g[r]]).sample()
                for r in range(n_rows)
            ]).double())
        _, _, objective = fit_offset_law([transfer] * members, counts)
        assert float(objective) >= _brute_force_offset_fit([transfer] * members, counts) - 1e-6


@pytest.mark.slow
def test_locations_beyond_the_value_range_fit_saturating_mechanisms():
    """x = clip((p + q) // 2 + 3 + U): rows whose location exceeds K - 1
    saturate at the edge; one shared law must still explain every row.  Only
    the table is identified (a saturated row fits several location/law pairs)."""
    gen = torch.Generator().manual_seed(0)
    n = 20_000
    p = torch.randint(0, 8, (n,), generator=gen)
    q = torch.randint(0, 8, (n,), generator=gen)
    u = torch.multinomial(torch.tensor([0.2, 0.6, 0.2]), n, replacement=True, generator=gen) - 1
    x = ((p + q) // 2 + 3 + u).clamp(0, 7)                                 # locations up to 10
    data = {"p": p.float(), "q": q.float(), "x": x.float()}
    model = StructuralCausalModel([("p", "x"), ("q", "x")], dict.fromkeys(data, ("discrete", 8)),
                                  device="cpu")
    model.set_mechanism("x", DiscreteAdditiveNoiseMechanism(max_offset=3))
    model.auto_mechanisms()
    model.fit(data)
    grid = torch.cartesian_prod(torch.arange(8), torch.arange(8))
    location = (grid[:, 0] + grid[:, 1]) // 2 + 3
    truth = torch.zeros(64, 8, dtype=torch.float64)
    for offset, prob in zip((-1, 0, 1), (0.2, 0.6, 0.2)):
        truth[torch.arange(64), (location + offset).clamp(0, 7)] += prob
    assert torch.allclose(model.mechanisms["x"].table().double(), truth, atol=0.04)
    assert noise_sharing_test(model, data, "x", n_boot=100).p_value > 0.01


def test_offset_fit_rejects_zero_iterations():
    from nbn.scm.fitting import fit_offset_law, offset_transfer

    with pytest.raises(ValueError, match="n_iter"):
        fit_offset_law([offset_transfer(3, torch.arange(-1, 2))], [torch.ones(1, 3)], n_iter=0)


def _too_wide_data(n=1000):
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(0, 5, (n,), generator=gen)
    y = (x + 2 * torch.randint(-1, 2, (n,), generator=gen)).clamp(0, 4)  # offsets -2, 0, +2
    return {"x": x.float(), "y": y.float()}


def test_data_beyond_max_offset_raise_a_clear_error():
    model = StructuralCausalModel([("x", "y")], {"x": ("discrete", 5), "y": ("discrete", 5)}, device="cpu")
    model.set_mechanism("y", DiscreteAdditiveNoiseMechanism(max_offset=1))
    model.auto_mechanisms()
    with pytest.raises(ValueError, match="max_offset"):
        model.fit(_too_wide_data())


def test_noise_posterior_explains_impossible_observations():
    model, data = _discrete_root_scm()
    mech = model.mechanisms["y"]
    far = torch.tensor([3.0])               # from x = 0 with |u| <= 1, y = 3 is impossible
    with pytest.raises(ValueError, match="cannot be produced"):
        mech.noise_posterior(far, torch.tensor([[0.0]]))


def test_abduction_with_sample_dimension_and_fewer_rows_than_samples():
    model, data = _discrete_root_scm()
    mech = model.mechanisms["y"]
    pa = model.parent_values(data, "y")[:2]                       # [B=2, 1]
    x = data["y"][:2].reshape(2, 1, 1).expand(2, 3, 1)           # [B=2, S=3, 1]
    assert mech.abduct(x, pa, n_samples=4).shape == (4, 2, 3, 1)
    cont = _continuous_scm()
    xy = _xy()
    cont.fit(xy)
    y3 = xy["y"][:2].unsqueeze(1).expand(2, 3, 1)
    assert cont.mechanisms["y"].abduct(y3, xy["x"][:2], n_samples=4).shape == (4, 2, 3, 1)


def test_white_test_counts_binary_parents_once():
    gen = torch.Generator().manual_seed(0)
    b = torch.randint(0, 2, (2000, 1), generator=gen).float()
    data = {"b": b, "y": 0.5 * b + 0.3 * torch.randn(2000, 1, generator=gen)}
    model = StructuralCausalModel([("b", "y")], {"b": ("continuous", 1), "y": ("continuous", 1)},
                                  device="cpu")
    model.set_mechanism("y", AdditiveNoiseMechanism())
    model.auto_mechanisms(default_continuous="linear")
    model.fit(data)
    assert noise_sharing_test(model, data, "y").details["df"] == 1


# ===========================================================================
# Round 4
# ===========================================================================


def test_fit_is_canonical_when_the_offset_range_is_wide():
    """With max_offset = K - 1 the location/noise split is ambiguous; the
    canonical fit centres the law at 0 and recovers the generating locations."""
    from tests.unit.scm.simulators import TRUE_Q, rehab_location

    data = rehab_iid(20_000, generator=torch.Generator().manual_seed(0)).data
    model = rehab_model(max_offset=None)                  # offsets -3 .. 3
    model.fit(data)
    mech = model.mechanisms["h'"]
    probs = mech.noise.probs.detach()
    assert torch.allclose(probs[2:5], TRUE_Q, atol=0.015)  # offsets -1, 0, 1
    grid = torch.cartesian_prod(torch.arange(4), torch.arange(2), torch.arange(2)).float()
    expected = rehab_location(grid[:, 0].long(), grid[:, 1].long(), grid[:, 2].long())
    assert torch.equal(mech.location(grid), expected)


def test_canonicalisation_keeps_the_likelihood():
    from nbn.scm.fitting import _coordinate_ascent, _objective, canonicalise, offset_transfer

    torch.manual_seed(0)
    transfer = offset_transfer(5, torch.arange(-4, 5)).double()
    counts = torch.distributions.Multinomial(200, probs=torch.rand(6, 5)).sample().double()
    q0 = torch.distributions.Dirichlet(torch.ones(9)).sample().double()
    locations, q, _ = _coordinate_ascent([transfer], [counts], q0, 200, 1e-12)
    before = _objective([transfer], [counts], locations, q)
    locations_c, q_c, _ = canonicalise([transfer], [counts], locations, q, None)
    assert float(_objective([transfer], [counts], locations_c, q_c)) == pytest.approx(float(before), abs=1e-8)


def _saturating_pair(device):
    """x = clip((p + q) // 2 + U) and a tied y = clip(x + U'), wide offset range."""
    gen = torch.Generator().manual_seed(0)
    n = 20_000
    p = torch.randint(0, 8, (n,), generator=gen)
    q = torch.randint(0, 8, (n,), generator=gen)
    law = torch.tensor([0.2, 0.6, 0.2])
    x = ((p + q) // 2 + torch.multinomial(law, n, replacement=True, generator=gen) - 1).clamp(0, 7)
    y = (x + torch.multinomial(law, n, replacement=True, generator=gen) - 1).clamp(0, 7)
    data = {"p": p.float(), "q": q.float(), "x": x.float(), "y": y.float()}
    model = StructuralCausalModel([("p", "x"), ("q", "x"), ("x", "y")],
                                  dict.fromkeys(data, ("discrete", 8)), device=device)
    model.set_mechanism("x", DiscreteAdditiveNoiseMechanism())
    model.set_mechanism("y", DiscreteAdditiveNoiseMechanism())
    model.auto_mechanisms()
    model.tie_noise(["x", "y"])
    model.fit(data)
    return model


@pytest.mark.parametrize("device", ACCELERATORS or ["cpu"])
def test_fit_is_the_same_on_every_device(device):
    data = rehab_iid(5000, generator=torch.Generator().manual_seed(0)).data
    cpu, other = rehab_model(max_offset=None), rehab_model(device=device, max_offset=None)
    cpu.fit(data)
    other.fit(data)
    assert torch.equal(cpu.mechanisms["h'"].locations, other.mechanisms["h'"].locations.cpu())
    assert torch.allclose(cpu.mechanisms["h'"].noise.probs.detach(),
                          other.mechanisms["h'"].noise.probs.detach().cpu(), atol=1e-4)


@pytest.mark.parametrize("device", ACCELERATORS or ["cpu"])
def test_saturated_rows_do_not_block_the_canonical_shift(device):
    cpu, other = _saturating_pair("cpu"), _saturating_pair(device)
    for model in (cpu, other):
        probs = model.mechanisms["x"].noise.probs.detach().cpu()
        assert torch.allclose(probs[6:9], torch.tensor([0.2, 0.6, 0.2]), atol=0.02)  # offsets -1..1
        loc = model.mechanisms["x"].location(torch.tensor([[0.0, 0.0], [5.0, 4.0]], device=model.device))
        assert loc.cpu().tolist() == [0, 4]
    for node in ("x", "y"):
        assert torch.equal(cpu.mechanisms[node].locations, other.mechanisms[node].locations.cpu())
