"""Agreement rule (#313): one valid set per row under structure uncertainty.

Ground truth: a four-machine one-way ring.  Machine i's next status depends on
(m_i, m_{i-1}, rep_i); two loggers each repair one half of the ring (plus 20%
random actions), so many joint input values are never visited.
"""
from __future__ import annotations

import math

import pytest
import torch

from nbn.structure import HypothesisSet, agreement_set
from nbn.uncertainty import parent_row_index
from tests.conftest import available_devices

N = 4
INPUTS = [f"m{i}" for i in range(N)] + [f"rep{i}" for i in range(N)]
TRUE_PARENTS = ("m0", "m3", "rep0")


def p_up(up, left, rep):
    stay = torch.where(left == 1, 0.95, 0.60)
    return torch.where(rep == 1, 0.95, torch.where(up == 1, stay, 0.05))


def _collect(half, n_episodes, horizon, seed):
    gen = torch.Generator().manual_seed(seed)
    up = torch.ones(n_episodes, horizon + 1, N, dtype=torch.long)
    out = {k: [] for k in INPUTS + [f"m{i}'" for i in range(N)]}
    for t in range(horizon):
        broken = (up[:, t, list(half)] == 0).float()
        pick = torch.multinomial(broken + 1e-9, 1, generator=gen).squeeze(-1)
        a = torch.where(broken.sum(-1) > 0, torch.tensor(half)[pick] + 1, 0)
        explore = torch.rand(n_episodes, generator=gen) < 0.2
        a = torch.where(explore, torch.randint(0, N + 1, (n_episodes,), generator=gen), a)
        rep = torch.stack([(a == i + 1).long() for i in range(N)], -1)
        up[:, t + 1] = torch.bernoulli(p_up(up[:, t], up[:, t].roll(1, -1), rep), generator=gen).long()
        for i in range(N):
            out[f"m{i}"].append(up[:, t, i])
            out[f"rep{i}"].append(rep[:, i])
            out[f"m{i}'"].append(up[:, t + 1, i])
    return {k: torch.cat(v) for k, v in out.items()}


def ring_data(n_episodes=300, horizon=30, seed=0, device="cpu"):
    left = _collect((0, 1), n_episodes, horizon, seed * 2 + 1)
    right = _collect((2, 3), n_episodes, horizon, seed * 2 + 2)
    return {k: torch.cat([left[k], right[k]]).to(device) for k in left}


VARIABLES = dict.fromkeys(INPUTS + [f"m{i}'" for i in range(N)], ("discrete", 2))


def truth_on(parents, device="cpu"):
    """True P(m0' | parents) on every row of `parents` (must contain TRUE_PARENTS)."""
    cards = [2] * len(parents)
    grid = torch.cartesian_prod(*[torch.arange(c) for c in cards]).reshape(-1, len(parents))
    col = {p: grid[:, k] for k, p in enumerate(parents)}
    p1 = p_up(col["m0"], col["m3"], col["rep0"])
    return torch.stack([1 - p1, p1], -1).to(device)


def _contains_truth(res):
    truth = truth_on(res.parents, res.set.center.device).to(res.set.center.dtype)
    seen = ~res.set.unseen
    return bool(res.set.row_contains(truth)[seen].all())


@pytest.mark.parametrize("device", available_devices())
def test_true_parents_survive_and_rows_never_visited_get_valid_sets(device):
    data = ring_data(device=device)
    res = agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, delta=0.1, sampling="trajectory")
    assert not res.assumption_refuted
    assert any(set(s) >= set(TRUE_PARENTS) for s in res.survivors)
    assert len(res.discarded) > 0                     # the data do refute wrong candidates
    assert _contains_truth(res)
    # structure buys coverage: joint input values never visited still get a set
    assert res.flat_rows == 2 ** len(INPUTS)
    assert res.flat_with_set > res.flat_visited and res.n_rows_gained > 0
    assert res.level == pytest.approx(0.9)
    assert res.set.guarantee == "pac_under_model"
    assert res.set.center.device.type == torch.device(device).type


def test_bound_below_the_true_in_degree_is_detected_and_falls_back_to_flat():
    data = ring_data()
    res = agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=2, delta=0.1, sampling="trajectory")
    assert res.assumption_refuted and res.survivors == []
    assert res.parents == tuple(INPUTS)                # the flat model
    assert res.n_rows_gained == 0                       # no gain without a valid bound
    assert _contains_truth(res)                         # but still valid: Pa ⊆ inputs


def test_explicit_candidates_and_the_full_input_set():
    data = ring_data(n_episodes=100)
    res = agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, delta=0.1,
                        candidates=[TRUE_PARENTS, ("m0", "m1", "rep0")])
    assert TRUE_PARENTS in res.survivors
    small = ["m0", "m3", "rep0"]
    res = agreement_set(VARIABLES, data, "m0'", small, max_parents=5, delta=0.1)
    assert res.survivors == [tuple(small)] and res.discarded == {}   # no extension to test against


def test_accepts_a_hypothesis_set_for_the_variables():
    data = ring_data(n_episodes=50)
    hs = HypothesisSet(VARIABLES, {"m0'": [TRUE_PARENTS]})
    res = agreement_set(hs, data, "m0'", ["m0", "m3", "rep0", "m1"], max_parents=3, delta=0.1)
    assert res.node == "m0'"


@pytest.mark.parametrize("kwargs, err", [
    ({"inputs": "m0"}, TypeError),
    ({"inputs": ["m0", "m0'"]}, ValueError),
    ({"delta": 0.0}, ValueError),
    ({"max_parents": -1}, ValueError),
    ({"candidates": [("m0", "zzz")]}, ValueError),
    ({"candidates": ["m0"]}, TypeError),
    ({"candidates": []}, ValueError),
])
def test_validation(kwargs, err):
    data = ring_data(n_episodes=10, horizon=5)
    args = {"inputs": INPUTS, "max_parents": 3, "delta": 0.1, **kwargs}
    with pytest.raises(err):
        agreement_set(VARIABLES, data, "m0'", **args)


def test_union_too_large_raises():
    data = ring_data(n_episodes=10, horizon=5)
    with pytest.raises(ValueError, match="max_rows"):
        agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, delta=0.1, max_rows=4)


def test_known_rows_have_radius_at_most_epsilon():
    data = ring_data()
    res = agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, delta=0.1, epsilon=0.2)
    assert bool((res.set.radius[res.known] <= 0.2).all())
    assert bool((~res.set.unseen[res.known]).all())


@pytest.mark.slow
def test_coverage_over_replicates():
    """Truth inside every row's set, and the true superset never discarded, in >= 1 - delta of runs."""
    delta, reps = 0.1, 40
    hits = kept = 0
    for r in range(reps):
        data = ring_data(n_episodes=150, seed=100 + r)
        res = agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, delta=delta,
                            sampling="trajectory")
        kept += int(TRUE_PARENTS in res.survivors)
        hits += int(_contains_truth(res))
    floor = 1 - delta - 3 * math.sqrt(delta * (1 - delta) / reps)
    assert hits / reps >= floor
    assert kept / reps >= floor


def test_row_layout_matches_parent_row_index():
    data = ring_data(n_episodes=100)
    res = agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, delta=0.1)
    grid = torch.cartesian_prod(*[torch.arange(c) for c in res.parent_cards]).reshape(-1, len(res.parents))
    assert torch.equal(parent_row_index(grid, res.parent_cards), torch.arange(grid.shape[0]))
