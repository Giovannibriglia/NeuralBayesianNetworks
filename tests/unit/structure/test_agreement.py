"""Agreement rule (#313): one valid set per row under structure uncertainty.

Ground truth: a four-machine one-way ring.  Machine i's next status depends on
(m_i, m_{i-1}, rep_i); two loggers each repair one half of the ring (plus 20%
random actions), so many joint input values are never visited.
"""
from __future__ import annotations

import itertools
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


def _collect(half, n_episodes, horizon, seed, explore=0.2):
    gen = torch.Generator().manual_seed(seed)
    up = torch.ones(n_episodes, horizon + 1, N, dtype=torch.long)
    out = {k: [] for k in INPUTS + [f"m{i}'" for i in range(N)]}
    for t in range(horizon):
        broken = (up[:, t, list(half)] == 0).float()
        pick = torch.multinomial(broken + 1e-9, 1, generator=gen).squeeze(-1)
        a = torch.where(broken.sum(-1) > 0, torch.tensor(half)[pick] + 1, 0)
        rand = torch.rand(n_episodes, generator=gen) < explore
        a = torch.where(rand, torch.randint(0, N + 1, (n_episodes,), generator=gen), a)
        rep = torch.stack([(a == i + 1).long() for i in range(N)], -1)
        up[:, t + 1] = torch.bernoulli(p_up(up[:, t], up[:, t].roll(1, -1), rep), generator=gen).long()
        for i in range(N):
            out[f"m{i}"].append(up[:, t, i])
            out[f"rep{i}"].append(rep[:, i])
            out[f"m{i}'"].append(up[:, t + 1, i])
    return {k: torch.cat(v) for k, v in out.items()}


def ring_data(n_episodes=300, horizon=30, seed=0, device="cpu", explore=0.2):
    left = _collect((0, 1), n_episodes, horizon, seed * 2 + 1, explore)
    right = _collect((2, 3), n_episodes, horizon, seed * 2 + 2, explore)
    return {k: torch.cat([left[k], right[k]]).to(device) for k in left}


VARIABLES = dict.fromkeys(INPUTS + [f"m{i}'" for i in range(N)], ("discrete", 2))
# The rep_i flags encode one 5-valued action: at most one machine is repaired per step.
FEASIBLE = torch.tensor([r for r in itertools.product(range(2), repeat=len(INPUTS)) if sum(r[N:]) <= 1])


def truth_on(parents, device="cpu"):
    """True P(m0' | parents) on every row of `parents` (must contain TRUE_PARENTS)."""
    cards = [2] * len(parents)
    grid = torch.cartesian_prod(*[torch.arange(c) for c in cards]).reshape(-1, len(parents))
    col = {p: grid[:, k] for k, p in enumerate(parents)}
    p1 = p_up(col["m0"], col["m3"], col["rep0"])
    return torch.stack([1 - p1, p1], -1).to(device)


def _contains_truth(res):
    if not set(TRUE_PARENTS) <= set(res.parents):
        return False            # the true parents were refuted (the delta-failure event): a miss
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
    assert res.flat_rows == 2 ** len(INPUTS)       # default: every joint value of the inputs
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


def test_a_hypothesis_set_supplies_its_candidates():
    data = ring_data(n_episodes=50)
    hs = HypothesisSet(VARIABLES, {"m0'": [TRUE_PARENTS, ("m0", "m1", "rep0")]})
    res = agreement_set(hs, data, "m0'", ["m0", "m3", "rep0", "m1"], max_parents=3, delta=0.1)
    assert res.candidates == [TRUE_PARENTS, ("m0", "m1", "rep0")]


@pytest.mark.parametrize("kwargs, err", [
    ({"inputs": "m0"}, TypeError),
    ({"inputs": ["m0", "m0'"]}, ValueError),
    ({"delta": 0.0}, ValueError),
    ({"max_parents": -1}, ValueError),
    ({"candidates": [("m0", "zzz")]}, ValueError),
    ({"candidates": ["m0"]}, TypeError),
    ({"candidates": []}, ValueError),
    ({"candidates": [TRUE_PARENTS, ("rep0", "m0", "m3")]}, ValueError),   # same family twice
    ({"max_families": 10}, ValueError),
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



def test_level_does_not_depend_on_the_memory_guard():
    data = ring_data(n_episodes=100)
    small = agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, delta=0.1, max_rows=200)
    big = agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, delta=0.1)
    assert small.delta_per_family == big.delta_per_family
    assert small.survivors == big.survivors
    assert small.flat_rows is None and big.flat_rows == 256


def test_flat_with_set_matches_a_dense_projection():
    from nbn.uncertainty import parent_grid

    data = ring_data(n_episodes=100)
    res = agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, delta=0.1)
    full = parent_grid([2] * len(INPUTS))
    proj = parent_row_index(full[:, [INPUTS.index(p) for p in res.parents]], res.parent_cards)
    assert res.flat_with_set == int((~res.set.unseen.cpu())[proj].sum())



@pytest.mark.parametrize("device", available_devices())
def test_structure_buys_coverage_on_feasible_rows(device):
    """Little exploration: some feasible (state, action) rows are never visited, yet get a set."""
    data = ring_data(n_episodes=100, explore=0.02, device=device)
    res = agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, delta=0.1,
                        sampling="trajectory", coverage_rows=FEASIBLE)
    assert res.flat_rows == FEASIBLE.shape[0] == 80
    assert res.flat_visited < res.flat_with_set and res.n_rows_gained > 0
    assert _contains_truth(res)


def test_coverage_rows_validation():
    data = ring_data(n_episodes=10, horizon=5)
    with pytest.raises(ValueError, match="coverage_rows"):
        agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, delta=0.1,
                      coverage_rows=FEASIBLE[:, :3])



# --- review round 2 regressions (#320) ------------------------------------------------------
def test_explicit_candidates_declare_their_own_assumption():
    data = ring_data(n_episodes=50)
    res = agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, delta=0.1,
                        candidates=[("m0", "m1", "rep0")])
    assert res.assumptions[0].startswith("some given candidate contains Pa")


def test_default_family_count_estimate_is_exact():
    """16 inputs, d = 4: C(16,4) + C(16,5) = 6188 families, under the 16384 limit."""
    v = {f"u{i}": ("discrete", 2) for i in range(16)}
    v["y'"] = ("discrete", 2)
    g = torch.Generator().manual_seed(0)
    data = {k: torch.randint(0, 2, (50,), generator=g) for k in v}
    with pytest.raises(ValueError, match="max_rows"):        # gets past the family guard
        agreement_set(v, data, "y'", [f"u{i}" for i in range(16)], max_parents=4, delta=0.1, max_rows=10)


def test_refuted_fallback_without_a_flat_table_reports_no_parents():
    data = ring_data()
    res = agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=2, delta=0.1, max_rows=200)
    assert res.assumption_refuted and res.parents == () and bool(res.set.unseen.all())


def test_repeated_coverage_rows_count_once():
    data = ring_data(n_episodes=100, explore=0.02)
    once = agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, delta=0.1, coverage_rows=FEASIBLE)
    twice = agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, delta=0.1,
                          coverage_rows=torch.cat([FEASIBLE, FEASIBLE[:30]]))
    assert (twice.flat_rows, twice.flat_visited, twice.flat_with_set) == \
        (once.flat_rows, once.flat_visited, once.flat_with_set)


def test_no_budget_share_is_reserved_when_the_flat_model_is_a_family():
    data = ring_data(n_episodes=50)
    res = agreement_set(VARIABLES, data, "m0'", ["m0", "m3", "rep0"], max_parents=5, delta=0.1)
    assert res.n_families == 1 and res.delta_per_family == pytest.approx(0.1)
