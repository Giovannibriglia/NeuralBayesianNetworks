"""``agreement_set(refute=False)``: the row-wise union over a given family of parent sets (#329)."""
from __future__ import annotations

import pytest
import torch

from nbn.structure import HypothesisSet, agreement_set, export_factored_sets, load_factored_sets
from nbn.structure.hypotheses import _reorder_parents
from nbn.uncertainty.builders import l1_ball_set_from_counts
from nbn.uncertainty.counts import count_table_for
from tests.conftest import available_devices
from tests.unit.structure.test_agreement import INPUTS, N, TRUE_PARENTS, VARIABLES, ring_data, truth_on

# An ICP-like family: one superset of the true parents, plus two wrong sets.
FAMILY = [("m0", "m3", "rep0", "m1"), ("m0", "rep0"), ("m1", "m2")]


def _union(data, delta=0.1, family=FAMILY, **kw):
    return agreement_set(VARIABLES, data, "m0'", INPUTS, delta=delta, candidates=family,
                         sampling="trajectory", refute=False, **kw)


def _lift(table, s, union):
    """A candidate's [R_s, C] table on the rows of `union` (C order)."""
    cards = [2] * len(union)
    in_u = tuple(p for p in union if p in s)
    t = _reorder_parents(table, [2] * len(s), s, in_u)
    shape = [2 if p in s else 1 for p in union] + [t.shape[-1]]
    return t.reshape(shape).expand(*cards, t.shape[-1]).reshape(-1, t.shape[-1])


@pytest.mark.parametrize("device", available_devices())
def test_union_contains_every_candidate_ball_and_keeps_every_candidate(device):
    data = ring_data(device=device)
    res = _union(data)
    assert res.survivors == FAMILY and res.discarded == {} and not res.refute
    assert res.parents == tuple(p for p in INPUTS if any(p in c for c in FAMILY))
    # No union bound: every ball at the full delta (the result holds when one superset's ball holds).
    assert res.n_families == len(FAMILY) and res.delta_per_family == pytest.approx(0.1)
    assert res.max_parents is None and not res.assumption_refuted
    assert res.assumptions[0].startswith("some given candidate contains Pa(node)")
    any_unseen = torch.zeros_like(res.set.unseen)
    for c in FAMILY:
        counts, _ = count_table_for(VARIABLES, data, "m0'", c)
        ball = l1_ball_set_from_counts(counts, delta=res.delta_per_family, sampling="trajectory")
        center = _lift(ball.center, c, res.parents)
        radius = _lift(ball.radius.unsqueeze(-1), c, res.parents)[:, 0]
        unseen = _lift(ball.unseen.unsqueeze(-1).float(), c, res.parents)[:, 0] > 0.5
        any_unseen |= unseen
        seen = ~res.set.unseen
        reach = ((center - res.set.center).abs().sum(-1) + radius).clamp_max(2.0)  # radius 2 = simplex
        assert bool((reach[seen] <= res.set.radius[seen] + 1e-5).all())
    assert torch.equal(res.set.unseen, any_unseen)                    # simplex where any candidate is blind


def test_union_contains_the_truth_across_seeds():
    hits = 0
    for seed in range(10):
        res = _union(ring_data(n_episodes=150, seed=seed))
        truth = truth_on(res.parents).to(res.set.center.dtype)
        hits += bool(res.set.row_contains(truth)[~res.set.unseen].all())
    assert hits >= 9                                                  # level 0.9 per run; in practice all


def test_the_agreement_rule_would_discard_the_wrong_candidates_the_union_keeps_them():
    data = ring_data()
    union = _union(data)
    rule = agreement_set(VARIABLES, data, "m0'", INPUTS, delta=0.1, candidates=FAMILY, sampling="trajectory")
    assert rule.refute and set(rule.discarded) == {("m0", "rep0"), ("m1", "m2")}
    assert set(union.survivors) == set(FAMILY)
    # The rule splits delta over its families (candidates and extensions) plus a fallback share;
    # the union does not split it.
    assert rule.n_families > len(FAMILY)
    assert rule.delta_per_family == pytest.approx(0.1 / (rule.n_families + 1))
    assert union.delta_per_family == pytest.approx(0.1)
    assert set(TRUE_PARENTS) <= set(union.parents)


def test_union_needs_the_family():
    data = ring_data(n_episodes=50)
    with pytest.raises(ValueError, match="refute=False needs the family"):
        agreement_set(VARIABLES, data, "m0'", INPUTS, max_parents=3, refute=False)
    hs = HypothesisSet(VARIABLES, {"m0'": FAMILY})
    res = agreement_set(hs, data, "m0'", INPUTS, refute=False)        # candidates from the HypothesisSet
    assert res.survivors == FAMILY


def test_single_candidate_union_is_that_candidate_ball():
    data = ring_data(n_episodes=100)
    res = _union(data, family=[TRUE_PARENTS])
    counts, _ = count_table_for(VARIABLES, data, "m0'", TRUE_PARENTS)
    ball = l1_ball_set_from_counts(counts, delta=0.1, sampling="trajectory")
    assert res.parents == ("m0", "m3", "rep0")                        # order of inputs
    order = _reorder_parents(ball.radius.unsqueeze(-1), [2, 2, 2], TRUE_PARENTS, res.parents)[:, 0]
    torch.testing.assert_close(res.set.radius, order)


def test_export_records_the_mode(tmp_path):
    data = ring_data(n_episodes=100)
    sets = {f"m{i}'": agreement_set(VARIABLES, data, f"m{i}'", INPUTS, delta=0.1 / N, sampling="trajectory",
                                    candidates=[(f"m{i}", f"m{(i - 1) % N}", f"rep{i}"), (f"m{i}",)],
                                    refute=False) for i in range(N)}
    out = export_factored_sets(tmp_path / "u", sets, VARIABLES, [f"m{i}" for i in range(N)],
                               [f"rep{i}" for i in range(N)])
    for f in load_factored_sets(out).manifest["factors"]:
        assert f["agreement"]["refute"] is False and f["agreement"]["max_parents"] is None
        assert f["assumptions"][0].startswith("some given candidate contains Pa(node)")


def test_union_options_pickle_and_schema(tmp_path):
    import json
    import pickle

    from nbn.structure.export import SCHEMA_FILE
    from tests.unit.structure.test_agreement import FEASIBLE

    data = ring_data(n_episodes=100)
    res = _union(data, family=[(), TRUE_PARENTS], coverage_rows=FEASIBLE)
    assert res.parents == TRUE_PARENTS and res.flat_rows == FEASIBLE.shape[0]
    assert res.flat_with_set is not None and res.flat_visited is not None
    back = pickle.loads(pickle.dumps(res))
    assert back.refute is False and torch.equal(back.set.radius, res.set.radius)
    with pytest.raises(ValueError, match="fewer or smaller candidates"):
        _union(data, max_rows=4)
    jsonschema = pytest.importorskip("jsonschema")
    sets = {f"m{i}'": agreement_set(VARIABLES, data, f"m{i}'", INPUTS, delta=0.1 / N, sampling="trajectory",
                                    candidates=[(f"m{i}", f"m{(i - 1) % N}", f"rep{i}"), (f"m{i}",)],
                                    refute=False) for i in range(N)}
    out = export_factored_sets(tmp_path / "s", sets, VARIABLES, [f"m{i}" for i in range(N)],
                               [f"rep{i}" for i in range(N)], exact=True)
    jsonschema.validate(json.loads((out / "manifest.json").read_text()), json.loads(SCHEMA_FILE.read_text()))
